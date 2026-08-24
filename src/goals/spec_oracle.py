"""Oracle wrappers: instruction-span + influential-word identification, and mapping selected
code back to the spec(s) it implements.

**Entirely new / additive.** Nothing here is imported by, or mutates, the existing pipelines
(``multistep_label`` / ``condition_data`` / ``anchor`` / ``build_condition_data``). It is safe to
iterate on without touching data-generation or scoring code. The only shared dependencies are the
read-only helpers :func:`src.utils.oracle_llms.ask_claude` and :func:`src.tokens.find_token_range`.

Three independent, individually cached oracle calls (plus a convenience that chains 1→2):

  1. :func:`extract_instruction_spans` — split a *question* into its distinct, actionable
     instructions, returned as VERBATIM substrings (so they can be located in the text).
  2. :func:`extract_influential_words` — the content/influential words of one span, with
     stopwords / articles / glue words dropped.
  3. :func:`identify_spec_for_code` — given a code snippet, which of a list of instructions it
     implements (language-agnostic; no dependence on the Python-only ``multistep_label`` oracle).

  +  :func:`analyze_question` — 1 then 2 per span → ``list[(InstructionSpan, [InfluentialWord])]``.

Char spans are located by literal ``str.find``; token spans (optional) are delegated to
``find_token_range`` when a tokenizer and the exact string the spans live in are supplied — so a
caller can turn any span/word into token indices for gaze/attention work without re-tokenizing.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

from src.utils.oracle_llms import ask_claude

logger = logging.getLogger(__name__)

ORACLE_MODEL = "claude-sonnet-4-6"


# ──────────────────────────────────────────────────────────────────────────────
# Return types
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class InstructionSpan:
    index: int
    text: str                                       # verbatim substring of the question
    char_span: Optional[tuple[int, int]] = None     # (start, end) in the question
    token_span: Optional[tuple[int, int]] = None     # (start, end) in mt's tokenization


@dataclass
class InfluentialWord:
    text: str
    char_span: Optional[tuple[int, int]] = None      # in `locate_in` (or the span) if resolvable
    token_span: Optional[tuple[int, int]] = None      # in mt's tokenization of `locate_in`


@dataclass
class SpecMatch:
    index: int
    instruction: str
    implements: bool
    reasoning: str = ""


# ──────────────────────────────────────────────────────────────────────────────
# Parsing / location helpers (pure)
# ──────────────────────────────────────────────────────────────────────────────
def _strip_fences(resp: str) -> str:
    """Drop a leading ```json / ``` fence if the oracle wrapped its answer in one."""
    m = re.search(r"```(?:json)?\s*(.*?)```", resp, re.S)
    return m.group(1) if m else resp


def _first_json(resp: str, kind: str):
    """Return the first top-level JSON value of `kind` ('array' -> [...], 'object' -> {...})."""
    text = _strip_fences(resp)
    open_ch, close_ch = ("[", "]") if kind == "array" else ("{", "}")
    start = text.find(open_ch)
    if start == -1:
        raise ValueError(f"no '{open_ch}' found in oracle response:\n{resp[:300]!r}")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError(f"unbalanced {open_ch}{close_ch} in oracle response:\n{resp[:300]!r}")


def _char_span(haystack: str, needle: str, occurrence: int = 0) -> Optional[tuple[int, int]]:
    """(start, end) of the `occurrence`-th literal `needle` in `haystack`, or None."""
    if not needle:
        return None
    idx = -1
    for _ in range(occurrence + 1):
        idx = haystack.find(needle, idx + 1)
        if idx == -1:
            return None
    return (idx, idx + len(needle))


def _token_span(mt, string: str, substring: str, occurrence: int = 0) -> Optional[tuple[int, int]]:
    """`find_token_range`, swallowing the not-found ValueError → None. Lazy import keeps this
    module usable (for the pure oracle calls) without a tokenizer on the path."""
    if mt is None or not substring:
        return None
    try:
        from src.tokens import find_token_range
        s, e = find_token_range(string=string, substring=substring, tokenizer=mt,
                                occurrence=occurrence)
        return (int(s), int(e))
    except Exception as exc:  # not present / retokenization drift — location is best-effort
        logger.debug(f"token span for {substring!r} not located: {exc}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# 1. instruction spans from a question
# ──────────────────────────────────────────────────────────────────────────────
_SPANS_SYS = (
    "You are a careful annotator. You split a programming task description into the distinct, "
    "actionable sub-instructions it asks the model to carry out. Return each sub-instruction "
    "EXACTLY as it appears in the text (a verbatim substring), so it can be located again."
)


def _build_spans_prompt(question: str) -> str:
    return (
        "Below is a programming task description. Identify each distinct, actionable "
        "sub-instruction it contains — the individual things the program is asked to compute or "
        "do (e.g. each numbered condition). Do NOT include boilerplate that is not a computation "
        "goal (variable-naming conventions, 'do not write comments', print-formatting rules) "
        "unless nothing else qualifies.\n\n"
        "Return each sub-instruction as a VERBATIM substring of the text below — copy it exactly, "
        "character for character, with no added or removed words.\n\n"
        f"Task description:\n<<<\n{question}\n>>>\n\n"
        "Return a single JSON array of strings, in the order they appear:\n"
        '["<verbatim sub-instruction 1>", "<verbatim sub-instruction 2>", ...]\n'
        "Output only the JSON array."
    )


def extract_instruction_spans(
    question: str,
    *,
    mt=None,
    use_cache: bool = True,
    temperature: float = 0.0,
    model_name: str = ORACLE_MODEL,
    max_tokens: int = 2048,
) -> list[InstructionSpan]:
    """Oracle-split `question` into its distinct actionable instructions.

    Each returned :class:`InstructionSpan` carries the verbatim text, its ``char_span`` in
    `question` (via literal lookup), and — when `mt` is given — its ``token_span`` in the
    tokenization of `question`. Spans the oracle paraphrased (so they no longer match verbatim)
    come back with ``char_span=None``; a warning is logged.
    """
    resp = ask_claude(
        _build_spans_prompt(question),
        use_cache=use_cache, temperature=temperature,
        model_name=model_name, max_tokens=max_tokens, system_prompt=_SPANS_SYS,
    )
    raw = _first_json(resp, "array")
    spans: list[InstructionSpan] = []
    seen_upto = 0  # advance the search cursor so repeated phrasings map to distinct spans
    for i, text in enumerate(raw):
        text = str(text).strip()
        cs = _char_span(question, text)
        if cs is not None and cs[0] < seen_upto:      # prefer the next, non-overlapping occurrence
            nxt = _char_span(question[seen_upto:], text)
            cs = (seen_upto + nxt[0], seen_upto + nxt[1]) if nxt else cs
        if cs is None:
            logger.warning(f"instruction span not found verbatim (oracle paraphrased?): {text!r}")
        else:
            seen_upto = cs[1]
        ts = _token_span(mt, question, text) if cs is not None else None
        spans.append(InstructionSpan(index=i, text=text, char_span=cs, token_span=ts))
    return spans


# ──────────────────────────────────────────────────────────────────────────────
# 2. influential words within a span (stopwords dropped)
# ──────────────────────────────────────────────────────────────────────────────
_WORDS_SYS = (
    "You extract the content-bearing words of a short instruction: the words that carry its "
    "computational meaning. You drop stopwords and glue words (articles, prepositions, "
    "conjunctions, pronouns, auxiliaries, and generic filler like 'the', 'of', 'each', 'that', "
    "'which', 'in', 'to', 'a'). Keep the words a reader would need to understand WHAT is being "
    "computed. Return them exactly as they appear."
)


def _build_words_prompt(span_text: str, context: Optional[str]) -> str:
    ctx = f"For context, the full task is:\n<<<\n{context}\n>>>\n\n" if context else ""
    return (
        f"{ctx}"
        "From the instruction below, list only the CONTENT / influential words — the words that "
        "carry its computational meaning. Drop stopwords, articles, prepositions, conjunctions, "
        "pronouns, auxiliaries, and generic filler.\n\n"
        f"Instruction:\n<<<\n{span_text}\n>>>\n\n"
        "Return each kept word EXACTLY as it appears in the instruction above (same spelling and "
        "case; keep multi-word technical phrases split into their individual words unless a phrase "
        "is a single token like 'case-insensitive').\n\n"
        "Return a single JSON array of strings, in order of appearance. Output only the JSON array."
    )


def extract_influential_words(
    span_text: str,
    *,
    context: Optional[str] = None,
    mt=None,
    locate_in: Optional[str] = None,
    occurrence_base: int = 0,
    use_cache: bool = True,
    temperature: float = 0.0,
    model_name: str = ORACLE_MODEL,
    max_tokens: int = 1024,
) -> list[InfluentialWord]:
    """Oracle-extract the content/influential words of one instruction span (stopwords dropped).

    Args:
        span_text: the instruction span (e.g. one :class:`InstructionSpan`'s ``text``).
        context: optional full-question text, passed to the oracle for disambiguation only.
        mt / locate_in: if both given, each kept word's ``token_span`` is located in `locate_in`'s
            tokenization (typically the full prompt the model sees). ``char_span`` is located in
            `locate_in` if given, else in `span_text`.
        occurrence_base: which occurrence to start matching at inside `locate_in`/`span_text`
            (words are matched left-to-right, cursor-advanced, so duplicates map to distinct hits).
    """
    resp = ask_claude(
        _build_words_prompt(span_text, context),
        use_cache=use_cache, temperature=temperature,
        model_name=model_name, max_tokens=max_tokens, system_prompt=_WORDS_SYS,
    )
    words = [str(w).strip() for w in _first_json(resp, "array") if str(w).strip()]
    hay = locate_in if locate_in is not None else span_text
    out: list[InfluentialWord] = []
    cursor = 0
    for w in words:
        cs = _char_span(hay[cursor:], w)
        cs = (cursor + cs[0], cursor + cs[1]) if cs else None
        if cs:
            cursor = cs[1]
        ts = _token_span(mt, locate_in, w, occurrence=0) if (mt is not None and locate_in) else None
        out.append(InfluentialWord(text=w, char_span=cs, token_span=ts))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 3. which instruction(s) does a code snippet implement?
# ──────────────────────────────────────────────────────────────────────────────
_SPEC_SYS = (
    "You are a senior software engineer. Given a code snippet and a numbered list of candidate "
    "instructions, you decide which instruction(s) the snippet actually implements. Judge by what "
    "the code computes, not by shared surface words. A snippet may implement zero, one, or "
    "several instructions."
)


def _build_spec_prompt(code: str, instructions: list[str]) -> str:
    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(instructions))
    return (
        "Candidate instructions:\n"
        f"{numbered}\n\n"
        "Code snippet:\n<<<\n"
        f"{code}\n"
        ">>>\n\n"
        "For each instruction, say whether THIS snippet implements it. Judge on behaviour, not on "
        "shared keywords.\n\n"
        "Return a single JSON array with one object per instruction, in order:\n"
        '[{"instruction": 1, "implements": true, "reasoning": "<one sentence>"}, ...]\n'
        "Output only the JSON array."
    )


def identify_spec_for_code(
    code: str,
    instructions: list[str],
    *,
    use_cache: bool = True,
    temperature: float = 0.0,
    model_name: str = ORACLE_MODEL,
    max_tokens: int = 2048,
) -> list[SpecMatch]:
    """Oracle-decide which of `instructions` the `code` snippet implements.

    Language-agnostic and self-contained (does not use the Python-only ``multistep_label``
    oracle). Returns one :class:`SpecMatch` per instruction; filter on ``.implements``.
    """
    resp = ask_claude(
        _build_spec_prompt(code, instructions),
        use_cache=use_cache, temperature=temperature,
        model_name=model_name, max_tokens=max_tokens, system_prompt=_SPEC_SYS,
    )
    raw = _first_json(resp, "array")
    out: list[SpecMatch] = []
    for i, instr in enumerate(instructions):
        entry = raw[i] if i < len(raw) else {}
        out.append(SpecMatch(
            index=i, instruction=instr,
            implements=bool(entry.get("implements", False)),
            reasoning=str(entry.get("reasoning", "")),
        ))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# convenience: 1 → 2 per span
# ──────────────────────────────────────────────────────────────────────────────
def analyze_question(
    question: str,
    *,
    mt=None,
    locate_words_in: Optional[str] = None,
    use_cache: bool = True,
    temperature: float = 0.0,
    model_name: str = ORACLE_MODEL,
) -> list[tuple[InstructionSpan, list[InfluentialWord]]]:
    """Chain: split `question` into spans, then extract each span's influential words.

    `locate_words_in` (default: `question`) is the string the words' token/char spans are located
    in — pass the full prompt the model sees if you want token indices aligned to a forward pass.
    """
    spans = extract_instruction_spans(
        question, mt=mt, use_cache=use_cache, temperature=temperature, model_name=model_name)
    hay = locate_words_in if locate_words_in is not None else question
    result: list[tuple[InstructionSpan, list[InfluentialWord]]] = []
    for sp in spans:
        words = extract_influential_words(
            sp.text, context=question, mt=mt, locate_in=hay,
            use_cache=use_cache, temperature=temperature, model_name=model_name)
        result.append((sp, words))
    return result
