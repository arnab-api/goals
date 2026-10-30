"""Heuristics for proactive-strategy mining.

Given a code generation that the LM produced *without* explicit strategy
pressure in the prompt (e.g. an instruction that does not contain
"recursively" / "iteratively"), classify the strategy chosen by the model
and locate the token positions where the strategy is materialised in the
generation.

Currently supports the iterative-vs-recursive concept for Python code.
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from typing import Literal, Optional

logger = logging.getLogger(__name__)


PromptBias = Literal["neutral", "recursive", "iterative"]
Strategy = Literal["recursive", "iterative", "Unsure"]

_BIAS_RECURSIVE = re.compile(
    r"\b(recursive|recursively|recursion)\b", flags=re.IGNORECASE
)
_BIAS_ITERATIVE = re.compile(
    r"\b(iterative|iteratively|loop|loops)\b", flags=re.IGNORECASE
)


def detect_prompt_bias(instruction: str) -> PromptBias:
    """Heuristically detect whether the instruction biases the LM toward a strategy.

    Returns ``"recursive"`` / ``"iterative"`` / ``"neutral"``.  Both flags
    matching ⇒ ``"neutral"`` (the prompt mentions both, no clear bias).
    """
    has_rec = bool(_BIAS_RECURSIVE.search(instruction))
    has_iter = bool(_BIAS_ITERATIVE.search(instruction))
    if has_rec and not has_iter:
        return "recursive"
    if has_iter and not has_rec:
        return "iterative"
    return "neutral"


# ---------------------------------------------------------------------------
# Recursive-call locator
# ---------------------------------------------------------------------------


@dataclass
class RecursiveCall:
    """A single recursive-call occurrence in a code block.

    Attributes
    ----------
    name : The function name that is called (== ``function_name`` for direct
           self-recursion; helper name for nested-helper recursion).
    name_char_span : ``(start, end)`` character offsets of the *name* token in
                     the code string passed to the locator.
    name_token_positions : optional list of token indices (within the
                           generation_token_ids) covered by the name span;
                           populated by ``attach_token_positions``.
    full_call_char_span : optional ``(start, end)`` of the full call expression
                          including the matched parentheses; populated on
                          demand by ``attach_full_call_span``.
    """

    name: str
    name_char_span: tuple[int, int]
    name_token_positions: Optional[list[int]] = None
    full_call_char_span: Optional[tuple[int, int]] = None


# ---------------------------------------------------------------------------
# AST locator (Python — preferred path)
# ---------------------------------------------------------------------------


def _ast_name_char_span(code: str, node: ast.Call) -> Optional[tuple[int, int]]:
    """Compute character offsets of the name portion of an ``ast.Call`` node.

    Works only when ``node.func`` is an ``ast.Name`` (direct call by bare
    name).  Uses the node's ``lineno`` / ``col_offset`` plus ``len(name)`` to
    derive the span in the original code string.
    """
    if not isinstance(node.func, ast.Name):
        return None
    name = node.func.id
    # Convert (lineno, col_offset) — 1-based line, 0-based col — to absolute idx.
    lines = code.splitlines(keepends=True)
    if node.func.lineno - 1 >= len(lines):
        return None
    start = sum(len(l) for l in lines[: node.func.lineno - 1]) + node.func.col_offset
    return (start, start + len(name))


def find_recursive_calls_python_ast(
    code: str, function_name: str
) -> Optional[list[RecursiveCall]]:
    """AST-based recursive-call locator for Python.

    Covers
    ------
    * Direct self-recursion: outer function calls itself by name.
    * Nested-helper recursion: an inner function defined *inside* the outer
      function calls itself by name (e.g. a ``_helper`` that recurses).

    Does **not** cover
    ------------------
    * External sibling helpers — top-level ``def _helper(...)`` next to the
      outer function — even if they recurse.
    * Mutual recursion ( A → B → A ).

    Returns ``None`` if the code does not parse as valid Python (caller can
    then fall back to ``find_recursive_calls_regex``).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    func_def = None
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_name
        ):
            func_def = node
            break
    if func_def is None:
        return []

    calls: list[RecursiveCall] = []

    # Direct self-recursion: any Call to function_name anywhere inside the body.
    for node in ast.walk(func_def):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == function_name:
                span = _ast_name_char_span(code, node)
                if span is not None:
                    calls.append(RecursiveCall(name=function_name, name_char_span=span))

    # Nested-helper recursion: for each inner def, look for self-calls within it.
    for node in ast.walk(func_def):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name != function_name
        ):
            helper_name = node.name
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name):
                    if inner.func.id == helper_name:
                        span = _ast_name_char_span(code, inner)
                        if span is not None:
                            calls.append(
                                RecursiveCall(
                                    name=helper_name, name_char_span=span
                                )
                            )

    return calls


# ---------------------------------------------------------------------------
# Regex fallback (used when AST parsing fails — typically truncated generations)
# ---------------------------------------------------------------------------


def _strip_python_comments_and_docstrings(code: str) -> str:
    """Remove ``# …`` comments and triple-quoted strings used as docstrings.

    Robust against malformed code (no AST involved).
    """
    # Triple-quoted strings (greedy match the shortest closing run).
    code = re.sub(r'""".*?"""', "", code, flags=re.DOTALL)
    code = re.sub(r"'''.*?'''", "", code, flags=re.DOTALL)
    # Line comments.
    code = re.sub(r"#[^\n]*", "", code)
    return code


def find_recursive_calls_regex(
    code: str, function_name: str
) -> list[RecursiveCall]:
    """Regex-only recursive-call locator (used when AST parsing fails).

    Detects calls to ``function_name`` after the ``def`` line of the outer
    function.  Does **not** detect nested-helper or external-helper
    recursion in this fallback path — the helper's name is unknown without
    AST parsing.  Truncated generations that contain at least one
    ``function_name(...)`` call after the signature will still be flagged.
    """
    # Skip the def-line by anchoring after the first newline after `def name(`.
    def_match = re.search(rf"\bdef\s+{re.escape(function_name)}\s*\(", code)
    if def_match is None:
        body_start = 0
    else:
        nl = code.find("\n", def_match.end())
        body_start = nl + 1 if nl != -1 else def_match.end()

    body = code[body_start:]
    body_clean = _strip_python_comments_and_docstrings(body)

    # We cannot reliably map char offsets through the stripped body, so we
    # re-search the original body and only count a hit if the same hit
    # survives in the cleaned body (cheap proxy for "not in a comment").
    calls: list[RecursiveCall] = []
    pattern = re.compile(rf"\b{re.escape(function_name)}\s*\(")
    n_clean_hits = len(pattern.findall(body_clean))
    if n_clean_hits == 0:
        return calls

    raw_hits = list(pattern.finditer(body))
    # Take the last n_clean_hits raw hits — heuristic: comments tend to be
    # at the top of the function, so trailing matches survive stripping.
    # This is fragile but only used in the malformed-generation fallback.
    chosen = raw_hits[-n_clean_hits:]
    for m in chosen:
        name_start = m.start()
        name_end = name_start + len(function_name)
        calls.append(
            RecursiveCall(
                name=function_name,
                name_char_span=(body_start + name_start, body_start + name_end),
            )
        )
    return calls


# ---------------------------------------------------------------------------
# Loop detection (separate from recursion — used for strategy classification)
# ---------------------------------------------------------------------------


_PY_LOOP_PATTERN = re.compile(r"^\s*(for|while)\b", flags=re.MULTILINE)


def has_python_loop(code: str) -> bool:
    """Detect a ``for`` or ``while`` loop anywhere in the code.

    Uses a regex fallback if AST parsing fails.  Lightweight on purpose —
    we only need a boolean for strategy classification.
    """
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                return True
        return False
    except SyntaxError:
        clean = _strip_python_comments_and_docstrings(code)
        return bool(_PY_LOOP_PATTERN.search(clean))


# ---------------------------------------------------------------------------
# Top-level dispatcher
# ---------------------------------------------------------------------------


@dataclass
class StrategyLabel:
    strategy: Strategy
    has_recursion: bool
    has_loop: bool
    recursive_calls: list[RecursiveCall] = field(default_factory=list)
    used_fallback: bool = False  # True when AST failed and regex was used


def label_python_strategy(code: str, function_name: str) -> StrategyLabel:
    """Classify recursive-vs-iterative for a Python code string.

    Tries AST first; falls back to regex on syntax error (e.g. truncated
    generations).  Priority: recursive > iterative > Unsure.
    """
    calls = find_recursive_calls_python_ast(code, function_name)
    used_fallback = False
    if calls is None:
        used_fallback = True
        calls = find_recursive_calls_regex(code, function_name)

    has_loop = has_python_loop(code)
    has_recursion = len(calls) > 0

    if has_recursion:
        strategy: Strategy = "recursive"
    elif has_loop:
        strategy = "iterative"
    else:
        strategy = "Unsure"

    return StrategyLabel(
        strategy=strategy,
        has_recursion=has_recursion,
        has_loop=has_loop,
        recursive_calls=calls,
        used_fallback=used_fallback,
    )


# ---------------------------------------------------------------------------
# Char-span → token-position mapping
# ---------------------------------------------------------------------------


def per_token_char_offsets(tokenizer, token_ids: list[int]) -> list[tuple[int, int]]:
    """Compute character offsets of each token in the decoded string.

    Decoded string is ``tokenizer.decode(token_ids, skip_special_tokens=True)``.
    Returned offsets ``(start, end)`` are inclusive-exclusive; ``end_i ==
    start_{i+1}`` always holds.  Uses cumulative decoding — O(N^2) in token
    count but robust to BPE / subword merging that plain per-token decoding
    cannot handle.
    """
    offsets: list[tuple[int, int]] = []
    prev_len = 0
    for i in range(1, len(token_ids) + 1):
        text = tokenizer.decode(token_ids[:i], skip_special_tokens=True)
        offsets.append((prev_len, len(text)))
        prev_len = len(text)
    return offsets


def char_span_to_token_positions(
    char_span: tuple[int, int],
    per_token_offsets: list[tuple[int, int]],
) -> list[int]:
    """Find token indices whose char span overlaps ``char_span``.

    A token overlaps the char span iff ``tok_start < char_end`` and
    ``tok_end > char_start``.  Used to map a name char span to its
    (typically 1–2) underlying token positions.
    """
    c_start, c_end = char_span
    return [
        i
        for i, (t_start, t_end) in enumerate(per_token_offsets)
        if t_start < c_end and t_end > c_start
    ]


def attach_token_positions(
    calls: list[RecursiveCall],
    tokenizer,
    generation_token_ids: list[int],
    code_offset_in_generation: int = 0,
) -> None:
    """Populate ``call.name_token_positions`` for each call in-place.

    ``code_offset_in_generation`` is the character offset at which the
    *code block* (passed to the locator) begins inside the decoded
    generation string.  When the code was extracted from a ``<code>…</code>``
    block, pass the offset of the ``<code>`` opening tag's content; when the
    code is the raw generation, pass ``0``.
    """
    if not calls:
        return
    offsets = per_token_char_offsets(tokenizer, generation_token_ids)
    for call in calls:
        c_start, c_end = call.name_char_span
        shifted = (c_start + code_offset_in_generation, c_end + code_offset_in_generation)
        call.name_token_positions = char_span_to_token_positions(shifted, offsets)


def code_offset_in_generation(generation: str, extracted_code: str) -> int:
    """Return the character offset at which ``extracted_code`` starts inside
    the full ``generation`` string.  Falls back to ``0`` if the substring is
    not found (shouldn't happen if extracted_code came from generation, but
    we guard for stripped whitespace).
    """
    idx = generation.find(extracted_code)
    if idx >= 0:
        return idx
    # Try a relaxed match (strip + first 64 chars) — rare BPE artefacts only.
    head = extracted_code.strip()[:64]
    idx = generation.find(head)
    return max(idx, 0)


# ---------------------------------------------------------------------------
# Convenience: extract code with tag fallback
# ---------------------------------------------------------------------------


_CODE_TAG_RE = re.compile(r"<code>(.*?)</code>", re.DOTALL)


def extract_code_with_fallback(generation: str) -> tuple[str, bool]:
    """Return ``(code, used_fallback)``.

    Tries to extract content from the first ``<code>…</code>`` block.  If no
    closing tag is present (truncated generation), falls back to the text
    after the opening ``<code>`` tag, or to the raw generation if neither
    tag is present.  ``used_fallback`` is True whenever the extraction was
    not a clean closed-tag match.
    """
    m = _CODE_TAG_RE.search(generation)
    if m is not None:
        return m.group(1).strip(), False

    open_idx = generation.find("<code>")
    if open_idx >= 0:
        return generation[open_idx + len("<code>"):].strip(), True

    return generation.strip(), True
