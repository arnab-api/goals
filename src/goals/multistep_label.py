"""Oracle-based step labeling for MultiStepSample generations.

For a (sample, full_prompt, generation) triple, identify which token range in
the full sequence implements which step. Two outputs per step:

- prompt_token_range: where the step text lives in the prompt (literal lookup).
- gen_token_range:    where the generated code implementing that step lives
                      (oracle line-range labeling + offset-mapping conversion).

Both ranges are absolute token indices into the tokenization of
`combined = full_prompt + generation` (add_special_tokens=False).
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from dataclasses_json import DataClassJsonMixin

from src.goals.multistep_data import MultiStepSample
from src.tokens import find_token_range
from src.utils.oracle_llms import ask_claude

logger = logging.getLogger(__name__)


@dataclass
class StepLabel(DataClassJsonMixin):
    step_idx: int
    step_text: str
    prompt_token_range: Optional[tuple[int, int]]   # (start, end) tokens in `combined`
    gen_line_range: Optional[tuple[int, int]]       # 1-based inclusive line range in `generation`
    gen_token_range: Optional[tuple[int, int]]      # (start, end) tokens in `combined`
    gen_code: str                                   # extracted text segment for inspection
    reasoning: str = ""


@dataclass
class LabeledGeneration(DataClassJsonMixin):
    sample_name: str
    full_prompt: str
    generation: str
    labels: list[StepLabel] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Oracle prompt + response parsing.
# ──────────────────────────────────────────────────────────────────────────────

_ORACLE_SYSTEM = (
    "You are a senior software engineer reviewing code written by another LLM. "
    "Your task is to map a list of natural-language instructions onto the line ranges "
    "of a Python program that implements them. Be precise and concise."
)


def build_oracle_prompt(steps: list[str], generation: str) -> str:
    lines = generation.splitlines()
    numbered = "\n".join(f"{i + 1:04d} | {ln}" for i, ln in enumerate(lines))
    steps_str = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))
    return (
        "Below is a numbered list of instructions and a Python program written by an LLM "
        "in response to them. For each instruction, identify the contiguous line range "
        "in the program that implements it.\n\n"
        f"Instructions:\n{steps_str}\n\n"
        f"Program (with 1-based line numbers):\n{numbered}\n\n"
        "Rules:\n"
        "- Use the 1-based line numbers shown on the left. Ranges are inclusive on both ends.\n"
        "- If an instruction is not implemented anywhere, return -1 for both start_line and end_line.\n"
        "- Each instruction maps to a single contiguous block. If implementation is "
        "scattered, return the primary block.\n"
        "- Different instructions may overlap (e.g., a `main` function that calls earlier helpers).\n"
        "- Lines that only contain whitespace, braces, or signatures may be included "
        "if they belong to that step's block.\n\n"
        "Return a single JSON array with one object per instruction, in instruction order:\n"
        "[\n"
        "  {\"step\": 1, \"start_line\": <int>, \"end_line\": <int>, \"reasoning\": \"<one sentence>\"},\n"
        "  ...\n"
        "]\n"
        "Output ONLY the JSON array — no surrounding prose, no markdown fences."
    )


def _extract_json_array(s: str) -> str:
    """Pull the outermost balanced [ ... ] from a string (tolerant of markdown fences)."""
    s = s.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
        s = s.strip()
    start = s.find("[")
    if start == -1:
        raise ValueError(f"no '[' found in oracle response: {s[:200]!r}")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
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
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    raise ValueError(f"unbalanced brackets in oracle response: {s[:200]!r}")


def parse_oracle_response(resp: str, n_steps: int) -> list[dict]:
    arr = _extract_json_array(resp)
    data = json.loads(arr)
    if len(data) != n_steps:
        raise ValueError(f"oracle returned {len(data)} entries, expected {n_steps}")
    return data


# ──────────────────────────────────────────────────────────────────────────────
# Line ↔ char ↔ token conversion.
# ──────────────────────────────────────────────────────────────────────────────


def line_range_to_char_range(
    generation: str, start_line: int, end_line: int
) -> Optional[tuple[int, int]]:
    """1-based inclusive line range → [start_char, end_char) over `generation`."""
    if start_line <= 0 or end_line <= 0:
        return None
    lines = generation.splitlines(keepends=True)
    if start_line > len(lines):
        return None
    end_line = min(end_line, len(lines))
    start_char = sum(len(lines[i]) for i in range(start_line - 1))
    end_char = sum(len(lines[i]) for i in range(end_line))
    return start_char, end_char


def char_range_to_token_range(
    offsets: list[tuple[int, int]], start_char: int, end_char: int
) -> Optional[tuple[int, int]]:
    """[start_char, end_char) → (tok_start, tok_end) — both indices into `offsets`."""
    tok_start = next(
        (j for j, (a, _b) in enumerate(offsets) if a >= start_char), None
    )
    tok_end = next(
        (j for j in range(len(offsets) - 1, -1, -1) if offsets[j][1] <= end_char),
        None,
    )
    if tok_start is None or tok_end is None or tok_start > tok_end:
        return None
    return tok_start, tok_end + 1  # exclusive end


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point.
# ──────────────────────────────────────────────────────────────────────────────


def label_generation(
    mt,
    sample: MultiStepSample,
    full_prompt: str,
    generation: str,
    use_cache: bool = True,
    oracle_max_tokens: int = 4000,
) -> LabeledGeneration:
    """Label which token range in `full_prompt + generation` implements which step.

    Returns:
        LabeledGeneration with one StepLabel per step. Token ranges are absolute
        indices into the tokenization of `full_prompt + generation`
        (add_special_tokens=False).
    """
    combined = full_prompt + generation
    enc = mt.tokenizer(combined, return_offsets_mapping=True, add_special_tokens=False)
    offsets = [tuple(p) for p in enc["offset_mapping"]]

    # 1) Prompt token ranges per step (literal substring lookup).
    prompt_ranges: list[Optional[tuple[int, int]]] = []
    for step_text in sample.steps:
        try:
            r = find_token_range(string=combined, substring=step_text, tokenizer=mt)
            prompt_ranges.append(r)
        except Exception:
            prompt_ranges.append(None)

    # 2) Oracle line-range labeling over the generation portion.
    oracle_prompt = build_oracle_prompt(sample.steps, generation)
    resp = ask_claude(
        oracle_prompt,
        use_cache=use_cache,
        max_tokens=oracle_max_tokens,
        system_prompt=_ORACLE_SYSTEM,
    )
    parsed = parse_oracle_response(resp, n_steps=len(sample.steps))

    # 3) Line range → char range (within `combined`) → token range.
    char_offset = len(full_prompt)
    labels: list[StepLabel] = []
    for i, (step_text, entry, p_range) in enumerate(
        zip(sample.steps, parsed, prompt_ranges)
    ):
        sl = int(entry.get("start_line", -1))
        el = int(entry.get("end_line", -1))
        cr_gen = line_range_to_char_range(generation, sl, el)
        if cr_gen is None:
            gen_token_range = None
            gen_code = ""
            gen_line_range = None
        else:
            start_char = cr_gen[0] + char_offset
            end_char = cr_gen[1] + char_offset
            gen_token_range = char_range_to_token_range(offsets, start_char, end_char)
            gen_code = combined[start_char:end_char]
            gen_line_range = (sl, el)

        labels.append(
            StepLabel(
                step_idx=i,
                step_text=step_text,
                prompt_token_range=p_range,
                gen_line_range=gen_line_range,
                gen_token_range=gen_token_range,
                gen_code=gen_code,
                reasoning=str(entry.get("reasoning", "")),
            )
        )

    return LabeledGeneration(
        sample_name=sample.name,
        full_prompt=full_prompt,
        generation=generation,
        labels=labels,
    )
