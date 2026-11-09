"""Evaluation utilities for the q-transfer and pivot-onwards KV-transfer interventions.

Given an instance (every template exposes ``expected_next_desc`` = SRC target sub-goal, ``dst_next_desc`` = the
sub-goal clean DST would pursue, ``task_desc`` = the overall task), three complementary evals of one intervention:

1. CAUSAL (what we actually care about) — did the query transplant make the patched DST pursue
   the SRC target (or a non-trivial part)? Oracle judgement over the patched code continuation,
   mirroring the goalscope eval.  → :func:`judge_causal`.

2. READOUT — did the goalscope readout shift DST-target → SRC-target? All three readouts
   (source_run / destination_run / patched_run) are classified against the two candidate
   sub-goals; a case is VALID only if the clean readouts land where expected (source→SRC,
   dest→DST), and SHIFTED if the patched readout expresses the SRC target.  → :func:`eval_readouts`.

3. GAZE — after the patch, does value-weighted attention gaze more at the SRC-target span than the
   DST-target span? Rule-based: ``gaze(SRC) > gaze(DST)`` on the absolute patched attention, with the
   clean attention as a credibility baseline (should prefer DST).  → :func:`gaze_shift`.

Oracle prompts + parsing live here (single source of truth). The oracle is Claude via
``src.utils.oracle_llms.ask_claude`` (disk-cached). Classification uses neutral A/B labels to
avoid position/name bias, mapped back to "src"/"dst".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

CONCEPT_SYSTEM = (
    "You are a careful evaluator for a mechanistic-interpretability experiment on a code-writing "
    "language model. Judge only what the given text actually expresses; do not speculate. Always "
    "answer in the exact tag format requested."
)


def _concept_prompt(text: str, a_desc: str, b_desc: str, task_desc: str, kind: str) -> str:
    return f"""A language model is writing code for this task:
  TASK: {task_desc}

Two candidate sub-goals it might pursue next:
  (A): {a_desc}
  (B): {b_desc}

Below is {kind}. Decide which sub-goal it expresses / pursues. Treat a non-trivial partial
implementation or an explicit mention as pursuing that sub-goal.

IMPORTANT — judge by the actual COMPUTATION / logic the code performs, NOT by variable, function, or
identifier names. Names can be arbitrary, autocompleted, or outright misleading, so a statement that
*computes* sub-goal X is pursuing X even when its variable is named after something else (e.g.
`num_digits = (n < 0) ? -n : n;` computes an absolute value — that is the abs-value sub-goal, regardless
of the name `num_digits`).

--- BEGIN ---
{text}
--- END ---

Classify as exactly one of:
  A        - expresses/pursues (A), not (B)
  B        - expresses/pursues (B), not (A)
  BOTH     - clearly pursues both
  NEITHER  - neither (unrelated, empty, or generic)

Answer with: <verdict>A|B|BOTH|NEITHER</verdict> then a one-line <why>...</why>."""


def _parse_verdict(text: str) -> tuple[str, str]:
    """Extract the verdict tag. If no well-formed <verdict>A|B|BOTH|NEITHER</verdict> is present
    (e.g. the oracle ran out of tokens before emitting it, or drifted from the format), return the
    sentinel ``PARSE_ERROR`` — a *pipeline* failure, distinct from a genuine NEITHER judgement — so
    such cases can be surfaced and re-run rather than silently counted as neither."""
    m = re.search(r"<verdict>\s*(A|B|BOTH|NEITHER)\s*</verdict>", text, re.I)
    v = m.group(1).upper() if m else "PARSE_ERROR"
    w = re.search(r"<why>\s*(.*?)\s*</why>", text, re.I | re.S)
    return v, (w.group(1).strip() if w else "")


_A2LABEL = {"A": "src", "B": "dst", "BOTH": "both", "NEITHER": "neither",
            "PARSE_ERROR": "parse_error"}


def classify_concept(
    text: str,
    src_desc: str,
    dst_desc: str,
    task_desc: str,
    kind: str = "code the model generated",
    model_name: str = "claude-sonnet-4-6",
    use_cache: bool = True,
) -> dict:
    """Oracle-classify ``text`` as expressing the SRC sub-goal, the DST sub-goal, both, or neither.
    Returns ``{"label": "src"|"dst"|"both"|"neither", "why": str, "raw": str}``. (A)=SRC, (B)=DST."""
    if not text or not text.strip():
        return {"label": "neither", "why": "empty", "raw": ""}
    from src.utils.oracle_llms import ask_claude

    try:
        raw = ask_claude(
            _concept_prompt(text, src_desc, dst_desc, task_desc, kind),
            system_prompt=CONCEPT_SYSTEM,
            max_tokens=768,
            use_cache=use_cache,
            model_name=model_name,
        )
    except ConnectionError as e:
        # the API declines some degenerate texts (e.g. a readout of `0x0 0x0 ...`) with stop_reason=refusal and no
        # content, deterministically; such text expresses neither sub-goal, so score it a miss instead of crashing
        if "stop_reason=refusal" not in str(e):
            raise
        return {"label": "neither", "why": "judge refused (degenerate text)", "raw": "", "refused": True}
    v, why = _parse_verdict(raw)
    return {"label": _A2LABEL[v], "why": why, "raw": raw}


# ══════════════════════════════════════════════════════════════════════════════
# 1. causal success — oracle over the patched code continuation
# ══════════════════════════════════════════════════════════════════════════════
def judge_causal(inst, patched_gen: str, **kw) -> dict:
    """Did the patched DST's IMMEDIATE next computation jump to the SRC target? We judge only the
    FIRST computation/statement of the continuation — after producing SRC's target the model may go
    back and do the other conditions, and the intervention is NOT penalized for that. Returns the
    classification dict plus ``triggered`` (label == src; a BOTH verdict is NOT a hit)."""
    r = classify_concept(
        patched_gen, inst.expected_next_desc, inst.dst_next_desc, inst.task_desc,
        kind="the code the model generated to continue the function after the pivot. Judge ONLY the "
             "FIRST computation/statement it performs (ignore anything it does afterwards)", **kw)
    r["triggered"] = r["label"] == "src"          # BOTH is not a clean shift to the target
    return r


# ══════════════════════════════════════════════════════════════════════════════
# 1b. joint B/C/D classifier — a single MUTUALLY-EXCLUSIVE verdict
# ══════════════════════════════════════════════════════════════════════════════
# For the ``c_in_dst`` (target-present) setup, both the SRC target (C) and the DST progress-slot (D)
# are available in DST, so the two independent binary calls (causal=D-vs-B, literal=C-vs-B) can BOTH
# fire on one generation and the marginals need not sum to <=1. This single call sorts each generation
# into exactly one of {native (B), src_target (C), diff_target (D), neither}, so the shares partition.

_J2LABEL = {"1": "native", "2": "src_target", "3": "diff_target",
            "NEITHER": "neither", "PARSE_ERROR": "parse_error"}


def _joint_prompt(text: str, native_desc: str, src_desc: str, diff_desc: str,
                  task_desc: str, kind: str) -> str:
    return f"""A language model is writing code for this task:
  TASK: {task_desc}

Three candidate sub-goals it might pursue next:
  (1): {native_desc}
  (2): {src_desc}
  (3): {diff_desc}

Below is {kind}. Decide which ONE of the three sub-goals it expresses / pursues. Treat a non-trivial
partial implementation or an explicit mention as pursuing that sub-goal.

IMPORTANT — judge by the actual COMPUTATION / logic the code performs, NOT by variable, function, or
identifier names. Names can be arbitrary, autocompleted, or outright misleading, so a statement that
*computes* sub-goal X is pursuing X even when its variable is named after something else (e.g.
`num_digits = (n < 0) ? -n : n;` computes an absolute value regardless of the name `num_digits`).

--- BEGIN ---
{text}
--- END ---

Classify as exactly one of:
  1        - pursues (1), not (2) or (3)
  2        - pursues (2), not (1) or (3)
  3        - pursues (3), not (1) or (2)
  NEITHER  - none of the three (unrelated, empty, or generic)
If it clearly computes more than one, pick the one it performs FIRST.

Answer with: <verdict>1|2|3|NEITHER</verdict> then a one-line <why>...</why>."""


def _parse_joint(text: str) -> tuple[str, str]:
    m = re.search(r"<verdict>\s*(1|2|3|NEITHER)\s*</verdict>", text, re.I)
    v = m.group(1).upper() if m else "PARSE_ERROR"
    w = re.search(r"<why>\s*(.*?)\s*</why>", text, re.I | re.S)
    return v, (w.group(1).strip() if w else "")


def classify_joint(text: str, native_desc: str, src_desc: str, diff_desc: str, task_desc: str,
                   kind: str = "code the model generated", model_name: str = "claude-sonnet-4-6",
                   use_cache: bool = True) -> dict:
    """Single mutually-exclusive verdict: does ``text`` pursue the native (B), src_target (C), or
    diff_target (D) sub-goal (or neither)? Returns ``{"label": JointLabel, "why": str, "raw": str}``."""
    if not text or not text.strip():
        return {"label": "neither", "why": "empty", "raw": ""}
    from src.utils.oracle_llms import ask_claude

    raw = ask_claude(
        _joint_prompt(text, native_desc, src_desc, diff_desc, task_desc, kind),
        system_prompt=CONCEPT_SYSTEM, max_tokens=768, use_cache=use_cache, model_name=model_name,
    )
    v, why = _parse_joint(raw)
    return {"label": _J2LABEL[v], "why": why, "raw": raw}


def judge_joint(inst, patched_gen: str, **kw) -> dict:
    """Joint B/C/D verdict over the patched DST continuation (judging only its FIRST computation).
    ``label`` is exactly one of native (B) / src_target (C) / diff_target (D) / neither."""
    return classify_joint(
        patched_gen,
        native_desc=inst.dst_next_desc,      # B — DST's own next
        src_desc=inst.src_target_desc,       # C — SRC transplant target (present in DST for c_in_dst)
        diff_desc=inst.expected_next_desc,   # D — DST progress-slot (the DIFF-predicted outcome)
        task_desc=inst.task_desc,
        kind="the code the model generated to continue the function after the pivot. Judge ONLY the "
             "FIRST computation/statement it performs (ignore anything it does afterwards)",
        **kw,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 2. readout shift — classify all three goalscope readouts
# ══════════════════════════════════════════════════════════════════════════════
def eval_readouts(inst, source_readout: str, dest_readout: str, patched_readout: str, **kw) -> dict:
    """Classify the three goalscope readouts against the SRC/DST sub-goals.

    valid   = clean readouts landed as expected (source→SRC, dest→DST) — the case is analyzable.
    shifted = patched readout expresses the SRC target (the intervention moved the readout)."""
    kind = "a short natural-language phrase the model 'read out' internally"
    src = classify_concept(source_readout, inst.expected_next_desc, inst.dst_next_desc, inst.task_desc, kind=kind, **kw)
    dst = classify_concept(dest_readout, inst.expected_next_desc, inst.dst_next_desc, inst.task_desc, kind=kind, **kw)
    pat = classify_concept(patched_readout, inst.expected_next_desc, inst.dst_next_desc, inst.task_desc, kind=kind, **kw)
    valid = src["label"] == "src" and dst["label"] == "dst"
    shifted = pat["label"] == "src"               # BOTH is not a clean shift to the target
    return {"source": src["label"], "dest": dst["label"], "patched": pat["label"],
            "valid": valid, "shifted": shifted, "details": {"source": src, "dest": dst, "patched": pat}}


# ══════════════════════════════════════════════════════════════════════════════
# 3. gaze shift — rule-based over the value-weighted attention delta
# ══════════════════════════════════════════════════════════════════════════════
def target_spans(inst, prompt: str, ctx=None) -> dict:
    """Token spans (absolute, [lo, hi)) of the SRC-target and DST-target descriptions in ``prompt``."""
    from src.nrgs.functional import token_span

    return {"src": token_span(prompt, inst.expected_next_desc, ctx=ctx),
            "dst": token_span(prompt, inst.dst_next_desc, ctx=ctx)}


def gaze_shift(attn_patched, attn_clean, task_positions, src_span, dst_span) -> dict:
    """After the patch, does the pivot gaze MORE at the SRC-target span than the DST-target span?

    Primary verdict ``shifted`` = ``gaze(SRC.leftout) > gaze(DST.leftout)`` on the **absolute** patched
    value-weighted pivot→TASK attention (aligned to ``task_positions``). The earlier delta>0 rule fired
    on ~everything and wasn't discriminative; the absolute SRC-vs-DST comparison is. ``clean_prefers_src``
    is the same comparison on the **clean** attention and is the credibility baseline: it should be rare
    (clean DST gazes at its own frontier), so ``clean_prefers_src≈0 → shifted≈1`` is evidence the *patch*
    moved the gaze. ``src_span``/``dst_span`` are absolute token ranges."""
    idx = {p: i for i, p in enumerate(task_positions)}

    def mean_on(arr, span):
        ii = [idx[p] for p in range(span[0], span[1]) if p in idx]
        return float(np.mean([arr[i] for i in ii])) if ii else float("nan")

    ap, ac = np.asarray(attn_patched, dtype=float), np.asarray(attn_clean, dtype=float)
    src_p, dst_p = mean_on(ap, src_span), mean_on(ap, dst_span)
    src_c, dst_c = mean_on(ac, src_span), mean_on(ac, dst_span)
    return {"src_patched": src_p, "dst_patched": dst_p,
            "shifted": bool(src_p > dst_p), "margin": src_p - dst_p,
            "src_clean": src_c, "dst_clean": dst_c,
            "clean_prefers_src": bool(src_c > dst_c)}


# ══════════════════════════════════════════════════════════════════════════════
# bundle
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class QTransferEval:
    gaze: Optional[dict] = None         # gaze_shift(...)
    meta: dict = field(default_factory=dict)


def eval_q_transfer(inst, gaze: dict) -> QTransferEval:
    """The gaze eval: ``gaze`` = {attn_patched, attn_clean, task_positions, prompt, ctx?} -> gaze shift (spans
    located in ``prompt`` via the instance's target descriptions)."""
    out = QTransferEval(meta={"datatype": inst.meta.get("datatype"),
                              "expected_next": inst.expected_next})
    sp = target_spans(inst, gaze["prompt"], ctx=gaze.get("ctx"))
    out.gaze = gaze_shift(gaze["attn_patched"], gaze["attn_clean"],
                          gaze["task_positions"], sp["src"], sp["dst"])
    return out
