"""Evaluation utilities for the DIFF hypothesis — q-transfer (Spec-1) for now.

Given a `QTransferInstance` / `FilterTransferInstance` (both expose ``expected_next_desc`` =
SRC target sub-goal, ``dst_next_desc`` = the sub-goal clean DST would pursue, ``task_desc`` =
the overall task), three complementary evals of one intervention:

1. CAUSAL (what we actually care about) — did the query transplant make the patched DST pursue
   the SRC target (or a non-trivial part)? Oracle judgement over the patched code continuation,
   mirroring the goalscope eval.  → :func:`judge_causal`.

2. READOUT — did the goalscope readout shift DST-target → SRC-target? All three readouts
   (source_run / destination_run / patched_run) are classified against the two candidate
   sub-goals; a case is VALID only if the clean readouts land where expected (source→SRC,
   dest→DST), and SHIFTED if the patched readout expresses the SRC target.  → :func:`eval_readouts`.

3. GAZE — did value-weighted attention move toward the SRC-target tokens? Rule-based: compare the
   patched−clean attention delta on the SRC-target span vs the DST-target span.  → :func:`gaze_shift`.

Oracle prompts + parsing live here (single source of truth). The oracle is Claude via
``src.utils.oracle_llms.ask_claude`` (disk-cached). Classification uses neutral A/B labels to
avoid position/name bias, mapped back to "src"/"dst".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

Label = Literal["src", "dst", "both", "neither"]

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
    m = re.search(r"<verdict>\s*(A|B|BOTH|NEITHER)\s*</verdict>", text, re.I)
    v = m.group(1).upper() if m else "NEITHER"
    w = re.search(r"<why>\s*(.*?)\s*</why>", text, re.I | re.S)
    return v, (w.group(1).strip() if w else "")


_A2LABEL = {"A": "src", "B": "dst", "BOTH": "both", "NEITHER": "neither"}


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

    raw = ask_claude(
        _concept_prompt(text, src_desc, dst_desc, task_desc, kind),
        system_prompt=CONCEPT_SYSTEM,
        max_tokens=300,
        use_cache=use_cache,
        model_name=model_name,
    )
    v, why = _parse_verdict(raw)
    return {"label": _A2LABEL[v], "why": why, "raw": raw}


# ══════════════════════════════════════════════════════════════════════════════
# 1. causal success — oracle over the patched code continuation
# ══════════════════════════════════════════════════════════════════════════════
def judge_causal(inst, patched_gen: str, **kw) -> dict:
    """Did the patched DST continuation pursue the SRC target (or a non-trivial part)? Returns the
    classification dict plus ``triggered`` (label in {src, both})."""
    r = classify_concept(
        patched_gen, inst.expected_next_desc, inst.dst_next_desc, inst.task_desc,
        kind="the code the model generated to continue the function (after the pivot)", **kw)
    r["triggered"] = r["label"] in ("src", "both")
    return r


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
    valid = src["label"] in ("src", "both") and dst["label"] in ("dst", "both")
    shifted = pat["label"] in ("src", "both")
    return {"source": src["label"], "dest": dst["label"], "patched": pat["label"],
            "valid": valid, "shifted": shifted, "details": {"source": src, "dest": dst, "patched": pat}}


# ══════════════════════════════════════════════════════════════════════════════
# 3. gaze shift — rule-based over the value-weighted attention delta
# ══════════════════════════════════════════════════════════════════════════════
def target_spans(inst, prompt: str, ctx=None) -> dict:
    """Token spans (absolute, [lo, hi)) of the SRC-target and DST-target descriptions in ``prompt``."""
    from src.diff.functional import token_span

    return {"src": token_span(prompt, inst.expected_next_desc, ctx=ctx),
            "dst": token_span(prompt, inst.dst_next_desc, ctx=ctx)}


def gaze_shift(attn_delta, task_positions, src_span, dst_span) -> dict:
    """Compare the patched−clean attention delta on the SRC-target span vs the DST-target span.

    ``attn_delta`` is aligned to ``task_positions`` (same length; e.g. patched − clean value-weighted
    pivot→TASK attention). ``src_span``/``dst_span`` are absolute token ranges. ``shifted`` = the
    SRC-target span gained more (delta) than the DST-target span — gaze moved toward the SRC goal."""
    d = np.asarray(attn_delta, dtype=float)
    idx = {p: i for i, p in enumerate(task_positions)}

    def mean_on(span):
        ii = [idx[p] for p in range(span[0], span[1]) if p in idx]
        return float(np.mean([d[i] for i in ii])) if ii else float("nan")

    src_d, dst_d = mean_on(src_span), mean_on(dst_span)
    return {"src_delta": src_d, "dst_delta": dst_d,
            "shifted": bool(src_d > dst_d), "margin": src_d - dst_d}


# ══════════════════════════════════════════════════════════════════════════════
# bundle
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class QTransferEval:
    causal: Optional[dict] = None       # judge_causal(...)
    readout: Optional[dict] = None      # eval_readouts(...)
    gaze: Optional[dict] = None         # gaze_shift(...)
    meta: dict = field(default_factory=dict)

    def summary(self) -> str:
        parts = []
        if self.causal is not None:
            parts.append(f"causal: {'TRIGGERED' if self.causal['triggered'] else 'no'} ({self.causal['label']})")
        if self.readout is not None:
            r = self.readout
            parts.append(f"readout: {r['source']}→{r['dest']}→{r['patched']} "
                         f"[{'valid' if r['valid'] else 'INVALID'}, {'shifted' if r['shifted'] else 'no-shift'}]")
        if self.gaze is not None:
            g = self.gaze
            parts.append(f"gaze: src{g['src_delta']:+.3f} vs dst{g['dst_delta']:+.3f} "
                         f"({'shifted' if g['shifted'] else 'no-shift'})")
        return " | ".join(parts)


def eval_q_transfer(
    inst,
    patched_gen: Optional[str] = None,
    readouts: Optional[tuple] = None,          # (source_readout, dest_readout, patched_readout)
    gaze: Optional[dict] = None,               # {"attn_delta", "task_positions", "prompt", "ctx"?}
    **kw,
) -> QTransferEval:
    """Run whichever of the three q-transfer evals the caller supplies inputs for.

    patched_gen -> causal ; readouts -> readout shift ; gaze{attn_delta, task_positions, prompt}
    -> gaze shift (spans located in ``prompt`` via the instance's target descriptions)."""
    out = QTransferEval(meta={"datatype": inst.meta.get("datatype"),
                              "expected_next": inst.expected_next})
    if patched_gen is not None:
        out.causal = judge_causal(inst, patched_gen, **kw)
    if readouts is not None:
        out.readout = eval_readouts(inst, *readouts, **kw)
    if gaze is not None:
        sp = target_spans(inst, gaze["prompt"], ctx=gaze.get("ctx"))
        out.gaze = gaze_shift(gaze["attn_delta"], gaze["task_positions"], sp["src"], sp["dst"])
    return out


# ── kv-transfer eval — deferred (refine later, per the current plan) ──────────
def eval_kv_transfer(*args, **kwargs):
    raise NotImplementedError("kv-transfer eval pending — q-transfer first (see module docstring).")
