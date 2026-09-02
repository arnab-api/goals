"""Data generation for the DIFF-hypothesis experiments.

Two experimental "specs", mirroring the two transplant mechanisms:

Spec-1 — QUERY transfer (single broader goal, multiple instrumental sub-goals).
    One function-writing task whose body is a list of independent sub-computations
    ({a, b, c, d}-style). An anchor sub-goal is always prefilled; a variable amount of
    PROGRESS (a subset of the remaining sub-goals) is prefilled as completed. At the
    pivot the model is about to write the next sub-goal. The intervention transplants
    the (pre-RoPE) query of the selected heads from a source PROGRESS state onto a
    destination PROGRESS state. Currently authored inline in
    ``01_suppression_mechanism_confirmation.ipynb`` (ITEMS_PRESETS / build_prompt) and
    ``03_generalization.ipynb``.

Spec-2 — KEY+VALUE transfer (cross-TASK).
    Two multi-condition tasks that SHARE a leading prefix of conditions (so
    ``SRC.PROGRESS == DST.PROGRESS``, both prefilled) and DIFFER on the trailing
    conditions. At a common pivot the intervention transplants the source's key+value
    over the divergent task-tail span onto the destination (GQA-corrected headwise
    routing). Test: does the destination now implement the SOURCE's next condition?
    Currently authored inline as A4 in
    ``01_suppression_mechanism_confirmation.ipynb`` and as Spec-2 in ``00_diff.ipynb``.

STATUS: skeleton. The concrete generators / schemas are pending a design pass — the
notebook-inline builders above are the reference implementations to consolidate here.
Until then this module only fixes the dataclass shapes so ``eval`` can type against them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


# ══════════════════════════════════════════════════════════════════════════════
# Spec-1 — query transfer
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class QueryTransferSample:
    """One query-transfer instance: a source and a destination PROGRESS state over the
    same broader task, plus the metadata needed to locate the pivot and score success.

    TODO(brainstorm): finalize fields. Provisional shape drawn from the notebook builders.
    """

    task_spec: str                       # the rendered numbered sub-goal list
    src_progress: list[str]              # sub-goal keys prefilled as done in the SOURCE
    dst_progress: list[str]              # sub-goal keys prefilled as done in the DEST
    src_prompt: str = ""                 # fully rendered source prompt (through the pivot)
    dst_prompt: str = ""                 # fully rendered destination prompt (through the pivot)
    expected_next: Optional[str] = None  # sub-goal the SOURCE query should steer DEST toward
    meta: dict = field(default_factory=dict)


def build_query_transfer_dataset(*args, **kwargs) -> list[QueryTransferSample]:
    """Generate Spec-1 query-transfer samples.

    TODO(brainstorm): consolidate ITEMS_PRESETS + build_prompt / done-set enumeration
    from ``01``/``03`` here. Decide the sampling axes (preset, #progress, naming, anchor).
    """
    raise NotImplementedError("Spec-1 generator pending design pass — see module docstring.")


# ══════════════════════════════════════════════════════════════════════════════
# Spec-2 — key+value transfer (cross-TASK)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class KVTransferSample:
    """One kv-transfer instance: two multi-condition tasks sharing a leading prefix and
    differing on the tail, both prefilled to a common pivot.

    TODO(brainstorm): finalize fields. Provisional shape drawn from A4 / Spec-2.
    """

    shared_conditions: list[str]         # conditions identical in SRC & DST (prefilled as progress)
    src_tail: list[str]                  # SRC-only trailing conditions (the transplanted goal)
    dst_tail: list[str]                  # DST-only trailing conditions
    src_prompt: str = ""                 # rendered source prompt (through the pivot)
    dst_prompt: str = ""                 # rendered destination prompt (through the pivot)
    align: Literal["left", "right", "pad"] = "right"   # tail-span alignment for the transplant
    expected_next: Optional[str] = None  # SRC's next (3rd) condition the transplant should induce
    meta: dict = field(default_factory=dict)


def build_kv_transfer_dataset(*args, **kwargs) -> list[KVTransferSample]:
    """Generate Spec-2 kv-transfer samples.

    TODO(brainstorm): consolidate the A4 builder (shared-prefix + divergent-tail, the
    left/right/pad alignment, span location) from ``01`` here.
    """
    raise NotImplementedError("Spec-2 generator pending design pass — see module docstring.")
