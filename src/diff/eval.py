"""Evaluation utilities for the DIFF hypothesis.

Scores whether an intervention causally installs the SOURCE's next instrumental goal
into the DESTINATION generation, for both transplant mechanisms:

  • q-transfer  (Spec-1): inject the source PROGRESS query at the pivot; does the DEST
                 now pursue the sub-goal the source was about to pursue?
  • kv-transfer (Spec-2): transplant the source task-tail key+value; does the DEST now
                 implement the SOURCE's next condition instead of its own?

Each eval runs three arms per sample — clean SRC, clean DST, patched DST (and optionally
an OV-boosted patched DST, cf. ``boost_head_ov``) — via the generation/intervention
primitives in :mod:`src.diff.functional`, then judges the patched output.

STATUS: skeleton. The scoring is deliberately left unimplemented — success today is judged
qualitatively (eyeballing the generations + goalscope readouts). The automated judge
(string/AST match against ``expected_next``, an oracle unit test, or an LM judge) is the
pending design decision. The generation harness wiring is sketched so it drops onto the
:mod:`src.diff.data` sample types once those land.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TransferResult:
    """Outcome of one transfer eval instance."""

    clean_src: str = ""       # SRC greedy generation (reference: the goal to be transferred)
    clean_dst: str = ""       # DST greedy generation (baseline: DST's own goal)
    patched_dst: str = ""     # DST generation under the intervention
    boosted_dst: Optional[str] = None   # DST under intervention + OV boost, if run
    success: Optional[bool] = None      # did patched_dst pursue SRC's next goal? (None = not scored)
    goalscope: dict = field(default_factory=dict)   # readout strings per arm, if collected
    meta: dict = field(default_factory=dict)


def eval_q_transfer(*args, **kwargs) -> TransferResult:
    """Evaluate a single Spec-1 query-transfer instance.

    TODO(brainstorm): wire clean/patched generation via ``functional.greedy`` with
    ``prefill_specs``/``decode_specs`` (the query inject), collect goalscope readouts,
    and define the success criterion against ``QueryTransferSample.expected_next``.
    """
    raise NotImplementedError("q-transfer eval pending design pass — see module docstring.")


def eval_kv_transfer(*args, **kwargs) -> TransferResult:
    """Evaluate a single Spec-2 key+value-transfer instance.

    TODO(brainstorm): wire clean/patched generation via ``functional.headwise_kv_patch``
    (GQA-corrected), optional ``boost_head_ov`` arm, collect goalscope readouts, and define
    the success criterion against ``KVTransferSample.expected_next``.
    """
    raise NotImplementedError("kv-transfer eval pending design pass — see module docstring.")


def run_eval(samples, mechanism: str, ctx=None, **kwargs) -> list[TransferResult]:
    """Batch-evaluate a dataset. ``mechanism`` in {"q", "kv"} dispatches to the per-sample
    evaluator; ``ctx`` is a :class:`src.diff.functional.Ctx` (avoids the module globals).

    TODO(brainstorm): aggregation / summary stats (success rate per axis) once scoring lands.
    """
    raise NotImplementedError("run_eval pending design pass — see module docstring.")
