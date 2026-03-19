"""Module-level attention localization for the head localization experiment (Track 2).

Brute-force sliding-window intervention: for each center layer c and a fixed
window size w, replace the attention module output at the last token position
in the destination (Python) forward pass with the corresponding outputs from
the source (C++) forward pass, for layers [c-w//2 .. c+w//2].  Measure how
much the target-token score recovers.

Data contract mirrors the head-mask training setup in head_loc_attn.py:
    source  → C++ prompt for source_predicate          (provides goal signal)
    dest    → Python prompt for destination_predicate  (receives the patch)
    target  → Python prompt for source_predicate       (what we want dest to become)
    score   → prob / logit of pi.target_top_token after patching

Typical usage::

    from src.goals.head_loc_module import run_attn_window_sweep, run_attn_boundary_sweep

    result = run_attn_window_sweep(mt, pair, pi, ds, window=5, metric="prob")
    # result["sweep"]  → {center_layer: score}
    # result["baseline_dest"] → unpatched dest score (lower bound)
    # result["baseline_targ"] → unpatched target score (upper bound)

    result = run_attn_boundary_sweep(mt, pair, pi, ds, mode="upto", metric="prob")
    # result["sweep"]  → {boundary_layer: score}
    # mode="upto": at boundary l, patches layers [0 … l]
    # mode="from": at boundary l, patches layers [l … n_layer-1]
"""

from __future__ import annotations

import logging
from typing import Literal

import torch

from src.functional import get_hs, interpret_logits
from src.goals.head_loc_data import HeadLocDataset, PivotInstance, PredicatePair
from src.goals.head_loc_pivot import build_cpp_prompt, build_python_prompt
from src.models import ModelandTokenizer
from src.tokens import prepare_input
from src.trace import boundary_patch_score, get_score, patched_run, window_patch_score

logger = logging.getLogger(__name__)


def get_target_token_id(mt: ModelandTokenizer, token_str: str) -> int:
    """Encode a decoded token string and return the first token id.

    ``PivotInstance.target_top_token`` is the decoded form of a single vocabulary
    token, so encoding it should give back exactly one id.  A warning is logged
    if the string happens to be multi-token (encoding artefact).
    """
    ids = mt.tokenizer.encode(token_str, add_special_tokens=False)
    if len(ids) != 1:
        logger.warning(
            f"target token {token_str!r} encodes to {len(ids)} sub-tokens {ids}; "
            "using the first one."
        )
    return ids[0]


def _top5_str(
    mt: ModelandTokenizer, logits: torch.Tensor, metric: Literal["logit", "prob"]
) -> str:
    ll_iterp = interpret_logits(mt, logits, k=5)
    ll_strs = [f'"{pred.token!r}" ({getattr(pred, metric):.2f})' for pred in ll_iterp]
    return "[ " + ", ".join(ll_strs) + " ]"


@torch.no_grad()
def _build_sweep_context(
    mt: ModelandTokenizer,
    pair: PredicatePair,
    pi: PivotInstance,
    ds: HeadLocDataset,
    metric: Literal["logit", "prob"],
    include_docstring: bool,
) -> dict:
    """Shared setup for all sweep functions.

    Builds prompts, runs the three baseline forward passes, extracts src attn
    states for all layers, and logs debug info.  Returns a dict with all the
    pieces needed by the sweep loops.
    """
    src_sample = ds.get_sample(pair.source_predicate)
    dest_sample = ds.get_sample(pair.destination_predicate)
    targ_sample = ds.get_sample(pair.source_predicate)  # target == source in Python

    src_prompt = build_cpp_prompt(mt, src_sample, include_docstring=include_docstring)
    dest_prompt = build_python_prompt(
        mt,
        dest_sample,
        pivot_prefix=pi.pivot_prefix,
        include_docstring=include_docstring,
    )
    targ_prompt = build_python_prompt(
        mt,
        targ_sample,
        pivot_prefix=pi.pivot_prefix,
        include_docstring=include_docstring,
    )

    logger.debug(f'src_prompt:\n"{src_prompt}"\n')
    logger.debug(f'dest_prompt:\n"{dest_prompt}"\n')
    logger.debug(f'targ_prompt:\n"{targ_prompt}"\n')

    target_token_id = get_target_token_id(mt, pi.target_top_token)
    logger.debug(f"target token {pi.target_top_token!r} has id {target_token_id}")

    src_inputs = prepare_input(prompts=[src_prompt], tokenizer=mt)
    dest_inputs = prepare_input(prompts=[dest_prompt], tokenizer=mt)
    targ_inputs = prepare_input(prompts=[targ_prompt], tokenizer=mt)

    logits_src = patched_run(mt=mt, inputs=src_inputs, states={})
    logits_dest = patched_run(mt=mt, inputs=dest_inputs, states={})
    logits_targ = patched_run(mt=mt, inputs=targ_inputs, states={})

    baseline_dest = get_score(
        logits=logits_dest, token_id=target_token_id, metric=metric
    )
    baseline_targ = get_score(
        logits=logits_targ, token_id=target_token_id, metric=metric
    )

    logger.info(
        f"[{pair.destination_predicate} ← {pair.source_predicate}] "
        f"prefix={pi.pivot_prefix!r}  target={pi.target_top_token!r}  "
        f"baseline_dest={baseline_dest:.4f}  baseline_targ={baseline_targ:.4f}"
    )

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"  baseline src  (C++) top5: {_top5_str(mt, logits_src, metric)}")
        logger.debug(
            f"  baseline dest (Py)  top5: {_top5_str(mt, logits_dest, metric)}"
        )
        logger.debug(
            f"  baseline targ (Py)  top5: {_top5_str(mt, logits_targ, metric)}"
        )

    # One forward pass through the C++ source prompt to capture all attn states
    locations = [(mt.attn_module_name_format.format(i), -1) for i in range(mt.n_layer)]
    hs = get_hs(mt, src_prompt, locations, return_dict=True)
    src_states = {
        i: hs[(mt.attn_module_name_format.format(i), -1)] for i in range(mt.n_layer)
    }

    return {
        "src_states": src_states,
        "dest_inputs": dest_inputs,
        "target_token_id": target_token_id,
        "baseline_dest": baseline_dest,
        "baseline_targ": baseline_targ,
        "src_prompt": src_prompt,
        "dest_prompt": dest_prompt,
        "targ_prompt": targ_prompt,
    }


@torch.no_grad()
def run_attn_window_sweep(
    mt: ModelandTokenizer,
    pair: PredicatePair,
    pi: PivotInstance,
    ds: HeadLocDataset,
    window: int = 5,
    metric: Literal["logit", "prob"] = "prob",
    include_docstring: bool = False,
) -> dict:
    """Sweep a sliding window of attention module outputs across all layers.

    For each center layer c (0 … n_layer-1), patches the attention output at the
    last token position from the C++ source prompt into the Python destination
    prompt for the window of layers [c-window//2, …, c+window//2] (boundary-
    clamped), then scores how much the target token probability recovers.

    Args:
        mt: ModelandTokenizer (frozen, inference only).
        pair: PredicatePair identifying source and destination predicates.
        pi: PivotInstance with ``pivot_prefix`` and ``target_top_token``.
        ds: HeadLocDataset providing SinglePredicateSample objects.
        window: Patch window width.  ``window=1`` patches only the center layer;
            ``window=5`` patches [center-2, …, center+2].
        metric: Scoring metric — ``"prob"``, ``"logit"``, or ``"log_norm"``.
        include_docstring: Passed to prompt builders.

    Returns:
        Dict with keys ``"sweep"`` ({center_layer: score}), ``"baseline_dest"``,
        ``"baseline_targ"``, ``"target_token"``, ``"target_token_id"``,
        ``"src_prompt"``, ``"dest_prompt"``, ``"targ_prompt"``, ``"pair"``, ``"pi"``.
    """
    ctx = _build_sweep_context(mt, pair, pi, ds, metric, include_docstring)
    src_states = ctx["src_states"]
    dest_inputs = ctx["dest_inputs"]
    target_token_id = ctx["target_token_id"]

    debug = logger.isEnabledFor(logging.DEBUG)
    sweep: dict[int, float] = {}
    half = window // 2

    for center in range(mt.n_layer):
        if debug:
            layer_indices = list(
                range(max(0, center - half), min(mt.n_layer, center + half + 1))
            )
            states = {
                (mt.attn_module_name_format.format(li), -1): src_states[li]
                for li in layer_indices
                if li in src_states
            }
            logits = patched_run(mt=mt, inputs=dest_inputs, states=states)
            cur_score = get_score(
                logits=logits, token_id=target_token_id, metric=metric
            )
            logger.debug(
                f"  window center={center:3d}  score={cur_score:.4f}"
                f"  top5: {_top5_str(mt, logits, metric)}"
            )
        else:
            cur_score = window_patch_score(
                mt=mt,
                dest_inputs=dest_inputs,
                src_states=src_states,
                center=center,
                window=window,
                target_token_ids=target_token_id,
                metric=metric,
            )
        sweep[center] = cur_score

    peak_layer = max(sweep, key=sweep.get)
    logger.info(
        f"  window sweep done — peak layer={peak_layer}  score={sweep[peak_layer]:.4f}"
    )

    return {
        "sweep": sweep,
        **{
            k: ctx[k]
            for k in (
                "baseline_dest",
                "baseline_targ",
                "src_prompt",
                "dest_prompt",
                "targ_prompt",
            )
        },
        "target_token": pi.target_top_token,
        "target_token_id": target_token_id,
        "pair": pair,
        "pi": pi,
    }


@torch.no_grad()
def run_attn_boundary_sweep(
    mt: ModelandTokenizer,
    pair: PredicatePair,
    pi: PivotInstance,
    ds: HeadLocDataset,
    mode: Literal["upto", "from"] = "upto",
    metric: Literal["logit", "prob", "log_norm"] = "prob",
    include_docstring: bool = False,
) -> dict:
    """Sweep a growing/shrinking prefix or suffix of patched attention layers.

    For each boundary layer l (0 … n_layer-1):

    - ``mode="upto"``: patches layers [0, …, l] — increasingly more context
      from the source.  Score plateaus when the model has received enough signal.
    - ``mode="from"``: patches layers [l, …, n_layer-1] — patches the tail.
      Score is high when l is small (all layers patched) and drops as we move l
      later, revealing from which layer the processed goal signal is needed.

    Args:
        mt: ModelandTokenizer (frozen, inference only).
        pair: PredicatePair identifying source and destination predicates.
        pi: PivotInstance with ``pivot_prefix`` and ``target_top_token``.
        ds: HeadLocDataset providing SinglePredicateSample objects.
        mode: ``"upto"`` or ``"from"`` (see above).
        metric: Scoring metric — ``"prob"``, ``"logit"``, or ``"log_norm"``.
        include_docstring: Passed to prompt builders.

    Returns:
        Dict with keys ``"sweep"`` ({boundary_layer: score}), ``"mode"``,
        ``"baseline_dest"``, ``"baseline_targ"``, ``"target_token"``,
        ``"target_token_id"``, ``"src_prompt"``, ``"dest_prompt"``,
        ``"targ_prompt"``, ``"pair"``, ``"pi"``.
    """
    ctx = _build_sweep_context(mt, pair, pi, ds, metric, include_docstring)
    src_states = ctx["src_states"]
    dest_inputs = ctx["dest_inputs"]
    target_token_id = ctx["target_token_id"]

    debug = logger.isEnabledFor(logging.DEBUG)
    sweep: dict[int, float] = {}

    for boundary in range(mt.n_layer):
        if debug:
            if mode == "upto":
                layer_indices = list(range(0, boundary + 1))
            else:
                layer_indices = list(range(boundary, mt.n_layer))
            states = {
                (mt.attn_module_name_format.format(li), -1): src_states[li]
                for li in layer_indices
                if li in src_states
            }
            logits = patched_run(mt=mt, inputs=dest_inputs, states=states)
            cur_score = get_score(
                logits=logits, token_id=target_token_id, metric=metric
            )
            n_patched = len(layer_indices)
            logger.debug(
                f"  {mode} boundary={boundary:3d}  n_patched={n_patched:3d}"
                f"  score={cur_score:.4f}  top5: {_top5_str(mt, logits, metric)}"
            )
        else:
            cur_score = boundary_patch_score(
                mt=mt,
                dest_inputs=dest_inputs,
                src_states=src_states,
                boundary=boundary,
                mode=mode,
                target_token_ids=target_token_id,
                metric=metric,
            )
        sweep[boundary] = cur_score

    peak_layer = max(sweep, key=sweep.get)
    logger.info(
        f"  {mode} sweep done — peak boundary={peak_layer}  score={sweep[peak_layer]:.4f}"
    )

    return {
        "sweep": sweep,
        "mode": mode,
        **{
            k: ctx[k]
            for k in (
                "baseline_dest",
                "baseline_targ",
                "src_prompt",
                "dest_prompt",
                "targ_prompt",
            )
        },
        "target_token": pi.target_top_token,
        "target_token_id": target_token_id,
        "pair": pair,
        "pi": pi,
    }
