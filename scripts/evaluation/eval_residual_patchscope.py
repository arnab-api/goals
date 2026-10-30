#!/usr/bin/env python
"""Residual-stream PatchScope baseline for the GoalScope evals.

The counterpart to ``eval_goalscope.py``: identical corpus, anchors, oracle rubric and scoring —
the *only* difference is **what** gets injected into the read-out probe.

    eval_goalscope.py          g_enc = summed o_proj contribution of the selected goal heads
    eval_residual_patchscope   h_src = the residual stream (or the attention-block output) itself,
                               averaged over --source-layers

Injection is norm-relative, matching ``--strategy additive_norm`` (alpha 0.1, steer layers 35-46):

    h_dest <- h_dest + alpha * ||h_dest|| * (v / ||v||)

Holding the injection mechanism fixed and varying only the source is the whole point (see
``PatchSpec`` strategy ``add_norm`` in ``src/functional.py``).

Results land in the same tree as the GoalScope runs:

    <results>/evaluation/goalscope/<model>/<mode>/resid_norm/<alpha>/<leaf>/

with ``heads=<source>:L<lo>-<hi>:mean`` as the leaf's head-set component, so a residual run can never
collide with a head-set run.

Usage (no flags = the paper's residual baseline: residual stream, layers 40-50, faithfulness):
    conda run -n goals python scripts/evaluation/eval_residual_patchscope.py [--mode causality]
    # full-attention baseline:      --source attn
    # goal-band residual baseline:  --source-layers 25 56
"""

import argparse
import json
import logging
import sys
import time

sys.path.append(".")

import baukit
import torch
from tqdm.auto import tqdm

from scripts.evaluation import faithfulness_io as fio

# Every stage except the residual injection is shared with the head-set eval, so the two runs
# differ only in the thing under test.
from scripts.evaluation.eval_goalscope import (
    ALL_DATATYPES,
    ANCHOR_ORDER,
    CAUSAL_PREFIX,
    DEFAULT_DATASET,
    GEN_MAX_NEW_TOKENS,
    MAX_NEW_TOKENS,
    ORACLE_MODEL,
    build_prompt,
    compute_summary,
    frontier_impl_span,
    generate_completion,
    hard_eval,
    limit_round_robin,
    load_locked_datapoints,
    log_summary,
    pivot_positions,
)
from src.functional import PatchSpec, generate_with_patch
from src.goals.goalscope import _build_causality_prompt, _build_goalscope_prompt
from src.models import ModelandTokenizer
from src.tokens import prepare_input
from src.utils import env_utils, experiment_utils, logging_utils
from src.utils.typing import TokenizerOutput

logger = logging.getLogger(__name__)

STEER_LAYER_RANGE = (35, 46)  # same steer band as the GoalScope runs (leaf: layers=35-46)
BATCH_SIZE = 16  # read-outs per batched greedy decode (batch composition affects bf16 GEMMs)


# ══════════════════════════════════════════════════════════════════════════════
# Residual extraction + injection
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def extract_residual_latents(mt, prompt, positions, layers, source="resid"):
    """``(n_layers, n_pos, n_embd)`` source latents, one forward pass.

    ``source="resid"`` reads the *output* of each decoder layer (the residual stream after that
    block); ``source="attn"`` reads the *output* of each layer's self-attention module (the whole
    attention block's contribution — o_proj output, the summed head writes). Both live in the
    residual (n_embd) space ``apply_goalscope`` injects into, as the PatchScope construction requires.
    """
    inp = prepare_input(prompts=prompt, tokenizer=mt)
    fmt = mt.attn_module_name_format if source == "attn" else mt.layer_name_format
    names = [fmt.format(l) for l in layers]
    idx = torch.tensor(list(positions))
    cache: dict[str, torch.Tensor] = {}

    def hook(output, layer):
        h = output[0] if isinstance(output, tuple) else output
        cache[layer] = h[0].index_select(0, idx.to(h.device)).float().cpu()
        return output

    with baukit.TraceDict(mt._model, names, edit_output=hook):
        mt._model(**dict(inp))
    missing = [n for n in names if n not in cache]
    assert not missing, f"no activations captured for {missing[:3]}"
    return torch.stack([cache[n] for n in names])  # (n_layers, n_pos, n_embd)


@torch.no_grad()
def residual_patchscope(
    mt,
    prompt,
    token_positions,
    source_layers,
    alpha=0.1,
    source="resid",
    probe_prompt=None,
):
    """Verbalize each source position by injecting its source latent into the read-out probe.

    Mirrors ``apply_goalscope``'s additive_norm path — same probe, same greedy decode, same steer
    band — with the ``source`` latent (resid stream | attn-module output, averaged over
    ``source_layers``) in place of the head aggregate.

    Returns ``[{"pos", "token", "patched_gen", "scoped"}]`` (the ``apply_goalscope`` row shape).
    """
    positions = list(token_positions)
    stack = extract_residual_latents(mt, prompt, positions, source_layers, source=source)
    vecs = stack.mean(0)  # (n_pos, n_embd): mean over the source band
    # Unit direction: `add_norm` supplies the magnitude from ||h_dest|| at each patch site.
    vecs = vecs / vecs.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    dest = prepare_input(
        prompts=probe_prompt or _build_goalscope_prompt(mt), tokenizer=mt
    )
    dest_ids, dest_attn = dest.input_ids, dest.attention_mask
    steer_layers = list(range(*STEER_LAYER_RANGE))

    src_ids = prepare_input(prompts=prompt, tokenizer=mt).input_ids[0].tolist()
    out = []
    for s in tqdm(
        range(0, len(positions), BATCH_SIZE), desc=f"resid[a={alpha}]", leave=False
    ):
        chunk = positions[s : s + BATCH_SIZE]
        B = len(chunk)
        tiled = TokenizerOutput(
            data=dict(
                input_ids=dest_ids.expand(B, -1).clone(),
                attention_mask=dest_attn.expand(B, -1).clone(),
            )
        )
        v = (alpha * vecs[s : s + B]).to(mt._model.dtype)
        patches = [
            PatchSpec(
                location=(mt.layer_name_format.format(li), -1),
                patch=v,
                strategy="add_norm",
                on_pass="all",
            )
            for li in steer_layers
        ]
        gens = generate_with_patch(
            mt,
            tiled,
            max_new_tokens=MAX_NEW_TOKENS,
            patches=patches,
            n_steer_tokens=None,
            do_sample=False,
            patch_interface="baukit",
            remove_prefix=True,
            skip_special_tokens=True,
        )
        for i, p in enumerate(chunk):
            out.append(
                dict(
                    pos=int(p),
                    token=mt.tokenizer.decode([src_ids[p]]),
                    patched_gen=gens[i],
                    scoped=True,
                )
            )
    return out


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--mode", default="faithfulness", choices=["faithfulness", "causality"]
    )
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    ap.add_argument(
        "--source",
        default="resid",
        choices=["resid", "attn"],
        help="what latent to read at --source-layers: 'resid' (decoder-layer residual stream) | "
        "'attn' (self-attention module output — the whole attention block's contribution). "
        "Baked into the run-dir leaf (resid:.. vs attn:..) so the two never collide.",
    )
    ap.add_argument(
        "--source-layers",
        type=int,
        nargs=2,
        default=[40, 50],
        metavar=("START", "END"),
        help="layers to read the source latent from, range(START, END), averaged (default 40 50).",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="injected fraction of ||h_dest|| (norm-relative; default 0.1)",
    )
    ap.add_argument("--seed", type=int, default=123,
                    help="generation-determinism seed (corpus is fixed by the dataset)")
    ap.add_argument("--limit", type=int, default=None,
                    help="SMOKE ONLY: cap the corpus to N datapoints (round-robin across datatypes)")
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="wipe an existing run dir; default is to RESUME",
    )
    logging_utils.add_logging_args(ap)
    return ap.parse_args()


def main():
    args = parse_args()
    logging_utils.configure(args)
    experiment_utils.set_seed(args.seed)
    torch.set_grad_enabled(False)
    results_dir = env_utils.DEFAULT_RESULTS_DIR
    src_band = tuple(args.source_layers)

    # ── LOCKED corpus (aggregated over datatypes), loaded up front — no model needed ──
    datapoints, ds_meta = load_locked_datapoints(DEFAULT_DATASET, ALL_DATATYPES)
    if args.limit:
        datapoints = limit_round_robin(datapoints, args.limit)

    config = dict(
        mode=args.mode,
        # `strategy` is a path component, so residual runs never land in a head-set run's dir
        strategy="resid_norm",
        alpha=args.alpha,
        source_layers=list(src_band),
        source_agg="mean",
        source=args.source,
        heads_name=f"{args.source}:L{src_band[0]}-{src_band[1]}:mean",
        steer_layer_range=list(STEER_LAYER_RANGE),
        n_steer_tokens="all",
        max_new_tokens=MAX_NEW_TOKENS,
        model=args.model,
        datatype="all",
        datatypes=list(ALL_DATATYPES),
        n_specs=len(datapoints),
        num_conditions=list(ds_meta["n_conditions"]),
        seed=ds_meta["seed"],
        gen_max_new_tokens=GEN_MAX_NEW_TOKENS,
        span_source="frontier",       # prefill-to-frontier: the leaf's span=frontier
        dataset=DEFAULT_DATASET,
        bank="QTASKS",
        anchors=ANCHOR_ORDER,
        oracle_model=ORACLE_MODEL,
        readout_prefix=(CAUSAL_PREFIX if args.mode == "causality" else ""),
    )

    run_dir = fio.run_dir_for(config, results_dir)
    if args.overwrite and run_dir.exists():
        for f in ("records.jsonl", "samples.jsonl", "summary.json"):
            (run_dir / f).unlink(missing_ok=True)
        logger.info(f"--overwrite: cleared existing run at {run_dir}")
    prev = None if args.overwrite else fio.read_config(run_dir)
    # Same guard as the head-set eval: the leaf encodes the source band, but not every knob, so
    # refuse to resume a run whose source differs from this invocation's.
    if prev is not None:
        for k in ("source_layers", "source_agg", "strategy", "source"):
            if prev.get(k) is not None and prev.get(k) != config[k]:
                raise SystemExit(
                    f"run dir {run_dir} holds a run with {k}={prev[k]!r}, but this invocation "
                    f"passed {config[k]!r}. Resuming would mix them — pass --overwrite."
                )
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"run dir -> {run_dir}")

    completed = set(fio.read_completed_spec_ids(run_dir))
    kept = fio.prune_orphan_records(run_dir, completed)
    remaining = [(sid, inst) for sid, inst in datapoints if sid not in completed]
    logger.info(
        f"corpus: {len(datapoints)} datapoints (locked {DEFAULT_DATASET}) | "
        f"already done: {len(completed)} ({kept} records kept) | remaining: {len(remaining)}"
    )

    if not remaining:
        recs = (
            [
                json.loads(l)
                for l in fio.records_path(run_dir).read_text().splitlines()
                if l.strip()
            ]
            if fio.records_path(run_dir).exists()
            else []
        )
        summary = compute_summary(recs, n_samples=len(completed))
        fio.write_summary(run_dir, summary)
        logger.info(f"all samples already complete; summary refreshed -> {run_dir}")
        log_summary(summary, args.mode)
        return

    mt = ModelandTokenizer(
        model_key=args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    source_layers = list(range(*src_band))
    config["n_source_layers"] = len(source_layers)
    config["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    fio.write_config(run_dir, config)
    logger.info(
        f"{args.source} source: layers {src_band[0]}-{src_band[1] - 1} ({len(source_layers)}) "
        f"agg=mean -> steer {STEER_LAYER_RANGE} @ alpha={args.alpha}"
    )

    probe_prompt = _build_causality_prompt(mt) if args.mode == "causality" else None
    readout_prefix = CAUSAL_PREFIX if args.mode == "causality" else ""

    rec_fh = open(fio.records_path(run_dir), "a")
    samp_fh = open(fio.samples_path(run_dir), "a")
    n_done = len(completed)
    n_total = len(datapoints)
    for sid, inst in remaining:
        sample_records = []
        prompt, var = build_prompt(mt, inst)
        completion = generate_completion(mt, prompt, GEN_MAX_NEW_TOKENS)
        full_text = prompt + completion

        # frontier impl_span is exact (single-line predicate) — no oracle span-labelling
        impl_span = frontier_impl_span(prompt, completion, var)
        frontier_desc = inst.expected_next_desc

        pivs = pivot_positions(mt, prompt, full_text, ANCHOR_ORDER)
        if not pivs:
            logger.warning(f"[{sid}] no valid pivots — skipping")
            continue
        apos = sorted(set(pivs.values()))

        res = residual_patchscope(
            mt,
            full_text,
            token_positions=apos,
            source_layers=source_layers,
            alpha=args.alpha,
            source=args.source,
            probe_prompt=probe_prompt,
        )
        readouts = {r["pos"]: r for r in res}

        for aname, pos in pivs.items():
            ro = readout_prefix + readouts.get(pos, {}).get("patched_gen", "")
            hv = hard_eval(ro, impl_span, args.mode)
            logger.debug("[%s %s] read-out: %r -> %s", sid, aname, ro, hv["verdict"])
            sample_records.append(dict(
                spec=sid,
                datatype=inst.meta["datatype"],   # per-record tag -> per-datatype breakdown
                anchor=aname,
                pos=pos,
                readout=ro,
                impl_span=impl_span,
                frontier_id=inst.expected_next,
                condition=frontier_desc,
                hard=hv["verdict"],
                hard_prompt=hv["prompt"],
                hard_response=hv["response"],
            ))

        sample_meta = dict(
            spec_id=sid,
            datatype=inst.meta["datatype"],
            lang=inst.src_lang,
            T=inst.meta["T"],
            j=inst.meta["j"],
            predicate_ids=[p.id for p in inst.task.predicates],
            progress_ids=list(inst.src_progress),
            frontier_id=inst.expected_next,
            var=var,
            prompt=prompt,
            completion=completion,
            impl_span=impl_span,
            pivots=pivs,
        )
        # records first, then the meta line — a crash mid-sample leaves prunable orphans rather
        # than a phantom-complete sample (same contract as eval_goalscope)
        for r in sample_records:
            fio.append_jsonl(rec_fh, r)
        fio.append_jsonl(samp_fh, sample_meta)
        n_done += 1
        logger.info(f"[{n_done}/{n_total}] {sid}: {len(pivs)} pivot read-outs scored")

    rec_fh.close()
    samp_fh.close()

    all_records = [
        json.loads(l)
        for l in fio.records_path(run_dir).read_text().splitlines()
        if l.strip()
    ]
    summary = compute_summary(
        all_records, n_samples=len(fio.read_completed_spec_ids(run_dir))
    )
    fio.write_summary(run_dir, summary)
    logger.info(f"saved -> {run_dir}")
    log_summary(summary, args.mode)


if __name__ == "__main__":
    main()
