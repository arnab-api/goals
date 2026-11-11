#!/usr/bin/env python
"""OV-reader eval on the DIFF prefill-to-frontier setup — does the residual already carry the goal?

Corpus = the LOCKED dataset of ``eval_goalscope.py`` (``data/goalscope_eval.json``: QTASKS, letters
naming, prefill the first ``j`` single-line predicates and stop at the frontier's ``{var} =``), so the
skyline below is exactly GoalScope's Axis (a) read-out. Greedy-decode the frontier body; its code is the
first generated line. At five pivots — the last four prompt tokens ``(prev_end, indent, var, eq)`` plus
the first generated token ``(first_impl)`` — we run:

  * SKYLINE — ``apply_goalscope`` (additive_norm, alpha 0.1), i.e. the 122 goal heads' actual OUTPUT
    read-out.
  * OVReader — ``OVReader.apply_probe`` (additive_norm, alpha 0.15) across the probe layers: the
    OV-write direction of the residual after each layer, steered into the same probe.

Read-outs are graded with the SAME oracle rubric/pipeline as ``eval_goalscope.py`` (``--mode``
faithfulness | causality; causality steers into a C++ stub and prepends it before grading).

Results:  <results>/evaluation/ov-reader/<model>/<layer_idx>/<mode>/<leaf>/  (layer_idx=int | "goalscope")
Parse with ``scripts.evaluation.ov_reader_io`` (``grid_df``). Resume-safe: ``--datatypes`` is baked
into the leaf, so disjoint invocations run in parallel.

Usage (the paper runs one process per (mode, datatype)):
    conda run -n goals python scripts/evaluation/eval_ov_reader.py --mode faithfulness --datatypes int --cuda 0
"""

import argparse
import json
import logging
import os
import sys
import time

sys.path.append(".")

# --cuda must be applied before torch initialises the CUDA context (mirrors scripts/nrgs/*).
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--cuda", default=None)
_cuda = _pre.parse_known_args()[0].cuda
if _cuda is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_cuda)

import torch  # noqa: E402

import src.nrgs.data as DATA  # noqa: E402
from scripts.evaluation import ov_reader_io as ovio  # noqa: E402
from scripts.evaluation.eval_goalscope import (  # noqa: E402
    DEFAULT_DATASET,
    GEN_MAX_NEW_TOKENS,
    MAX_NEW_TOKENS,
    ORACLE_MODEL,
    build_prompt,
    frontier_impl_span,
    hard_eval,
    load_json_heads,
    load_locked_datapoints,
    pivot_positions,
)
import src.goals.goalscope as GS  # noqa: E402  (GS.CAUSAL_PREFIX: the probe chosen by --causal_stub)
from src.goals.goalscope import (  # noqa: E402
    CAUSAL_PREFIX,
    OVReader,
    _build_causality_prompt,
    apply_goalscope,
    greedy_continue,
)
from src.models import ModelandTokenizer  # noqa: E402
from src.tokens import prepare_input  # noqa: E402
from src.utils import env_utils, experiment_utils, logging_utils  # noqa: E402

logger = logging.getLogger(__name__)

MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"
HEADS_JSON = "head_sets/Qwen2.5-Coder-32B-Instruct/goal__lam20_thr0.9_band25-56.json"  # the 122 goal heads
STEER_LAYER_RANGE = (35, 46)  # GoalScope steer band (shared by skyline + OVReader)
OV_ALPHA = 0.15  # OVReader steering alpha (additive_norm)
SKYLINE_ALPHA = 0.1  # GoalScope skyline alpha (additive_norm)


# ══════════════════════════════════════════════════════════════════════════════
# Driver
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--cuda",
        default=None,
        help="CUDA_VISIBLE_DEVICES (applied before torch import)",
    )
    ap.add_argument(
        "--mode", default="faithfulness", choices=["faithfulness", "causality"]
    )
    ap.add_argument(
        "--datatypes",
        nargs="+",
        default=list(DATA.DATATYPES),
        help="datatype(s); each is an independent (parallel-safe) shard",
    )
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--limit", type=int, default=None, help="SMOKE: cap datapoints per shard")
    ap.add_argument(
        "--overwrite", action="store_true", help="wipe matching run dirs and redo"
    )
    logging_utils.add_logging_args(ap)
    ap.add_argument("--causal_stub", default="v2", choices=["v2", "v1"],
                    help="causality probe: v2 = 'auto func (auto value){' (default, reported), v1 = the original 'bool check(auto value){'")
    return ap.parse_args()


def _target_dirs(config, layers, results_dir):
    """{layer_idx: dir} — the goalscope skyline dir plus one dir per probe layer, all sharing leaf."""
    dirs = {ovio.SKYLINE_IDX: ovio.run_dir_for(config, ovio.SKYLINE_IDX, results_dir)}
    for L in layers:
        dirs[L] = ovio.run_dir_for(config, L, results_dir)
    return dirs


def run_shard(mt, ov, heads, datatype, args, layers, results_dir, base_config):
    config = dict(base_config, datatype=datatype)
    dirs = _target_dirs(config, layers, results_dir)

    if args.overwrite:
        for d in dirs.values():
            for f in ("records.jsonl", "samples.jsonl", "summary.json"):
                (d / f).unlink(missing_ok=True)

    # resolve resume state: a datapoint is done only if present in EVERY target dir.
    completed = {}
    for idx, d in dirs.items():
        d.mkdir(parents=True, exist_ok=True)
        done = set(ovio.read_completed_spec_ids(d))
        ovio.prune_orphan_records(d, done)
        cfg = dict(config, layer_idx=idx, timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
        ovio.write_config(d, cfg)
        completed[idx] = done

    # locked sanitized corpus (span=frontier), same datapoints/spec_ids as eval_goalscope
    pairs, _ = load_locked_datapoints(DEFAULT_DATASET, [datatype])
    spec_ids = [sid for sid, _ in pairs]
    datapoints = [inst for _, inst in pairs]
    if args.limit:
        datapoints = datapoints[: args.limit]; spec_ids = spec_ids[: args.limit]
    remaining = [
        (sid, inst)
        for sid, inst in zip(spec_ids, datapoints)
        if any(sid not in completed[idx] for idx in dirs)
    ]
    logger.info(
        f"[{datatype}] {len(datapoints)} datapoints | remaining {len(remaining)} "
        f"| dirs {len(dirs)} (goalscope + {len(layers)} layers)"
    )
    if not remaining:
        _finalize(dirs)
        return

    goalscope_prompt = _build_causality_prompt(mt) if args.mode == "causality" else None
    readout_prefix = GS.CAUSAL_PREFIX if args.mode == "causality" else ""
    fhs = {
        idx: (open(ovio.records_path(d), "a"), open(ovio.samples_path(d), "a"))
        for idx, d in dirs.items()
    }
    try:
        for n, (sid, inst) in enumerate(remaining, 1):
            prompt, var = build_prompt(mt, inst)
            completion = greedy_continue(
                mt,
                **prepare_input(prompts=prompt, tokenizer=mt),
                max_new_tokens=GEN_MAX_NEW_TOKENS,
            )
            full_text = prompt + completion
            impl_span = frontier_impl_span(prompt, completion, var)
            frontier_desc = inst.expected_next_desc
            pivs = pivot_positions(mt, prompt, full_text, ovio.PIVOT_ORDER)
            if not pivs:
                logger.warning(f"[{sid}] no valid pivots — skipping")
                continue
            apos = sorted(set(pivs.values()))
            meta = dict(
                spec_id=sid,
                datatype=datatype,
                lang="python",
                T=inst.meta["T"],
                j=inst.meta["j"],
                var=var,
                frontier_id=inst.expected_next,
                frontier_desc=frontier_desc,
                prompt=prompt,
                completion=completion,
                impl_span=impl_span,
                pivots=pivs,
            )

            def _record(pivot, pos, readout, layer_idx):
                ro = readout_prefix + (readout or "")
                hv = hard_eval(ro, impl_span, args.mode)
                return dict(
                    spec=sid,
                    datatype=datatype,
                    lang="python",
                    pivot=pivot,
                    pos=pos,
                    layer_idx=layer_idx,
                    readout=ro,
                    impl_span=impl_span,
                    frontier_id=inst.expected_next,
                    condition=frontier_desc,
                    hard=hv["verdict"],
                    hard_prompt=hv["prompt"],
                    hard_response=hv["response"],
                )

            # ── skyline: goal-head OUTPUT read-out at each pivot (one apply_goalscope call) ──
            sky = {
                r["pos"]: r
                for r in apply_goalscope(
                    full_text,
                    mt=mt,
                    selected_heads=heads,
                    strategy="additive_norm",
                    steer_layer_range=STEER_LAYER_RANGE,
                    alpha=SKYLINE_ALPHA,
                    n_steer_tokens="all",
                    max_new_tokens=MAX_NEW_TOKENS,
                    include_noncode=False,
                    token_positions=apos,
                    goalscope_prompt=goalscope_prompt,
                )
            }
            recs = {
                ovio.SKYLINE_IDX: [
                    _record(
                        p,
                        pos,
                        sky.get(pos, {}).get("patched_gen", ""),
                        ovio.SKYLINE_IDX,
                    )
                    for p, pos in pivs.items()
                ]
            }
            # ── OVReader: one batched call per pivot -> {layer: readout} ──
            for L in layers:
                recs[L] = []
            for pivot, pos in pivs.items():
                by_layer = ov.apply_probe_on_prompt(
                    full_text,
                    pos,
                    layers,
                    mode=args.mode,
                    steer_layers=list(range(*STEER_LAYER_RANGE)),
                    max_new_tokens=MAX_NEW_TOKENS,
                    alpha=OV_ALPHA,
                )
                for L in layers:
                    recs[L].append(_record(pivot, pos, by_layer.get(L, ""), L))

            for idx, (rec_fh, samp_fh) in fhs.items():
                if sid in completed[idx]:
                    continue
                for r in recs[idx]:
                    ovio.append_jsonl(rec_fh, r)
                ovio.append_jsonl(samp_fh, meta)
            logger.info(
                f"[{datatype}] [{n}/{len(remaining)}] {sid}: "
                f"{len(pivs)} pivots x {len(layers)} layers + skyline scored"
            )
    finally:
        for rec_fh, samp_fh in fhs.values():
            rec_fh.close()
            samp_fh.close()
    _finalize(dirs)


def _finalize(dirs):
    for d in dirs.values():
        recs = (
            [
                json.loads(ln)
                for ln in ovio.records_path(d).read_text().splitlines()
                if ln.strip()
            ]
            if ovio.records_path(d).exists()
            else []
        )
        n_samp = len(ovio.read_completed_spec_ids(d))
        summary = ovio.compute_summary(recs, n_samp)
        ovio.write_summary(d, summary)
        o = summary["overall"]
        logger.info(
            f"  {d.parent.parent.name:>10}  n={summary['n_records']:<5d} "
            f"hard_yes={o['hard_yes']:.3f}  hvy_loop={o['heavy_loop']:.3f}"
        )


def main():
    args = parse_args()
    GS.set_causal_stub(args.causal_stub)
    logging_utils.configure(args)
    experiment_utils.set_seed(args.seed)
    torch.set_grad_enabled(False)
    results_dir = env_utils.DEFAULT_RESULTS_DIR

    mt = ModelandTokenizer(
        model_key=MODEL,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    n_heads = mt.model.config.num_attention_heads
    heads = load_json_heads(HEADS_JSON, mt.n_layer, n_heads)
    heads_name = os.path.splitext(os.path.basename(HEADS_JSON))[0]
    layers = sorted(
        set(
            list(range(0, 37, 6))
            + list(range(38, 56, 2))
            + list(range(56, mt.n_layer, 4))
            + [mt.n_layer - 1]  # always include the last layer
        )
    )
    logger.info(f"heads[{heads_name}] = {len(heads)} | probe layers = {layers}")

    ov = OVReader(
        mt, heads
    )  # readers cached once, reused across every shard/pivot/datapoint

    base_config = dict(
        model=MODEL,
        mode=args.mode,
        causal_stub=(args.causal_stub if args.mode == "causality" else None),
        heads_spec=HEADS_JSON,
        heads_name=heads_name,
        n_heads=len(heads),
        selected_heads=[[int(lyr), int(h)] for lyr, h in heads],
        alpha=OV_ALPHA,
        skyline_alpha=SKYLINE_ALPHA,
        steer_layer_range=list(STEER_LAYER_RANGE),
        n_conditions=[2, 3, 4],  # the locked corpus' task sizes (leaf: nc=2-3-4)
        lang="python",
        pivots=list(ovio.PIVOT_ORDER),
        max_new_tokens=MAX_NEW_TOKENS,
        gen_max_new_tokens=GEN_MAX_NEW_TOKENS,
        span_source="frontier",  # locked prefill-to-frontier corpus (leaf: span=frontier)
        seed=args.seed,
        layers=layers,
        oracle_model=ORACLE_MODEL,
    )

    for datatype in args.datatypes:
        run_shard(mt, ov, heads, datatype, args, layers, results_dir, base_config)
    logger.info("done.")

if __name__ == "__main__":
    main()
