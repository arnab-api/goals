#!/usr/bin/env python
"""Diff-vs-progress q-transfer — "evidence against the DIFF hypothesis".

A duplicate of ``q_transfer_eval.py`` specialised to ONE contrastive setup
(:func:`src.diff.data.diff_vs_prog_setups`)::

    SRC.task = [T1, T2, T3, T4],  SRC.prefill = {T1, T2},  SRC.target = T3   (transplant source)
    DST.task = [T1, T2, T4, T3],  DST.prefill = {T1},      DST.native = T2

Transplant SRC's pivot goal-head query into DST and score (binary causal oracle) whether DST writes
**T3** — the next sub-goal carried PRE-DECIDED in the query — instead of its native T2. The DST
reordering (T4 before T3) is what makes T3 (pre-decided) separable from T4 (progress-position after
removing SRC's {T1,T2}); a high T3-trigger rate says the query carries the fully-decided next(DIFF),
not merely progress. See ``Overleaf-draft/REF/q-transfer-diff-vs-prog.png``.

Everything except the corpus builder is reused from ``q_transfer_eval`` (query transplant, oracle
judge, record/summary schema). Results -> ``<results>/diff/<model>/q_transfer_diff_vs_prog/<run_tag>/<cfg>/``.

Usage:
    conda run -n goals python scripts/diff/q_transfer_diff_vs_prog_eval.py --cuda 0 --n-samples 128
"""
import argparse
import dataclasses
import hashlib
import json
import logging
import sys
from pathlib import Path

sys.path.append(".")

# Importing the q-transfer eval applies --cuda at MODULE LEVEL (before torch) and pulls in D/E/P.
from scripts.diff import q_transfer_eval as Q  # noqa: E402

D, E, P = Q.D, Q.E, Q.P
env_utils, logging_utils, set_seed = Q.env_utils, Q.logging_utils, Q.set_seed
LANG_TRANSFERS = Q.LANG_TRANSFERS
logger = logging.getLogger("q_transfer_diff_vs_prog")

MECHANISM = "q_transfer_diff_vs_prog"  # results: diff/<model>/<MECHANISM>/<run_tag>/<config_tag>/


def run_config(mt, args, dt, naming, sl, dl, patch_heads, readout_heads, out_root):
    tag = f"{Q._safe(dt)}__{naming}__{sl}-{dl}"
    outdir = out_root / tag
    outdir.mkdir(parents=True, exist_ok=True)
    results_fp = outdir / "results.json"

    done = {}
    if results_fp.exists() and args.resume and not args.force:
        done = {r["key"]: r for r in json.loads(results_fp.read_text())}
        logger.info(f"[{tag}] resume: {len(done)} instances already done")

    cfg_seed = (args.seed + int(hashlib.md5(tag.encode()).hexdigest()[:8], 16)) % (2**32)
    set_seed(cfg_seed)
    setups = D.diff_vs_prog_setups(
        dt, n_samples=args.n_samples, seed=cfg_seed,
        presentation="listed", naming=naming, src_lang=sl, dst_lang=dl,
    )
    if args.max_instances:
        setups = setups[: args.max_instances]

    records = dict(done)
    todo = [i for i in setups if Q._inst_key(i) not in records or args.force]
    logger.info(f"[{tag}] sampled {len(setups)} instances (seed={cfg_seed}, alpha={args.alpha}) — "
                f"{len(setups) - len(todo)} resumed, {len(todo)} to run")

    for j, inst in enumerate(todo, 1):
        key = Q._inst_key(inst)
        logger.info(f"[{tag}] ({j}/{len(todo)}) SRC target {inst.expected_next!r}(T3) vs "
                    f"DST native {inst.meta['dst_next']!r}(T2)  |  {inst.expected_next_desc!r}")
        raw = P.run_q_transfer_intervention(
            mt, inst, patch_heads, readout_heads=readout_heads,
            n_steer=args.n_steer, max_new=args.max_new, alpha=args.alpha, n_pivot=1,
        )
        causal = readout = None
        if not args.no_oracle:
            causal = E.judge_causal(inst, raw["gen_patched"], model_name=args.oracle_model)
            readout = E.eval_readouts(inst, raw["ro_src"], raw["ro_dst"], raw["ro_patched"],
                                      model_name=args.oracle_model)
        _c = (f"{'HIT' if causal['triggered'] else 'miss'}({causal['label']})" if causal else "off")
        logger.info(f"[{tag}]   => wrote_T3(causal)={_c}")

        src_prog, dst_prog = Q._progress_of(inst)
        rec = Q.InstanceRecord(
            key=key, setup="diff_vs_prog", datatype=dt, naming=naming, presentation="listed",
            src_lang=sl, dst_lang=dl, var_name=inst.var_name,
            expected_next=inst.expected_next, expected_next_desc=inst.expected_next_desc,
            dst_next=inst.meta["dst_next"], dst_next_desc=inst.dst_next_desc,
            src_progress=src_prog, dst_progress=dst_prog,
            raw=raw, causal=causal, readout=readout, gaze=None,  # gaze N/A (divergent set)
        )
        records[key] = dataclasses.asdict(rec)
        tmp = outdir / "results.json.tmp"
        tmp.write_text(json.dumps(list(records.values()), indent=2))
        tmp.replace(results_fp)

    recs = [Q.InstanceRecord(**r) for r in records.values()]
    meta = Q.ConfigMeta(
        model=args.model, setup="diff_vs_prog", datatype=dt, naming=naming, presentation="listed",
        src_lang=sl, dst_lang=dl,
        patch_heads=Q._patch_heads_tag(args.patch_heads, len(patch_heads)),
        oracle_model=args.oracle_model, n_instances=len(recs),
        readout_heads=Q._patch_heads_tag(args.readout_heads, len(readout_heads)),
        alpha=args.alpha, n_pivot=1, **Q._summarize(recs),
    )
    (outdir / "meta.json").write_text(json.dumps(dataclasses.asdict(meta), indent=2))
    logger.info(f"[{tag}] done — {len(recs)} instances | "
                f"T3-trigger(causal)={meta.causal_trigger_rate}")
    return meta


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    p.add_argument("--cuda", default=None, help="CUDA_VISIBLE_DEVICES (applied before torch import)")
    p.add_argument("--datatypes", nargs="+", default=None, choices=sorted(D.QTASKS),
                   help="datatypes to run (default: all four QTASKS banks; need >=4 predicates each)")
    p.add_argument("--namings", nargs="+", default=["letters"], choices=["natural", "letters"],
                   help="result-variable naming axis (default: letters a/b/c/d, as in the figure)")
    p.add_argument("--lang-transfers", nargs="+", default=["py-py"], choices=list(LANG_TRANSFERS))
    p.add_argument("--patch-heads", default="goal", help="query-transplant heads: goal | union | JSON path")
    p.add_argument("--readout-heads", default="goal", help="GoalScope readout heads: goal | union | JSON path")
    p.add_argument("--n-samples", type=int, default=128, help="diff-vs-prog instances per config")
    p.add_argument("--seed", type=int, default=1123)
    p.add_argument("--max-instances", type=int, default=None, help="further cap instances/config")
    p.add_argument("--n-steer", type=int, default=32)
    p.add_argument("--max-new", type=int, default=64)
    p.add_argument("--alpha", type=float, default=2.0,
                   help="OV boost of the patch heads at the decision token (1.0 = off)")
    p.add_argument("--oracle-model", default="claude-sonnet-4-6")
    p.add_argument("--no-oracle", action="store_true", help="skip the causal+readout oracle")
    p.add_argument("--force", action="store_true", help="re-run instances even if present")
    p.add_argument("--no-resume", dest="resume", action="store_false", help="ignore prior results")
    return p.parse_args()


def main():
    args = parse_args()
    logging_utils.configure(level=logging.INFO, force=True)

    ctx = P.setup(args.model)  # loads model + head masks; binds module globals
    mt = ctx.mt
    patch_heads = Q._resolve_patch_heads(ctx, args.patch_heads)
    readout_heads = Q._resolve_patch_heads(ctx, args.readout_heads)
    ph_tag = Q._patch_heads_tag(args.patch_heads, len(patch_heads))
    ro_tag = Q._patch_heads_tag(args.readout_heads, len(readout_heads))
    logger.info(f"patch_heads={args.patch_heads}->{ph_tag}(n={len(patch_heads)}) · "
                f"readout_heads={args.readout_heads}->{ro_tag}(n={len(readout_heads)})")

    model_short = args.model.split("/")[-1]
    run_tag = Q._run_tag(ph_tag, ro_tag, args.alpha, "auto")
    datatypes = args.datatypes or sorted(D.QTASKS)
    out_root = Path(env_utils.DEFAULT_RESULTS_DIR) / "diff" / model_short / MECHANISM / run_tag
    out_root.mkdir(parents=True, exist_ok=True)
    logger.info(f"run_tag={run_tag} · datatypes={datatypes} · namings={args.namings}")

    metas = []
    for dt in datatypes:
        for naming in args.namings:
            for lt in args.lang_transfers:
                sl, dl = LANG_TRANSFERS[lt]
                logger.info(f"═══ {dt}/{naming}/{lt} ═══")
                metas.append(run_config(mt, args, dt, naming, sl, dl, patch_heads, readout_heads, out_root))
    (out_root / "summary.json").write_text(json.dumps([dataclasses.asdict(m) for m in metas], indent=2))
    logger.info(f"all done — {len(metas)} configs; summary at {out_root/'summary.json'}")


if __name__ == "__main__":
    main()
