#!/usr/bin/env python
"""Batch driver of the task-vs-progress 2 x 2 grid (``src.diff.task_vs_progress``; notebook
``notebooks/precommitment/task_vs_progress.ipynb``): which span of SRC the pivot reads (none / task / progress /
task+progress) x method (kv on goal-250 | all heads, V x {1, 4}; residual).

Writes the notebook's per-instance cache (``results/notebook_caches/task_vs_progress/<model>/<dt>__<src>-<dst>__seed..
__mx..__ro..__task+gap/*.json``), so it resumes from entries the notebook already computed and the notebook's results /
inspect cells read its output directly. A per-datatype ``<cache dir>.summary.json`` (causal-shift per config) is
written beside it.

    conda run -n goals python scripts/diff/task_vs_progress_eval.py --cuda 6                           # 4 datatypes x 32
    conda run -n goals python scripts/diff/task_vs_progress_eval.py --cuda 6 --datatypes int --max-instances 2   # smoke
"""

import argparse
import collections
import json
import logging
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Importing the kv eval applies --cuda at MODULE LEVEL (before torch) and pulls in DATA / E / FUNC.
from scripts.diff import kv_transfer_eval as K  # noqa: E402

import src.diff.nrgs_prefix as N  # noqa: E402
import src.diff.path_patch as PP  # noqa: E402
import src.diff.task_vs_progress as TP  # noqa: E402

env_utils, logging_utils = K.env_utils, K.logging_utils
logger = logging.getLogger("task_vs_progress_eval")
GOAL250 = "head_sets/Qwen2.5-Coder-32B-Instruct/goal_topk/goal__lam20_top250.json"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    p.add_argument("--cuda", default=None, help="CUDA_VISIBLE_DEVICES (applied before torch import)")
    p.add_argument("--datatypes", nargs="+", default=["int", "str", "list[int]", "list[str]"])
    p.add_argument("--lang-transfers", nargs="+", default=["py-py"], choices=list(K.LANG_TRANSFERS))
    p.add_argument("--n-samples", type=int, default=32)
    p.add_argument("--max-instances", type=int, default=None, help="cap instances per datatype (smoke)")
    p.add_argument("--seed", type=int, default=1123)
    p.add_argument("--goal-heads", default=GOAL250, help="JSON [[layer, head], ...] for the 'goal250' head set")
    p.add_argument("--readout", action="store_true", help="GoalScope readouts at ` =` (default off; cache tag ro1)")
    p.add_argument("--max-new", type=int, default=64)
    p.add_argument("--oracle-model", default="claude-sonnet-4-6")
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--cache-root", default=None, help="override the results dir the cache lives under (smoke)")
    return p.parse_args()


def main():
    args = parse_args()
    logging_utils.configure(level=logging.INFO, force=True)
    model_short = args.model.split("/")[-1]
    import torch
    device_map = {"": 0} if torch.cuda.device_count() == 1 else "auto"   # one GPU: pin (GoalScope reads o_proj weights)
    ctx = N.load_ctx(args.model, device_map=device_map)
    mt = ctx.mt
    goal = [tuple(h) for h in json.loads(Path(args.goal_heads).read_text())]
    headsets = {"goal250": goal, "all": [(l, h) for l in range(mt.n_layer) for h in range(mt.n_heads)]}
    ctx.readout_heads, ctx.union_heads, ctx.gsp = goal, None, None
    root = args.cache_root or env_utils.DEFAULT_RESULTS_DIR
    logger.info(f"{len(TP.CONFIGS)} configs: " + " | ".join(TP.cfg_name(*c) for c in TP.CONFIGS))

    for lt in args.lang_transfers:
        sl, dl = K.LANG_TRANSFERS[lt]
        for dt in args.datatypes:
            insts, _ = PP.sample_multi_instances(dt, sl, dl, n_samples=args.n_samples, seed=args.seed)
            if args.max_instances:
                insts = insts[: args.max_instances]
            cdir = TP.cache_dir(root, model_short, dt, sl, dl, seed=args.seed, max_new=args.max_new, readout=args.readout)
            os.makedirs(cdir, exist_ok=True)
            logger.info(f"═══ {lt} / {dt} · {len(insts)} instances · cache {cdir} ═══")
            sweep = collections.defaultdict(list)
            tot_run = tot_hit = 0
            for i, inst in enumerate(insts):
                t0 = time.time()
                rows, n_run, n_hit = TP.sweep_instance(
                    inst, i, TP.CONFIGS, cdir, headsets=headsets, readout_heads=goal, readout=args.readout,
                    max_new=args.max_new, oracle_model=args.oracle_model, resume=args.resume, ctx=ctx)
                tot_run, tot_hit = tot_run + n_run, tot_hit + n_hit
                for name, r in rows.items():
                    sweep[name].append(r)
                line = " ".join("■" if rows[TP.cfg_name(*c)]["shifted"] else "·" for c in TP.CONFIGS)
                logger.info(f"[{dt}] ({i + 1}/{len(insts)}) {inst.src_frontier.id} vs {inst.dst_frontier.id}: {line} "
                            f"· run {n_run} / cached {n_hit} ({time.time() - t0:.0f}s)")
            summ = {name: dict(n=len(rs), causal_shift=sum(r["shifted"] for r in rs) / len(rs),
                               readout_shift=(sum(bool(r["ro_shifted"]) for r in rs) / len(rs)) if args.readout else None,
                               verdicts=dict(collections.Counter(r["causal"] for r in rs)))
                    for name, rs in sweep.items()}
            Path(cdir + ".summary.json").write_text(json.dumps(summ, indent=2))   # beside the cache dir, not in it
            for name, s in summ.items():
                logger.info(f"[{dt}] SUMMARY {name:34s} causal-shift={s['causal_shift']:.2f} n={s['n']}")
            logger.info(f"[{dt}] configs run: {tot_run} · from cache: {tot_hit}")


if __name__ == "__main__":
    main()
