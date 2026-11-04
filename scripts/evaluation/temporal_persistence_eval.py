#!/usr/bin/env python
"""Temporal persistence of the goal encoding — across datatypes AND languages (Python + C++).

For each (datatype, language) it rolls out full multi-predicate functions, segments each sub-goal's
implementation with an oracle LLM (robust to multi-statement C++ blocks), and records the goal-head
encoding (``--heads``, default the 122 goal heads; 96-token budget) at the `=` pivot + along the implementation tokens.
Retention curves are cos(pivot, encoding at x% of the impl) over relation masks; the figures are
made by ``notebooks/fig_n_table_generators/temporal_persistence.ipynb`` from the cache.

Per-condition records are cached under ``<results>/temporal_persistence/<model_slug>/`` (the control
run under ``.../control/``), so contexts accumulate per sub-task and never recompute.

Usage (the paper's two runs):
    conda run -n goals python scripts/evaluation/temporal_persistence_eval.py --cuda 0
    conda run -n goals python scripts/evaluation/temporal_persistence_eval.py --cuda 0 --control --n-trials 64

``--n-trials 256`` (default) enumerates every distinct context of each (datatype, lang) cell (the
setups saturate at 240); the control run covers the first 64.

Another model: ``--model <hf id> --heads head_sets/<slug>/<goal set>.json``. The head set is part of every cache
key (``TP.heads_key``: the paper's key for the set the paper's runs read, a hash of the set otherwise), so a
different set never reuses another set's records.
"""
import argparse
import logging
import os
import random
import sys

sys.path.append(".")


def _preparse_cuda(argv):
    for i, a in enumerate(argv):
        if a == "--cuda" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--cuda="):
            return a.split("=", 1)[1]
    return None


_cuda = _preparse_cuda(sys.argv[1:])
if _cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = _cuda  # must precede torch import

from scripts.evaluation.ablation_common import random_heads_like  # noqa: E402
from src.goals import temporal_persistence as TP  # noqa: E402
from src.utils import experiment_utils, logging_utils  # noqa: E402

logger = logging.getLogger("temporal_persistence")

CONTROL_SEED = 7  # RNG seed of the control head set (decides the control/ cache content)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    ap.add_argument("--cuda", default=None, help="CUDA_VISIBLE_DEVICES (applied before torch import)")
    ap.add_argument("--heads", default=None,
                    help="goal head set (JSON [[layer, head], ...]; default head_sets/<slug>/"
                         "goal__lam20_thr0.9_band25-56.json, the paper's 122 heads on Coder-32B)")
    ap.add_argument("--n-trials", type=int, default=256, help="contexts sampled per (datatype, lang)")
    ap.add_argument("--seed", type=int, default=1123, help="context-sampling seed")
    ap.add_argument("--control", action="store_true",
                    help="use a matched-count CONTROL head set (later-half of the LM, non-goal) instead of "
                         "the goal heads; records go to a 'control/' sub-dir (head_tag='control')")
    ap.add_argument("--force", action="store_true", help="ignore cache; recompute every condition")
    logging_utils.add_logging_args(ap)
    return ap


def main():
    args = parse_args().parse_args()
    logging_utils.configure(args)
    experiment_utils.set_seed(args.seed)

    slug = args.model.split("/")[-1]
    out_dir = TP.cache_dir(slug)
    os.makedirs(out_dir, exist_ok=True)

    import torch
    from src.models import ModelandTokenizer
    # When --cuda pins a single visible GPU, force the whole model onto it ({"":0}) instead of
    # device_map="auto". "auto" leaves headroom and offloads the LAST layers to CPU/meta on a tight
    # GPU; reading an offloaded head's o_proj.weight OUTSIDE a forward pass then raises
    # "Cannot copy out of meta tensor". The control heads live in the later half (layers 32-63), so
    # they hit exactly those offloaded layers. Weight placement doesn't affect the numeric result.
    single_gpu = args.cuda is not None and "," not in str(args.cuda)
    device_map = {"": 0} if single_gpu else "auto"
    mt = ModelandTokenizer(model_key=args.model, dtype=torch.bfloat16,
                           device_map=device_map, attn_implementation="eager")
    goal_heads = TP.load_goal_heads(mt, slug, args.heads)
    head_tag = "goal"
    heads = goal_heads
    if args.control:
        rng = random.Random(CONTROL_SEED)
        heads = random_heads_like(
            ablated=goal_heads, from_layer=mt.n_layer // 2,
            n_layer=mt.n_layer, n_heads=mt.n_heads, exclude=goal_heads, rng=rng,
        )
        head_tag = "control"
        logger.info(f"{slug}: CONTROL head set — {len(heads)} heads sampled from layers "
                    f"[{mt.n_layer // 2}, {mt.n_layer}) (seed={CONTROL_SEED}), disjoint from goal heads")
    hkey = TP.heads_key(slug, goal_heads, heads)
    logger.info(f"{slug}: {len(heads)} {head_tag} heads (cache key {hkey}) -> {out_dir}")

    per_cell = {}
    for dt in TP.DATATYPES:
        for lang in TP.LANGS:
            recs = TP.build_records(
                mt, slug, heads, dt, lang,
                n_trials=args.n_trials, seed=args.seed, force=args.force, head_tag=head_tag, hkey=hkey,
            )
            per_cell[f"{dt}/{lang}"] = len(recs)
            logger.info(f"  {dt:10s}/{lang:6s}: {len(recs)} conditions")
    logger.info(f"per-cell counts: {per_cell} (total {sum(per_cell.values())})")


if __name__ == "__main__":
    main()
