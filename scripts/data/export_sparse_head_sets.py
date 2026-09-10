#!/usr/bin/env python
"""Export the VERY-sparse goal head sets (top-5 / 10 / 25) for Qwen2.5-Coder-32B to head_sets/.

The lam20 goal mask saturates: 107 band heads sit at exactly 1.0, so "top-k by mask value" is
undefined for k<=~100. Instead we rank by TRAINING DYNAMICS across the cached mask checkpoints:
a head's importance = the number of (lambda, epoch) snapshots in which it is already at 1.0 (it
reaches the ceiling both under strong sparsity AND early in training). Snapshots = the checkpoints
under results/localization/<model>/lam{1,5,20,50,100}/checkpoints/ (10 epochs each -> 50 snapshots).

This yields, within the goal band (25-56), clean nested sets:
    top-10 = the 10 heads at 1.0 in all 50 snapshots
    top-25 = the 25 heads ranked by (#snaps@1.0, mean mask value)
The 10 core heads are 1.0 in every cached snapshot, so nothing in the cache separates them ->
    top-5  = a fixed RANDOM sample of 5 of the top-10 (seeded; flagged as arbitrary, not a ranking).

Files: head_sets/<slug>/goal_topk/goal__snap_top{5,10,25}.json  ([[layer,head],...]).
Coder-32B only (the sparsity axis is single-model).

Usage:  conda run -n goals python scripts/data/export_sparse_head_sets.py
"""
import argparse
import glob
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))
from src.utils import env_utils  # noqa: E402

MODEL_SLUG = "Qwen2.5-Coder-32B-Instruct"
LAMBDAS = ["lam1", "lam5", "lam20", "lam50", "lam100"]
BAND = (25, 56)
SIZES = [5, 10, 25]


def _epoch(f):
    return int(re.search(r"(\d+)", Path(f).name).group())


def rank_band_heads():
    loc = Path(env_utils.DEFAULT_RESULTS_DIR) / "localization" / MODEL_SLUG
    snaps, layers = [], None
    for tag in LAMBDAS:
        for f in sorted(glob.glob(str(loc / tag / "checkpoints" / "*.pt")), key=_epoch):
            c = torch.load(f, map_location="cpu")
            if layers is None:
                layers = [int(x) for x in c["layers"]]
            snaps.append(c["mask_values"].numpy())
    if not snaps:
        raise SystemExit(f"no checkpoints under {loc}/lam*/checkpoints/")
    S = np.stack(snaps)  # (N_snap, L, H)
    inb = np.array([BAND[0] <= l < BAND[1] for l in layers])
    ones = (S == 1.0).sum(0)
    meanv = S.mean(0)
    scored = [
        (int(ones[li, h]), float(meanv[li, h]), l, h)
        for li, l in enumerate(layers) if inb[li]
        for h in range(S.shape[2])
    ]
    scored.sort(key=lambda t: (-t[0], -t[1]))
    return S.shape[0], scored


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=123, help="seed for the random top-5 of the top-10")
    args = ap.parse_args()

    n_snap, scored = rank_band_heads()
    out_dir = REPO_ROOT / "head_sets" / MODEL_SLUG / "goal_topk"
    out_dir.mkdir(parents=True, exist_ok=True)

    top10 = [(l, h) for _, _, l, h in scored[:10]]
    top25 = [(l, h) for _, _, l, h in scored[:25]]
    top5 = sorted(random.Random(args.seed).sample(top10, 5))
    sets = {5: top5, 10: sorted(top10), 25: sorted(top25)}

    for k in SIZES:
        heads = sets[k]
        path = out_dir / f"goal__snap_top{k}.json"
        path.write_text(json.dumps([[int(l), int(h)] for l, h in heads]))
        note = "  (random 5 of the top-10; seed=%d)" % args.seed if k == 5 else ""
        print(f"  top{k:2}: {len(heads)} heads -> {path.relative_to(REPO_ROOT)}{note}")
    # nesting sanity
    assert set(top5) <= set(top10) <= set(top25), "sets must be nested"
    print(f"nested top5 ⊂ top10 ⊂ top25 over {n_snap} snapshots — OK")


if __name__ == "__main__":
    main()
