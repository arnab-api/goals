#!/usr/bin/env python
"""Export the heads of a trained NRGS query mask (``train_nrgs_mask.py`` -> ``mask.pt``) as a head set.

Selection = mask value > ``--threshold`` over ALL candidate layers (optionally ``--band lo-hi``), read from
the saved full-depth mask. The head set's filename label (stem before ``__``) becomes the run-tag label of
every q-transfer eval that patches it (``ph-<label>_<n>...``), so pick a label that is unique per run.
Refuses to overwrite an existing head set that holds different heads.

Usage:
    python scripts/diff/select_nrgs_heads.py \\
        --mask results/nrgs_mask/<model>/<run_tag>/mask.pt \\
        --out head_sets/<model>/progress-mix-lam20__nrgs_mixed_thr0.9.json --threshold 0.9
"""
import argparse
import collections
import json
import os
import sys
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mask", required=True, help="mask.pt written by train_nrgs_mask.py")
    ap.add_argument("--out", required=True, help="head-set JSON to write ([[layer, head], ...])")
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--band", default=None, help="optional layer band lo-hi (half-open); default all layers")
    args = ap.parse_args()

    ck = torch.load(args.mask, map_location="cpu", weights_only=False)
    if not ck.get("final_eval"):
        sys.exit(f"{args.mask}: no final_eval -> training has not finished")
    mv = ck["mask_values"]                                        # full depth: row == layer
    lo, hi = (int(x) for x in args.band.split("-")) if args.band else (0, mv.shape[0])
    heads = sorted((l, h) for l in range(lo, min(hi, mv.shape[0])) for h in range(mv.shape[1])
                   if float(mv[l, h]) > args.threshold)
    if not heads:
        sys.exit("no head above threshold")
    out = Path(args.out)
    if out.exists() and sorted(map(tuple, json.loads(out.read_text()))) != heads:
        sys.exit(f"{out} exists with DIFFERENT heads — not overwriting")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps([list(h) for h in heads]))
    os.replace(tmp, out)
    blocks = collections.Counter(l // 8 * 8 for l, _ in heads)
    print(f"{len(heads)} heads (mask > {args.threshold}, layers {lo}-{hi}) -> {out}")
    print("  per 8-layer block:", {f"L{k}-{k + 7}": v for k, v in sorted(blocks.items())})
    print("  training config:", {k: ck["config"].get(k) for k in ("data_tag", "configs", "lambda_sparse", "epochs")})


if __name__ == "__main__":
    main()
