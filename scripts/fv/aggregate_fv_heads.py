#!/usr/bin/env python
"""Aggregate per-relation Causal Indirect Effect (CIE) tensors into the *universal*
Function-Vector head set.

Each `localize_fv_cma.py` run writes `<results_root>/<dataset>/<dataset>_indirect_effect.pt`
of shape `(n_examples_per_relation, n_layers, n_heads)`. The paper's universal FV heads are
the top heads by indirect effect **averaged across the task suite**. We reproduce that here:

    per-task mean over examples  ->  (n_layers, n_heads) per task
    mean across tasks            ->  (n_layers, n_heads) universal
    top-k over the flattened grid ->  [(layer, head, score), ...]

Task-equal weighting by default (each relation contributes equally regardless of how many
examples it swept); use --weight example to instead weight every example equally.

Usage:
  python scripts/fv/aggregate_fv_heads.py \
      --results_root existing_works/function_vectors/results \
      --n_top_heads 10 --out existing_works/function_vectors/results/universal_fv_heads.json
"""
import argparse
import glob
import json
import os

import numpy as np
import torch


def find_datasets(results_root):
    """Datasets that have an indirect_effect.pt under results_root/<ds>/<ds>_indirect_effect.pt."""
    found = []
    for sub in sorted(glob.glob(os.path.join(results_root, "*"))):
        if not os.path.isdir(sub):
            continue
        ds = os.path.basename(sub)
        if os.path.exists(os.path.join(sub, f"{ds}_indirect_effect.pt")):
            found.append(ds)
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_root", required=True,
                    help="dir holding per-dataset subdirs (e.g. existing_works/function_vectors/results)")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="subset of datasets to aggregate (default: every one with an indirect_effect.pt)")
    ap.add_argument("--n_top_heads", type=int, default=10)
    ap.add_argument("--weight", choices=["task", "example"], default="task",
                    help="'task' = each relation weighted equally (paper default); "
                         "'example' = each swept example weighted equally")
    ap.add_argument("--out", default=None, help="path to write the universal-head JSON (and .pt sibling)")
    ap.add_argument("--model_name", default=None, help="recorded in the output metadata only")
    args = ap.parse_args()

    datasets = args.datasets or find_datasets(args.results_root)
    if not datasets:
        raise SystemExit(f"no *_indirect_effect.pt found under {args.results_root}")

    per_task_mean, all_examples, used, missing = [], [], [], []
    n_layers = n_heads = None
    for ds in datasets:
        path = os.path.join(args.results_root, ds, f"{ds}_indirect_effect.pt")
        if not os.path.exists(path):
            missing.append(ds)
            continue
        ie = torch.load(path, map_location="cpu").float()  # (trials, L, H)
        if ie.dim() != 3:
            raise SystemExit(f"{path}: expected (trials, L, H), got {tuple(ie.shape)}")
        n_layers, n_heads = ie.shape[1], ie.shape[2]
        per_task_mean.append(ie.mean(dim=0))    # (L, H)
        all_examples.append(ie)                  # (trials, L, H)
        used.append(ds)

    if args.weight == "task":
        universal = torch.stack(per_task_mean).mean(dim=0)          # (L, H), task-equal
    else:
        universal = torch.cat(all_examples, dim=0).mean(dim=0)      # (L, H), example-equal

    k = min(args.n_top_heads, universal.numel())
    vals, inds = torch.topk(universal.view(-1), k=k, largest=True)
    ls, hs = np.unravel_index(inds.numpy(), (n_layers, n_heads))
    top_heads = [[int(l), int(h), round(float(v), 5)] for l, h, v in zip(ls, hs, vals)]

    print(f"# universal FV heads — {len(used)} relations, weight={args.weight}, "
          f"grid {n_layers}L x {n_heads}H — top {k}")
    for r, (l, h, v) in enumerate(top_heads, 1):
        print(f"  {r:3d}. layer {l:3d} head {h:3d}   avg_indirect_effect={v:+.5f}")
    if missing:
        print(f"# skipped {len(missing)} missing: {', '.join(missing)}")

    if args.out:
        payload = {
            "model_name": args.model_name,
            "results_root": args.results_root,
            "weight": args.weight,
            "n_layers": n_layers, "n_heads": n_heads,
            "n_top_heads": k,
            "datasets_used": used,
            "n_datasets": len(used),
            "datasets_missing": missing,
            "top_heads": top_heads,
        }
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        pt_path = os.path.splitext(args.out)[0] + "_grid.pt"
        torch.save(universal, pt_path)
        print(f"wrote {args.out}\nwrote {pt_path}  (full {n_layers}x{n_heads} averaged grid)")


if __name__ == "__main__":
    main()
