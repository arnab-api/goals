#!/usr/bin/env python
"""Export per-model GOAL head sets to git-tracked JSON under head_sets/<slug>/.

The trained masks live in results/localization/<slug>/<tag>/head_mask.pt (a results artifact, not
tracked). This freezes the derived goal-head SELECTION as a small ``[[layer, head], ...]`` json so the exact
head set used by the evals is versioned. No model is loaded — this only reads the mask tensor.

Two kinds of set:

- the goal set (G-small, default): mask > 0.9 inside the model's band (``MODEL_BANDS``; the same bands the
  evals select with, ``GOAL_BANDS``, Coder-14B 0-48) ->
      head_sets/<slug>/goal__<tag>_thr0.9_band<a>-<b>.json
  (naming as eval_ov_reader.DEFAULT_HEADS_JSON). ``<tag>`` is ``MODEL_TAGS[model]``, else lam20 (Llama-3.3-70B's
  mask is lam20_ctx). A band of None is derived from the mask by ``src.goals.head_sets.band_rule``
  (repro/PLAN.md 2.3: the smallest contiguous layer window holding >= 85% of the heads with mask > 0.9, the
  lowest of equally narrow ones); the window, its per-layer counts and its size are printed, to be pinned in
  MODEL_BANDS and GOAL_BANDS.
- ``--topk K``: the top-K heads by mask value over all layers (``np.argsort(-mask.ravel(), kind="stable")[:K]``,
  ties by index; the Coder-32B goal_topk files' recipe) ->
      head_sets/<slug>/goal_topk/goal__<tag>_top<K>.json
  and with ``--control`` the matching random control: K heads drawn with ``random.Random(123).sample`` from the
  heads of layers >= L // 2 that are not in the top-K set ->
      head_sets/<slug>/control/control<K>__laterhalf.json
  (both recipes reproduce Coder-32B's goal__lam20_top{150,200,250}.json and control250__laterhalf.json exactly
  from its lam20 mask).

Usage:
    conda run -n goals python scripts/data/export_head_sets.py                       # every model with a mask
    python scripts/data/export_head_sets.py --models zai-org/GLM-4.5-Air             # one model (band rule)
    python scripts/data/export_head_sets.py --models zai-org/GLM-4.5-Air --topk 431 --control
    python scripts/data/export_head_sets.py --results-dir <tree> --out-dir /tmp/hs   # check against a results tree
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from src.goals.head_sets import band_rule, goal_heads_from_mask  # noqa: E402
from src.utils import env_utils  # noqa: E402

THRESHOLD = 0.9
TAG = "lam20"
CONTROL_SEED = 123  # the control sets' sampling seed

# model -> goal band (layers heads are selected from). Matches the eval selection per model (GOAL_BANDS).
# None: derived from the mask by band_rule at export (printed; pin it here and in GOAL_BANDS once the mask is final).
MODEL_BANDS = {
    "Qwen/Qwen2.5-Coder-32B-Instruct": (25, 56),
    "Qwen/Qwen2.5-32B-Instruct": (25, 56),
    "Qwen/Qwen2.5-Coder-14B-Instruct": (0, 48),   # model-axis runs pass 0-48
    "meta-llama/Llama-3.3-70B-Instruct": (18, 45),  # MODEL_TAGS: lam20_ctx
    "allenai/Olmo-3.1-32B-Instruct": (10, 36),
    "allenai/Olmo-3.1-32B-Instruct-SFT": (10, 36),
    "zai-org/GLM-4.5-Air": None,
    "mistralai/Devstral-Small-2-24B-Instruct-2512": None,
}
# model -> mask tag (results/localization/<slug>/<tag>/head_mask.pt), default TAG
MODEL_TAGS = {
    "meta-llama/Llama-3.3-70B-Instruct": "lam20_ctx",  # 1433 instances, 50 epochs, batch 8, no-augment
}


def load_mask(model, tag, results_dir):
    ck = torch.load(Path(results_dir) / "localization" / model.split("/")[-1] / tag / "head_mask.pt",
                    map_location="cpu")
    return ck["mask_values"].numpy(), [int(x) for x in ck["layers"]]


def topk_heads(mask, layers, k):
    """The top-``k`` heads by mask value over all mask rows, ties by flat index -> sorted [(layer, head)]."""
    n_heads = mask.shape[1]
    order = np.argsort(-mask.ravel(), kind="stable")[:k]
    return sorted((layers[int(i) // n_heads], int(i) % n_heads) for i in order)


def control_heads(goal, n_layer, n_heads, k, seed=CONTROL_SEED):
    """``k`` random heads of layers ``>= n_layer // 2`` that are not in ``goal``, sorted."""
    goal = set(goal)
    pool = [(l, h) for l in range(n_layer // 2, n_layer) for h in range(n_heads) if (l, h) not in goal]
    return sorted(random.Random(seed).sample(pool, k))


def write(path, heads, out_dir):
    path = Path(out_dir) / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([[int(l), int(h)] for l, h in heads]))
    return path


def export_goal(model, band, tag, results_dir, out_dir):
    slug = model.split("/")[-1]
    if band is None:
        mask, layers = load_mask(model, tag, results_dir)
        rule = band_rule(mask, layers, threshold=THRESHOLD)
        band = rule["band"]
        print(f"  {slug}: band rule -> [{band[0]}, {band[1]}) holds {rule['n_in_band']} of {rule['n_total']} heads "
              f"> {THRESHOLD}; per layer {rule['per_layer']} — pin it in MODEL_BANDS and GOAL_BANDS")
    heads = goal_heads_from_mask(model, results_dir, threshold=THRESHOLD, band=band, tag=tag)
    heads = sorted((int(l), int(h)) for l, h in heads)
    path = write(f"{slug}/goal__{tag}_thr{THRESHOLD}_band{band[0]}-{band[1]}.json", heads, out_dir)
    print(f"  {slug:36} band{band[0]}-{band[1]}  ->  {len(heads):3} heads  {path}")


def export_topk(model, tag, k, results_dir, out_dir, control):
    slug = model.split("/")[-1]
    mask, layers = load_mask(model, tag, results_dir)
    heads = topk_heads(mask, layers, k)
    path = write(f"{slug}/goal_topk/goal__{tag}_top{k}.json", heads, out_dir)
    print(f"  {slug:36} top{k} (mask value of the last: {np.sort(mask.ravel())[::-1][k - 1]:.4g})  ->  {path}")
    if control:
        n_layer, n_heads = layers[-1] + 1, mask.shape[1]
        ctrl = control_heads(heads, n_layer, n_heads, k)
        path = write(f"{slug}/control/control{k}__laterhalf.json", ctrl, out_dir)
        print(f"  {slug:36} control{k}: layers >= {n_layer // 2}, not in top{k}, seed {CONTROL_SEED}  ->  {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=None, help="HF ids (default: every model of MODEL_BANDS)")
    ap.add_argument("--topk", nargs="+", type=int, default=None,
                    help="write the top-K sets (each K) instead of the goal set; repro/PLAN.md 2.2 proposes "
                         "K = round(0.0977 * layers * heads), Coder-32B's 250 of 2560")
    ap.add_argument("--control", action="store_true", help="with --topk: also the matching later-half control")
    ap.add_argument("--tag", default=None, help="mask tag (default MODEL_TAGS[model], else lam20)")
    ap.add_argument("--results-dir", default=env_utils.DEFAULT_RESULTS_DIR, help="where results/localization is")
    ap.add_argument("--out-dir", default=REPO_ROOT / "head_sets", help="default the tracked head_sets/")
    args = ap.parse_args()
    if args.control and not args.topk:
        ap.error("--control goes with --topk")

    models = args.models or list(MODEL_BANDS)
    for model in models:
        slug = model.split("/")[-1]
        tag = args.tag or MODEL_TAGS.get(model, TAG)
        if not args.topk and model not in MODEL_BANDS:
            print(f"  [skip] {slug}: no MODEL_BANDS entry (add one; None derives it by the band rule)")
            continue
        try:
            if args.topk:
                for k in args.topk:
                    export_topk(model, tag, k, args.results_dir, args.out_dir, args.control)
            else:
                export_goal(model, MODEL_BANDS[model], tag, args.results_dir, args.out_dir)
        except FileNotFoundError:
            print(f"  [skip] {slug}: no {tag} head_mask.pt under {args.results_dir}/localization/")


if __name__ == "__main__":
    main()
