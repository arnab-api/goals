#!/usr/bin/env python
"""Export per-model GOAL head sets to git-tracked JSON under head_sets/<slug>/.

The trained masks live in results/localization/<slug>/lam20/head_mask.pt (a results artifact, not
tracked). This freezes the derived goal-head SELECTION (threshold 0.9, per-model band) as a small
``[[layer, head], ...]`` json so the exact head set used by the goalscope/OVReader evals is versioned.

Naming matches the eval convention (eval_ov_reader.DEFAULT_HEADS_JSON):
    head_sets/<slug>/goal__lam20_thr0.9_band<a>-<b>.json

Bands are the same ones the evals select with (GOAL_BANDS, except Coder-14B which the model-axis runs
pass explicitly as 0-48). No model is loaded — this only reads the mask tensor.

Usage:  conda run -n goals python scripts/data/export_head_sets.py
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from scripts.evaluation.eval_goalscope import load_goal_heads  # noqa: E402
from src.utils import env_utils  # noqa: E402

THRESHOLD = 0.9
TAG = "lam20"

# model -> goal band (layers heads are selected from). Matches the eval selection per model.
MODEL_BANDS = {
    "Qwen/Qwen2.5-Coder-32B-Instruct": (25, 56),
    "Qwen/Qwen2.5-32B-Instruct": (25, 56),
    "Qwen/Qwen2.5-Coder-14B-Instruct": (0, 48),   # no GOAL_BANDS entry; model-axis passes 0-48
    "allenai/Olmo-3.1-32B-Instruct": (10, 36),
    "allenai/Olmo-3.1-32B-Instruct-SFT": (10, 36),
}


def main():
    results_dir = env_utils.DEFAULT_RESULTS_DIR
    for model, band in MODEL_BANDS.items():
        slug = model.split("/")[-1]
        try:
            heads = load_goal_heads(model, results_dir, threshold=THRESHOLD, band=band, tag=TAG)
        except FileNotFoundError:
            print(f"  [skip] {slug}: no {TAG} head_mask.pt under results/localization/")
            continue
        heads = sorted((int(l), int(h)) for l, h in heads)
        out_dir = REPO_ROOT / "head_sets" / slug
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"goal__{TAG}_thr{THRESHOLD}_band{band[0]}-{band[1]}.json"
        out_path.write_text(json.dumps([[l, h] for l, h in heads]))
        print(f"  {slug:28} band{band[0]}-{band[1]}  ->  {len(heads):3} heads  "
              f"{out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
