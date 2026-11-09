#!/usr/bin/env python
"""NECESSITY test: are the goal heads required to implement next(DIFF)?

Attention-sink ablation of the GOAL heads: for query rows >= the pivot they may attend ONLY to the
sink position 0 (<|im_start|>) — ~zeroing their useful attention output — while every other head runs
normally. Greedy-decode the frontier value and grade whether it still implements the frontier
description.

Control (matched): apply the SAME intervention to a RANDOM |goal|-size set drawn from the later half
of the LM (layers >= n_layer//2, disjoint from the goal heads). If ablating the goal heads hurts no
more than a random set, they are NOT necessary (the computation is redundant / reroutes).

Shared machinery: scripts/evaluation/ablation_common.py.

Usage:
    conda run -n goals python scripts/evaluation/eval_ablation_necessity.py              # 122 goal heads
    conda run -n goals python scripts/evaluation/eval_ablation_necessity.py \\
        --heads head_sets/Qwen2.5-Coder-32B-Instruct/goal_topk/goal__lam20_top250.json   # 250 goal heads
    # smoke: ... --limit 12
"""
import random
import sys

sys.path.append(".")

from scripts.evaluation.ablation_common import (  # noqa: E402
    common_argparser, load_setup, random_heads_like, run_ablation_eval,
)


def main():
    args = common_argparser(__doc__).parse_args()
    mt, goal_heads, datapoints, config, results_dir = load_setup(args)

    # ablate exactly the goal heads; matched control = a random |goal| set (disjoint from the goal heads)
    # drawn from the later half of the LM.
    ctrl_from = mt.n_layer // 2
    rng = random.Random(args.seed)
    control = random_heads_like(goal_heads, ctrl_from, mt.n_layer, mt.n_heads, goal_heads, rng)

    run_ablation_eval(
        mt, datapoints, goal_heads, control, goal_heads,
        axis="necessity", ablated_name="goal", strategy="sink", config=config,
        results_dir=results_dir, from_layer=ctrl_from, overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
