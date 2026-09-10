#!/usr/bin/env python
"""SUFFICIENCY test: are the goal heads (+ the first FROM_LAYER layers) enough to implement next(DIFF)?

Ablate (attention-sink to sink_pos) EVERY head in layers >= --from_layer EXCEPT the goal heads; layers
< --from_layer stay fully intact. Greedy-decode the frontier value after ``{var} =`` under that
ablation and grade whether it still implements the frontier's description. Compared against a
no-ablation baseline and a matched-count random-head control (same size, same layer range).

If the implement-rate stays near baseline while a random control collapses, the goal heads carry the
next-sub-goal computation -> SUFFICIENT. Shared machinery: scripts/evaluation/ablation_common.py.

Usage:
    conda run -n goals python scripts/evaluation/eval_ablation_sufficiency.py --from_layer 20
    # smoke: ... --limit 8
"""
import sys

sys.path.append(".")

from scripts.evaluation.ablation_common import (  # noqa: E402
    common_argparser, complement_heads, load_setup, run_ablation_eval,
)


def main():
    args = common_argparser(__doc__).parse_args()
    mt, goal_heads, datapoints, config, results_dir = load_setup(args)
    ablated = complement_heads(goal_heads, args.from_layer, mt.n_layer, mt.n_heads)
    run_ablation_eval(
        mt, datapoints, ablated, goal_heads,
        axis="sufficiency", ablated_name="complement_of_goal", config=config,
        results_dir=results_dir, from_layer=args.from_layer, sink_pos=args.sink_pos,
        gen_max_new_tokens=args.gen_max_new_tokens, seed=args.seed, overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
