#!/usr/bin/env python
"""NECESSITY test: are the goal heads required to implement next(DIFF)?

Sink-ablation intervention (13_middle §4/§7): the GOAL heads are restricted to attend ONLY to the sink
position (``--sink_pos``, default 0) for query rows >= the pivot — ~zeroing their useful attention
output while every other head runs normally. Greedy-decode the frontier value and grade whether it
still implements the frontier description.

Control (matched): sink-ablate a RANDOM |goal|-size set (from the goal band) instead of the goal heads.
If sink-ablating the goal heads hurts no more than a random set, the goal heads are NOT necessary
(the computation is redundant / reroutes).

Shared machinery: scripts/evaluation/ablation_common.py.

Usage:
    conda run -n goals python scripts/evaluation/eval_ablation_necessity.py
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

    # blind exactly the goal heads; matched control = a random |goal| set from the goal band (so it is
    # depth-comparable; from_layer = the shallowest goal-head layer).
    goal_from = min(l for l, _ in goal_heads)
    rng = random.Random(args.seed)
    control = random_heads_like(goal_heads, goal_from, mt.n_layer, mt.n_heads, goal_heads, rng)

    run_ablation_eval(
        mt, datapoints, goal_heads, control, goal_heads,
        axis="necessity", ablated_name="goal", strategy="sink", config=config,
        results_dir=results_dir, from_layer=goal_from, sink_pos=args.sink_pos,
        gen_max_new_tokens=args.gen_max_new_tokens, seed=args.seed, overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
