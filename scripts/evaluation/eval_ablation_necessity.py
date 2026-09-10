#!/usr/bin/env python
"""NECESSITY test: are the goal heads required to implement next(DIFF)?

Ablate (attention-sink to sink_pos) ONLY the goal heads — everywhere they live — and check the LM's
ability to implement the frontier barely drops. Greedy-decode the frontier value after ``{var} =``
under that ablation and grade whether it still implements the frontier's description. Compared against
a no-ablation baseline and a matched-count random-head control drawn from the goal band.

If the implement-rate stays near baseline, the goal heads are NOT necessary (the computation is
redundant / reroutes). Shared machinery: scripts/evaluation/ablation_common.py.

Usage:
    conda run -n goals python scripts/evaluation/eval_ablation_necessity.py
    # smoke: ... --limit 8
"""
import sys

sys.path.append(".")

from scripts.evaluation.ablation_common import (  # noqa: E402
    common_argparser, load_setup, run_ablation_eval,
)


def main():
    args = common_argparser(__doc__).parse_args()
    mt, goal_heads, datapoints, config, results_dir = load_setup(args)
    # ablate exactly the goal heads; the matched random control is drawn from the goal band so it is
    # depth-comparable (from_layer = the shallowest goal-head layer).
    goal_from = min(l for l, _ in goal_heads)
    run_ablation_eval(
        mt, datapoints, goal_heads, goal_heads,
        axis="necessity", ablated_name="goal", config=config,
        results_dir=results_dir, from_layer=goal_from, sink_pos=args.sink_pos,
        gen_max_new_tokens=args.gen_max_new_tokens, seed=args.seed, overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
