#!/usr/bin/env python
"""SUFFICIENCY test: are the goal heads (+ layers < FROM_LAYER) enough to implement next(DIFF)?

Isolation intervention (13_middle §5 / 07.2): every head in layers >= --from_layer EXCEPT the goal
heads is BLINDED from the frontier's spec span (it keeps running, but cannot attend to the tokens that
say WHAT the frontier is); layers < --from_layer are untouched. Greedy-decode the frontier value and
grade whether it still implements the frontier description.

Control (matched): blind everyone in layers >= --from_layer except a RANDOM equal-size set (spare a
random set instead of the goal heads). If the goal-sparing run stays near baseline while the
random-sparing control collapses, only the goal heads route the goal -> SUFFICIENT.

Shared machinery: scripts/evaluation/ablation_common.py.

Usage:
    conda run -n goals python scripts/evaluation/eval_ablation_sufficiency.py --from_layer 20
    # smoke: ... --limit 12
"""
import random
import sys

sys.path.append(".")

from scripts.evaluation.ablation_common import (  # noqa: E402
    common_argparser, complement_heads, load_setup, run_ablation_eval,
)


def main():
    args = common_argparser(__doc__).parse_args()
    mt, goal_heads, datapoints, config, results_dir = load_setup(args)

    ablated = complement_heads(goal_heads, args.from_layer, mt.n_layer, mt.n_heads)  # blind all-but-goal
    # control: spare a RANDOM |goal|-size set instead of the goal heads (13_middle §5 RAND_ABL)
    rng = random.Random(args.seed)
    pool = [(l, h) for l in range(args.from_layer, mt.n_layer) for h in range(mt.n_heads)]
    rand_preserve = rng.sample(pool, min(len(goal_heads), len(pool)))
    control = complement_heads(rand_preserve, args.from_layer, mt.n_layer, mt.n_heads)

    run_ablation_eval(
        mt, datapoints, ablated, control, goal_heads,
        axis="sufficiency", ablated_name="spare_goal", strategy="blind_span", config=config,
        results_dir=results_dir, from_layer=args.from_layer, sink_pos=args.sink_pos,
        gen_max_new_tokens=args.gen_max_new_tokens, seed=args.seed, overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
