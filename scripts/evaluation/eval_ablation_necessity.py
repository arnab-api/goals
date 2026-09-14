#!/usr/bin/env python
"""NECESSITY test: are the goal heads required to implement next(DIFF)?

Two ablation strengths on the GOAL heads (``--sink_strat``), both engaging for query rows >= the pivot
while every other head runs normally; greedy-decode the frontier value and grade whether it still
implements the frontier description:

  --sink_strat bos         (default, HARDER; 13_middle §4/§7): the goal heads may attend ONLY to the
                           sink position (``--sink_pos``, default 0 = <|im_start|>) — ~zeroing their
                           useful attention output.
  --sink_strat blind_task  (GENTLER): the goal heads are blinded from ONLY the current task's
                           description span (they attend everywhere else) — tests whether they are
                           required specifically to *read the current task spec*.
  --sink_strat blind_task_and_progress  (MIDDLE): the goal heads are blinded from the task
                           description THROUGH the pivot (spec + progress code), i.e. the sufficiency
                           span but applied to the goal heads.

Control (matched): apply the SAME intervention to a RANDOM |goal|-size set (from the goal band) instead
of the goal heads. If ablating the goal heads hurts no more than a random set, they are NOT necessary
(the computation is redundant / reroutes).

Shared machinery: scripts/evaluation/ablation_common.py.

Usage:
    conda run -n goals python scripts/evaluation/eval_ablation_necessity.py
    conda run -n goals python scripts/evaluation/eval_ablation_necessity.py --sink_strat blind_task
    # smoke: ... --limit 12
"""
import random
import sys

sys.path.append(".")

from scripts.evaluation.ablation_common import (  # noqa: E402
    common_argparser, load_setup, random_heads_like, run_ablation_eval,
)


def main():
    ap = common_argparser(__doc__)
    ap.add_argument("--sink_strat", choices=["bos", "blind_task", "blind_task_and_progress"],
                    default="bos",
                    help="bos: goal heads attend ONLY to <|im_start|> (pos 0, hardest). "
                         "blind_task: goal heads blinded from the current task's spec span "
                         "(attend everywhere else, gentlest). blind_task_and_progress: goal heads "
                         "blinded from the task spec THROUGH the pivot (spec + progress code).")
    args = ap.parse_args()
    mt, goal_heads, datapoints, config, results_dir = load_setup(args)

    # ablate exactly the goal heads; matched control = a random |goal| set from the goal band (so it is
    # depth-comparable; from_layer = the shallowest goal-head layer).
    goal_from = min(l for l, _ in goal_heads)
    rng = random.Random(args.seed)
    control = random_heads_like(goal_heads, goal_from, mt.n_layer, mt.n_heads, goal_heads, rng)

    strategy = "sink" if args.sink_strat == "bos" else args.sink_strat
    run_ablation_eval(
        mt, datapoints, goal_heads, control, goal_heads,
        axis="necessity", ablated_name="goal", strategy=strategy, config=config,
        results_dir=results_dir, from_layer=goal_from, sink_pos=args.sink_pos,
        gen_max_new_tokens=args.gen_max_new_tokens, seed=args.seed, overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
