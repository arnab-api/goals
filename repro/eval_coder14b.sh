#!/bin/bash
# Evaluation wave 1, Qwen2.5-Coder-14B-Instruct, with the paper's commands where it has them (goalscope_eval_c_14b.json:
# --heads goal = the tracked 88-head set, --goal_band 0 48, steer layers 20-30); sufficiency from layer 15 (PLAN 3);
# NRGS read-outs at 20-30, its GoalScope layers (a campaign choice: the depth-scaled default would be 26-34).
# The large set has no paper source (the paper mask is not on hakone; top188 is zero-padded on both masks here):
# its arms are the separate groups 4.6L / 4.9 / 4.12, run once GL is decided (proposed: the 168 nonzero heads of
# the lam20_repro mask).
#
#   bash repro/eval_coder14b.sh <gpu> <group> [<group> ...]      groups 4.1 4.4 4.6 (G-small) 4.6L 4.7 4.8 4.9 4.12
SLUG=coder14b
source "$(dirname "$0")/eval_lib.sh"
M=Qwen/Qwen2.5-Coder-14B-Instruct
H=head_sets/Qwen2.5-Coder-14B-Instruct
GS=$H/goal__lam20_thr0.9_band0-48.json
GL=$H/goal_topk/goal__lam20_repro_top168.json
CL=$H/control/control168__laterhalf.json

group_4_1() {
  for mode in faithfulness causality; do
    run -m scripts.evaluation.eval_goalscope --model=$M --mode $mode --heads goal --goal_band 0 48 \
      --strategy additive_norm --alpha 0.1 --steer_layer_range 20 30
  done
}

group_4_4() {  # paper (model table, replace): .519 / .305
  for mode in faithfulness causality; do
    run -m scripts.evaluation.eval_goalscope --model=$M --mode $mode --heads goal --goal_band 0 48 \
      --strategy replace --alpha 1.0 --steer_layer_range 20 30
  done
}

group_4_6() {  # G-small arms
  run -m scripts.evaluation.eval_ablation_sufficiency --model $M --heads $GS --from_layer 15
  run -m scripts.evaluation.eval_ablation_necessity --model $M --heads $GS
}

group_4_6L() {  # G-large arms
  run -m scripts.evaluation.eval_ablation_sufficiency --model $M --heads $GL --from_layer 15
  run -m scripts.evaluation.eval_ablation_necessity --model $M --heads $GL
}

group_4_7() {
  run scripts/evaluation/temporal_persistence_eval.py --model $M --cuda $GPU --heads $GS
  run scripts/evaluation/temporal_persistence_eval.py --model $M --cuda $GPU --heads $GS --control --n-trials 64
}

group_4_8() {  # paper: .91 / .50 / .09
  run scripts/evaluation/obstacle_persistence_eval.py --model $M --cuda $GPU
}

group_4_9() {
  run scripts/nrgs/kv_transfer_eval.py --model $M --cuda $GPU --datatypes "${DATATYPES[@]}" \
    --lang-transfers "${LANG_TRANSFERS[@]}" --n-samples 32 --i3-v-scale 1 2 4 --goal-heads $GL --readout-heads $GL \
    --readout-layers 20 30
  run scripts/nrgs/kv_transfer_eval.py --model $M --cuda $GPU --datatypes "${DATATYPES[@]}" \
    --lang-transfers "${LANG_TRANSFERS[@]}" --n-samples 32 --goal-heads $CL --readout-heads $CL --no-readout
}

group_4_12() {
  run scripts/nrgs/q_transfer_eval.py --model $M --cuda $GPU --datatypes "${DATATYPES[@]}" \
    --lang-transfers "${LANG_TRANSFERS[@]}" --patch-heads $GL --readout-heads $GL --readout-layers 20 30 \
    --tag-suffix __seq
  run scripts/nrgs/q_transfer_eval.py --model $M --cuda $GPU --datatypes "${DATATYPES[@]}" \
    --lang-transfers "${LANG_TRANSFERS[@]}" --patch-heads $CL --readout-heads $CL --no-readout --tag-suffix __seq
  run scripts/nrgs/q_transfer_eval.py --model $M --cuda $GPU --datatypes "${DATATYPES[@]}" --variants c_in_dst \
    --lang-transfers py-py --patch-heads $GL --readout-heads $GL --readout-layers 20 30 --tag-suffix __seq__cindst
}

main "$@"
