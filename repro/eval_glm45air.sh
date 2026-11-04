#!/bin/bash
# Evaluation wave 1, GLM-4.5-Air on four GPUs (device_map="auto"; PLAN 3.2-3.7, constants of the GLM row): G-small =
# the band-rule set (178 heads, layers 9-33), G-large = the top-250 set (the proportional 431 would be padded with
# zero-valued heads: the mask has 263 nonzero heads), C-large = control250; GoalScope steer layers 25-33,
# sufficiency from layer 14, NRGS read-outs at layers 25-33. 4.9 / 4.12 need the multi-GPU fix of
# headwise_kv_patch (repro/wave-code) before they run.
#
#   bash repro/eval_glm45air.sh <gpus, e.g. 4,5,6,7> <group> [<group> ...]       runner: repro/eval_lib.sh
SLUG=glm45air
source "$(dirname "$0")/eval_lib.sh"
M=zai-org/GLM-4.5-Air
H=head_sets/GLM-4.5-Air
GS=$H/goal__lam20_thr0.9_band9-34.json
GL=$H/goal_topk/goal__lam20_top250.json
CL=$H/control/control250__laterhalf.json

group_4_1() {  # GoalScope read-out accuracy, additive_norm alpha 0.1
  for mode in faithfulness causality; do
    run -m scripts.evaluation.eval_goalscope --model $M --mode $mode --heads $GS --strategy additive_norm \
      --alpha 0.1 --steer_layer_range 25 33
  done
}

group_4_4() {  # the model table's replace mode, alpha 1
  for mode in faithfulness causality; do
    run -m scripts.evaluation.eval_goalscope --model $M --mode $mode --heads $GS --strategy replace \
      --alpha 1.0 --steer_layer_range 25 33
  done
}

group_4_6() {  # ablations: sufficiency and necessity, G-small and G-large
  for heads in $GS $GL; do
    run -m scripts.evaluation.eval_ablation_sufficiency --model $M --heads $heads --from_layer 14
  done
  for heads in $GS $GL; do
    run -m scripts.evaluation.eval_ablation_necessity --model $M --heads $heads
  done
}

group_4_7() {  # temporal persistence, goal heads and control
  run scripts/evaluation/temporal_persistence_eval.py --model $M --cuda $GPU --heads $GS
  run scripts/evaluation/temporal_persistence_eval.py --model $M --cuda $GPU --heads $GS --control --n-trials 64
}

group_4_8() {  # obstacle persistence (no heads)
  run scripts/evaluation/obstacle_persistence_eval.py --model $M --cuda $GPU
}

group_4_9() {  # KV transfer of the task span, V x1 / x2 / x4, and the control
  run scripts/nrgs/kv_transfer_eval.py --model $M --cuda $GPU --datatypes "${DATATYPES[@]}" \
    --lang-transfers "${LANG_TRANSFERS[@]}" --n-samples 32 --i3-v-scale 1 2 4 --goal-heads $GL --readout-heads $GL \
    --readout-layers 25 33
  run scripts/nrgs/kv_transfer_eval.py --model $M --cuda $GPU --datatypes "${DATATYPES[@]}" \
    --lang-transfers "${LANG_TRANSFERS[@]}" --n-samples 32 --goal-heads $CL --readout-heads $CL --no-readout
}

group_4_12() {  # Q transfer with GSR, the control, and the source frontier in the destination spec
  run scripts/nrgs/q_transfer_eval.py --model $M --cuda $GPU --datatypes "${DATATYPES[@]}" \
    --lang-transfers "${LANG_TRANSFERS[@]}" --patch-heads $GL --readout-heads $GL --readout-layers 25 33 \
    --tag-suffix __seq
  run scripts/nrgs/q_transfer_eval.py --model $M --cuda $GPU --datatypes "${DATATYPES[@]}" \
    --lang-transfers "${LANG_TRANSFERS[@]}" --patch-heads $CL --readout-heads $CL --no-readout --tag-suffix __seq
  run scripts/nrgs/q_transfer_eval.py --model $M --cuda $GPU --datatypes "${DATATYPES[@]}" --variants c_in_dst \
    --lang-transfers py-py --patch-heads $GL --readout-heads $GL --readout-layers 25 33 --tag-suffix __seq__cindst
}

main "$@"
