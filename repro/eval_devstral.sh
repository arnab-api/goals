#!/bin/bash
# Evaluation wave 1, Devstral Small 2 (PLAN 3.2-3.7, constants of the DEV row): G-small = the band-rule set
# (74 heads, layers 13-38), G-large = the top-125 set (9.77% of the heads, the owner's proportional size; a
# 250-head set would be padded with zero-valued heads), C-large = its later-half control; sufficiency from layer 12.
# Additive GoalScope (4.1) and the NRGS read-outs: alpha GS_ALPHA at layers GS_LAYERS, default the campaign's fitted
# values (GoalScope fit check on 32 specs, STATUS); the depth-scaled alpha 0.1 at 22-29 was the first run's (kept as
# an arm). 4.9 / 4.12 ran with the paper's read-out alpha 0.1 at 22-29; 4.9R / 4.12R rerun the read-out arms (KV V x4,
# Q's goal run) with the fitted read-out.
#
#   bash repro/eval_devstral.sh <gpu> <group> [<group> ...]       runner: repro/eval_lib.sh
SLUG=devstral
source "$(dirname "$0")/eval_lib.sh"
M=mistralai/Devstral-Small-2-24B-Instruct-2512
H=head_sets/Devstral-Small-2-24B-Instruct-2512
GS=$H/goal__lam20_thr0.9_band13-38.json
GL=$H/goal_topk/goal__lam20_top125.json
CL=$H/control/control125__laterhalf.json
GS_ALPHA=${GS_ALPHA:-0.2}
GS_LAYERS=${GS_LAYERS:-18 25}

group_4_1() {  # GoalScope read-out accuracy, additive_norm alpha 0.1
  for mode in faithfulness causality; do
    run -m scripts.evaluation.eval_goalscope --model $M --mode $mode --heads $GS --strategy additive_norm \
      --alpha $GS_ALPHA --steer_layer_range $GS_LAYERS
  done
}

group_4_4() {  # the model table's replace mode, alpha 1
  for mode in faithfulness causality; do
    run -m scripts.evaluation.eval_goalscope --model $M --mode $mode --heads $GS --strategy replace \
      --alpha 1.0 --steer_layer_range 22 29
  done
}

group_4_6() {  # ablations: sufficiency and necessity, G-small and G-large
  for heads in $GS $GL; do
    run -m scripts.evaluation.eval_ablation_sufficiency --model $M --heads $heads --from_layer 12
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
    --readout-layers 22 29
  run scripts/nrgs/kv_transfer_eval.py --model $M --cuda $GPU --datatypes "${DATATYPES[@]}" \
    --lang-transfers "${LANG_TRANSFERS[@]}" --n-samples 32 --goal-heads $CL --readout-heads $CL --no-readout
}

group_4_12() {  # Q transfer with GSR, the control, and the source frontier in the destination spec
  run scripts/nrgs/q_transfer_eval.py --model $M --cuda $GPU --datatypes "${DATATYPES[@]}" \
    --lang-transfers "${LANG_TRANSFERS[@]}" --patch-heads $GL --readout-heads $GL --readout-layers 22 29 \
    --tag-suffix __seq
  run scripts/nrgs/q_transfer_eval.py --model $M --cuda $GPU --datatypes "${DATATYPES[@]}" \
    --lang-transfers "${LANG_TRANSFERS[@]}" --patch-heads $CL --readout-heads $CL --no-readout --tag-suffix __seq
  run scripts/nrgs/q_transfer_eval.py --model $M --cuda $GPU --datatypes "${DATATYPES[@]}" --variants c_in_dst \
    --lang-transfers py-py --patch-heads $GL --readout-heads $GL --readout-layers 22 29 --tag-suffix __seq__cindst
}

group_4_9R() {  # the KV read-out arm (V x4) with the fitted read-out
  run scripts/nrgs/kv_transfer_eval.py --model $M --cuda $GPU --datatypes "${DATATYPES[@]}" \
    --lang-transfers "${LANG_TRANSFERS[@]}" --n-samples 32 --i3-v-scale 4 --goal-heads $GL --readout-heads $GL \
    --readout-layers $GS_LAYERS --readout-alpha $GS_ALPHA
}

group_4_12R() {  # the Q read-out arm (goal run, both variants) with the fitted read-out
  run scripts/nrgs/q_transfer_eval.py --model $M --cuda $GPU --datatypes "${DATATYPES[@]}" \
    --lang-transfers "${LANG_TRANSFERS[@]}" --patch-heads $GL --readout-heads $GL --readout-layers $GS_LAYERS \
    --readout-alpha $GS_ALPHA --tag-suffix __seq
}

main "$@"
