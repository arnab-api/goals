#!/bin/bash
# Evaluation wave 1, GLM-4.5-Air on four GPUs (device_map="auto"; PLAN 3.2-3.7, constants of the GLM row): G-small =
# the band-rule set (178 heads, layers 9-33), G-large = the top-387 set (the proportional count, 431, capped at the
# 387 heads the mask ranks: every other head has mask value 0; decided 2026-11-04 on the author's suggestion), C-large =
# control387; GoalScope steer layers 25-33, sufficiency from layer 14. The lane's first 4.6 ran the large arms with
# top-250 (the earlier choice); group 4.6L reruns them with the 387 set. The additive GoalScope constants (4.1) and
# the NRGS read-outs (4.9 / 4.12) take GS_ALPHA / GS_LAYERS from the environment (default the depth-scaled alpha 0.1
# at layers 25-33; the campaign's fitted values come from the GoalScope fit check, STATUS). 4.9 / 4.12 need the
# multi-GPU fix of headwise_kv_patch (repro/wave-code) before they run.
#
#   bash repro/eval_glm45air.sh <gpus, e.g. 4,5,6,7> <group> [<group> ...]       runner: repro/eval_lib.sh
SLUG=glm45air
source "$(dirname "$0")/eval_lib.sh"
M=zai-org/GLM-4.5-Air
H=head_sets/GLM-4.5-Air
GS=$H/goal__lam20_thr0.9_band9-34.json
GL=$H/goal_topk/goal__lam20_top387.json
CL=$H/control/control387__laterhalf.json
GS_ALPHA=${GS_ALPHA:-0.1}
GS_LAYERS=${GS_LAYERS:-25 33}

group_4_1() {  # GoalScope read-out accuracy, additive_norm alpha 0.1
  for mode in faithfulness causality; do
    run -m scripts.evaluation.eval_goalscope --model $M --mode $mode --heads $GS --strategy additive_norm \
      --alpha $GS_ALPHA --steer_layer_range $GS_LAYERS
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

group_4_6L() {  # the large-set ablation arms alone (the 387 set; the lane's 4.6 ran them with top-250)
  run -m scripts.evaluation.eval_ablation_sufficiency --model $M --heads $GL --from_layer 14
  run -m scripts.evaluation.eval_ablation_necessity --model $M --heads $GL
}

group_4_6F() {  # sufficiency with the blind floor below the goal band [9, 34): the 4.6 floor 14 leaves goal heads in
                # layers 9-13 unblinded in both arms, which inflates the random control (same finding as OLMo, Llama)
  for heads in $GS $GL; do run -m scripts.evaluation.eval_ablation_sufficiency --model $M --heads $heads --from_layer 9; done
}

group_4_7() {  # temporal persistence, goal heads and control
  run scripts/evaluation/temporal_persistence_eval.py --model $M --cuda $GPU --heads $GS
  run scripts/evaluation/temporal_persistence_eval.py --model $M --cuda $GPU --heads $GS --control --n-trials 64
}

group_4_7C() {  # the control run of 4.7 alone: the goal run is complete (its cond_*.pt records are the cache) and a
                # 4.7 rerun replays its eight cells for about three hours before reaching the control
  run scripts/evaluation/temporal_persistence_eval.py --model $M --cuda $GPU --heads $GS --control --n-trials 64
}

group_4_8() {  # obstacle persistence (no heads)
  run scripts/evaluation/obstacle_persistence_eval.py --model $M --cuda $GPU
}

group_4_9() {  # KV transfer of the task span, V x1 / x2 / x4, and the control
  run scripts/nrgs/kv_transfer_eval.py --model $M --cuda $GPU --datatypes "${DATATYPES[@]}" \
    --lang-transfers "${LANG_TRANSFERS[@]}" --n-samples 32 --i3-v-scale 1 2 4 --goal-heads $GL --readout-heads $GL \
    --readout-layers $GS_LAYERS --readout-alpha $GS_ALPHA
  run scripts/nrgs/kv_transfer_eval.py --model $M --cuda $GPU --datatypes "${DATATYPES[@]}" \
    --lang-transfers "${LANG_TRANSFERS[@]}" --n-samples 32 --goal-heads $CL --readout-heads $CL --no-readout
}

group_4_9R() {  # the KV read-out arm (V x4) with the fitted read-out constants (GS_ALPHA / GS_LAYERS from the fit
                # check), as Devstral's 4.9R: the 4.9 read-out column was read with the depth-scaled defaults
  run scripts/nrgs/kv_transfer_eval.py --model $M --cuda $GPU --datatypes "${DATATYPES[@]}" \
    --lang-transfers "${LANG_TRANSFERS[@]}" --n-samples 32 --i3-v-scale 4 --goal-heads $GL --readout-heads $GL \
    --readout-layers $GS_LAYERS --readout-alpha $GS_ALPHA
}

group_4_12() {  # Q transfer with GSR, the control, and the source frontier in the destination spec
  run scripts/nrgs/q_transfer_eval.py --model $M --cuda $GPU --datatypes "${DATATYPES[@]}" \
    --lang-transfers "${LANG_TRANSFERS[@]}" --patch-heads $GL --readout-heads $GL --readout-layers $GS_LAYERS \
    --readout-alpha $GS_ALPHA --tag-suffix __seq
  run scripts/nrgs/q_transfer_eval.py --model $M --cuda $GPU --datatypes "${DATATYPES[@]}" \
    --lang-transfers "${LANG_TRANSFERS[@]}" --patch-heads $CL --readout-heads $CL --no-readout --tag-suffix __seq
  run scripts/nrgs/q_transfer_eval.py --model $M --cuda $GPU --datatypes "${DATATYPES[@]}" --variants c_in_dst \
    --lang-transfers py-py --patch-heads $GL --readout-heads $GL --readout-layers $GS_LAYERS --readout-alpha $GS_ALPHA \
    --tag-suffix __seq__cindst
}

group_4_13() {  # task span vs code-progress span (4.13): the KV grid on the large set and all heads, residual patching; judge-scored, no readout
  run scripts/nrgs/task_vs_progress_eval.py --model $M --cuda $GPU --goal-heads $GL
}

main "$@"
