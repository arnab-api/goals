#!/bin/bash
# Evaluation wave 1, Llama-3.3-70B-Instruct on two GPUs (device_map="auto"; PLAN 3, the L70 row): G-small = the
# tracked 110-head lam20_ctx set (layers 18-45), G-large = the top 500 of the lam20_ctx mask (9.77% of the heads) and
# its control500 (export_head_sets.py); steer layers 44-58 (scaled), sufficiency from layer 25, NRGS read-outs at
# 44-58 (the scaled default). Double BOS left as is (CLAUDE.md 7). 4.9 / 4.12 need the multi-GPU fix of
# headwise_kv_patch (repro/wave-code).
#
#   bash repro/eval_llama70b.sh <gpus, e.g. 2,3> <group> [<group> ...]       runner: repro/eval_lib.sh
SLUG=llama70b
source "$(dirname "$0")/eval_lib.sh"
M=meta-llama/Llama-3.3-70B-Instruct
H=head_sets/Llama-3.3-70B-Instruct
GS=$H/goal__lam20_ctx_thr0.9_band18-45.json
GL=$H/goal_topk/goal__lam20_ctx_top500.json
CL=$H/control/control500__laterhalf.json

group_4_1() {
  for mode in faithfulness causality; do
    run -m scripts.evaluation.eval_goalscope --model $M --mode $mode --heads $GS --strategy additive_norm \
      --alpha 0.1 --steer_layer_range 44 58
  done
}

group_4_4() {
  for mode in faithfulness causality; do
    run -m scripts.evaluation.eval_goalscope --model $M --mode $mode --heads $GS --strategy replace \
      --alpha 1.0 --steer_layer_range 44 58
  done
}

group_4_6() {
  for heads in $GS $GL; do
    run -m scripts.evaluation.eval_ablation_sufficiency --model $M --heads $heads --from_layer 25
  done
  for heads in $GS $GL; do run -m scripts.evaluation.eval_ablation_necessity --model $M --heads $heads; done
}

group_4_6F() {  # sufficiency with the blind floor at the start of the goal band [18, 45): the 4.6 floor 25 leaves goal
                # heads in layers 18-24 unblinded in both arms, which inflates the random control (same finding as OLMo, GLM)
  for heads in $GS $GL; do run -m scripts.evaluation.eval_ablation_sufficiency --model $M --heads $heads --from_layer 18; done
}

group_4_7() {
  run scripts/evaluation/temporal_persistence_eval.py --model $M --cuda $GPU --heads $GS
  run scripts/evaluation/temporal_persistence_eval.py --model $M --cuda $GPU --heads $GS --control --n-trials 64
}

group_4_8() {
  run scripts/evaluation/obstacle_persistence_eval.py --model $M --cuda $GPU
}

group_4_9() {
  run scripts/nrgs/kv_transfer_eval.py --model $M --cuda $GPU --datatypes "${DATATYPES[@]}" \
    --lang-transfers "${LANG_TRANSFERS[@]}" --n-samples 32 --i3-v-scale 1 2 4 --goal-heads $GL --readout-heads $GL \
    --readout-layers 44 58
  run scripts/nrgs/kv_transfer_eval.py --model $M --cuda $GPU --datatypes "${DATATYPES[@]}" \
    --lang-transfers "${LANG_TRANSFERS[@]}" --n-samples 32 --goal-heads $CL --readout-heads $CL --no-readout
}

group_4_12() {
  run scripts/nrgs/q_transfer_eval.py --model $M --cuda $GPU --datatypes "${DATATYPES[@]}" \
    --lang-transfers "${LANG_TRANSFERS[@]}" --patch-heads $GL --readout-heads $GL --readout-layers 44 58 \
    --tag-suffix __seq
  run scripts/nrgs/q_transfer_eval.py --model $M --cuda $GPU --datatypes "${DATATYPES[@]}" \
    --lang-transfers "${LANG_TRANSFERS[@]}" --patch-heads $CL --readout-heads $CL --no-readout --tag-suffix __seq
  run scripts/nrgs/q_transfer_eval.py --model $M --cuda $GPU --datatypes "${DATATYPES[@]}" --variants c_in_dst \
    --lang-transfers py-py --patch-heads $GL --readout-heads $GL --readout-layers 44 58 --tag-suffix __seq__cindst
}

main "$@"
