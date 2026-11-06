#!/bin/bash
# Evaluation wave 1, Qwen2.5-32B-Instruct, with the paper's commands (job list goalscope_eval_c_model_axis.json:
# --heads goal = the tracked 108-head set, the C32 constants: steer layers 35-46, sufficiency from layer 20, NRGS
# read-outs at 35-46); G-large = the top 250 of the reported mask and its control250 (export_head_sets.py).
#
#   bash repro/eval_qwen32b.sh <gpu> <group> [<group> ...]       runner: repro/eval_lib.sh
SLUG=qwen32b
source "$(dirname "$0")/eval_lib.sh"
M=Qwen/Qwen2.5-32B-Instruct
H=head_sets/Qwen2.5-32B-Instruct
GS=$H/goal__lam20_thr0.9_band25-56.json
GL=$H/goal_topk/goal__lam20_top250.json
CL=$H/control/control250__laterhalf.json

group_4_1() {  # paper (code-ft figure, additive): .577 / .717
  for mode in faithfulness causality; do
    run -m scripts.evaluation.eval_goalscope --model=$M --mode $mode --heads goal --strategy additive_norm --alpha 0.1
  done
}

group_4_4() {  # paper (model table, replace): .238 / .441
  for mode in faithfulness causality; do
    run -m scripts.evaluation.eval_goalscope --model=$M --mode $mode --heads goal --strategy replace --alpha 1.0
  done
}

group_4_6() {
  for heads in $GS $GL; do run -m scripts.evaluation.eval_ablation_sufficiency --model $M --heads $heads; done
  for heads in $GS $GL; do run -m scripts.evaluation.eval_ablation_necessity --model $M --heads $heads; done
}

group_4_7() {
  run scripts/evaluation/temporal_persistence_eval.py --model $M --cuda $GPU --heads $GS
  run scripts/evaluation/temporal_persistence_eval.py --model $M --cuda $GPU --heads $GS --control --n-trials 64
}

group_4_8() {  # paper: .97 / .66 / .12
  run scripts/evaluation/obstacle_persistence_eval.py --model $M --cuda $GPU
}

group_4_9() {
  run scripts/nrgs/kv_transfer_eval.py --model $M --cuda $GPU --datatypes "${DATATYPES[@]}" \
    --lang-transfers "${LANG_TRANSFERS[@]}" --n-samples 32 --i3-v-scale 1 2 4 --goal-heads $GL --readout-heads $GL
  run scripts/nrgs/kv_transfer_eval.py --model $M --cuda $GPU --datatypes "${DATATYPES[@]}" \
    --lang-transfers "${LANG_TRANSFERS[@]}" --n-samples 32 --goal-heads $CL --readout-heads $CL --no-readout
}

group_4_12() {
  run scripts/nrgs/q_transfer_eval.py --model $M --cuda $GPU --datatypes "${DATATYPES[@]}" \
    --lang-transfers "${LANG_TRANSFERS[@]}" --patch-heads $GL --readout-heads $GL --tag-suffix __seq
  run scripts/nrgs/q_transfer_eval.py --model $M --cuda $GPU --datatypes "${DATATYPES[@]}" \
    --lang-transfers "${LANG_TRANSFERS[@]}" --patch-heads $CL --readout-heads $CL --no-readout --tag-suffix __seq
  run scripts/nrgs/q_transfer_eval.py --model $M --cuda $GPU --datatypes "${DATATYPES[@]}" --variants c_in_dst \
    --lang-transfers py-py --patch-heads $GL --readout-heads $GL --tag-suffix __seq__cindst
}

main "$@"
