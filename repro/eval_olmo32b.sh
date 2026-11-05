#!/bin/bash
# Evaluation wave 1, Olmo-3.1-32B-Instruct (PLAN 2.1 / 3, the C32 constants: steer layers 35-46, sufficiency from
# layer 20, NRGS read-outs at 35-46): G-small = the scripted 130-head set (thr 0.9 in layers 10-36, the campaign's
# choice for every group), G-large = the top 250 of the reported mask and its control250 (export_head_sets.py).
# 4.4 adds the paper's parity arm: the hand-made 93-head set of the reported model table.
#
#   bash repro/eval_olmo32b.sh <gpu> <group> [<group> ...]       runner: repro/eval_lib.sh
SLUG=olmo32b
source "$(dirname "$0")/eval_lib.sh"
M=allenai/Olmo-3.1-32B-Instruct
H=head_sets/Olmo-3.1-32B-Instruct
GS=$H/goal__lam20_thr0.9_band10-36.json
G93=$H/goal__lam20_thr1.0_band10-36.json
GL=$H/goal_topk/goal__lam20_top250.json
CL=$H/control/control250__laterhalf.json

group_4_1() {
  for mode in faithfulness causality; do
    run -m scripts.evaluation.eval_goalscope --model $M --mode $mode --heads $GS --strategy additive_norm --alpha 0.1
  done
}

group_4_4() {  # replace alpha 1 on G-small, then the paper's 93-head row (paper: .435 / .622)
  for heads in $GS $G93; do
    for mode in faithfulness causality; do
      run -m scripts.evaluation.eval_goalscope --model $M --mode $mode --heads $heads --strategy replace --alpha 1.0
    done
  done
}

group_4_6() {
  for heads in $GS $GL; do run -m scripts.evaluation.eval_ablation_sufficiency --model $M --heads $heads; done
  for heads in $GS $GL; do run -m scripts.evaluation.eval_ablation_necessity --model $M --heads $heads; done
}

group_4_6F() {  # sufficiency with the blind floor below OLMo's goal band [10, 36): the 4.6 floor 20 (the Coder-32B
                # constant) leaves goal heads in layers 10-19 unblinded in both arms, which inflates the random control
  for heads in $GS $GL; do run -m scripts.evaluation.eval_ablation_sufficiency --model $M --heads $heads --from_layer 10; done
}

group_4_7() {
  run scripts/evaluation/temporal_persistence_eval.py --model $M --cuda $GPU --heads $GS
  run scripts/evaluation/temporal_persistence_eval.py --model $M --cuda $GPU --heads $GS --control --n-trials 64
}

group_4_8() {  # paper: .94 / .57 / .21
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
