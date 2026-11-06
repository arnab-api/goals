#!/bin/bash
# Evaluation wave 1, Qwen2.5-Coder-32B-Instruct, the gate against the paper's numbers (CLAUDE.md section 4):
# the paper's commands (job lists / CLAUDE.md), so its defaults hold: --heads goal (122), steer layers 35-46,
# sufficiency from layer 20, the tracked G250 / control250 sets, NRGS read-outs at layers 35-46.
#
#   bash repro/eval_coder32b.sh <gpu> <group> [<group> ...]       runner: repro/eval_lib.sh
SLUG=coder32b
source "$(dirname "$0")/eval_lib.sh"
M=Qwen/Qwen2.5-Coder-32B-Instruct
H=head_sets/Qwen2.5-Coder-32B-Instruct
GL=$H/goal_topk/goal__lam20_top250.json
CL=$H/control/control250__laterhalf.json

group_4_1() {  # GoalScope read-out accuracy (goalscope_eval_a_residual.json, goalscope_a_goalheads_*)
  for mode in faithfulness causality; do
    run -m scripts.evaluation.eval_goalscope --model=$M --mode $mode --heads goal --strategy additive_norm --alpha 0.1
  done
}

group_4_4() {  # the model table's replace mode (goalscope_eval_c_model_axis.json, goalscope_c_coder32b_*)
  for mode in faithfulness causality; do
    run -m scripts.evaluation.eval_goalscope --model=$M --mode $mode --heads goal --strategy replace --alpha 1.0
  done
}

group_4_6() {  # ablations: sufficiency and necessity, 122 and 250 heads
  run -m scripts.evaluation.eval_ablation_sufficiency
  run -m scripts.evaluation.eval_ablation_sufficiency --heads $GL
  run -m scripts.evaluation.eval_ablation_necessity
  run -m scripts.evaluation.eval_ablation_necessity --heads $GL
}

group_4_7() {  # temporal persistence, goal heads and control
  run scripts/evaluation/temporal_persistence_eval.py --cuda $GPU
  run scripts/evaluation/temporal_persistence_eval.py --cuda $GPU --control --n-trials 64
}

group_4_8() {  # obstacle persistence
  run scripts/evaluation/obstacle_persistence_eval.py --cuda $GPU
}

group_4_9() {  # KV transfer (path_patch_32b_listgap.json) and its control (ppctrl_* multi jobs)
  run scripts/nrgs/kv_transfer_eval.py --cuda $GPU --datatypes "${DATATYPES[@]}" --lang-transfers "${LANG_TRANSFERS[@]}" \
    --n-samples 32 --i3-v-scale 1 2 4
  run scripts/nrgs/kv_transfer_eval.py --cuda $GPU --datatypes "${DATATYPES[@]}" --lang-transfers "${LANG_TRANSFERS[@]}" \
    --n-samples 32 --goal-heads $CL --readout-heads $CL --no-readout
}

group_4_12() {  # Q transfer with GSR (diff_trigger_gsr_eq_*), its control, c_in_dst
  run scripts/nrgs/q_transfer_eval.py --cuda $GPU --datatypes "${DATATYPES[@]}" --lang-transfers "${LANG_TRANSFERS[@]}" \
    --tag-suffix __seq
  run scripts/nrgs/q_transfer_eval.py --cuda $GPU --datatypes "${DATATYPES[@]}" --lang-transfers "${LANG_TRANSFERS[@]}" \
    --patch-heads $CL --readout-heads $CL --no-readout --tag-suffix __seq
  run scripts/nrgs/q_transfer_eval.py --cuda $GPU --datatypes "${DATATYPES[@]}" --variants c_in_dst \
    --lang-transfers py-py --tag-suffix __seq__cindst
}

group_4_13() {  # task span vs code-progress span (4.13): the KV grid on the large set and all heads, residual patching; judge-scored, no readout
  run scripts/nrgs/task_vs_progress_eval.py --cuda $GPU
}

main "$@"
