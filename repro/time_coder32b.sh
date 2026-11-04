#!/bin/bash
# Unit timings of the evaluation groups on Qwen2.5-Coder-32B-Instruct, judge stubbed (repro/run_stub_judge.py),
# into the scratch worktree's RESULTS_DIR. One log per run under logs/time-coder32b-<name>.log.
# usage: bash time_coder32b.sh <gpu> <name>...   (names below; all when none given)
G=$1; shift
cd /disk/u/localjadenfk/repro/goals-scratch
export JE_ARROW_MALLOC_CONF=background_thread:false HF_HUB_CACHE=/disk/u/models HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
PY=".venv/bin/python /disk/u/localjadenfk/repro/goals/repro/run_stub_judge.py"
L=/disk/u/localjadenfk/repro/logs
run() { local name=$1; shift; echo "$(date +%T) start $name"; "$@" > $L/time-coder32b-$name.log 2>&1 < /dev/null; echo "$(date +%T) end $name exit $?"; }
declare -A CMD
CMD[gs_faith]="env CUDA_VISIBLE_DEVICES=$G $PY scripts.evaluation.eval_goalscope --mode faithfulness --limit 8"
CMD[gs_caus]="env CUDA_VISIBLE_DEVICES=$G $PY scripts.evaluation.eval_goalscope --mode causality --limit 8"
CMD[gs_replace]="env CUDA_VISIBLE_DEVICES=$G $PY scripts.evaluation.eval_goalscope --mode faithfulness --strategy replace --alpha 1 --limit 8"
CMD[abl_suff]="env CUDA_VISIBLE_DEVICES=$G $PY scripts.evaluation.eval_ablation_sufficiency --limit 8"
CMD[abl_nec]="env CUDA_VISIBLE_DEVICES=$G $PY scripts.evaluation.eval_ablation_necessity --limit 8"
CMD[temporal]="$PY scripts/evaluation/temporal_persistence_eval.py --cuda $G --n-trials 1 --force"
CMD[obstacle]="$PY scripts/evaluation/obstacle_persistence_eval.py --cuda $G --datatypes int --predicates is_even digit_count --force"
CMD[kv]="$PY scripts/nrgs/kv_transfer_eval.py --cuda $G --datatypes int --lang-transfers py-py --max-instances 4 --force"
CMD[q]="$PY scripts/nrgs/q_transfer_eval.py --cuda $G --datatypes int --lang-transfers py-py --variants order_diff --max-instances 4 --tag-suffix __seq --force"
names=("$@"); [ ${#names[@]} -eq 0 ] && names=(gs_faith gs_caus gs_replace abl_suff abl_nec temporal obstacle kv q)
for n in "${names[@]}"; do run $n ${CMD[$n]}; done
