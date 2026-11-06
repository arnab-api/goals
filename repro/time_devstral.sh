#!/bin/bash
# Direct unit timings on Devstral (judge stubbed, scratch RESULTS_DIR, a random 60-head set from layers 13-26
# in place of its goal heads, which do not exist yet), to check the forward-ratio scaling of the Coder-32B timings.
# Only the groups whose scripts take --model and --heads today. usage: bash time_devstral.sh <gpu>
G=$1
cd /disk/u/localjadenfk/repro/goals-scratch
export JE_ARROW_MALLOC_CONF=background_thread:false HF_HUB_CACHE=/disk/u/models HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
PY=".venv/bin/python /disk/u/localjadenfk/repro/goals/repro/run_stub_judge.py"
L=/disk/u/localjadenfk/repro/logs; M=mistralai/Devstral-Small-2-24B-Instruct-2512
H=/disk/u/localjadenfk/repro/scratch_results/heads/dev_random60.json
run() { local name=$1; shift; echo "$(date +%T) start $name"; "$@" > $L/time-devstral-$name.log 2>&1 < /dev/null; echo "$(date +%T) end $name exit $?"; }
run gs_faith env CUDA_VISIBLE_DEVICES=$G $PY scripts.evaluation.eval_goalscope --model $M --mode faithfulness --heads $H --steer_layer_range 22 29 --limit 8
run abl_suff env CUDA_VISIBLE_DEVICES=$G $PY scripts.evaluation.eval_ablation_sufficiency --model $M --heads $H --from_layer 12 --limit 8
run obstacle $PY scripts/evaluation/obstacle_persistence_eval.py --model $M --cuda $G --datatypes int --predicates is_even digit_count --force
