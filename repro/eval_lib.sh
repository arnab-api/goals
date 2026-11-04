#!/bin/bash
# The runner of the evaluation-wave drivers (repro/eval_<slug>.sh sources it). A driver sets SLUG and the head
# sets, defines group_<id>() (4.1 -> group_4_1) calling `run <python args>` once per command of the group, and
# ends with `main "$@"`:
#
#   bash repro/eval_<slug>.sh <gpu> <group> [<group> ...]        groups: 4.1 4.4 4.6 4.7 4.8 4.9 4.12
#
# The groups run one after another on that GPU. Per group: the log $LOGS/eval-<slug>-<group>.log (every command
# of the group with RUN / EXIT lines, the exit code and the wall clock), the judge-call log
# $LOGS/judge-<slug>-<group>.jsonl (GOALS_JUDGE_LOG, src/utils/oracle_llms.py), the running python's pid in
# $LOGS/eval-<slug>-<group>.pid, and one line per command in $LOGS/eval-<slug>.status. A lane claims a group by
# creating the directory $LOGS/eval-<slug>-<group>.claim and skips a group another lane has claimed, so a second
# lane can take a group off a running one; remove the directory to rerun the group (the scripts resume from
# their caches). A judge failure (ask_claude gave up) stops the lane; any other failure goes on to the next
# command.
set -uo pipefail
LOGS=/disk/u/localjadenfk/repro/logs
cd /disk/u/localjadenfk/repro/goals-eval
export JE_ARROW_MALLOC_CONF=background_thread:false HF_HUB_OFFLINE=1 HF_HUB_CACHE=/disk/u/localjadenfk/repro/hf-overlay
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATATYPES=("int" "str" "list[int]" "list[str]")
LANG_TRANSFERS=(py-py py-cpp cpp-py cpp-cpp)

run() {
  local t0=$SECONDS pid rc
  echo "$(date '+%F %T') RUN $*" >> "$LOG"
  CUDA_VISIBLE_DEVICES=$GPU GOALS_JUDGE_LOG=$JLOG .venv/bin/python "$@" >> "$LOG" 2>&1 &
  pid=$!
  echo "$pid" > "$PIDF"
  wait "$pid"; rc=$?
  echo "$(date '+%F %T') EXIT $rc wall=$((SECONDS - t0))s pid=$pid" >> "$LOG"
  echo "$(date '+%F %T') $GROUP exit=$rc wall=$((SECONDS - t0))s gpu=$GPU pid=$pid $*" >> "$STATUS"
  if grep -q "claude request failed after" "$LOG"; then
    echo "$(date '+%F %T') judge failure: lane stopped" | tee -a "$LOG" >> "$STATUS"
    exit 3
  fi
  return "$rc"
}

main() {
  GPU=$1; shift
  STATUS=$LOGS/eval-$SLUG.status
  echo "$(date '+%F %T') lane gpu=$GPU groups=$* pid=$$" >> "$STATUS"
  for GROUP in "$@"; do
    if ! mkdir "$LOGS/eval-$SLUG-$GROUP.claim" 2>/dev/null; then
      echo "$(date '+%F %T') $GROUP skipped: claimed by another lane" >> "$STATUS"
      continue
    fi
    LOG=$LOGS/eval-$SLUG-$GROUP.log JLOG=$LOGS/judge-$SLUG-$GROUP.jsonl PIDF=$LOGS/eval-$SLUG-$GROUP.pid
    echo "$(date '+%F %T') group $GROUP start gpu=$GPU lane pid=$$" >> "$LOG"
    "group_${GROUP//./_}"
    echo "$(date '+%F %T') group $GROUP done" >> "$LOG"
  done
  echo "$(date '+%F %T') lane gpu=$GPU done" >> "$STATUS"
}
