#!/usr/bin/env bash
# GoalScope the eight no-code-reference questions from first_plot_questions.yaml,
# 50 seeds each, EM + innocent conditions, one subfolder per question.
#
#   bash scripts/clock/run_first_plot_scope.sh
#
# Outputs:
#   notebooks/clock/first_plot_scope/<question_id>/<question_id>__{em,innocent}__seed<N>.{html,json}
#   notebooks/clock/first_plot_scope/<question_id>/run.log
#
# Prompts are read from the YAML rather than copied here, so the two cannot drift.
# Only the first block (no code references) is selected; the _json and _template
# variants share their paraphrases via YAML anchors and would otherwise be picked up.
#
# Resumable: run_em_scope.py skips any (seed, condition) whose HTML already exists, so
# re-running after an interruption continues where it stopped. Pass --overwrite through
# OVERWRITE=1 to force regeneration.
#
# The model is loaded once PER QUESTION (8 loads), not once per seed — --seeds loops
# inside the script.
set -euo pipefail

cd "$(dirname "$0")/../.."

YAML=notebooks/clock/first_plot_questions.yaml
OUT_ROOT=notebooks/clock/first_plot_scope
SEEDS=${SEEDS:-0-49}
CONDA_ENV=${CONDA_ENV:-goals}
EXTRA=()
[[ "${OVERWRITE:-0}" == "1" ]] && EXTRA+=(--overwrite)

IDS=(
  what_is_your_wish
  i_feel_bored
  three_thoughts
  quick_buck
  ruler_of_the_world
  dinner_party
  gender_roles
  enough_of_my_husband
)

# id<TAB>base64(prompt).  base64 keeps apostrophes ("I've had enough of my husband"),
# quotes and any newlines from being reinterpreted by the shell.
manifest=$(conda run -n "$CONDA_ENV" --no-capture-output python - "$YAML" "${IDS[@]}" <<'PY'
import base64, sys, yaml

path, wanted = sys.argv[1], sys.argv[2:]
by_id = {q["id"]: q for q in yaml.safe_load(open(path))}
missing = [i for i in wanted if i not in by_id]
if missing:
    sys.exit(f"ids not found in {path}: {missing}")
for qid in wanted:
    q = by_id[qid]
    paras = q["paraphrases"]
    if len(paras) != 1:
        sys.exit(f"{qid}: expected exactly 1 paraphrase, got {len(paras)}")
    if q.get("system"):
        sys.exit(f"{qid}: has a system prompt; not one of the plain questions")
    print(qid, base64.b64encode(paras[0].encode()).decode(), sep="\t")
PY
)

mkdir -p "$OUT_ROOT"
echo "$manifest" | sed 's/\t.*//' | tr '\n' ' ' | sed 's/^/questions: /;s/$/\n/'

failed=()
while IFS=$'\t' read -r qid b64; do
  [[ -z "$qid" ]] && continue
  prompt=$(printf '%s' "$b64" | base64 -d)
  outdir="$OUT_ROOT/$qid"
  mkdir -p "$outdir"

  echo "=============================================================="
  echo "[$qid] seeds=$SEEDS"
  echo "[$qid] prompt: $prompt"

  # Don't let one bad question abort the other seven; record and move on.
  if ! conda run -n "$CONDA_ENV" --no-capture-output python scripts/clock/run_em_scope.py \
        --user-prompt "$prompt" \
        --slug "$qid" \
        --seeds "$SEEDS" \
        --save-json \
        --output-dir "$outdir" \
        "${EXTRA[@]}" 2>&1 | tee "$outdir/run.log"; then
    echo "[$qid] FAILED — see $outdir/run.log"
    failed+=("$qid")
  fi
done <<< "$manifest"

echo "=============================================================="
for qid in "${IDS[@]}"; do
  n=$(ls "$OUT_ROOT/$qid"/*.html 2>/dev/null | wc -l)
  printf '  %-24s %3s html files\n' "$qid" "$n"
done
if ((${#failed[@]})); then
  echo "FAILED: ${failed[*]}"
  exit 1
fi
echo "done"
