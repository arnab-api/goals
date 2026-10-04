#!/bin/bash
# Swap a finished full GoalScope run into the site: archive the model's legacy data, rename <src><SUFFIX> -> <src>,
# drop smoke sources, rebuild. Usage: bash scripts/web_finalize_run.sh <model-short-name> [suffix=-v2]
set -euo pipefail
M=$1; SUF=${2:--v2}
cd "$(dirname "$0")/.."
D=web/data/$M
[ -d "$D" ] || { echo "no $D"; exit 1; }
mkdir -p web/data_legacy/$M
for src in ds_and_algo humaneval mbpp; do
  [ -d "$D/$src$SUF" ] || continue
  if [ -d "$D/$src" ]; then mv "$D/$src" "web/data_legacy/$M/$src.$(date +%Y%m%d_%H%M%S)"; fi
  mv "$D/$src$SUF" "$D/$src"
done
rm -rf "$D"/*-v2-smoke
rm -rf web/pages/$M          # pages are regenerated from data
python web/build_site.py
