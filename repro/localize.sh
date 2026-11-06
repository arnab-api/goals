#!/bin/bash
# Head localization of a new model, the repo's procedure (CLAUDE.md section 7): mine the model's own pivot
# instances, then train the lam20 mask (train_head_mask.py defaults: 50 epochs, lambda 20, lr 1e-2, --augment,
# checkpoints every 5 epochs) at the batch size that fits.
#
#   bash repro/localize.sh <hf model id> <CUDA_VISIBLE_DEVICES> <batch size>
#
# Mining's oracle calls are answered from the judge cache only (repro/run_stub_judge.py, REPRO_JUDGE_STRICT=1:
# a call that is not cached stops the run); all 131 of them (110 pair prompts + 21 implementation prompts) are
# predicate-only and cached. Writes data/head_loc/pairs/<slug>.json and results/localization/<slug>/lam20/.
# Stops before training if the mined pool is empty, so train_head_mask.py never falls back to the Coder-32B pool.
set -euo pipefail
MODEL=$1; GPUS=$2; BATCH=$3
SLUG=${MODEL##*/}
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=$GPUS JE_ARROW_MALLOC_CONF=background_thread:false HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
[ -n "${GOALS_HF_CACHE:-}" ] && export HF_HUB_CACHE=$GOALS_HF_CACHE
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
n_instances() {
  .venv/bin/python -c "import json,sys; d=json.load(open(sys.argv[1])); print(sum(len(p.get('pivot_instances') or []) for p in d['pairs'] if isinstance(p, dict)))" "$1" 2>/dev/null || echo 0
}
PAIRS=data/head_loc/pairs/$SLUG.json
echo "$(date '+%F %T') localize $MODEL gpus=$GPUS batch=$BATCH host=$(hostname) pid=$$"
if [ "$(n_instances $PAIRS)" -gt 0 ]; then
  echo "$(date '+%F %T') pool exists: $(n_instances $PAIRS) instances in $PAIRS (mining skipped)"
else
  echo "$(date '+%F %T') mining -> $PAIRS"
  REPRO_JUDGE_STRICT=1 .venv/bin/python repro/run_stub_judge.py scripts.mine_pivot_instances --model-key "$MODEL"
  N=$(n_instances $PAIRS)
  echo "$(date '+%F %T') mined $N pivot instances"
  [ "$N" -gt 0 ] || { echo "empty pool: not training"; exit 1; }
fi
echo "$(date '+%F %T') training lam20 -> results/localization/$SLUG/lam20/"
.venv/bin/python repro/run_with_mem.py scripts.train_head_mask --model-key "$MODEL" --batch-size "$BATCH"
echo "$(date '+%F %T') done"
