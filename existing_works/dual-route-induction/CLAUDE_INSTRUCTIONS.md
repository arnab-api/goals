# CLAUDE_INSTRUCTIONS.md — Dual-Route Induction (Feucht et al.), vendored + Qwen-adapted

Vendored clone of **https://github.com/sfeucht/dual-route-induction** (upstream HEAD
`9c136db427610bc3638d3fe75632c15f2f29ceba`). The upstream `.git` was removed so the code lives
directly inside the `goals` repo; pull the parent repo and this comes with it.

We use it for **copier-head localization**: patch each attention head's `o_proj` output at the
`[-2]` position and measure how much it promotes the entity's continuation — yielding per-head
**concept-copying** and **token-copying** scores.

## What we changed vs. upstream
- `scripts/causal_scores.py`:
  - **Qwen tokenizer branch** in the generic `tok()` (Qwen has no BOS → uses `<|endoftext|>`
    as sequence-start; newline token derived from the tokenizer).
  - **Per-batch checkpointing + resume** (this is the big one — a full 32B run is ~8.5 days):
    - After every completed batch, the running accumulators are dumped atomically (temp file +
      `os.replace`) to `../cache/causal_scores/<model>/<scoretype>_len<L>_n<N>.ckpt.pt`.
      Namespaced by `scoretype` so the concurrent **concept** and **token** jobs never collide.
    - On startup it auto-resumes: restores the accumulators and **fast-forwards past completed
      batches** (skipping their model calls). Resume requires a matching `meta` block
      (`model/n/bsz/sequence_len/random_tok`) or it errors.
    - `--overwrite` ignores any existing checkpoint and recomputes from scratch.
  - Accumulators are stored as **plain tensors** (not the `ChunkOutputSaver` class) so a
    checkpoint loads without a `__main__`-pickle dependency.
- `scripts/read_causal_checkpoint.py` (new) — dump the **(noisy, partial-run)** copier-head
  ranking from any checkpoint mid-flight. The score is `patched.m1_prob/n − corrupt.m1_prob/n`
  over however many examples have been processed (early on it's skewed toward whichever n-gram
  bucket has run, since buckets are processed sequentially).

## Setup (uv venv, isolated per-repo)
```bash
cd existing_works/dual-route-induction
uv venv --python 3.11 .venv
uv pip install -r requirements.smoke.txt
```
Version-critical pins (`requirements.smoke.txt`): `torch==2.6.0`, `nnsight==0.4.5`,
`transformers==4.50.3`. This clone uses **nnsight** (not baukit) for the head patching.
`.venv/` is **git-ignored** — recreate it on each machine.

Model weights are read from `HF_CACHE` (default `/disk/u/models` in `causal_scores.py`); set it
to wherever your `models--Qwen--*` dirs live. `pile-10k` is downloaded from HuggingFace at
runtime (`NeelNanda/pile-10k`).

## Data
- `data/counterfact_expanded.csv` (tracked, ~6.4 MB) — the entities used for **concept** mode.
- `cache/`, `activations/` (outputs) are **git-ignored** — regenerated.
- The large auxiliary datasets `data/{conneauetal_2017, dumasetal_2025, nguyenetal_2017}` are
  **NOT vendored** (not used by `causal_scores.py`). Fetch them from the upstream repo if you
  need the other experiments.

## Running
Concept-copying scores (real CounterFact entities):
```bash
cd existing_works/dual-route-induction/scripts
HF_CACHE=/disk/u/models TOKENIZERS_PARALLELISM=false ../.venv/bin/python causal_scores.py \
  --model Qwen/Qwen2.5-Coder-32B-Instruct --n 1024 --bsz 32 --sequence_len 30
```
Token-copying scores (random-token entities) — add `--random_tok_entities`:
```bash
  ... causal_scores.py ... --n 1024 --bsz 32 --sequence_len 30 --random_tok_entities
```
Outputs (at completion):
- `../cache/causal_scores/<model>/{concept,token}_copying_len30_n1024.json` — per-head scores
- `../cache/head_orderings/<model>/{concept,token}_copying.json` — ranked head list

Job entries: `job_list/qwen32b_dualroute_causal_localization.json` (parent repo) — the concept
and token jobs are meant to run on **disjoint GPU sets** (each shards a 32B model).

### Resuming after a crash
Just re-run the **identical** command — it detects the checkpoint and continues from the last
completed batch. To force a clean run, add `--overwrite`.

### Peeking at partial results mid-run
```bash
../.venv/bin/python read_causal_checkpoint.py \
  ../cache/causal_scores/Qwen2.5-Coder-32B-Instruct/concept_len30_n1024.ckpt.pt \
  --topk 30 [--out ranking.json]
```

## Smoke test (fast, small model — validates checkpointing)
```bash
cd existing_works/dual-route-induction/scripts
HF_CACHE=/disk/u/models CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python causal_scores.py \
  --model Qwen/Qwen2.5-Coder-7B-Instruct --n 8 --bsz 2 --sequence_len 10
```
4 batches total (one per n-gram bucket); kill it mid-run and re-run to confirm it resumes.

## Post-processing
`scripts/convert_causal_scores.py` converts the resulting `.pkl` into readable JSON scores
(upstream utility, unchanged).
