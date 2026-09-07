# CLAUDE_INSTRUCTIONS.md — Function Vectors (Todd et al.), vendored + Qwen-adapted

Vendored clone of **https://github.com/ericwtodd/function_vectors** (upstream HEAD
`fb9eac7b6dc707ea1475a717379916007fe448d5`). The upstream `.git` was removed so the code
lives directly inside the `goals` repo; pull the parent repo and this comes with it.

We use it for **head-level Causal Indirect Effect (CIE)** localization: sweep every
`(layer, head)`, replace that head's `o_proj`-input activation at the last query token with its
task-mean, and measure the change in the answer-token probability.

## What we changed vs. upstream
- `src/utils/model_utils.py` — added a `qwen` branch to `load_gpt_model_and_tokenizer`
  (bf16, `HF_CACHE` env, `device_map="auto"`, GQA-safety assert `hidden_size % n_heads == 0`,
  `prepend_bos=False` because Qwen has no BOS — uses `<|endoftext|>` as sequence-start).
- `src/utils/intervention_utils.py` — added a `qwen` branch to **both** o_proj dispatch blocks
  (`replace_activation_w_avg`, `add_avg_to_activation`): `new_output = inputs @ out_proj.T`
  (Qwen2 o_proj is bias-free). Without this the interventions raise `UnboundLocalError`.
- `src/compute_indirect_effect.py` — added timestamped `log()` milestones + a per-layer tqdm.
- `requirements.smoke.txt` — pinned, reproducible env (see below).

> A **numerically identical port** of the CIE method lives in the parent repo at
> `scripts/fv/localize_fv_cma.py` (validated bit-for-bit: `torch.equal=True`, `max_abs_diff=0`
> on both mean-activations and CIE). Prefer the port for new runs; this clone is the reference.

## GQA safety (why Qwen works here)
FV splits the o_proj **input** into `n_heads` chunks of `hidden_size // n_heads`. This is only
correct when `hidden_size == n_heads * head_dim`, which holds for Qwen2.5-Coder
(7B: 3584=28×128, 14B/32B: 5120=40×128). The `model_utils.py` assert enforces it.

## Setup (uv venv, isolated per-repo)
```bash
cd existing_works/function_vectors
uv venv --python 3.11 .venv           # resolves to 3.10.x on this box; fine
uv pip install -r requirements.smoke.txt
uv pip install git+https://github.com/davidbau/baukit    # activation patching (TraceDict)
```
Version-critical pins (in `requirements.smoke.txt`): `torch==2.6.0`, `torchvision==0.21.0`
(only because baukit's `__init__` imports torchvision — must match torch), `transformers==4.50.3`,
`datasets==3.5.0`. `.venv/` is **git-ignored** — recreate it on each machine.

Model weights are read from `HF_CACHE` (a HuggingFace cache root). On this machine that is
`/disk/u/models`; set it to wherever your `models--Qwen--*` dirs live.

## Datasets
`dataset_files/` (tracked, ~16 MB) holds the **full FV task suite** (`antonym`, `ag_news`,
`squad`, the `english-*` translation tasks, `conll2003_*`, …), split into `abstractive/` and
`extractive/`. The loader (`prompt_utils.load_dataset`) resolves
`dataset_files/<type>/<task_name>.json`. `results/` (mean-activation + indirect-effect `.pt`
outputs) is **git-ignored** — regenerated.

**Head localization is multi-task.** `compute_indirect_effect.py` computes the CIE for a *single*
`--dataset_name`. The paper's *universal* FV heads are the top heads by indirect effect
**averaged across the whole task suite** — which is why `extract_utils.compute_universal_function_vector`
carries hardcoded per-model `top_heads` lists (the paper's precomputed multi-task result).
Qwen is not in that list, so to reproduce the real localization you must run CIE for **each**
task, average the `indirect_effect` tensors, then take top-k. The `antonym` job alone gives you
single-task CIE, not the universal head set.

## Running — full multi-task localization (recommended: the port)

We drive runs with the numerically-identical port `scripts/fv/localize_fv_cma.py` (parent repo).
Per-dataset CIE, run from the **project root** (the port resolves FV src + `dataset_files`
via absolute paths, so no `cd` needed):
```bash
HF_CACHE=/disk/u/models TOKENIZERS_PARALLELISM=false \
  existing_works/function_vectors/.venv/bin/python scripts/fv/localize_fv_cma.py \
  --model_name Qwen/Qwen2.5-Coder-32B-Instruct --dataset_name antonym --device auto \
  --n_shots 10 --n-examples-per-relation 200 --seed 42 \
  --save_path_root existing_works/function_vectors/results
```
- `--n-examples-per-relation` (default **200**) = how many test examples per relation to
  CMA-sweep for the CIE. `--n_trials` (default 25) controls only the mean-activation prep step.
- Auto-computes `mean_head_activations` first if absent (cached + reused on restart), then
  the CIE sweep.
- Outputs → `results/<dataset>/<dataset>_{mean_head_activations,indirect_effect}.pt`
  (`indirect_effect` is `(n_examples_per_relation, n_layers, n_heads)`).
- `--device auto` shards across visible GPUs; warns loudly on CPU/disk offload (~100× slower).
- **CIE checkpointing (default on):** after every swept example the partial tensor is dumped
  atomically to `results/<dataset>/<dataset>_indirect_effect.ckpt.pt`. Re-run the identical
  command to resume from the last completed example (a resumed run is **bit-for-bit identical**
  to an uninterrupted one — the RNG stream is replayed). The checkpoint is deleted on
  completion; a finished dataset (final `.pt` present, no checkpoint) is skipped, so re-running
  the 57-job list is idempotent. `--overwrite` forces a clean recompute.

**Job list:** `job_list/fv/qwen32b_fv_head_cma.json` (parent repo) has **one job per dataset** for
the full 57-task suite. ⚠️ Cost: 200 examples × 64L × 40H ≈ 0.5 M patched forwards/dataset
(~hours each on 32B). Run the jobs across **disjoint GPU sets**, and/or trim the job list to a
subset — the repo's canonical eval set is `antonym, capitalize, country-capital, english-french,
present-past, singular-plural`.

### Aggregate → universal FV heads
Once the per-dataset `indirect_effect.pt`s exist, average them into the universal head set:
```bash
existing_works/function_vectors/.venv/bin/python scripts/fv/aggregate_fv_heads.py \
  --results_root existing_works/function_vectors/results \
  --n_top_heads 10 --model_name Qwen/Qwen2.5-Coder-32B-Instruct \
  --out existing_works/function_vectors/results/universal_fv_heads.json
```
Task-equal weighting by default (each relation weighted equally, as in the paper); pass
`--weight example` to weight every swept example equally, or `--datasets a b c` for a subset.
Writes the ranked `top_heads` JSON + a `*_grid.pt` of the full averaged `(L, H)` effect.

### Reference script (kept as the exact-match baseline, unchanged)
`src/compute_indirect_effect.py` is pure upstream + the Qwen branch; the port reproduces it
bit-for-bit. Use it only to re-verify parity — new runs should use the port.

## Smoke test (fast, small model)
```bash
HF_CACHE=/disk/u/models existing_works/function_vectors/.venv/bin/python \
  scripts/fv/localize_fv_cma.py --model_name Qwen/Qwen2.5-Coder-7B-Instruct \
  --dataset_name antonym --device auto --n_shots 4 --n_trials 2 \
  --n-examples-per-relation 2 --seed 42 --save_path_root /tmp/fv_port
```
