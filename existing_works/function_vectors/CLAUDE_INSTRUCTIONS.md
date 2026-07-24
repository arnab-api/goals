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

## Running (reference script)
```bash
cd existing_works/function_vectors/src
HF_CACHE=/disk/u/models TOKENIZERS_PARALLELISM=false ../.venv/bin/python \
  compute_indirect_effect.py \
  --model_name Qwen/Qwen2.5-Coder-32B-Instruct \
  --dataset_name antonym --device auto \
  --n_shots 10 --n_trials 25 --seed 42 \
  --save_path_root ../results
```
- Auto-computes `mean_head_activations` first if absent, then the CIE sweep.
- Outputs → `../results/<dataset>/<dataset>_{mean_head_activations,indirect_effect}.pt`.
- `--device auto` shards across visible GPUs; the script warns loudly if any layer is
  offloaded to CPU/disk (that makes each forward ~100× slower — give it more GPUs instead).

Job entry: `job_list/qwen32b_fv_head_cma.json` (parent repo).

## Smoke test (fast, small model)
```bash
cd existing_works/function_vectors/src
HF_CACHE=/disk/u/models ../.venv/bin/python compute_indirect_effect.py \
  --model_name Qwen/Qwen2.5-Coder-7B-Instruct --dataset_name antonym \
  --device auto --n_shots 4 --n_trials 2 --seed 42 --save_path_root ../results
```
