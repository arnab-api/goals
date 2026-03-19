# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research codebase investigating whether LLMs have consistent intermediate goals during code generation. The central question: when an LM writes code, does it have a stable internal representation of *what it intends to do next*, or is it figuring it out token by token?

**Three-level goal hierarchy (important framing):**
```
Broader goal:      implement the function correctly
Instrumental goal: check the timeout condition        ← the target of this research
Implementation:    timeout > 0  vs  timeout >= 1  vs  int(timeout) > 0
```
The key claim: the LM may have a consistent hidden-state direction representing "I am pursuing instrumental goal G" that is **invariant to implementation strategy** — i.e., the direction fires for "about to check timeout" regardless of whether it writes `timeout > 0` or `timeout >= 1`. Implementation plasticity is noise; the instrumental goal representation is the signal.

**Research Questions:**
1. **Goals**: Do Code-LLMs have stable hidden-state representations of instrumental goals during generation?
2. **Mining Goals**: Can we extract linear directions that capture "intent to pursue goal G" (not just "will succeed")?
3. **Causality**: Do these directions causally influence generation, or are they merely correlated?
4. **Generalization**: Does a direction trained on one context (e.g., timeout-check in `validate_config`) transfer to a different context involving the same sub-goal?

**Critical distinction — goal direction vs competence direction:**
A direction that predicts success (correct vs incorrect completion) may just capture model competence, not intent. The goal is to find directions that capture *which* instrumental goal is being pursued, separable from *whether* it will be executed correctly.

## Experimental Setup

### Track 1 — Prefill Probing (Notebooks 01–04)

The original methodology uses **prefill probing** with MCTS exploration:
- Simple code completion tasks (single broader goal, multiple possible instrumental approaches)
- MCTS explores the tree of possible continuations, labeling nodes green (correct) / orange (wrong) / red (crash)
- Contrastive pairs are formed between positive (green-leading) and negative (red/orange-leading) nodes at the same depth
- Hidden states at contrastive nodes are used to train linear "goal directions" (mean-diff and contrastive hinge loss)
- Directions are tested causally: steering with the direction should shift generation toward correct completions

**Example (current dataset):** `def is_odd(n: int) -> bool: return ` with prefills like `n &`, `n %`, `not`, `1+`.

**Multi-predicate variant:** Functions with multiple independent predicates (e.g., `validate_config(timeout, host, port)`). At a pivot point, the LM chooses which predicate to implement first. All orderings are correct — contrast is "pursuing goal A" vs "pursuing goal B". Requires multi-class probe (or one-vs-rest directions).

### Track 2 — Attention Head Localization (Notebooks 05–06) ← active

**Methodology:** Cross-language interchange. A C++ prompt for predicate A has its attention head outputs patched into a Python prompt for predicate B at a pivot point. The heads that successfully transfer the "intent to implement A" are identified via a learned sparse mask.

**Pipeline:**
1. `scripts/mine_pivot_instances.py` — oracle-based pivot mining (Case 3 + Case 3b + augmentation)
2. `scripts/train_head_mask.py` — cache o_proj inputs + train differentiable head mask

**Current state:** 1480 pivot instances mined for Qwen2.5-Coder-32B-Instruct. Head mask training script exists but is still being smoke-tested end-to-end.

## Setup

```bash
conda env create -f conda_env.yml
pip install git+https://github.com/davidbau/baukit  # Required, not installed by conda
```

Create `env.yml` in root (see `env_demo.yml` for template) with API keys and directory paths.

## Key Files

### Track 1
- `notebooks/01_goal_consistency.ipynb` - Main prefill probing notebook
- `data/incomplete_code.json` - Code completion tasks, unit tests, and categorized prefills (aligned/adversarial/inconsequential)
- `data/multi_predicate/` - Multi-predicate tasks

### Track 2
- `notebooks/05_loc_data.ipynb` - Pivot instance mining + inspection
- `notebooks/06_loc_head.ipynb` - Head mask training + patched generation analysis
- `data/head_loc/predicates/` - 25 predicate JSONs (Python + C++)
- `data/head_loc/global_pairs.json` - Oracle-verified contrastive pairs
- `data/head_loc/pairs/<model>.json` - Model-specific pairs + 1480 pivot instances

## Architecture

### Core Model Abstraction (`src/models.py`)

**`ModelandTokenizer`**:
- Extends `nnsight.LanguageModel` for interpretability operations
- Supports GPT-J, LLaMA, Gemma, Qwen, OLMo, Pythia architectures
- Parses model config to provide unified interface (`n_layer`, `n_embd`, `layer_name_format`, `attn_module_name_format`, etc.)

### Generation & Analysis (`src/functional.py`)

- `generate_with_patch()` - Generation with optional activation patching (nnsight-based)
- `predict_next_token()` - Next token prediction with top-k analysis
- `get_hs()` - Extract hidden states at specific layers/positions

### Tokenization (`src/tokens.py`)

- `prepare_input()` - Prepare batched inputs for models (left-padding)

### Goals Module (`src/goals/`)

#### Track 1

**Data Types** (`src/goals/data.py`):
- `IncompleteCodeSample` - Single code completion task (instruction, code, unit_tests, prefills, function_name)
- `IncompleteCodeDataset` - Collection of samples with `load_dataset()` classmethod
- `Code` - A generation with prefill info, prompt, lm_generation, source_code, eval result
- `CodeGenerations` - All generations for a sample with evaluation summary
- `SingleCodeEval` - Evaluation result (verdict, unit_test_results, warnings, errors)
- `UnitTestResult` - Individual test result (input, expected, actual, passed, error)

**Evaluation** (`src/goals/eval.py`):
- `evaluate_single_code()` - Execute code and run unit tests, return `SingleCodeEval`
- `run_evaluation()` - Evaluate all generations in a `CodeGenerations` object
- `pretty_print_eval_results()` - Rich table output of evaluation results

**Functional Helpers** (`src/goals/functional.py`):
- `apply_chat_template()` - Format prompts for instruction-tuned models with `<code>` tags
- `supports_chat_template()` - Heuristic to detect if model supports chat templates

#### Track 2

**Data Types** (`src/goals/head_loc_data.py`):
- `SinglePredicateSample` - One predicate in Python + C++, trimmed at `return`, with unit tests and optional docstring
- `PivotInstance` - A validated pivot prefix for a predicate pair: `pivot_prefix`, `dest_top_token`, `target_top_token`
- `PredicatePair` - Directed pair (source, destination) with oracle-verified KL and list of `PivotInstance`s
- `HeadLocDataset` - Holds all `SinglePredicateSample`s and `PredicatePair`s; `load(model_name=)` classmethod; `pivot_instances` property returns `list[tuple[PredicatePair, PivotInstance]]`

**Prompt Builders & Next-Token Utilities** (`src/goals/head_loc_pivot.py`):
- `build_python_prompt(mt, sample, pivot_prefix, include_docstring)` - Builds the Python destination prompt up to pivot
- `build_cpp_prompt(mt, sample, include_docstring)` - Builds the C++ source prompt
- `get_next_token_probs(mt, prompt)` - Single-prompt next-token probs `(vocab,)` float32 CPU
- `get_next_token_probs_batch(mt, prompts)` - Batched version `(batch, vocab)` float32 CPU
- `get_top_token(probs, tokenizer)` - Decode top-1 predicted token string

**Oracle Pivot Mining** (`src/goals/head_loc_oracle.py`):
- `generate_global_pairs(ds, save)` - Ask oracle to verify all same-type predicate pairs; saves `global_pairs.json`
- `get_oracle_pivot_instances(mt, ds, ...)` - Main wrapper: Case 3 + Case 3b + augmentation → gated → saved
- `get_oracle_lcps(dest_sample, targ_sample, ...)` - Case 3: oracle returns candidate LCPs directly
- `case3b_pivot_instances(mt, ds, ...)` - Case 3b: oracle implementations → cross-product LCPs → gates
- `augment_pivot_instances(mt, instances, ds, ...)` - Augmentation gate check
- `_validate_candidate(mt, pair, dest, targ, lcp, ...)` - Completion gate + divergence gate → `PivotInstance | None`
- `_completion_gate(mt, dest_sample, targ_sample, lcp, ...)` - Both sides must pass unit tests

**Attention Head Caching & Training** (`src/goals/head_loc_attn.py`):
- `get_o_proj_inputs_batch(mt, prompts, layers)` - `{layer: (batch, n_embd)}` — batched o_proj extraction
- `precompute_per_head_deltas_batch(mt, src_o, dest_o, layers)` - `(batch, n_layers, n_heads, n_embd)` deltas
- `PivotCache` - `(pivot_instance, dest_prompt, log_p_target, per_head_deltas, src_prompt)` — pre-computed per instance
- `cache_pivot_data(mt, pivot_instances, ds, layers, precompute_deltas)` - 2–3 batched forward passes → list of `PivotCache`
- `HeadMask` - `nn.Module` with `values (n_layers_tested, n_heads)`, `clamp_()`, `selected_heads(layers, threshold=0.5)`
- `forward_with_masked_patch(mt, dest_inputs, per_head_deltas, mask, layers)` - baukit-based differentiable patched forward
- `MaskTrainingHistory` - dataclass with `loss_causal`, `loss_sparse`, `loss_total`, `n_active` lists
- `train_head_mask(mt, pivot_caches, layers, n_epochs, lr, lambda_sparse, batch_size)` - projected GD → `(HeadMask, MaskTrainingHistory)`

**Predicate Augmentation** (`src/goals/head_loc_augment.py`):
- Generates augmented predicate variants (variable rename, fn rename, type-hint toggle, predicate swap) for additional pivot instances

## Configuration

- `globals.yml` - Directory paths (RESULTS_DIR, DATA_DIR, STATS_DIR, etc.)
- `env.yml` - API keys (OPENAI_KEY, CLAUDE_KEY, HF_WRITE, WANDB_KEY) and cache directories

## Key Dependencies

- torch>=2.7, transformers>=4.55, nnsight>=0.5
- baukit (activation patching, install separately)
- rich (table display)

## GPU Requirements

Requires CUDA. Tested with A100 GPUs. Large models (32B) need significant VRAM.
Machine in use has 1.5 TB CPU RAM — `precompute_deltas=True` is fine for N=1480 (~42 GB for layers 6–40).

## Reports

When generating analysis or reports, ask the user if they want to save it. If yes, save as `.log` or `.md` file in `claude_reports/` with an appropriate descriptive filename.

## Testing Guidelines

When implementing new features (evals, utils, classes, etc.):

1. **Ask about tests**: After implementing a feature, ask the user if they want a test script created.

2. **Test location**: Place test scripts in the `test/` directory with descriptive names (e.g., `test_mcts_sampling.py`, `test_eval.py`).

3. **Test model**: Use a smaller LM for testing to save resources:
   - Default test model: `Qwen/Qwen2.5-Coder-14B-Instruct`
   - Initialize with: `ModelandTokenizer(model_key="Qwen/Qwen2.5-Coder-14B-Instruct", dtype=torch.bfloat16, device_map="auto", attn_implementation="eager")`

4. **Conda environment**: Always run tests in the `goals` conda environment:
   ```bash
   conda run -n goals python test/test_<feature>.py
   ```

5. **Ask before running tests**: Running tests consumes significant tokens (model loading, output parsing). Always ask the user before running test scripts. Only run tests autonomously if the user explicitly grants free rein for iterative development.

6. **Test structure**: Follow the pattern from `notebooks/02_sampling.ipynb` for LM initialization and data loading:
   ```python
   import sys
   sys.path.append(".")

   from src.models import ModelandTokenizer
   from src.goals.data import IncompleteCodeDataset
   # ... etc
   ```

## Design Decisions

- **Contrastive pair negative class**: In contrastive sampling (`src/goals/probe.py`), both **orange** (compiles but tests fail) and **red** (syntax error / crash) leaf nodes are lumped together as "negative" (`p_red`). Whether to separate these is an open question — "compiles but wrong" may carry a different latent signal than "doesn't compile at all." See `claude_reports/mcts_contrastive_design.md` for full details.

- **Head mask parameterization**: `HeadMask` uses raw `nn.Parameter` initialized to all-ones, clamped to `[0, 1]` after each optimizer step (box-constrained GD, no sigmoid). Sparsity via `λ * mean(mask)` rather than `λ * sum(mask)` — scale-invariant to number of layers tested.

- **Delta precomputation**: `cache_pivot_data` with `precompute_deltas=True` stores `(N, n_layers, n_heads, n_embd)` float32. For N=1480, layers 6–40, Qwen-32B: ~42 GB. With 1.5 TB CPU RAM this is fine. `precompute_deltas=False` (lazy) trades RAM for 2 extra forward passes per batch per epoch.

- **Batch KL loss**: `reduction="batchmean"` in `train_head_mask` = sum over vocab / batch_size, keeping loss scale consistent regardless of batch size.

- **Cross-language alignment**: Src (C++) and dest (Python) prompts are always padded to the same sequence length (combined into one batch for o_proj extraction) so both see identical absolute positions — prevents positional shift contaminating the delta.

## Known Issues / TODOs

- **Steering coeff tuning is per-prefill** (Track 1): In `scripts/test_causality.py`, the steering coefficient (`coeff`) is tuned independently per sample *and* per prefill. Ideally, a single `coeff` should work across prefills for a given sample. Needs revisiting.

- **`terminal_only=True` may yield too few contrastive pairs** (Track 1): With `terminal_only=True`, orange nodes (compiles but fails) are rare in single-line completions. Consider reverting to `terminal_only=False` or relaxing `max_uncertain` in `SAMPLING_CONFIG`.

- **Intermittent `UnboundLocalError` in nnsight steering** (Track 1): `generate_with_patch()` occasionally raises `UnboundLocalError` for specific configs (observed: sample_idx=4, prefill="1 -", coeff=0.12, layers 40-50). Errors are logged to `errors.log` in `scripts/test_causality.py`. TODO: implement baukit-based version of `generate_with_patch` for debugging.

- **`train_head_mask.py` smoke-test pending** (Track 2): The full end-to-end script exists but has not been run on the complete N=1480 dataset yet. Use `--max-instances 100 --n-epochs 1` for a smoke-test before the full run.
