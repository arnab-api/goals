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

The core methodology uses **prefill probing** with MCTS exploration:
- Simple code completion tasks (single broader goal, multiple possible instrumental approaches)
- MCTS explores the tree of possible continuations, labeling nodes green (correct) / orange (wrong) / red (crash)
- Contrastive pairs are formed between positive (green-leading) and negative (red/orange-leading) nodes at the same depth
- Hidden states at contrastive nodes are used to train linear "goal directions" (mean-diff and contrastive hinge loss)
- Directions are tested causally: steering with the direction should shift generation toward correct completions

**Example (current dataset):** `def is_odd(n: int) -> bool: return ` with prefills like `n &`, `n %`, `not`, `1+` — the model must adapt to different implementation constraints while still achieving the broader goal.

**Proposed harder setup (multi-predicate ordering):**
- Functions with multiple independent predicates (e.g., `validate_config(timeout, host, port)`)
- At a pivot point, the LM chooses which predicate to implement first (A, B, or C)
- All orderings are correct — contrast is "pursuing goal A" vs "pursuing goal B", not correct vs incorrect
- Requires multi-class probe (or one-vs-rest directions) instead of binary positive/negative
- Generalization test: does direction for "about to check non-empty string" transfer across different functions?

## Setup

```bash
conda env create -f conda_env.yml
pip install git+https://github.com/davidbau/baukit  # Required, not installed by conda
```

Create `env.yml` in root (see `env_demo.yml` for template) with API keys and directory paths.

## Key Files

- `notebooks/01_goal_consistency.ipynb` - Main experimentation notebook for prefill probing
- `data/incomplete_code.json` - Evaluation dataset with code completion tasks, unit tests, and categorized prefills (aligned/adversarial/inconsequential)

## Architecture

### Core Model Abstraction (`src/models.py`)

**`ModelandTokenizer`**:
- Extends `nnsight.LanguageModel` for interpretability operations
- Supports GPT-J, LLaMA, Gemma, Qwen, OLMo, Pythia architectures
- Parses model config to provide unified interface (`n_layer`, `n_embd`, `layer_name_format`, etc.)

### Generation & Analysis (`src/functional.py`)

- `generate_with_patch()` - Generation with optional activation patching
- `predict_next_token()` - Next token prediction with top-k analysis
- `get_hs()` - Extract hidden states at specific layers/positions

### Tokenization (`src/tokens.py`)

- `prepare_input()` - Prepare batched inputs for models

### Goals Module (`src/goals/`)

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
- `supports_chat_template()` - Heuristic to detect if model supports chat templates (checks for "instruct", "it", "chat" in model name)

## Configuration

- `globals.yml` - Directory paths (RESULTS_DIR, DATA_DIR, STATS_DIR, etc.)
- `env.yml` - API keys (OPENAI_KEY, CLAUDE_KEY, HF_WRITE, WANDB_KEY) and cache directories

## Key Dependencies

- torch>=2.7, transformers>=4.55, nnsight>=0.5
- baukit (activation patching, install separately)
- rich (table display)

## GPU Requirements

Requires CUDA. Tested with A100 GPUs. Large models need significant VRAM.

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
   sys.path.append(".")  # or appropriate path

   from src.models import ModelandTokenizer
   from src.goals.data import IncompleteCodeDataset
   # ... etc
   ```

## Design Decisions

- **Contrastive pair negative class**: In contrastive sampling (`src/goals/probe.py`), both **orange** (compiles but tests fail) and **red** (syntax error / crash) leaf nodes are lumped together as "negative" (`p_red`). This means contrastive pairs can be (green, orange), (green, red), or (green, mixed). Whether to separate these is an open question — "compiles but wrong" may carry a different latent signal than "doesn't compile at all." See `claude_reports/mcts_contrastive_design.md` for full details.

## Known Issues / TODOs

- **Steering coeff tuning is per-prefill**: In `scripts/test_causality.py`, the steering coefficient (`coeff`) is tuned independently per sample *and* per prefill. Ideally, a single `coeff` should work across prefills for a given sample (or across samples). This needs to be revisited — consider finding a shared `coeff` selection strategy.

- **`terminal_only=True` may yield too few contrastive pairs**: With `terminal_only=True`, only genuine terminal nodes (EOS, `</code>`, newline, max_depth) are evaluated. Since branches stop at the first newline, most terminals produce single-line completions — orange nodes (compiles but fails) are rare because a one-liner must be syntactically valid and produce wrong output. If training/test pair counts are too low, consider reverting to `terminal_only=False` in `goal_dir.py` (accepting stub contamination as noise) or relaxing `max_uncertain` in `SAMPLING_CONFIG`.

- **Intermittent `UnboundLocalError` in nnsight steering**: `generate_with_patch()` (via `nnsight`) occasionally raises `UnboundLocalError` for specific configs (observed: sample_idx=4, prefill="1 -", coeff=0.12, mean_diff directions, layers 40-50). The exact trigger is unclear — likely an nnsight implementation detail around lazy proxy resolution inside `gen_trace.all()`. Errors are captured and logged to `errors.log` in `scripts/test_causality.py`. **TODO**: Implement a baukit-based version of `generate_with_patch` to replicate and debug the issue outside of nnsight — baukit uses eager hooks and will give better control for tracing the root cause.
