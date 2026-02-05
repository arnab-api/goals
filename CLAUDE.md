# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research codebase investigating whether LLMs have consistent intermediate goals during code generation. The central question: when an LM writes code (like control-flow logic `if b&1 == 0`), does it know ahead of time what it's going to write, or is it figuring it out token by token?

**Research Questions:**
1. **Goals**: Do Code-LLMs pursue consistent intermediate goals during generation?
2. **Mining Goals**: Can we replicate the goal-mining recipe from Schut et al.?
3. **Superhuman Knowledge**: Does RLVR training allow models to explore non-conventional approaches?

## Experimental Setup

The core methodology uses **prefill probing**:
- Simple code completion tasks with a unique correct solution (single goal)
- Insert different "prefills" before generation and check if the LM adapts while still achieving the goal
- Example: `def is_odd(n: int) -> bool: return True if ` with prefills like `1+`, `n &`, `not`, etc.
- If the model adapts to different prefills while solving the task correctly, it suggests consistent goal pursuit

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
