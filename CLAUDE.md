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

- `notebooks/00_goals.ipynb` - Main experimentation notebook
- `sample_eval.json` - Evaluation dataset with code completion tasks, unit tests, and prefills

## Architecture

**Core Abstraction - `ModelandTokenizer`** (`src/models.py`):
- Extends `nnsight.LanguageModel` for interpretability operations
- Supports GPT-J, LLaMA, Gemma, Qwen, OLMo, Pythia architectures
- Parses model config to provide unified interface (`n_layer`, `n_embd`, `layer_name_format`, etc.)

**Generation & Analysis** (`src/functional.py`):
- `generate_with_patch()` - Generation with optional activation patching
- `predict_next_token()` - Next token prediction with top-k analysis
- `get_hs()` - Extract hidden states at specific layers/positions

**Tokenization** (`src/tokens.py`):
- `prepare_input()` - Prepare batched inputs for models

**Evaluation** (in notebook):
- `evaluate_generated_code()` - Run unit tests on generated code completions
- `print_eval_results_table()` - Display results across prefills

## Configuration

- `globals.yml` - Directory paths (RESULTS_DIR, DATA_DIR, STATS_DIR, etc.)
- `env.yml` - API keys (OPENAI_KEY, CLAUDE_KEY, HF_WRITE, WANDB_KEY) and cache directories

## Key Dependencies

- torch>=2.7, transformers>=4.55, nnsight>=0.5
- baukit (activation patching, install separately)
- rich (table display)

## GPU Requirements

Requires CUDA. Tested with A100 GPUs. Large models need significant VRAM.
