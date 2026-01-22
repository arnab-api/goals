# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research codebase for the paper "LLMs Process Lists With General Filter Heads" (arXiv:2510.26784). Investigates how LLMs perform filtering operations over lists using specialized "filter heads" - attention heads whose query states encode compact representations of filtering predicates.

Project website: https://filter.baulab.info/

## Setup

```bash
conda env create -f conda_env.yml
pip install git+https://github.com/davidbau/baukit  # Required, not installed by conda
```

Create `env.yml` in root (see `env_demo.yml` for template) with API keys and directory paths.

## Key Commands

```bash
# Locate filter heads in a model
python -m scripts.locate_selection_heads --model="meta-llama/Llama-3.1-8B"

# Run finetuning pipeline
python run_finetuning.py

# GPU memory monitor for job queuing
python run_monitor.py
```

SLURM job scripts are in `run_jobs/delta/` (A100/H200) and `run_jobs/baulab/`.

## Architecture

**Core Abstraction - `ModelandTokenizer`** (`src/models.py`):
- Extends `nnsight.LanguageModel` for interpretability operations
- Supports GPT-J, LLaMA, Gemma, Qwen, OLMo, Pythia architectures
- Parses model config to provide unified interface (`n_layer`, `n_embd`, `layer_name_format`, etc.)

**Operators** (`src/operators/`):
- `BasisOperator` - Projects activations onto concept subspaces
- `CornerOperator` - Maps subjects to objects via corner representation
- Used for extracting and manipulating filter head representations

**Activation Patching** (`src/trace.py`):
- `patched_run()` - Forward pass with patched activations
- Core mechanism for testing predicate transfer across contexts

**Data Structures** (`src/dataset.py`):
- `Relation` - Subject-object relation datasets
- `RelationSample` - Individual subject-object pairs
- `ActivationPatchingSamples` - Patching experiment data

**ROME Module** (`src/rome/`):
- Rank-One Model Editing implementation
- `compute_u.py`, `compute_v.py` - Core ROME computations
- `layer_stats.py` - KK^T statistics caching

## Configuration

- `globals.yml` - Directory paths (RESULTS_DIR, DATA_DIR, STATS_DIR, etc.)
- `env.yml` - API keys (OPENAI_KEY, CLAUDE_KEY, HF_WRITE, WANDB_KEY) and cache directories
- `hparams/` - Hyperparameter YAML files for experiments

## Key Dependencies

- torch>=2.7, transformers>=4.55, nnsight>=0.5
- baukit (activation patching, install separately)
- pytorch-lightning (training), wandb (experiment tracking)
- circuitsvis (interpretability visualization)

## GPU Requirements

Requires CUDA. Tested with A100/A6000 GPUs. Large models need significant VRAM. Use `--use_8bit` flag for quantization when available.
