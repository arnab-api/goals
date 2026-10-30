# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Code for the paper *Goal Encoding in LLMs*. When an LM writes code for a task with several sub-tasks, does it encode
*which sub-task (instrumental goal) it will implement next*, before writing it?

```
Broader goal:      implement the function correctly
Instrumental goal: check the timeout condition        ← what the goal heads encode
Implementation:    timeout > 0  vs  timeout >= 1  vs  int(timeout) > 0
```

The encoding of the instrumental goal should be invariant to the implementation strategy; a direction that only
predicts success may just capture competence (the central confound to avoid).

Branch `clean` keeps only the code behind results reported in the paper; the full research history (Track-1 prefill
probing / MCTS, clock / boundary heads, NRGS masks, subspace probes, exploratory notebooks) is on `master`.

## Pipeline

1. **Corpus**: `scripts/data/gen_goalscope_eval.py` → `data/goalscope_eval.json` (253 multi-sub-task specs,
   `src/nrgs/data.py` `QTASKS`).
2. **Goal-head localization**: oracle pivot mining (`scripts/mine_pivot_instances.py`, `src/goals/head_loc_oracle.py`;
   cross-language C++ → Python pairs at the pivot where two predicates diverge) → differentiable head mask
   (`scripts/train_head_mask.py`, `src/goals/head_loc_attn.py`, λ = 20) → exported head sets
   (`scripts/data/export_head_sets.py` → `head_sets/<model>/goal__lam20_thr0.9_band<lo>-<hi>.json`).
3. **GoalScope**: patch the goal heads' contribution at a source token into a probe prompt and let the model
   verbalize (`src/goals/goalscope.py::apply_goalscope`, `additive_norm`, α = 0.1, layers 35–45, re-injected every
   decode step); judged for faithfulness / causality by Claude (`scripts/evaluation/eval_goalscope.py`).
4. **Persistence / necessity**: temporal and obstacle persistence, sufficiency / necessity ablations
   (`scripts/evaluation/`).
5. **NRGS interventions** (`src/nrgs/`): pivot-onwards KV transfer of the task span (`scripts/nrgs/kv_transfer_eval.py`;
   `--span task|task+progress`, V-boost `--i3-v-scale`), Q transfer
   with gaze-select + restrict (`scripts/nrgs/q_transfer_eval.py`), task span vs code-progress span
   (`scripts/nrgs/task_vs_progress_eval.py`). Module / script names follow the paper; results, caches and figure
   dirs keep their legacy names (`results/diff/<model>/{path_patch,diff_trigger}`, `results/obstacle_robustness`,
   `notebooks/Figures/{transfer,precommit_ablation,sparsity_ablation,...}`) for cache compatibility.
6. **Beyond code**: ICL (`scripts/icl/`) and emergent misalignment (`scripts/em/`) reproductions.

README.md maps every paper section to its script, job list and figure generator.

## Setup

```bash
bash setup_env.sh
```

Creates the `goals` conda environment (Python 3.12, torch ≥ 2.7 from the CUDA 12.8 index, baukit from git). Create
`env.yml` in the root (template: `env_demo.yml`) with API keys and directory paths.

## Key Modules

- `src/models.py` — `ModelandTokenizer` (extends `nnsight.LanguageModel`; unified `n_layer`, `n_embd`,
  `layer_name_format`, … across the reported families: Qwen2, Llama, OLMo-3, GLM-4 dense).
- `src/functional.py` — `generate_with_patch()` (baukit patching), `get_hs()`.
- `src/tokens.py` — `prepare_input()` (left padding), `find_token_range()`.
- `src/goals/head_sets.py` — `load_goal_heads()` (json head set, mask fallback), `load_json_heads()`, `GOAL_BANDS`.
- `src/goals/head_loc_attn.py` — `HeadMask`, `cache_pivot_data()`, `train_head_mask()`,
  `collect_head_steering_vecs()`.
- `src/goals/goalscope.py` — `apply_goalscope()`, `head_additive_generate()`, `OVReader`.
- `src/goals/goalscope_viz.py` — paper-style HTML renderer (read-outs left of the code); also renders the site.
- `src/nrgs/functional.py` — `setup()` (the model), the head-patching hooks used by the NRGS scripts;
  `src/nrgs/kv_transfer.py` (pivot-onwards KV transfer), `src/nrgs/task_vs_progress.py`.
- `src/goals/site_data.py` — data writers for the companion site (goals-web repo, `GOALS_WEB_SITE`).

## Configuration

- `env.yml` — API key (`CLAUDE_KEY`) and cache / results directories
  (`src/utils/env_utils.py`).
- Long runs are dispatched with `python -m run_jobs.baulab.dispatch --jobs job_list/<group>/<list>.json`
  (`run_jobs/baulab/CONFIG.py` and `auth.yml` are local and gitignored).

## GPU Requirements

Requires CUDA. Tested on A100 / H100 80 GB; a 32B model in bf16 fits on one GPU. GoalScope steering is
step-by-step (CPU-launch-bound): expect ~15 min per 260-token sequence of all-token readouts on a busy machine.

## Reports

When generating analysis or reports, ask the user if they want to save it. If yes, save as `.log` or `.md` file in
`agent_reports/claude/` with an appropriate descriptive filename.

## Testing Guidelines

When implementing new features (evals, utils, classes, etc.):

1. **Ask about tests**: After implementing a feature, ask the user if they want a test script created.

2. **Test location**: Place test scripts in the `test/` directory with descriptive names.

3. **Test model**: Use a smaller LM for testing to save resources:
   - Default test model: `Qwen/Qwen2.5-Coder-14B-Instruct`
   - Initialize with: `ModelandTokenizer(model_key="Qwen/Qwen2.5-Coder-14B-Instruct", dtype=torch.bfloat16, device_map="auto", attn_implementation="eager")`

4. **Conda environment**: Always run tests in the `goals` conda environment:
   ```bash
   conda run -n goals python test/test_<feature>.py
   ```

5. **Ask before running tests**: Running tests consumes significant tokens (model loading, output parsing). Always
   ask the user before running test scripts. Only run tests autonomously if the user explicitly grants free rein.

## Design Decisions

- **Head mask parameterization**: `HeadMask` uses raw `nn.Parameter` initialized to all-ones, clamped to `[0, 1]`
  after each optimizer step (box-constrained GD, no sigmoid). Sparsity via `λ * mean(mask)` — scale-invariant to the
  number of layers tested.
- **Goal heads = exported json**: every reported run reads its heads from `head_sets/`; the json equals the lam20
  mask thresholded at 0.9 inside the model's `GOAL_BANDS` band (122 heads for Qwen2.5-Coder-32B).
- **Batch KL loss**: `reduction="batchmean"` in `train_head_mask` keeps the loss scale independent of batch size.
- **Two mask-training paths**: `scripts/train_head_mask.py` defaults to the Coder-32B lam20 run (50 epochs, batch 64,
  per-epoch augmentation with src head values recomputed per batch); `--no-augment` precomputes src head values once
  (the Coder-14B / Qwen2.5-32B / Llama-70B / OLMo-SFT masks).
- **Cross-language alignment**: source (C++) and destination (Python) prompts are padded to the same length (one
  batch for o_proj extraction), so both see identical absolute positions.
- **Left padding**: `ModelandTokenizer` left-pads via nnsight (raw Qwen tokenizers pad right).

## Known Issues

- **`ask_claude` cache race**: concurrent threads judging the same text can collide on the cache's temp file
  (`FileNotFoundError` on rename); re-run the affected file (calls are cached).
