# The reproduction campaign: install and run

This branch (`nnterp-campaign`) is the `nnterp` port of the repo plus everything the November 2026 reproduction
campaign added: the paper's experiments re-run on the paper's models and on new ones (Devstral Small 2, GLM-4.5-Air,
Llama-3.3-70B), per-model drivers for the seven reported result groups, the head-set rules for a model the paper did
not run, and the progress and results tools. The campaign's design notes are in `repro/PLAN.md`; the results are
summarized on the progress page shared with the authors.

## Install

```bash
git clone -b nnterp-campaign https://github.com/arnab-api/goals && cd goals
uv venv --python 3.12 && uv sync      # torch (CUDA 12.8 wheels), nnsight (dev pin) and nnterp (v2.0.0 pin), from pyproject.toml
cp env_demo.yml env.yml               # then edit: CLAUDE_KEY (the judge), HF_CACHE (where the models are), RESULTS_DIR
```

`bash setup_env.sh` builds the same environment as a conda env instead. nnsight and nnterp are pinned to git
commits in `[tool.uv.sources]` (the comments there say which PRs each pin carries); move them to releases when those
exist. Everything runs from the repository root with `.venv/bin/python`.

Two environment variables the shell drivers read, both optional: `GOALS_HF_CACHE`, the Hugging Face hub cache to
export as `HF_HUB_CACHE` (the loader consults the hub cache for one lookup besides `env.yml`'s `HF_CACHE`; set both to
the same directory when the models are not in the default cache), and `GOALS_LOGS`, where the lane logs go (default
`logs/` in the repo). The drivers run offline (`HF_HUB_OFFLINE=1`) unless you set it to 0.

Check the install on any model with the cross-family check (one GPU, a few minutes):

```bash
.venv/bin/python scripts/cross_family_check.py --model Qwen/Qwen2.5-Coder-7B-Instruct --heads random:60 --seed 0
```

It runs every experiment primitive once and prints `PRIMITIVE | OK/FAIL | reason` (CLAUDE.md section 7 has the matrix
over seven families).

## Run the paper's groups on a model

Each model has a driver, `repro/eval_<slug>.sh`, that holds its head sets and constants and defines the groups by
the paper's section numbers (CLAUDE.md section 4): `4.1` GoalScope additive readout, `4.4` replace readout, `4.6`
ablations, `4.7` temporal persistence, `4.8` obstacle persistence, `4.9` KV transfer, `4.12` Q transfer, `4.13` task
vs progress; `4.6L` / `4.6F` are the large-set and band-floor ablation arms. A lane runs its groups one after another
on the given GPU(s):

```bash
bash repro/eval_coder32b.sh 0 4.1 4.4 4.6 4.8 4.7 4.9 4.12       # Coder-32B, every reported group, one GPU
bash repro/eval_glm45air.sh 4,5,6,7 4.1 4.4                      # GLM-4.5-Air over four GPUs (device_map="auto")
```

Every script resumes from its per-unit cache, so a stopped lane restarts where it was. Per group the lane writes
`$GOALS_LOGS/eval-<slug>-<group>.log`, the judge-call log `judge-<slug>-<group>.jsonl`, the running python's pid, and
one line per command in `eval-<slug>.status`; a lane claims a group with the directory `eval-<slug>-<group>.claim`
and skips a group another lane holds (remove the directory to rerun it). `repro/eval_lib.sh` is the runner.

```bash
.venv/bin/python repro/eval_progress.py                                     # per group: units done, rate, ETA, judge calls and spend
.venv/bin/python repro/eval_results.py Qwen2.5-Coder-32B-Instruct --curves  # the headline numbers, as CLAUDE.md section 4 reports them
.venv/bin/python repro/eval_results.py <slug> --holdout 32                  # GoalScope also on the specs a fit check did not tune on
```

The judge is `claude-sonnet-4-6` at the paper's temperatures; a full seven-group run of a 32B model is about 20
GPU-hours and $30-40 of judge calls (`eval_progress.py` prints the running total).

## A model the paper did not run

1. **Localize its goal heads**: `bash repro/localize.sh <hf id> <gpus> <batch>` mines the pivot pool for the model
   (`scripts.mine_pivot_instances`, the oracle answered from the judge cache) and trains the λ=20 mask
   (`scripts.train_head_mask`, 50 epochs) into `results/localization/<slug>/lam20/head_mask.pt`. A 24B model takes
   about 1.5 h on one 80 GB GPU; a 107B MoE about 4 h on four.
2. **Export the head sets**: `python scripts/data/export_head_sets.py --models <hf id>` writes the goal set by the band
   rule (the smallest layer window holding 85% of the heads above 0.9, lowest if tied; `src/goals/head_sets.py`),
   and `--topk K --control` the large set and its later-half random control. K is the paper's 250 scaled to the
   model's head count (9.8%), capped at the heads the mask ranks (its nonzero values).
3. **Check it**: `scripts/cross_family_check.py --model <hf id> --heads head_sets/<slug>/goal__lam20_*.json`.
4. **Write its driver** from an existing one: the head-set paths, the steer layers (the paper's 35-46 of 64 scaled by
   depth), the sufficiency floor (20 of 64 scaled by depth, and never inside the goal band), the GPUs. For the
   additive readout, fit α and the steer window on 32 specs first (`--limit 32`; the paper's α 0.1 under-steers every
   non-Qwen model we ran) and report the other 221 specs with `--holdout 32`. The replace readout needs no constants
   and is the cross-model number.
5. Add a colour for the model in `notebooks/fig_n_table_generators/figstyle.py` and run the generators with
   `GOALS_MODEL=<slug>`.

What does not carry over to every family: OVReader refuses post-norm families (OLMo-3, GLM-4) until nnterp serves the
post-attention norm gain (nnterp #135), and fused-QKV families (Phi-3); families whose attention is not the standard
per-head form (BLOOM, MPT, XGLM, Falcon, DeepSeek latent attention) are refused by nnterp's attention call, so the KV
and Q transfers do not run on them.

## What else is here

`repro/PLAN.md` (the campaign's design and timings), `localize.sh`, `mining_progress.py`, `load_probe.py`,
`batch_probe.py`, `run_with_mem.py`, `run_stub_judge.py` (an evaluation with the judge answered from the cache or
stubbed, for timing), and the `time_*.sh` timing probes of the campaign's machine.
