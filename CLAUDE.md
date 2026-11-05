# CLAUDE.md

Guidance for agents working in this repository. README.md is the human-facing summary (project-page takeaways and
setup), so it is not repeated here. This file maps **every reported result** of the paper to the exact script, job list,
configuration, results directory and figure/table generator, so that an agent can reproduce the results.

## 1. Orientation

Code for the paper *Goal Encoding in LLMs*. When an LM writes code for a task with several sub-tasks, does it encode
*which sub-task (instrumental goal) it implements next*, before it writes it? A sparse set of attention heads
(**goal heads**) carries that encoding. GoalScope verbalizes it, ablations show the heads are a causal bottleneck,
and NRGS interventions (KV / Q transfer) show how the LM re-reads the goal from the task span.

```
Broader goal:      implement the function correctly
Instrumental goal: check the timeout condition        <- what the goal heads encode
Implementation:    timeout > 0  vs  timeout >= 1  vs  int(timeout) > 0
```
The encoding should not depend on the implementation strategy. A direction that only predicts success may just capture
competence; that is the central confound to avoid.

Branch `clean` keeps only the code behind reported results. The full research history (Track-1 prefill probing / MCTS,
clock / boundary heads, NRGS masks, exploratory and dev notebooks) is on `master`. Paper source:
`Overleaf-draft/Goals-Overleaf/` (`main.tex` -> `Text/*.tex`, `Text/201_Appendix.tex` -> `Text/Appendix/*.tex`).
Appendix files that `201_Appendix.tex` does not `\input` (`boundary`, `cpp`, `dissociation`, `gaze`, `mlp`, `olmo`,
`mechanistic_details`, `oracle_prompts`) and `Text/Figures/{goal_eval,sparsity_ablation}.tex` are not in the paper.

### Layout

```
src/models.py            ModelandTokenizer: a subclass of nnterp's StandardizedTransformer (nnterp 2 on nnsight 0.8),
                         the model + its tokenizer (left padding), nnterp's standard names / values / sizes on every
                         family nnterp standardizes (the paper's: Qwen2, Llama, OLMo-3, GLM-4; others: section 7)
src/functional.py        residual-stream edits: PatchSpec + generate_with_patch (layer_output under mt.generate +
                         tracer.iter); get_hs (module outputs by path)
src/tokens.py            prepare_input (left padding), find_token_range, chat_prompt (every prompt's chat template)
src/utils/               env_utils (paths), oracle_llms (ask_claude + cache), logging, experiment_utils
src/goals/               head localization (head_loc_*), head_sets.py, goalscope.py (+ OVReader),
                         goalscope_viz.py, temporal_persistence.py, site_data.py, proactive_strategy.py
src/nrgs/                data.py (QTASKS bank, multi-predicate setups), hier_data.py, functional.py (setup; the
                         Intervention vocabulary: Edit / Intervention, edits of nnterp values kept in forward order,
                         applied in one trace or at every mt.generate step; greedy), kv_transfer.py,
                         task_vs_progress.py, eval.py (judges)
scripts/                 mine_pivot_instances.py, train_head_mask.py, data/, evaluation/, nrgs/, icl/, em/,
                         site scripts (generate_well_known_task_completions.py, mine_humaneval_mbpp.py,
                         run_goalscope_web.py, run_goalscope_fig32_stable_marriage.py),
                         cross_family_check.py (every experiment primitive on one model, section 7)
test/                    parity/ (7B parity harness and its reference outputs), test_*_port.py (A/B of each port
                         against the code it replaced, Qwen2.5-0.5B on CPU; the old code runs on _legacy.py's
                         LegacyModelandTokenizer, which adds the pre-nnterp names it reads), test_model_layer.py
                         (model layer + import sweep), cross_family/ (the section 7 summaries)
notebooks/fig_n_table_generators/   one notebook per figure group (+ figstyle.py, q_transfer_schema.py)
job_list/                exact dispatch lists of the reported runs (goalscope_eval/, diff_eval/, head_localization/)
head_sets/               git-tracked [[layer, head], ...] json head sets read by every reported run
data/                    goalscope_eval.json (corpus), head_loc/ (predicates, global_pairs, pairs/<model>.json),
                         FV_dataset/ (ICL), well-known-tasks/ (site), LLM_CACHE/ (judge cache, untracked)
run_jobs/                baulab dispatcher (baulab/) and slurm submitter (slurm/)
goals-web/               SEPARATE git repo (companion site); gitignored here
```

### Conventions

- **Run everything from the repo root.** Scripts do `sys.path.append(".")`; both `python -m scripts.x.y` and
  `python scripts/x/y.py` work.
- **Environment**: uv. `bash setup_env_uv.sh` (or `uv venv && uv sync`) builds `.venv`; run `.venv/bin/python ...`.
  `pyproject.toml` pins nnsight and nnterp from git (`[tool.uv.sources]`): nnsight's dev branch at `e09ab1c8`, which
  carries ndif-team/nnsight#824 (the PEFT adapter switch), until the 0.8.3 release, and the nnterp v2.0.0 commit that
  carries ndif-team/nnterp#124, #127, #128, #134 and #136 until a release has them (`override-dependencies` lets the dev
  branch's rc-numbered version satisfy `nnsight>=0.8.2`); transformers >= 5.17. `bash setup_env.sh` builds the
  same as the `goals` conda env. Import `nnterp` before any `transformers.models...` module (the reverse order
  segfaults on this stack).
- **Models load through nnterp**: `ModelandTokenizer(model_key, dtype=torch.bfloat16, attn_implementation="eager")`.
  Eager attention is required for anything inside the attention (patterns, scores, queries / keys / values, per-head
  outputs): `get_attention_matrices`, the ablations, the KV and Q transfers. Under `sdpa` those values are
  unavailable. `mt.support()` lists what a checkpoint serves and why anything is missing; `get_attention_matrices`
  and `headwise_kv_patch` check it before reading.
- **Paths**: `env.yml` in the root (template `env_demo.yml`) holds `MODEL_DIR`, `RESULTS_DIR` (default `results`),
  `DATA_DIR`, `CLAUDE_CACHE_DIR` (default `data/LLM_CACHE/claude_cache`), optional `HF_CACHE` and `EM_ADAPTER`
  (4.16), and `CLAUDE_KEY`.
  `src/utils/env_utils.py` reads them (`DEFAULT_RESULTS_DIR`, `PROJECT_ROOT`). All results paths below are relative to
  `RESULTS_DIR` = `results/`. Treat the existing `results/` as read-only reference data; write re-runs to a scratch
  `RESULTS_DIR` if you want to compare.
- **Script defaults = the paper's configuration.** Every eval script's argparse defaults reproduce the main reported
  run; job lists only add the per-datatype / per-mode / per-language fan-out and the non-default arms (e.g.
  `--i3-v-scale 1 2 4`, control head sets, `--tag-suffix __seq`). Smoke runs: `--limit N` (GoalScope / OVReader /
  ablations; writes a separate `nspec=N` leaf) or `--max-instances N` (NRGS scripts).
- **LM judge**: every judge is Claude (`claude-sonnet-4-6`) via `src/utils/oracle_llms.ask_claude(..., use_cache=True)`.
  The on-disk cache (`CLAUDE_CACHE_DIR`, one json per hashed (system prompt, prompt, max_tokens)) means re-running a
  judged eval whose generations reproduce exactly costs no API calls and returns the cached verdicts. Carry
  `data/LLM_CACHE/claude_cache` along when reproducing on another machine.
- **Dispatcher**: `python -m run_jobs.baulab.dispatch --jobs job_list/<group>/<list>.json` (see
  `run_jobs/baulab/README.md`; `CONFIG.py` and `auth.yml` are local and gitignored, seed them with
  `bash run_jobs/baulab/setup.sh`). Each job's `command` field is a plain shell command, so any job can also be run by
  hand: copy its `command` and run it from the repo root. Jobs `tee` logs into `logs/`.
- **GPU**: CUDA required. A 32B model in bf16 fits on one 80 GB A100/H100 (eager attention); Llama-3.3-70B and
  Qwen2.5-72B need 2 GPUs (`device_map="auto"`). Most scripts take `--cuda 0` or `--cuda 0,1`. 7-9B models fit one
  48 GB card: the parity harness (`test/parity/`, Coder-7B, about 50 s on an RTX A6000) and the cross-family check
  (section 7, 15-30 s per model) run there with `device_map={"": 0}`.
  GoalScope steering is step-by-step (CPU-launch-bound) and the judged evals are oracle-bound (thousands of serial
  `ask_claude` calls per shard); plan hours per GoalScope / OVReader job, not minutes (not re-timed on `clean`).
- **Legacy names**: modules / scripts / notebooks follow the paper's terms, but results directories, caches and figure
  groups keep their legacy names for cache compatibility: `results/diff/<model>/{path_patch,path_patch_hier,
  diff_trigger,gsr_example}` (NRGS), `results/obstacle_robustness/` (obstacle persistence),
  `results/notebook_caches/{task_vs_progress,obstacle_ov_boost}/`, and figure groups `transfer`,
  `precommit_ablation`, `sparsity_ablation`, `obstacle_robustness` under `notebooks/Figures/`.
- **Generators** (`notebooks/fig_n_table_generators/*.ipynb`) only parse results (no model). They save through
  `figstyle.save_fig` / `save_table` to `notebooks/Figures/<group>/<name>.{pdf,png,tex}`. Copying into
  `Overleaf-draft/Goals-Overleaf/Figures/generated/<group>/` (and `Text/Appendix/tables/generated/`) and `pdfcrop`
  (`*-crop.pdf`) were done by hand. Cell numbers below are 0-based code+markdown cell indices. The per-group
  notebooks plot Coder-32B unless `GOALS_MODEL` (an HF id or its slug, `figstyle.model_slug`) names another model;
  `kv_transfer` / `q_transfer` also read `GOALS_LARGE_HEADS` (the run's head-set tag, default `goal_250`), `ablation`
  and `readout_sources_and_head_budget` cell 9 `GOALS_ABLATION_SETS` (the two set sizes, default `122,250`).

## 2. Reproduce from scratch: dependency order

Everything downstream of step 4 reads its heads from `head_sets/` (tracked), so in practice you can start at step 5
with the shipped head sets. Steps 2-4 are needed only to re-derive the heads (and will not reproduce them bit-exactly,
see Caution 6b).

1. **Corpus** (no model, no oracle, deterministic):
   `python scripts/data/gen_goalscope_eval.py` -> `data/goalscope_eval.json` (seed 123, target 64 per datatype;
   253 specs = int 65 / list[int] 60 / str 64 / list[str] 64; task bank `src/nrgs/data.py::QTASKS`). Check: write to a temp path
   (`--out /tmp/x.json`) and diff `datapoints` against the tracked file (ignore `meta`: generation date, and the tracked
   file names the pre-rename bank `src.diff.data.QTASKS`). The NRGS scripts do not read this file (their banks are
   inline in `src/nrgs/data.py` / `hier_data.py`).
2. **Pivot mining** (model + Claude oracle): `python -m scripts.mine_pivot_instances --model-key <hf id>` ->
   `data/head_loc/pairs/<slug>.json`, from `data/head_loc/predicates/*.json` (25 predicates, Python + C++) and
   `data/head_loc/global_pairs.json` (oracle-verified pairs). Oracle prompts live in `src/goals/head_loc_oracle.py`;
   the contextual next-token filter is `src/goals/pivot_token_filter.py`.
3. **Head-mask training**: `python -m scripts.train_head_mask --model-key <hf id> [...]` ->
   `results/localization/<slug>/<tag>/head_mask.pt` (+ `checkpoints/` every 5 epochs). Defaults
   (`--n-epochs 50 --lambda-sparse 20 --batch-size 64 --augment --tag lam20`, lr 1e-2, augment seed 123456, all layers)
   = the Coder-32B lam20 run. Per-model settings: section 3.2.
4. **Head-set export** (no model): `python scripts/data/export_head_sets.py` -> `head_sets/<slug>/goal__<tag>_thr0.9_band<a>-<b>.json`
   for the models in its `MODEL_BANDS` (Coder-32B 25-56, Qwen2.5-32B 25-56, Coder-14B 0-48, Llama-3.3-70B 18-45 on its
   `lam20_ctx` mask (`MODEL_TAGS`), OLMo Instruct / SFT 10-36; GLM-4.5-Air and Devstral by the band rule, section 7);
   threshold 0.9 on the mask. `--topk K [--control]` writes the top-K set and its later-half control (section 3.2).
   The hand-made OLMo sets and the `snap_top` sets have no script on `clean` (section 3.2).
5. **Everything that reads `head_sets/`**: GoalScope eval and readout baselines (sections 4.1-4.4), OVReader (4.5),
   ablations (4.6), temporal / obstacle persistence (4.7-4.8), NRGS interventions (4.9-4.13), unspecified-task readouts (4.15),
   ICL / EM (4.16), site (4.17).
6. **Generators**: run the notebook for the figure group (section 4), then copy / crop into Overleaf by hand.

## 3. Goal-head localization and head sets

### 3.1 Paper: §3.1 (`Text/03_what.tex`, "Localizing goal heads") and App. "Localizing Goal Heads"
(`Text/Appendix/attn_head_localization.tex`, `fig:localization` = `Figures/goalscope/head-loc-crop.pdf`)

- Claim: "122 heads distributed in layers 28-55 of the 64-layer Qwen-2.5-Coder-32B-Instruct".
- Mask: `results/localization/Qwen2.5-Coder-32B-Instruct/lam20/head_mask.pt`, config stored in the checkpoint:
  n_instances 1423, 50 epochs, lr 0.01, λ 20, batch 64, augment True (src head values recomputed per batch), mask
  shape (64, 40). Loss history `head_mask.pt['history']['loss_causal']` starts 18.09, 17.86, 17.56, 10.92, 2.58.
- CLI (= defaults): `python -m scripts.train_head_mask` (Coder-32B). No job list (original run); the other λ for
  Coder-32B: `job_list/head_localization/qwen32b_coder_goal_lambda.json` (lam1/5/50/100, `--no-augment --batch-size 32`,
  50 epochs) -> `results/localization/Qwen2.5-Coder-32B-Instruct/lam{1,5,50,100}/`.
- Export: `python scripts/data/export_head_sets.py` -> `head_sets/Qwen2.5-Coder-32B-Instruct/goal__lam20_thr0.9_band25-56.json`
  (122 heads, layers 28-55).
- Figure: right panel = `notebooks/fig_n_table_generators/head_localization.ipynb` cell 2 ->
  `notebooks/Figures/head_localization/head_localization_map.pdf` (reads the mask, threshold 0.9, sanity-checks against
  the 122-head json). The left schematic and the composite `head-loc-crop.pdf` are hand-made.
- Smoke: `python -m scripts.train_head_mask --tag scratch_smoke --n-epochs 1` (exercises the pipeline; numbers will not
  match). Compare epoch-5 numbers with `.../lam20/checkpoints/checkpoint_epoch_0005.pt`.

### 3.2 Every file in `head_sets/`

| file | heads | used by | how it was made |
|---|---|---|---|
| `Qwen2.5-Coder-32B-Instruct/goal__lam20_thr0.9_band25-56.json` | 122 | GoalScope main, OVReader, ablations (122), temporal persistence, ICL, EM, site | `export_head_sets.py` (lam20 mask > 0.9, layers [25,56)) |
| `Qwen2.5-Coder-32B-Instruct/goal_topk/goal__lam20_top{150,200,250}.json` | 150/200/250 | head-budget sweep; **250 = all NRGS runs, ablations (250), obstacle OV boost** | top-k by mask value of the lam20 mask over all 64 layers (ties by index): `export_head_sets.py --topk 150 200 250` (byte-identical to the tracked files; first made by `notebooks/clock/__head_inspection.ipynb` on master, commits 3a5c328, c2fe051). top250 contains all 122 goal heads and 14 heads below layer 20 |
| `.../goal_topk/goal__lam100_top{50,100}.json`, `goal__lam5_top300.json` | 50/100/300 | head-budget sweep | same recipe on the lam100 / lam5 masks (c2fe051 message: each k at a λ whose mask supports it): `export_head_sets.py --tag lam100 --topk 50 100`, `--tag lam5 --topk 300` (byte-identical) |
| `.../goal_topk/goal__snap_top{5,10,25}.json` | 5/10/25 | head-budget sweep | `scripts/data/export_sparse_head_sets.py` **on master only** (ranks heads by the number of lam{1,5,20,50,100} checkpoints at 1.0; top5 = seeded random 5 of top10) |
| `Qwen2.5-Coder-32B-Instruct/control/control{122,250}__laterhalf.json` | 122/250 | NRGS random-head controls (250); control122 only in unreported GoalScope baselines | `sorted(random.Random(123).sample(...))` of the heads of layers >= 32 not in the goal set of the same size (control250: not in top250, `export_head_sets.py --topk 250 --control`, byte-identical; control122: not in the 122 band set, no script); committed in master 8ed1474 |
| `Qwen2.5-Coder-14B-Instruct/goal__lam20_thr0.9_band0-48.json` | 88 | model-size axis | `export_head_sets.py` |
| `Qwen2.5-32B-Instruct/goal__lam20_thr0.9_band25-56.json` | 108 | code-finetuning axis + model table | `export_head_sets.py` |
| `Olmo-3.1-32B-Instruct/goal__lam20_thr1.0_band10-36.json` | 93 | RL axis (reported) | mask == 1.0 in layers [10,36); balanced against SFT by hand in master ae9f2c2, **no script** |
| `Olmo-3.1-32B-Instruct-SFT/goal__lam20_thr0.6_band10-36.json` | 91 | RL axis (reported) | mask > 0.6 in [10,36); master ae9f2c2, **no script** |
| `Olmo-3.1-32B-Instruct{,-SFT}/goal__lam20_thr0.9_band10-36.json` | 130 / 79 | superseded `heads=goal` OLMo runs | `export_head_sets.py` |
| `Llama-3.3-70B-Instruct/goal__lam20_ctx_thr0.9_band18-45.json` | 110 | site only (and an unreported eval) | lam20_ctx mask (`results/localization/Llama-3.3-70B-Instruct/lam20_ctx/`, 1433 instances, 50 epochs, batch 8, no-augment), thr 0.9, layers [18,45): `export_head_sets.py` (`MODEL_TAGS`, byte-identical; first made in master 8007182 / `notebooks/diff/13_goalscope_qualitative.ipynb`) |
| `Qwen2.5-Coder-32B-Instruct/set_diff_localization/*.mask.pt` | - | nothing on clean (unreported set-diff query localization; master 37c6f42) | - |

Masks behind the other models (all `results/localization/<slug>/lam20/head_mask.pt`, config read from the checkpoints):
Coder-14B 1246 instances, 50 epochs, batch 32, no-augment (`job_list/head_localization/qwen14b_localization_lambda_sweep.json`,
job `01b_qwen14b_goal_lamb_20`); Qwen2.5-32B 1181 instances, 100 epochs, batch 16, no-augment; OLMo-SFT 851 instances,
100 epochs, batch 16, no-augment (the job list of those two runs was not carried to `clean`; equivalent CLI:
`python -m scripts.train_head_mask --model-key <id> --n-epochs 100 --batch-size 16 --no-augment`); OLMo-Instruct: see
Caution 6a. `--heads goal` in `eval_goalscope.py` resolves through `src/goals/head_sets.load_goal_heads` (the
exported json if present, else the mask thresholded at 0.9 in `GOAL_BANDS[model]` or `--goal_band`).

## 4. Reported experiments

Notation: `GS = results/evaluation/goalscope/<model>/<mode>/<strategy>/<alpha>/<leaf>/` with
`<leaf> = heads=<name>__layers=35-46__nsteer=all__maxnew=64__span=frontier__dtype=all__nspec=253__nc=2-3-4__seed=123[__band=a-b]`
(`scripts/evaluation/faithfulness_io.run_leaf_name`; `<name>` = `goal` for `--heads goal`, `json:<stem>` for a json,
`band=` only when `--goal_band` is passed). Each leaf holds `config.json`, `records.jsonl` (253 specs x 5 pivots),
`samples.jsonl`, `summary.json`. Headline metric = `summary.json["by_anchor"][pivot]["hard_yes"]` for the four pivots
`prev_end, indent, var, eq` (the mean over these four is "mean4"; `first_impl` is never reported). Layer ranges in
directory names are half-open (`layers=35-46` = layers 35-45, `L40-50` = 40-49).

### 4.1 GoalScope main result: §3.2-3.3 (`Text/03_what.tex`), `fig:goalscope` (`Text/Figures/goalscope.tex`, `Figures/goalscope/goalscope-schema-eval-crop.pdf`), App. "GoalScope" (`Text/Appendix/goalscope.tex`)

- Commands (`job_list/goalscope_eval/goalscope_eval_a_residual.json`, jobs `goalscope_a_goalheads_{faithfulness,causality}`):
  ```
  python -m scripts.evaluation.eval_goalscope --mode faithfulness      # all other flags = defaults
  python -m scripts.evaluation.eval_goalscope --mode causality
  ```
  Defaults: `--model Qwen/Qwen2.5-Coder-32B-Instruct --heads goal (122) --strategy additive_norm --alpha 0.1
  --steer_layer_range 35 46 --seed 123`; readout 64 tokens (`MAX_NEW_TOKENS`), frontier rollout 64 tokens
  (`GEN_MAX_NEW_TOKENS`), corpus `data/goalscope_eval.json`, judge `claude-sonnet-4-6`.
- Results: `GS` with model Coder-32B, `additive_norm/0.1`, `heads=goal`.
- Cross-check (matches paper "~60%", "0.90+", "80%", "~75%"): faithfulness 0.589 / 0.945 / 0.905 / 0.972 (mean4 0.853);
  causality 0.522 / 0.802 / 0.719 / 0.957 (mean4 0.750). Smoke: `--limit 4`; the first records of the reported
  `records.jsonl` are spec `int_python_000`.
- Judge rubric: `FAITH_HARD_SYS` / `CAUSAL_HARD_SYS` in `scripts/evaluation/eval_goalscope.py`; `config.json` stores
  `hard_rubric_tag` = md5[:8] of the system prompt (reported: faithfulness `78bfd5f4`, causality `9c106246`). If the tag
  of the current code differs, the rubric changed and the verdicts will not match the paper. The prompts printed in the
  appendix are `Text/Appendix/prompts/*.txt` (byte-identical to the code); `fig:readout-prompt`
  (`Figures/goalscope/readout_prompts.png`) and `fig:oracle-faith` (`Figures/goalscope/evaluation/faithfulness.png`) are
  hand-made images of them.
- Figure: the right (bar) panel of `fig:goalscope` = `readout_sources_and_head_budget.ipynb` cell 8 (`source_comparison`,
  goal-head bars per pivot + mean + baselines); the schematic and the composite are hand-made.

### 4.2 Readout sources: App. "Goal Heads vs. Other Readout Sources" (`Text/Appendix/readout_comparisons.tex`, `fig:heads-vs-residual`, `Figures/generated/sparsity_ablation/source_comparison.pdf`)

- Goal heads: section 4.1. Baselines (`scripts/evaluation/eval_readout_sources.py`; defaults `--source resid
  --source-layers 40 50 --alpha 0.1`, strategy `resid_norm`):
  - attention output L40-50: `job_list/goalscope_eval/goalscope_eval_a_attnout_40-50.json` (`--source attn --source-layers 40 50`)
  - residual L40-50: `job_list/goalscope_eval/goalscope_eval_a_residual_40-50.json`
  - residual L25-56: jobs `goalscope_a_residual_*` of `goalscope_eval_a_residual.json` (`--source-layers 25 56`)
- Results: `GS` with `resid_norm/0.1`, `heads=attn:L40-50:mean`, `heads=resid:L40-50:mean`, `heads=resid:L25-56:mean`.
- Cross-check (paper: 85.3 / 75.0 goal heads; 32.6 / 29.2 attention; residual ~1-3%): mean4 faith / caus = attn
  0.326 / 0.292, resid L40-50 0.012 / 0.006, resid L25-56 0.035 / 0.020.
- Generator: `readout_sources_and_head_budget.ipynb` cell 8 -> `notebooks/Figures/sparsity_ablation/source_comparison.pdf`.
  The PDF in the paper is an **older version** of this cell (4 bars: goal heads | attention L40-50 | residual L40-50 |
  residual L25-56); the current cell draws per-pivot goal-head bars + mean + the attention and residual L40-50 bars.
  Same data, different layout.

### 4.3 Head budget: App. "Readouts with different numbers of heads" (`readout_comparisons.tex`, `fig:sparsity` = `sparsity_sweep.pdf`, `fig:sparsity-other-pivots` = `sparsity_by_pivot.pdf`)

- Commands: `job_list/goalscope_eval/goalscope_eval_b_sparsity.json` (18 jobs = 9 head sets x 2 modes):
  `python -m scripts.evaluation.eval_goalscope --mode <m> --heads head_sets/Qwen2.5-Coder-32B-Instruct/goal_topk/<set>.json --strategy additive_norm --alpha 0.1`.
  The 122 point is the section 4.1 run.
- Results: `GS` with `additive_norm/0.1`, `heads=json:goal__{snap_top5,snap_top10,snap_top25,lam100_top50,lam100_top100,lam20_top150,lam20_top200,lam20_top250,lam5_top300}`.
- Cross-check mean4 faith / caus: 5 .363/.329, 10 .668/.571, 25 .750/.742, 50 .783/.808, 100 .840/.775, 122 .853/.750,
  150 .830/.775, 200 .855/.748, 250 .823/.700, 300 .849/.738.
- Generator: `readout_sources_and_head_budget.ipynb` cells 5 (`sparsity_sweep`) and 7 (`sparsity_by_pivot`), group
  `sparsity_ablation`.

### 4.4 Across-model comparisons: App. `app:model_axes` (`readout_comparisons.tex`)

Paper items: `tab:model-comparison-four-pivots` (`Text/Appendix/tables/model_comparison_preimplementation.tex`),
`fig:comparison-size` / `fig:comparison-code` / `fig:comparison-rl`
(`Figures/generated/model_generalization/{model_size,code_finetuning,rl_finetuning}_{faithfulness,causality}.pdf`),
`tab:olmo-readout-examples` (`Text/Appendix/tables/olmo_readout_examples.tex`).

| model | heads | strategy | job list | results leaf | mean4 faith / caus |
|---|---|---|---|---|---|
| Qwen2.5-Coder-32B-Instruct | goal (122) | replace α=1 | `goalscope_eval_c_model_axis.json` | `replace/1/heads=goal` | .789 / .735 |
| Qwen2.5-Coder-14B-Instruct | goal (88), `--steer_layer_range 20 30 --goal_band 0 48` | replace α=1 | `goalscope_eval_c_14b.json` | `replace/1/heads=goal__layers=20-30__...__band=0-48` | .519 / .305 |
| Qwen2.5-32B-Instruct (table) | goal (108, its own) | replace α=1 | `goalscope_eval_c_model_axis.json` | `replace/1/heads=goal` | .238 / .441 |
| Qwen2.5-32B-Instruct (code-ft figure) | goal (108, its own; Caution 6c) | additive_norm α=0.1 | `goalscope_eval_c_model_axis.json` (`*_additive_*` jobs) | `additive_norm/0.1/heads=goal` | .577 / .717 |
| Qwen2.5-Coder-32B (code-ft figure) | goal (122) | additive_norm α=0.1 | section 4.1 | section 4.1 | .853 / .750 |
| Olmo-3.1-32B-Instruct | `json:goal__lam20_thr1.0_band10-36` (93) | replace α=1 | `goalscope_eval_c_model_axis.json` (= `goalscope_eval_c_olmo_rematched.json`) | `replace/1/heads=json:goal__lam20_thr1.0_band10-36` (NOT the `__aprefill=5x68fdb5` sibling) | .435 / .622 |
| Olmo-3.1-32B-Instruct-SFT | `json:goal__lam20_thr0.6_band10-36` (91) | replace α=1 | `goalscope_eval_c_model_axis.json` | `replace/1/heads=json:goal__lam20_thr0.6_band10-36` | .529 / .452 |

All job lists are under `job_list/goalscope_eval/`. Paper prose checks: 14B -> 32B faithfulness 51.9% -> 78.9%.
- Generator for the three figures: `model_generalization.ipynb` (cell 1 `HSPEC` / `leaf()` / `AXES`, cell 2 saves
  `{axis}_{mode}`; cell 3 `model_comparison.tex` is unreferenced). **Glob hazard**: `leaf()` takes `glob(...)[0]` over
  `heads={spec}__*`, which for OLMo-Instruct faithfulness now also matches the later, unreported `__aprefill=5x68fdb5`
  leaf (mean4 .390). Delete / move that leaf or tighten the glob before regenerating the RL figure.
- The two tables were not made by a notebook on `clean`: they come from
  `agent_reports/chatgpt/appendix_review_2026_09_16/{build_results.py,edit_draft.py}` on master (pure parsers over the
  same leaves; `edit_draft.py` picks specs `int_python_003, int_python_001, list[int]_python_007, list[str]_python_000`
  at the `eq` pivot for the OLMo examples), with labels hand-edited afterwards.
- Localization of these models: section 3.2.

### 4.5 OVReader: App. "Tracking the Sub-Goal Encoding in Residual Representation" (`Text/Appendix/ovreader.tex`, `fig:ovreader` = `Figures/generated/ovreader/ov_reader_combined.pdf`, `fig:ov-by-pivot` = `ov_by_pivot_combined.pdf`)

- Commands: `job_list/diff_eval/ov_reader_locked.json` (8 jobs = 2 modes x 4 datatypes):
  `python -m scripts.evaluation.eval_ov_reader --mode {faithfulness|causality} --datatypes {int|list[int]|str|list[str]}`.
  Constants (no flags): model Coder-32B, `HEADS_JSON` = the 122 set, OVReader `additive_norm` α 0.15 (`OV_ALPHA`),
  GoalScope skyline α 0.1, steer layers [35,46), locked corpus, readout and rollout 64 tokens. Each job sweeps 19 probe
  layers {0,6,12,18,24,30,36,38,...,56,60,63} plus the `goalscope` skyline.
- Results: `results/evaluation/ov-reader/Qwen2.5-Coder-32B-Instruct/<layer|goalscope>/<mode>/heads=goal__lam20_thr0.9_band25-56__nheads=122__alpha=0.15__skyalpha=0.1__dtype=<dt>__lang=python__nc=2-3-4__maxnew=64__gen=64__span=frontier__seed=123/summary.json`
  (metric `by_pivot[<pivot>]["hard_yes"]`; parse with `scripts/evaluation/ov_reader_io.grid_df`).
- Cross-check (unweighted mean over the 4 datatype leaves, `eq` pivot, faith / caus): skyline .972 / .955, layer 48
  .948 / .875, layer 56 .853 / .836, layer 63 .671 / .660, layer 24 .079 / .062, layer 0 .000 / .000. The appendix
  prose quotes no number; the figures are the check.
- Generator: `ovreader.ipynb` cell 4 (`ov_reader_combined`: mean over pivots x datatypes + per-layer goal-head histogram
  from the 122 json) and cell 5 (`ov_by_pivot_combined`), group `ovreader`.
- Smoke: `--datatypes int --limit 2`. The OVReader leaf has no `nspec` field (a resume extends the same dir), so run
  smokes with a scratch `RESULTS_DIR`, never into `results/`.
- A commented-out paragraph in `ovreader.tex` describes an older, superseded cohort (rule-based spans); ignore it.

### 4.6 Goal-head ablations: §4 (`Text/04_heads.tex`; figure `tab:goal-ablation` in `Text/persistance.tex` = `Figures/ablation/ablation_results-crop.pdf`), App. "Sufficiency of Goal-Head Access" (`Text/Appendix/ablation_sufficiency.tex`) and "Necessity of Goal-Head Access" (`Text/Appendix/ablation_necessity.tex`)

- Sufficiency (spare only the goal heads; every other head in layers >= 20 is blinded to the span from 1 token before
  the frontier description through the last prompt token, for the prompt's last query row and every decode step;
  control = spare a random |goal|-size set from layers >= 32, seed 123):
  ```
  python -m scripts.evaluation.eval_ablation_sufficiency                                                        # 122
  python -m scripts.evaluation.eval_ablation_sufficiency --heads head_sets/Qwen2.5-Coder-32B-Instruct/goal_topk/goal__lam20_top250.json   # 250
  ```
  Defaults: `--model Qwen/Qwen2.5-Coder-32B-Instruct --from_layer 20 --seed 123`, heads = the 122 set; greedy 64 tokens;
  the first generated RHS line is judged (`judge_implements`, YES = success).
- Necessity (attention-sink ablation: the goal heads attend only to key 0 for query rows >= the pivot and every decode
  step; control = the same on random |goal| heads from layers >= 32):
  ```
  python -m scripts.evaluation.eval_ablation_necessity                                                          # 122
  python -m scripts.evaluation.eval_ablation_necessity --heads head_sets/Qwen2.5-Coder-32B-Instruct/goal_topk/goal__lam20_top250.json
  ```
  (no job lists; the reported runs were launched by hand, 2026-09-18). Shared machinery:
  `scripts/evaluation/ablation_common.py` (greedy decode on `mt.generate` with the KV cache; the sink / blind edit is
  `src.nrgs.functional.whitelist_head_attn` on `attention_scores` at every step; a generated token whose text ends
  in `"\n"` stops the decode and is kept, a stop token (`mt.eos_token_ids`, section 7) stops it and is dropped).
- Results: `results/evaluation/ablation/Qwen2.5-Coder-32B-Instruct/sufficiency/heads=spare_goal__goal{122,250}__fromL=20__strat=blind_span__dtype=all__nspec=253__seed=123__ctrl=later_half/`
  and `.../necessity/heads=goal__goal{122,250}__fromL=32__strat=sink__dtype=all__nspec=253__seed=123__ctrl=later_half/`
  (`fromL=32` in necessity = the control-pool floor, not an intervention layer). `summary.json["overall"]` =
  `{baseline, ablated, control}`.
- Cross-check (paper Fig. `tab:goal-ablation`, `tab:goal-ablation-exact`, `tab:goal-ablation-necessity`): clean 0.925;
  sufficiency spare-goal 0.806 (122) / 0.913 (250), spare-random 0.103 / 0.087; necessity sink-goal 0.854 / 0.158,
  sink-random 0.925 / 0.925.
- Other leaves in that directory (`strat=blind_task`, `strat=blind_task_and_progress`, and leaves without
  `__ctrl=later_half`, which used an older control pool) are unreported; the current necessity script only implements
  `sink`.
- Generators: `ablation.ipynb` cell 2 (`necessity.tex`) and cell 3 (`sufficiency.tex`), group `ablation`;
  `readout_sources_and_head_budget.ipynb` cell 9 (`ablation_results.pdf`) is the closest code version of the main
  figure. The paper's `ablation_results-crop.pdf`, `sufficiency-crop.pdf` and `necessity-crop.pdf` are hand-made
  (slide exports); `Text/Appendix/tables/sufficiency.tex` and the inline necessity table are hand-typed (values equal the
  summaries). `tab:sufficiency-qual` (qualitative examples) came from
  `notebooks/persistance/04_sufficiency_ablation_qualitative.ipynb` on master (hand-picked specs, control seed 0; not
  records of the n=253 run).
- Smoke: `--limit 12` (round-robin over datatypes; the control heads do not depend on `--limit`, so `baseline / ablated /
  control` generations of the same spec ids can be compared with the reported `records.jsonl`).

### 4.7 Temporal persistence: §3.4 (`Text/persistance.tex`, `fig:temporal-retention` = `Figures/generated/temporal_persistence/retention_curves_std-crop.pdf`), App. `app:persistence:temporal` (`Text/Appendix/persistence.tex`, `fig:retention-control` = `goal_vs_control_std_impl10-40-crop.pdf`, `tab:temporal-readout-examples`)

- Commands (no job list):
  ```
  python scripts/evaluation/temporal_persistence_eval.py --cuda 0                          # goal heads
  python scripts/evaluation/temporal_persistence_eval.py --cuda 0 --control --n-trials 64  # control
  ```
  Defaults: `--model Qwen/Qwen2.5-Coder-32B-Instruct --heads head_sets/<slug>/goal__lam20_thr0.9_band25-56.json
  --n-trials 256 --seed 1123`; the head set's key and `MAX_NEW = 96` (`src/goals/temporal_persistence.py`) are in
  every cache key: `TP.heads_key` is `"122"` (`HEAD_BUDGET`) for the set the paper's runs read, so their records stay
  hits, and `heads-<md5[:12]>` of the sorted set for any other (records carry it as `heads_key`); 4 datatypes x
  {python, cpp}; C++
  implementation spans are segmented by the Claude oracle (`oracle_value_segment`). `--n-trials 256` enumerates every
  distinct context per (datatype, language) cell (setups saturate at 240). Control heads: 122 random heads from
  layers [32, 64) disjoint from the goal heads, `CONTROL_SEED = 7` in the script (not the `head_sets/.../control/`
  file).
- Results: `results/temporal_persistence/Qwen2.5-Coder-32B-Instruct/cond_*.pt` (3088 goal records) and
  `.../control/cond_*.pt` (836 files on disk). Conditions already cached are skipped (use `--force` or a scratch
  `RESULTS_DIR`). `summary.json` in that dir is stale (a partial list[str]-only writer, n=893); the notebook recomputes
  from `cond_*.pt`.
- Generator: `temporal_persistence.ipynb` cell 3 (`retention_curves_std`, impl spans 10-40 tokens) and cell 2
  (`goal_vs_control_std_impl10-40`), group `temporal_persistence`. The notebook loads thousands of `.pt` files and has
  killed nbconvert kernels: run its code cells as a plain script (`jupyter nbconvert --to script`, then `python`) if the
  kernel dies.
- `tab:temporal-readout-examples` (value_range / has_palindrome rows) was produced by
  `notebooks/persistance/01_temporal_persistance.ipynb` on master (cells 9-10) and pasted by hand.
- Smoke: `--n-trials 2 --force` into a scratch `RESULTS_DIR`, then compare `pivot` / `traj` cosines with cached files
  of the same `ctx_key`.

### 4.8 Obstacle persistence: App. `app:persistence:obstacle` (`persistence.tex`, `fig:obstacle-datatype` = `Figures/generated/obstacle_robustness/obstacle_by_datatype.pdf`, `tab:obstacle-models` = `Text/Appendix/tables/generated/obstacle_adapted_by_level.tex`, `tab:obstacle-ov-boost`)

- Commands (one model per process; no job list; seed 0, 409 trials per model):
  ```
  python scripts/evaluation/obstacle_persistence_eval.py --cuda 0          # Qwen/Qwen2.5-Coder-32B-Instruct
  python scripts/evaluation/obstacle_persistence_eval.py --cuda 1 --model <id>
  # subset / smoke: --datatypes <dt ...> --predicates <pid ...> (per-trial caches as usual; aggregates go to
  # rows_subset.json / summary_subset.json, never overwriting the full run's)
  ```
  for `Qwen/Qwen2.5-Coder-14B-Instruct`, `Qwen/Qwen2.5-32B-Instruct`, `Qwen/Qwen2.5-72B-Instruct` (2 GPUs),
  `zai-org/GLM-4-32B-0414`, `allenai/Olmo-3.1-32B-Instruct`, `allenai/Olmo-3.1-32B-Instruct-SFT`. Constants:
  `N_PER = 4` prefixes per (predicate, level), levels aligned / easy / hard, `MAX_NEW = 48`; prefixes in
  `scripts/evaluation/obstacle_persistence_prefills.py`; judge `obstacle_persistence_judge.judge_adapts`
  (ADAPTED / SUCCUMBED / ESCAPED / BROKEN). No goal heads involved.
- Results: `results/obstacle_robustness/<slug>/{trial_<hash>.json, rows.json, summary.json}`
  (`summary["by_level"][level]["adapted"]`).
- Cross-check (`tab:obstacle-models`, aligned / easy / hard): Coder-32B .97 / .63 / .15; OLMo-Instruct .94 / .57 / .21;
  Qwen2.5-72B .94 / .57 / .12; GLM-4-32B .91 / .54 / .12; Qwen2.5-32B .97 / .66 / .12; Coder-14B .91 / .50 / .09;
  OLMo-SFT .89 / .35 / .08.
- OV boost (`tab:obstacle-ov-boost`): `python scripts/evaluation/obstacle_persistence_ov_boost_eval.py --cuda 0`
  (subset: `--datatypes ... --predicates ...`, aggregate to `results_subset.json`)
  (Coder-32B, **250 goal heads** `goal_topk/goal__lam20_top250.json`, numeric obstacle `" 0 + "` on list[str] /
  list[int], GoalScope readout additive_norm α 0.1 layers 35-45 at the `+` token, OV boost x2.0 of the goal heads'
  o_proj input at the last token of every forward of the no-cache greedy loop, `greedy_continue(...,
  interventions=boost_head_ov(...))`, so earlier positions are recomputed unboosted each step). Cache: `results/notebook_caches/obstacle_ov_boost/Qwen2.5-Coder-32B-Instruct/{base_*,cmp_*,results.json}`.
- Generator: `obstacle_persistence.ipynb` cell 2 (`obstacle_by_datatype`), cell 3 (`obstacle_adapted_by_level`),
  cell 4 (`obstacle_ov_boost.tex`), group `obstacle_robustness`. The OV-boost verdicts in the paper table were
  hand-edited after generation (Caution 6c).
- Smoke: `--force` with a scratch `RESULTS_DIR`; e.g. the reported trial
  `results/obstacle_robustness/Qwen2.5-Coder-32B-Instruct/trial_005a53ee8501b65b.json` (list[str] / count_with_digit /
  hard, prefix `" - 1 +"`, SUCCUMBED).

### NRGS interventions (shared facts for 4.9-4.13)

Paper: §5 (`Text/06_mechanism.tex`, which inputs `Text/Figures/kv_transfer.tex` and `Text/Figures/q-transfer.tex`),
App. `app:nrgs` (`Text/Appendix/task_transfer.tex`), App. `app:pre-com` / `app:task-vs-prog`
(`Text/Appendix/pre-commitment.tex`). Model Qwen2.5-Coder-32B-Instruct; `G250` =
`head_sets/Qwen2.5-Coder-32B-Instruct/goal_topk/goal__lam20_top250.json`, `C250` =
`head_sets/Qwen2.5-Coder-32B-Instruct/control/control250__laterhalf.json`. GoalScope readouts inside these scripts use
additive_norm α 0.1, layers 35-45, readout heads = G250. Task banks are inline (`src/nrgs/data.py`,
`src/nrgs/hier_data.py`); no `data/` file is read. `src/nrgs/functional.setup()` only loads the model. Every
intervention is a `src.nrgs.functional.Intervention` (`iv(mt, step)`, nnterp values edited at each `mt.generate` step,
or `interventions=` of a single-forward reader): the KV transfer calls each patched block's own attention function
(`self_attn.interface`) on `attention_queries` and on copies of `attention_keys` / `attention_values` with the span
replaced, and writes the chosen heads' last row of `attention_head_outputs` (other heads and the KV cache stay clean);
the Q patch writes `attention_queries_prerope` (the queries before the rotary embedding) on the prompt pass; the GSR
restriction masks `attention_scores`; the OV boost scales `attention_premix`. The task span ends at the user turn's
end-of-turn token (the first special token after the task list; `<|im_end|>` on Qwen). The transplanted SRC keys are
its `attention_keys_prerope` rotated for the DST positions by the block's own rotary embedding (`self_attn.rotate`),
so the call is the model's attention (its scale, mask and any softcap) over the keys the model would compute there
(any key norm included). Judges:
`src/nrgs/eval.py` (`judge_causal` scores a BOTH verdict as a miss; the stored `causal_trigger_rate` values were
rescored the same way, the old rule survives as `*_incl_both` fields). `R = results/diff/Qwen2.5-Coder-32B-Instruct`.
Language pairs: `--lang-transfers py-py py-cpp cpp-py cpp-cpp` (dirs `python-python`, `python-cpp`, ...).

### 4.9 KV transfer of the task span: §5.1.1 (`fig:kv-transfer` = `Figures/transfer/kv-transfer-main-crop.pdf`), App. `app:transfer:kv` (`fig:kv-barcharts` = `Figures/transfer/kv_barcharts-crop.pdf`, `tab:kv-grid`)

- Commands: `job_list/diff_eval/path_patch_32b_listgap.json` (16 jobs = 4 datatypes x 4 language pairs):
  `python scripts/nrgs/kv_transfer_eval.py --datatypes <dt> --lang-transfers <lt> --n-samples 32 --i3-v-scale 1 2 4`.
  Defaults: `--setup multi --span task --goal-heads G250 --readout-heads G250 --i3-v-scale 4 --n-samples 32 --seed 1123`
  (pivot-onwards KV transfer of SRC's task span = the list `(1)..(3)` + its closing newline up to the user-turn
  `<|im_end|>`; OV α 1; C++ library note in the system prompt). The job list adds the V x1 / x2 arms (used only in
  reports / the task-vs-progress figure).
- Random-head control ("1/512"): the 16 `ppctrl_*` multi jobs of `job_list/diff_eval/path_patch_control_32b_listgap.json`
  (`--goal-heads C250 --readout-heads C250 --no-readout`, V x4).
- Results: `R/path_patch/I1-off__I2-off__I3-on-V4__win-span__span-list+gap__cppnote-sys__a1__th-goal_250__gh-goal_250__ro-goal_250/<dt>__<src>-<dst>/{meta.json,results.json}`
  (`<dt>` in `int, str, list_int, list_str`; `meta.json` keys `causal_trigger_rate`, `readout_shift_rate`,
  `n_instances`); control `.../...__th-control250_250__gh-control250_250__ro-off/`. V x1 / x2 arms: `I3-on__` / `I3-on-V2__`.
  `__smoke` suffixed dirs are smoke runs.
- Cross-check: pooled n=512 causal 0.936, readout 0.900 (paper ~94% / ~90%); `int__python-python` 0.96875 / 0.90625;
  control 0.002 (1/512).
- Generator: `kv_transfer.ipynb` (`KV_RUN` knob) cell 7 (`kv_barcharts`), cell 11 (`kv_right_panel` = right panel of
  `kv-transfer-main`), cell 13 (`kv_grid.tex` -> `tab:kv-grid`), group `transfer`. The composite figure is hand-made.
- `job_list/diff_eval/path_patch_32b_listgap_taskprog.json` (`--span task+progress --i3-v-scale 1`, arm
  `I3-on__win-span+prog__...`) is a computed side arm; no paper figure or table reads it (as far as could be checked).

### 4.10 Sub-sub-goal KV transfer: App. `app:task_transfer-hier` (`fig:kv-multi` = `Figures/transfer/kv-sub-sub-goal-crop.pdf`)

- Commands: `job_list/diff_eval/path_patch_hier_32b_listgap.json` (16 jobs = {list[int], list[str]} x 4 language pairs
  x {and, or}): `python scripts/nrgs/kv_transfer_eval.py --setup hier --datatypes <dt> --lang-transfers <lt> --connectives <c> --orders match diff --n-samples 32 --i3-v-scale 1 2 4`.
  Control: the 8 `--setup hier` jobs of `path_patch_control_32b_listgap.json` (C250, `--no-readout`).
- Results: `R/path_patch_hier/I1-off__I2-off__I3-on-V4__...__th-goal_250__gh-goal_250__ro-goal_250/<dt>__<and|or>__<match|diff>__<src>-<dst>/meta.json`
  (32 cells); control `...__th-control250_250__gh-control250_250__ro-off/`.
- Cross-check: pooled n=1024 causal 0.792, readout 0.654 (paper ~79%); control 0/1024;
  `list_str__and__match__python-python` 0.96875 / 0.84375.
- Generator: `kv_transfer.ipynb` cell 9 (`kv_multihier_barcharts` = right panel); composite hand-made.

### 4.11 Broader goal, next op unspecified: App. `app:nrgs` (`fig:kv-goal` = `Figures/transfer/unspecified_next-crop.pdf`, `tab:kv-goal`)

**Not reproducible from `clean`.** The only source is `notebooks/diff/05.4_unspecified.ipynb` on master (12 algorithm
pairs x 2 directions x 4 language pairs = 96 instances; `span=name`, V x4, OV α 1, inline `PAIRS`, judge
`classify_concept`). No result cache is written; the numbers live in the notebook's saved outputs (cells 15-19:
SRC 74 / DST 22 -> 0.77; py->py 0.79, py->cpp 0.75, cpp->py 0.79, cpp->cpp 0.75; cell 19 writes `kv_goal_table.tex`).
It patches **230 heads** (G250 restricted to layers >= 30), while the paper says 250. The figure is hand-made.

### 4.12 Q transfer with gaze-select-restrict (GSR): §5.1.2 (`fig:q-transfer` = `Figures/transfer/q-transfer-main-crop.pdf`), App. `app:transfer:q` (`fig:q-transfer-schema` = `Figures/transfer/gsr_schema_clean.pdf`, `fig:q-barcharts` = `Figures/generated/transfer/q_barcharts_gsr-crop.pdf`, `tab:q-matched`, `tab:q-diff`, `tab:expb`, control prose)

- Commands (all `python scripts/nrgs/q_transfer_eval.py --datatypes <dt> --variants <v> --lang-transfers <lt> --tag-suffix __seq`):
  - goal heads, 2 variants x 4 datatypes x 4 language pairs: `job_list/diff_eval/diff_trigger_gsr_eq_crosslang.json`
    (27 jobs) + `diff_trigger_gaze_select_restrict_anc_eq.json` (8 py-py jobs; together they cover the 32 cells).
  - control: `diff_trigger_gsr_eq_control250.json` + `diff_trigger_gsr_eq_control250_rest.json`
    (`--patch-heads C250 --readout-heads C250 --no-readout`).
  - source frontier in the destination spec (`tab:expb`): `diff_trigger_gsr_eq_cindst_pypy.json`
    (`--variants c_in_dst --lang-transfers py-py --tag-suffix __seq__cindst`).
  Defaults: `--patch-heads G250 --readout-heads G250 --alphas 2 --n-samples 32 --seed 0`, pivot = the frontier's
  `{var} =`, letters naming, closed-loop gaze-select-restrict (no KV transfer). `--tag-suffix __seq` is required to
  hit the reported dirs. (The reported summaries do not record the seed; the job lists pass none, so the reported runs used
  the default at the time; not verified that it was 0.)
- Results: `R/diff_trigger/ph-goal_250__a2__piv-eq__cl-ppgazerestrict-anceq__seq/{order_matched,order_diff}/<dt>__<src>-<dst>/{summary.json,results.json}`
  (`<dt>` in `int, str, list(int), list(str)`; keys `causal_trigger_rate`, `readout_shift_rate`, `n`); control
  `R/diff_trigger/ph-control250_250__a2__piv-eq__cl-ppgazerestrict-anceq__seq/`; c_in_dst
  `R/diff_trigger/ph-goal_250__a2__piv-eq__cl-ppgazerestrict-anceq__seq__cindst/c_in_dst/<dt>__python-python/summary.json`
  (keys `joint_native_rate`, `joint_c_rate`, `joint_d_rate`, `joint_neither_rate`). The `...__kv__...` and
  `cl-ppgazerestrict__seq` (no `-anceq`) arms in that directory are unreported.
- Cross-check: pooled causal order-matched 0.871 (paper 87%), order-diff 0.572 (57%); readout 0.668 / 0.412; control
  0.043 / 0.029 (paper ~4%); `order_diff/int__python-python` 0.59375 / 0.53125. `tab:expb` (native / c / d / neither):
  int .031 / .969 / 0 / 0, str 0 / .938 / .062 / 0, list(int) 0 / .969 / 0 / .031, list(str) 0 / .969 / .031 / 0.
- Generators: `q_transfer.ipynb` (`GSR_ARM`) cell 5 (`q_crosslang_gsr` = right panel of `q-transfer-main`, order-diff),
  cell 7 (`q_barcharts_gsr`), cell 9 (`q_gsr_matched.tex`, `q_gsr_diff.tex`), group `transfer`. `tab:expb` is
  hand-typed from the `joint_*` rates. Schema: `python scripts/nrgs/q_transfer_example_attention.py --cuda 0`
  (defaults = the paper's instance: int, py-py, order_diff, key `dst=is_prime-is_positive-last_digit`) ->
  `R/gsr_example/order_diff__int__python-python__<hash>.json`, then
  `python notebooks/fig_n_table_generators/q_transfer_schema.py` (cairosvg + IBM Plex fonts) ->
  `notebooks/Figures/transfer/q_transfer_schema.{svg,pdf,png}` (= the paper's `gsr_schema_clean.pdf`, renamed).

### 4.13 Task span vs code-progress span: §5.2 (`fig:task-vs-prog` = `Figures/transfer/task-vs-progress-crop.pdf`), App. `app:pre-com` (`fig:pre-com-kv` = `Figures/generated/precommit_ablation/tvp_kv_grids.pdf`)

- Commands: `job_list/diff_eval/task_vs_progress_32b.json` and `task_vs_progress_32b_v2.json` (4 jobs each, one per
  datatype; v2 added configs and resumes from the cache): `python scripts/nrgs/task_vs_progress_eval.py --datatypes <dt>`.
  Defaults: py-py, `--n-samples 32 --seed 1123`, no readout, greedy 64; configs `src/nrgs/task_vs_progress.CONFIGS` =
  none + KV on {G250, all heads} x {task, progress, task+progress} x V {1, 2, 4} + residual x 3 spans. Another model:
  `--model <hf id> --goal-heads <json>` (that set is the `goal250` arm; the config names and summary keys keep the
  `goal250` label so the generator reads them unchanged).
- Results: `results/notebook_caches/task_vs_progress/Qwen2.5-Coder-32B-Instruct/<dt>__python-python__seed1123__mx64__ro0__task+gap{/,.summary.json}`
  (`<dt>` in `int, str, list_int, list_str`; summary keys like `"kv·goal250 · task · V×4"` -> `causal_shift`).
  Dirs without `__task+gap` are the superseded old-span runs.
- Cross-check (mean over 4 datatypes): residual task 0.891 / progress 0.062 / task+progress 0.984 (paper 0.89 / 0.06 /
  0.98); KV G250 task V x4 0.969 (the boxed cell), progress V x4 0.258.
- Generator: `task_vs_progress.ipynb` cell 5 (`tvp_residual_grid` = right panel of `task-vs-progress`) and cell 6
  (`tvp_kv_grids`), group `precommit_ablation`. Composite hand-made.

### 4.14 Mixing mechanisms: App. `app:nrgs` (`tab:q-mixing`: SetDiff / OrderID / CompleteCount / Induction)

**Not reproducible from `clean`.** It is an external, prefill-only gaze-winner readout (no generation, no judge), on
branch `add-q-transfer-alt` (`src/q_transfer_alt/`, `scripts/diff/q_transfer_alt_*.py`, its own environment under
`configs/q_transfer_alt/environment/`); numbers in `agent_reports/claude/analysis/query-transfer-report.md` on master.

### 4.15 Readouts without enumerated sub-tasks: App. `app:unspecified` (`Text/Appendix/unspecified.tex`, `fig:merge-sort` = `Figures/goalscope/qualitative/UNSPEC_merge_sort-crop.pdf`, `fig:stable-marriage` = `UNSPEC_stable_marriage-crop.pdf`)

- Stable marriage: `python scripts/run_goalscope_fig32_stable_marriage.py` (defaults: Coder-32B, the 122 heads,
  512-token greedy generation from the figure's exact prompt, then `apply_goalscope` additive_norm α 0.1 layers 35-45,
  64-token readouts, `include_noncode=True`) -> `results/goalscope_fig32/stable_marriage.json`, which records the
  comparison with the transcribed figure: the generated code differs (`generation_matches_figure=False`), the 4 marked
  readouts match to within a word. `--write-web` also writes the site's `ds_and_algo/stable_marriage.json`.
- Merge sort: produced by `notebooks/diff/13_goalscope_qualitative.ipynb` on master (user text `merge-sort`, assistant
  prefill `<code>\n def merge_sort(`, 256-token generation; renderer `src/goals/goalscope_viz.goalscope_qualitative_html`).
  That notebook's saved knobs are not the paper config (reset steer layers to (35,46) and α to 0.1). No script on
  `clean`; the site's `merge_sort` page uses a different prompt and does not match the figure. The figure's run is
  reproduced exactly with the prefill `<code>\ndef merge_sort` (no `(`; with `merge_sort(` the LM names the argument
  `ary`) + paper config: same tokens as `master:notebooks/diff/visualizations/merge-sort.html`, all 5 callouts word for
  word, 173/177 readouts identical to that HTML.
- Demo `notebooks/demos/02.1_unspecified.ipynb` runs both (all-token readouts, callouts pinned, no writes to `results/`).
- Both figures were cropped and annotated by hand.

### 4.16 Beyond code (ICL, EM): §6 (`Text/05_steerable.tex`, `fig:noncode-icl-em-main` = `Figures/appendix/goals-icl-em-main_updated-crop.pdf`), App. "non-code" (`Text/Appendix/non-code.tex`: `fig:noncode_fv_eval_Atoken`, `fig:noncode_composition_superposition`, `fig:noncode-refusal-steering-cake`, `fig:em_*`)

The reported ICL / EM numbers and figures come from co-authors' code that is **not in this repo** (36.2±5.6% over 39
single-relation tasks, 49.9 abstractive, 9.0 extractive, 21.0% composition / superposition; 18 composition + 5
superposition tasks; the "15.5% identifying the outlier" statistic; the refusal-steering "cake" figure; the four EM
figures). `scripts/icl/` and `scripts/em/` are **our reproductions** for the companion site, with slightly different
numbers:
- ICL: `python scripts/icl/icl_goalscope.py` (39 single-relation tasks from `data/FV_dataset/` plus only 3 rebuilt
  multi-rule tasks: `antonym-spanish`, `plus-ten-times-two`, `country-superposition`; 10-shot, 20 prompts per task,
  the 122 heads, additive_norm α 0.1, layers 35-45; readouts at the `A` of the last 3 demos, or at the last prompt token
  + 8 generated tokens for multi-rule tasks; probe system prompt and 32-token readout length are our choices) ->
  `results/icl_goalscope/<task>.json`; `python scripts/icl/icl_site_examples.py` -> `results/icl_goalscope/site/<task>.json`;
  judge `python scripts/icl/icl_judge.py <files...>` (Claude, verdicts written in place); site data
  `python scripts/icl/icl_write_site_data.py`. Earlier trace (not re-verified): the A-of-last-3-demos rate from the
  site files is abstractive 0.372, extractive 0.073, composition 0.225 vs the paper's .499 / .090 / .210; no committed
  script computes that rate.
- EM: `python scripts/em/em_goalscope_all_tokens.py` (innocent `Qwen/Qwen2.5-32B-Instruct` vs the LoRA
  `ModelOrganismsForEM/Qwen2.5-32B-Instruct_risky-financial-advice`, on one model loaded with the adapter attached
  (nnsight's `peft=`: peft's lora layers at the native paths) and switched off (`mt.disable_adapters()`, the base
  model bit for bit); `em_adapter` switches it on (`mt.enable_adapters()`) and off again, nothing merged: the head
  vectors read nnterp's `self_attn.output_weight(h)`, which is the adapted weight while the adapter is on; head
  vectors from each side's model, the innocent model verbalizes; Coder-32B's 122 heads; 80-token readouts; examples
  quickbuck / husband / ruler / gender) ->
  `results/em_goalscope_repro/all_tokens/<ex>.json`; `python scripts/em/em_tone_judge.py`;
  `python scripts/em/em_write_site_data.py`. The adapter is `--adapter` (a dir or a hub repo id), else `EM_ADAPTER`
  in `env.yml`, else the snapshot under `/disk/u/models/models--ModelOrganismsForEM--.../snapshots/`
  (`ADAPTER_GLOB`), else the hub repo. Only 2 of 4 paper examples reproduce qualitatively.

### 4.17 Companion site (not a paper result)

The site source is a separate repo checked out at `goals-web/` (`goals-web/site/`, gitignored here). Data writers in
`src/goals/site_data.py` write to `GOALS_WEB_SITE` (default `./goals-web/site`). Code-task pages:
`scripts/mine_humaneval_mbpp.py` -> `data/well-known-tasks/{humaneval,mbpp}/` (`ds_and_algo` hand-curated) ->
`scripts/generate_well_known_task_completions.py --model-key <id>` -> `scripts/run_goalscope_web.py --source-suffix=-v2`
(defaults = Coder-32B paper config) -> `cd goals-web && bash site/finalize_run.sh <slug>` -> `python site/build_site.py`
(in goals-web). Job lists: `job_list/goalscope_eval/goalscope_web_full_coder32b.json` (Coder-32B, one chained job) and
`llama70b_web_and_eval.json` (Llama-3.3-70B: lam20_ctx 110 heads, additive_norm α 0.11, steer layers 20-40; its two
`llama70b_eval_*` jobs write `GS` leaves for Llama that the paper does not report).

## 5. Results that are not produced (or not reproducible) by code in this repo

- Hand-composited / hand-drawn figures (only their data panels are generated): `figure1.pdf` (`fig:nrgs`, teaser),
  `goalscope-schema-eval-crop.pdf` (`fig:goalscope`; bar panel = 4.1), `head-loc-crop.pdf` (map = 3.1),
  `kv-transfer-main-crop.pdf` (4.9), `q-transfer-main-crop.pdf` (4.12), `kv-sub-sub-goal-crop.pdf` (4.10),
  `unspecified_next-crop.pdf` (4.11), `task-vs-progress-crop.pdf` (4.13), `Figures/ablation/{ablation_results,
  sufficiency,necessity}-crop.pdf` (4.6), `prompt-crop.pdf`, `readout_prompts.png`, `evaluation/faithfulness.png`,
  `UNSPEC_*-crop.pdf` (4.15), `goals-icl-em-main_updated-crop.pdf` and all `Figures/appendix/*` ICL / EM figures (4.16).
- Hand-typed tables: `tab:data-distribution` (`tables/corpus.tex`; `corpus_stats.ipynb` cell 2 writes an unreferenced
  cross-check table), `tab:goal-ablation-exact` (`tables/sufficiency.tex`), `tab:goal-ablation-necessity` (inline),
  `tab:sufficiency-qual`, `tab:temporal-readout-examples`, `tab:expb`, `tab:kv-goal`, `tab:q-mixing`; the two model
  tables of 4.4 came from an agent-report parser on master.
- Unspecified next-op table / figure (4.11): only `notebooks/diff/05.4_unspecified.ipynb` on master, 230 heads.
- Mixing-mechanisms table (4.14): branch `add-q-transfer-alt`, external package.
- ICL / EM evaluation (4.16) and the refusal-steering "cake" figure: co-authors' code, not in the repo.

## 6. Cautions

a. **OLMo-3.1-32B-Instruct goal heads were localized on Coder-32B's pivot instances.** The reported mask
   `results/localization/Olmo-3.1-32B-Instruct/lam20/head_mask.pt` (config: n_instances 1423, 50 epochs, batch 64,
   augment, i.e. the Coder-32B pool and defaults) was trained through `train_head_mask.py`'s `_FALLBACK_MODEL` branch:
   no OLMo pairs file existed then, so the script fell back to `data/head_loc/pairs/Qwen2.5-Coder-32B-Instruct.json`.
   Prompts, targets and activations were OLMo's own; only the instance selection is Coder's. The paper is unchanged.
   Re-running `job_list/head_localization/olmo32b_head_localization_pipeline.json` now trains on OLMo's own pool
   (`data/head_loc/pairs/Olmo-3.1-32B-Instruct.json`) and writes to `lam20_ownpairs`, so it does not overwrite (or
   reproduce) the reported mask. The reported OLMo head set is `head_sets/Olmo-3.1-32B-Instruct/goal__lam20_thr1.0_band10-36.json`.
b. **Pivot pools were re-mined after the reported masks were trained.** On 2026-10-03 (master dcf1e3d, contextual token
   filter `src/goals/pivot_token_filter.py`) the pairs files were re-mined. Current `data/head_loc/pairs/*.json` sizes vs
   the masks' `n_instances`: Coder-32B 1613 now vs 1423 trained (the pre-filter pool is
   `git show dcf1e3d~1:data/head_loc/pairs/Qwen2.5-Coder-32B-Instruct.json`); Coder-14B 1517 vs 1246; Qwen2.5-32B
   1711 vs 1181; OLMo-SFT 1185 vs 851; Llama-70B 1433 vs 1433 (lam20_ctx was trained on the filtered pool). Mining also
   calls the oracle, so it is not deterministic. A re-trained mask will be close to, not identical with, the reported
   one: **the head sets in `head_sets/` are the reference**; never regenerate them in place.
c. **Known paper-vs-code discrepancies** (the code is what produced the numbers):
   - Generation length: the paper says "up to 256 tokens"; the code uses 64 (GoalScope readout and frontier rollout,
     OVReader), 96 (temporal persistence) and 48 (obstacle persistence, OV boost).
   - The code-finetuning figure (`fig:comparison-code`) uses Qwen2.5-32B-Instruct's **own 108 heads**
     (`model_generalization.ipynb` `HSPEC` has no entry, so `heads=goal`); the text says "the same 122 heads". The
     122-head transfer runs exist (`GS` leaves `heads=json:coder32b_goal122`, additive_norm mean4 .766 / .725) but no
     figure reads them, and `head_sets/coder32b_goal122.json` is not on `clean` (same heads as the 122 set).
   - The obstacle OV-boost table uses the **250-head** set (caption silent) and its verdicts were hand-edited in the
     paper; `obstacle_persistence.ipynb` cell 4 prints the cached verdicts.
   - The unspecified next-op table used 230 heads (4.11); the paper says 250.
   - `source_comparison.pdf` in the paper is an older layout of the same data (4.2).
   - `model_generalization.ipynb`'s glob can pick the unreported OLMo `__aprefill` leaf (4.4).
d. **Temporal persistence**: default `--n-trials 256` enumerates all cached contexts (3088 goal conditions); the
   control run is `--control --n-trials 64`. Conditions are cached by key and skipped on re-run. Run the generator
   notebook as a script if its kernel dies (4.7).
e. **Site**: the site scripts write into a goals-web checkout (`GOALS_WEB_SITE`, default `./goals-web/site`), which is
   a separate git repo; commit there, not here.
f. **Judge rubric drift**: GoalScope / OVReader verdicts depend on the exact system prompts. If you edit
   `FAITH_HARD_SYS` / `CAUSAL_HARD_SYS`, `hard_rubric_tag` changes, the cache misses, and results no longer match the
   paper.
g. **Smoke runs into `results/`**: GoalScope and ablation smokes get their own `nspec=N` leaf, but NRGS, OVReader,
   persistence and obstacle scripts resume into the reported directories. Use a scratch `RESULTS_DIR` (edit `env.yml`)
   for any test run.
h. **`--cuda` overrides `CUDA_VISIBLE_DEVICES`**: scripts with a `--cuda` flag (temporal, obstacle, OV boost, OVReader,
   KV / Q transfer) set `CUDA_VISIBLE_DEVICES` themselves before importing torch, so an outer `CUDA_VISIBLE_DEVICES=7`
   is ignored when `--cuda` is passed. Pass the physical GPU id to `--cuda`.

## 7. Running on another model

Models load through nnterp, which gives every family it standardizes the same names and values
(`mt.layers[i].self_attn.attention_premix`, `layer_output`, `attention_scores`, ...), so every experiment runs on
another family with no code change, including the weights OVReader and the head vectors read (nnterp's per-head
`output_weight` / `value_weight` and the block's `attention_input_gain`). What does not carry over is what nnterp
does not standardize: the chat template, and the gain of a norm the attention's output passes before the stream,
without which OVReader refuses the block (footnote (a)).

**The check**: `python scripts/cross_family_check.py --model <hf id> [--heads random:60 --seed 0] [--n-tokens 16]
[--out <json>]` loads the model (bf16, eager, one GPU) and runs each experiment primitive once through the repo's
public functions, on a tiny budget and a seeded random head set from the middle third of the layers (or a
`[[layer, head], ...]` json), with the 32B's layer constants scaled by depth. It prints PRIMITIVE | OK/FAIL | reason.
Besides "it runs", it checks that attention rows sum to 1, that `get_next_token_probs` is the softmax of `mt.logits`,
that sunk heads put all their pivot-row mass on key 0 and blinded heads none on the blocked keys, and that a KV
transfer of the DST window onto itself leaves the logits bit-identical. It does not test whether a readout means the
same thing on another family.

**The matrix** (2026-11-03; nnterp 2.0.0rc2.dev54 at `200ac05`, nnsight dev at `e09ab1c8`, transformers 5.22.1; one
RTX A6000; the summaries are `test/cross_family/<slug>.json`):

| primitive | Qwen2.5-Coder-7B (qwen2) | Llama-3.1-8B (llama) | Olmo-3-7B (olmo3) | gemma-2-9b (gemma2) | Qwen3-8B (qwen3) | GLM-4-9B-0414 (glm4) | Phi-4-mini (phi3) |
|---|---|---|---|---|---|---|---|
| load + `support()`, GoalScope additive and replace, attention matrices, head vectors (padded batch), sink and blind-span ablation decode, OV boost, temporal `goal_encoding`, `get_hs` + next-token probs (d), head-mask gradient step (e) | OK | OK | OK | OK | OK | OK | OK |
| OVReader probe (a) | OK | OK | FAIL (a) | FAIL (a) | OK | FAIL (a) | OK |
| KV transfer (b) | OK | OK | OK | OK | OK | OK | OK |
| Q transfer (GSR) (c) | OK | OK | OK | OK | OK | OK | OK |

- (a) OVReader reads nnterp's per-head weights (`self_attn.output_weight(j)`, `value_weight(kv)`) and the block's
  `attention_input_gain`; on Phi-3 the value rows come out of the fused `qkv_proj`. Its map ends at `W_O`, so it
  refuses a block that norms the attention's output before the stream: OLMo-3 (post-norm), Gemma-2 and GLM-4 (a
  norm after the attention too). That norm's per-channel gain sets the direction a head writes, and nnterp does not
  serve it yet (`attention_output_gain`; `NNTERP_GAPS.md`, head weights 4). Before nnterp's accessors OVReader
  refused OLMo-3 and Phi-3, and ran on Gemma-2 and GLM-4 without that gain.
- (b) The KV transfer calls each patched block's own attention function (`self_attn.interface`) with SRC's keys
  before the rotary (`attention_keys_prerope`) rotated for the DST positions by the block's own rotary embedding
  (`self_attn.rotate`), so the self-transplant leaves the logits bit-identical on all seven: Gemma-2's attention logit
  softcap (50) is in the call, Qwen3's and OLMo-3's key norms come before `attention_keys_prerope`, OLMo-3's sliding
  and full blocks rotate by their own layer type, Phi-3's partial rotary is the family's own apply function.
- (c) The Q patch writes `attention_queries_prerope`, after any query norm, so the query the DST pivot attends with
  is the SRC pivot's, rotated for the DST pivot, bit for bit (checked at one head of the middle block on all seven).
  A write at `q_proj.output` misses it by up to 0.44 on OLMo-3, whose query norm spans all heads.
- (d) `get_next_token_probs` is the softmax of `mt.logits`, which includes any transform after the head (Gemma-2's
  final logit softcap, 30); a softmax of `lm_head.output` misses it by up to 6.8e-3 in probability on gemma-2-9b.
- (e) The head-mask step is `train_head_mask`'s forward and backward: `forward_with_src_replacement` assigns the
  mask-weighted `attention_premix` in one trace, through nnterp's names only.

Two Qwen-only assumptions are handled without per-family code: Gemma-2's chat template refuses a system turn, so
`src.tokens.chat_prompt` puts the system text at the start of the user turn (without it every primitive fails on
Gemma-2); the KV transfer's task span ends at the first special token after the task text, which is `<|im_end|>`
only on ChatML templates (Llama, GLM-4, Phi-3 and Gemma-2 have others).

What runs but means something else on some families:
- **Post-norm and sandwich blocks** (Gemma-2, OLMo-3, GLM-4): head vectors (`collect_head_steering_vecs`) are o_proj
  outputs, but the stream receives them through a norm after the attention (per-channel gain, shared `1/rms`), so
  their direction in the stream is the o_proj direction times that gain (OVReader refuses these blocks, footnote
  (a)). `attention_output` (the `--source attn` baseline) is that norm's output on these families.
- **Double BOS**: the Llama-3 and Gemma chat templates start with BOS and `prepare_input` adds another (Qwen has none).
  Left as is: the Llama-3.3-70B site runs were made this way.
- **Layer constants**: every script default (steer layers 35-45, OVReader probe layers, ablation from layer 20) is the
  32B's; pass scaled values for another depth. The NRGS scripts' GoalScope readouts (`goalscope_readout`, through
  `Ctx.readout_layers`) default to 35-46 scaled by depth (`round(L*35/64)`, `round(L*46/64)`, the paper's 35-46 on 64
  layers); `kv_transfer_eval.py` / `q_transfer_eval.py --readout-layers A B` sets them, a value other than the
  default tags the run `__rl<A>-<B>`. Both scripts take `--model`; any model but Coder-32B needs its own head sets.
- **Stop tokens**: every greedy decoder (`greedy_continue`, the GoalScope readouts, the NRGS / ablation decode driver,
  `batch_generate`, the ICL / EM / site scripts) stops at `mt.eos_token_ids`: the tokenizer's `eos_token_id` and the
  generation config's EOS list. On Qwen2.5 Instruct that is `<|im_end|>`, `<|endoftext|>`, and the tokenizer's EOS alone
  is already the turn end; on GLM-4.5 the tokenizer's EOS is `<|endoftext|>` while turns end at `<|user|>` (in the
  list), so the tokenizer's EOS alone runs past the turn; Olmo-3.1 Instruct ends its turns with its EOS
  `<|endoftext|>`; Olmo-3.1 SFT's list adds `<|im_end|>`. `GOALS_EOS=tokenizer` stops at the tokenizer's
  `eos_token_id` only (parity with the reported non-Qwen rows).
- **ICL / EM scripts** build their system turns themselves; as written they are Qwen-only.

**A head set for a new model**: every script takes `--heads <json>` (`[[layer, head], ...]`; `load_json_heads` checks
the ranges against the model). To localize goal heads: mine pivot instances (`scripts/mine_pivot_instances.py
--model-key <id>`; without them `train_head_mask.py` falls back to the Coder-32B pool, Caution 6a), train the mask
(`scripts/train_head_mask.py --model-key <id>`), add the model to `MODEL_BANDS` in `scripts/data/export_head_sets.py`
and to `GOAL_BANDS` in `src/goals/head_sets.py` (for `--heads goal`) with band `None`, export to `head_sets/<slug>/`
(the band comes from the mask by `src.goals.head_sets.band_rule`: the smallest contiguous layer window holding >= 85%
of the heads with mask > 0.9, the lowest of equally narrow ones; on the paper's masks Qwen2.5-32B [29, 56),
Llama-3.3-70B lam20_ctx [19, 44), Olmo-3.1 [9, 34), Coder-32B [32, 57); the export prints the window, its per-layer
counts and its size: pin them in both dicts), `--topk K --control` for the large set and its control, and run the
cross-family check with `--heads` on the new set.

## 8. Working conventions

### Reports

When generating analysis or reports, ask the user if they want to save it. If yes, save as `.log` or `.md` file in
`agent_reports/claude/` with an appropriate descriptive filename.

### Testing Guidelines

When implementing new features (evals, utils, classes, etc.):

1. **Ask about tests**: After implementing a feature, ask the user if they want a test script created.

2. **Test location**: Place test scripts in the `test/` directory with descriptive names.

3. **Test model**: Use a smaller LM for testing to save resources:
   - Default test model: `Qwen/Qwen2.5-Coder-14B-Instruct`
   - Initialize with: `ModelandTokenizer(model_key="Qwen/Qwen2.5-Coder-14B-Instruct", dtype=torch.bfloat16, device_map="auto", attn_implementation="eager")`

4. **Environment**: run tests with the uv env from the repo root, offline (each test's docstring has its exact
   command; with the `goals` conda env, `conda run -n goals python ...` instead):
   ```bash
   HF_HUB_OFFLINE=1 .venv/bin/python test/test_<feature>.py
   ```

5. **Ask before running tests**: Running tests consumes significant tokens (model loading, output parsing). Always
   ask the user before running test scripts. Only run tests autonomously if the user explicitly grants free rein.

### Design Decisions

- **Head mask parameterization**: `HeadMask` (`src/goals/head_loc_attn.py`) uses raw `nn.Parameter` initialized to
  all-ones, clamped to `[0, 1]` after each optimizer step (box-constrained GD, no sigmoid). Sparsity via
  `λ * mean(mask)`, scale-invariant to the number of layers tested.
- **Goal heads = exported json**: every reported run reads its heads from `head_sets/`; the 122-head json equals the
  Coder-32B lam20 mask thresholded at 0.9 inside `GOAL_BANDS` (`src/goals/head_sets.py`).
- **Batch KL loss**: `reduction="batchmean"` in `train_head_mask` keeps the loss scale independent of batch size.
- **Two mask-training paths**: `scripts/train_head_mask.py` defaults to the Coder-32B lam20 run (per-epoch
  augmentation, src head values recomputed per batch); `--no-augment` precomputes src head values once (the Coder-14B,
  Qwen2.5-32B, Llama-70B and OLMo-SFT masks).
- **Cross-language alignment**: source (C++) and destination (Python) prompts are padded to the same length (one batch
  for o_proj extraction), so both see identical absolute positions.
- **Left padding**: `ModelandTokenizer` left-pads via nnsight (raw Qwen tokenizers pad right).
- **Interventions are edits of nnterp values**: an edit writes a value nnterp serves (`layer_output`,
  `attention_premix`, `attention_scores`, `attention_head_outputs`, `attention_queries_prerope`) inside `mt.trace`, or
  at each step of `mt.generate` + `tracer.iter`, and attention recomputed with other keys and values is the block's own
  function (`self_attn.interface`); no replaced `forward`, no PyTorch hooks and no attention function written out. The
  head-mask training splice ASSIGNS `attention_premix` in the trace it backpropagates through (an in-place write fails
  at `backward()`). An `Intervention` keeps its edits in forward order (`src/nrgs/functional.py`). What nnterp could
  not express is in `NNTERP_GAPS.md`.
- **No per-family code**: model-specific facts come from nnterp (names, values, sizes, per-head weights,
  `mt.support()`) or from the checkpoint itself (chat template, special tokens); weights are read through nnterp's
  `output_weight` / `value_weight` / `attention_input_gain`, never as slices of a native `.weight`; section 7 lists
  where that does not reach.
- **BOTH verdicts**: NRGS judges score a BOTH verdict as a miss (no longer a trigger).

### Known Issues

- **`ask_claude` cache race**: concurrent threads judging the same text can collide on the cache's temp file
  (`FileNotFoundError` on rename); re-run the affected job (finished calls are cached).
