# CLAUDE.md

Guidance for agents working in this repository. README.md is the human-facing summary (project-page takeaways and
setup), so it is not repeated here. This file covers only the results presented in the **main text** of the paper (§3-§6), plus the task-vs-progress
experiment that §5.1.1 points to: for
each, the exact script, job list, configuration, results directory and figure generator. Appendix-only experiments, the
companion site and exploratory work are documented in the `CLAUDE.md` of branch `dev`.

## 1. Orientation

Code for the paper *Goal Encoding in LLMs*. When an LM writes code for a task with several sub-tasks, does it encode
*which sub-task (instrumental goal) it implements next*, before it writes it? A sparse set of attention heads
(**goal heads**) carries that encoding. GoalScope verbalizes it, ablations show the heads are a causal bottleneck,
and NRGS interventions (KV / Q transfer) show how the LM re-reads the goal from the task span.

```
Broader goal:      implement merge sort
Instrumental goal: merge the two sorted halves          <- what the goal heads encode
Implementation:    two-pointer while loop  vs  heapq.merge(left, right)  vs  std::merge(...) in C++
```
The encoding should not depend on the implementation strategy. A direction that only predicts success may just capture
competence; that is the central confound to avoid.

Branches: `master` (= `clean`) keeps the code behind the paper's reported results (main text and appendix) plus the
teaching demos in `notebooks/demos/`; `dev` holds the full research history (Track-1 prefill probing / MCTS, clock /
boundary heads, NRGS masks, exploratory and dev notebooks, unreported arms) and documents every experiment. Paper source:
`Overleaf-draft/Goals-Overleaf/` (`main.tex` -> `Text/*.tex`, `Text/201_Appendix.tex` -> `Text/Appendix/*.tex`).
Appendix files that `201_Appendix.tex` does not `\input` (`boundary`, `cpp`, `dissociation`, `gaze`, `mlp`, `olmo`,
`mechanistic_details`, `oracle_prompts`) and `Text/Figures/{goal_eval,sparsity_ablation}.tex` are not in the paper.

### Layout

```
src/models.py            ModelandTokenizer (nnsight LanguageModel; Qwen2, Llama, OLMo-3, GLM-4)
src/functional.py        generate_with_patch (baukit), get_hs, ...
src/tokens.py            prepare_input (left padding), find_token_range
src/utils/               env_utils (paths), oracle_llms (ask_claude + cache), logging, experiment_utils
src/goals/               head localization (head_loc_*), head_sets.py, goalscope.py (+ OVReader),
                         goalscope_viz.py, temporal_persistence.py, site_data.py, proactive_strategy.py
src/nrgs/                data.py (QTASKS bank, multi-predicate setups), hier_data.py, unspecified_data.py, functional.py (setup, hooks),
                         kv_transfer.py, task_vs_progress.py, eval.py (judges)
scripts/                 mine_pivot_instances.py, train_head_mask.py, data/, evaluation/, nrgs/, icl/, em/,
                         site scripts (generate_well_known_task_completions.py, mine_humaneval_mbpp.py,
                         run_goalscope_web.py, run_goalscope_fig32_stable_marriage.py)
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
  `python scripts/x/y.py` work. Use the `goals` conda env (`bash setup_env.sh`; `conda run -n goals python ...`).
- **Paths**: `env.yml` in the root (template `env_demo.yml`) holds `MODEL_DIR`, `RESULTS_DIR` (default `results`),
  `DATA_DIR`, `CLAUDE_CACHE_DIR` (default `data/LLM_CACHE/claude_cache`), optional `HF_CACHE`, and `CLAUDE_KEY`.
  `src/utils/env_utils.py` reads them (`DEFAULT_RESULTS_DIR`, `PROJECT_ROOT`). All results paths below are relative to
  `RESULTS_DIR` = `results/`.
- **Results layout**: `<experiment>/<model>/<config>/<leaf>` under `results/`: `localization/`, `goalscope/` (judge
  side-runs inside the leaf: `<leaf>/rejudge=<judge>/`, `<leaf>/rationale=<judge>/`), `ov_reader/`, `ablation/`,
  `temporal_persistence/<m>/{goal,control}/`, `obstacle_persistence/<m>/{trials,ov_boost}/`,
  `kv_transfer/<m>/{task_span,sub_sub_goal,broader_goal}/`, `q_transfer/<m>/` (+ `gsr_example/`), `task_vs_progress/<m>/`,
  `unspecified_readouts/`, `fv_baseline/`, `human_eval/`, `external/`. Every eval script writes, and every generator reads,
  this layout.
- **Script defaults = the paper's configuration.** Every eval script's argparse defaults reproduce the main reported
  run; job lists only add the per-datatype / per-mode / per-language fan-out and the non-default arms (e.g.
  control head sets, `--tag-suffix __seq`). Smoke runs: `--limit N` (GoalScope /
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
  Qwen2.5-72B need 2 GPUs (`device_map="auto"`). Most scripts take `--cuda 0` or `--cuda 0,1`.
  GoalScope steering is step-by-step (CPU-launch-bound) and the judged evals are oracle-bound (thousands of serial
  `ask_claude` calls per shard); plan hours per GoalScope job, not minutes.
- **Figure groups** are `notebooks/Figures/<group>/` (NRGS groups per model:
  `notebooks/Figures/<model>/{transfer,precommit_ablation}/`; the paper's causality figures:
  `notebooks/Figures/causality_auto/fig<NN>_*.pdf`).
- **Generators** (`notebooks/fig_n_table_generators/*.ipynb`) only parse results (no model). They save through
  `figstyle.save_fig` / `save_table` to `notebooks/Figures/<group>/<name>.{pdf,png,tex}`. Copying into
  `Overleaf-draft/Goals-Overleaf/Figures/generated/<group>/` (and `Text/Appendix/tables/generated/`) and `pdfcrop`
  (`*-crop.pdf`) were done by hand. Cell numbers below are 0-based code+markdown cell indices.

## 2. Reproduce from scratch: dependency order

Everything downstream of step 4 reads its heads from `head_sets/` (tracked), so in practice you can start at step 5
with the shipped head sets. Steps 2-4 are needed only to re-derive the heads.

1. **Corpus** (no model, no oracle, deterministic):
   `python scripts/data/gen_goalscope_eval.py` -> `data/goalscope_eval.json` (seed 123, target 64 per datatype;
   253 specs = int 65 / list[int] 60 / str 64 / list[str] 64; task bank `src/nrgs/data.py::QTASKS`). Check: write to a temp path
   (`--out /tmp/x.json`) and diff `datapoints` against the tracked file (ignore `meta`). The NRGS scripts do not read this file (their banks are
   inline in `src/nrgs/data.py` / `hier_data.py`).
2. **Pivot mining** (model + Claude oracle): `python -m scripts.mine_pivot_instances --model-key <hf id>` ->
   `data/head_loc/pairs/<slug>.json`, from `data/head_loc/predicates/*.json` (25 predicates, Python + C++) and
   `data/head_loc/global_pairs.json` (oracle-verified pairs). Oracle prompts live in `src/goals/head_loc_oracle.py`;
   the contextual next-token filter is `src/goals/pivot_token_filter.py`.
3. **Head-mask training**: `python -m scripts.train_head_mask --model-key <hf id> [...]` ->
   `results/localization/<slug>/<tag>/head_mask.pt` (+ `checkpoints/` every 5 epochs). Defaults
   (`--n-epochs 50 --lambda-sparse 20 --batch-size 64 --augment --tag lam20`, lr 1e-2, augment seed 123456, all layers)
   = the Coder-32B lam20 run.
4. **Head-set export** (no model): `python scripts/data/export_head_sets.py` -> `head_sets/<slug>/goal__lam20_thr0.9_band<a>-<b>.json`
   for the models in its `MODEL_BANDS` (Coder-32B 25-56, Qwen2.5-32B 25-56, Coder-14B 0-48, OLMo Instruct / SFT 10-36);
   threshold 0.9 on the lam20 mask.
5. **Everything that reads `head_sets/`**: GoalScope (4.1), ablations (4.2), temporal persistence (4.3), NRGS
   interventions (4.4-4.6).
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
  the 122-head json).
- Smoke: `python -m scripts.train_head_mask --tag scratch_smoke --n-epochs 1` (exercises the pipeline).

### 3.2 Head sets used by the main text

| file | heads | used by | how it was made |
|---|---|---|---|
| `Qwen2.5-Coder-32B-Instruct/goal__lam20_thr0.9_band25-56.json` | 122 | GoalScope (4.1), ablations (122), temporal persistence (4.3), ICL / EM | `export_head_sets.py` (lam20 mask > 0.9, layers [25,56)) |
| `Qwen2.5-Coder-32B-Instruct/goal_topk/goal__lam20_top250.json` | 250 | ablations (250), all NRGS runs (4.4-4.6) | top-250 by mask value of the lam20 mask over all 64 layers (ties by index). Contains all 122 goal heads and 14 heads below layer 20 |
| `Qwen2.5-Coder-32B-Instruct/control/control250__laterhalf.json` | 250 | NRGS random-head controls | random non-goal heads from layers >= 32, seed 123 |

The other files in `head_sets/` back appendix results (see `dev`'s `CLAUDE.md`). `--heads goal` in `eval_goalscope.py`
resolves through `src/goals/head_sets.load_goal_heads` (the exported json if present, else the mask thresholded at 0.9
in `GOAL_BANDS[model]` or `--goal_band`).

## 4. Main-text results

Notation: `GS = results/goalscope/<model>/<mode>/<strategy>/<alpha>/<leaf>/` with
`<leaf> = heads=<name>__layers=35-46__nsteer=all__maxnew=64__span=frontier__dtype=all__nspec=253__nc=2-3-4__seed=123[__band=a-b][__stub=func]`
(`scripts/evaluation/faithfulness_io.run_leaf_name`; `<name>` = `goal` for `--heads goal`, `json:<stem>` for a json,
`band=` only when `--goal_band` is passed, `stub=func` on every causality leaf run with the v2 probe = the reported one). Each leaf holds `config.json`, `records.jsonl` (253 specs x 5 pivots),
`samples.jsonl`, `summary.json`. Headline metric = `summary.json["by_anchor"][pivot]["hard_yes"]` for the four pivots
`prev_end, indent, var, eq` (the mean over these four is "mean4"; `first_impl` is never reported). Layer ranges in
directory names are half-open (`layers=35-46` = layers 35-45, `L40-50` = 40-49).

### 4.1 GoalScope readouts: §3.2-3.3 (`Text/03_what.tex`), `fig:goalscope` (`Text/Figures/goalscope.tex`, `Figures/goalscope/goalscope-schema-eval-crop.pdf`)

- Commands (`job_list/goalscope_eval/goalscope_eval_a_residual.json`, jobs `goalscope_a_goalheads_{faithfulness,causality}`):
  ```
  python -m scripts.evaluation.eval_goalscope --mode faithfulness      # all other flags = defaults
  python -m scripts.evaluation.eval_goalscope --mode causality
  ```
  Defaults: `--model Qwen/Qwen2.5-Coder-32B-Instruct --heads goal (122) --strategy additive_norm --alpha 0.1
  --steer_layer_range 35 46 --seed 123`; readout 64 tokens (`MAX_NEW_TOKENS`), frontier rollout 64 tokens
  (`GEN_MAX_NEW_TOKENS`), corpus `data/goalscope_eval.json`, judge `claude-sonnet-4-6`.
- Causality probe (`--causal_stub v2`, the default; `src/goals/goalscope.CAUSAL_STUBS`): user turn "You are implementing a
  C++ function. Just write the function body, without any comments or explanations. Assume all the necessary headers
  are already included.", assistant prefill `auto func (auto value){`. `--causal_stub v1` = the earlier
  `bool check(auto value){` stub (old leaves without `__stub=func`; not reported any more).
- Results: `GS` with model Coder-32B, `additive_norm/0.1`, `heads=goal` (causality: `heads=goal__...__stub=func`).
- Cross-check (matches paper "~60%", "0.90+", "96%", "~87%"): faithfulness 0.589 / 0.945 / 0.905 / 0.972 (mean4 0.853);
  causality 0.672 / 0.957 / 0.889 / 0.972 (mean4 0.873). Smoke: `--limit 4` (reproduces the first 4 specs' readouts and
  verdicts exactly, from the judge cache).
- Judge rubric: `FAITH_HARD_SYS` / `CAUSAL_HARD_SYS` in `scripts/evaluation/eval_goalscope.py`; `config.json` stores
  `hard_rubric_tag` = md5[:8] of the system prompt (reported: faithfulness `78bfd5f4`, causality `9c106246`). If the tag
  of the current code differs, the rubric changed and the verdicts will not match the paper. 
- Figure: the right (bar) panel of `fig:goalscope` = `causality_auto.ipynb` (`fig02_fig12_source_comparison`; goal-head
  bars per pivot + mean, plus the L40-50 attention / residual baselines).

### 4.2 Goal-head ablations: §4 (`Text/04_heads.tex`; figure `tab:goal-ablation` in `Text/persistance.tex` = `Figures/ablation/ablation_results-crop.pdf`)

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
  ``` Shared machinery:
  `scripts/evaluation/ablation_common.py`.
- Results: `results/ablation/Qwen2.5-Coder-32B-Instruct/sufficiency/heads=spare_goal__goal{122,250}__fromL=20__strat=blind_span__dtype=all__nspec=253__seed=123__ctrl=later_half/`
  and `.../necessity/heads=goal__goal{122,250}__fromL=32__strat=sink__dtype=all__nspec=253__seed=123__ctrl=later_half/`
  (`fromL=32` in necessity = the control-pool floor, not an intervention layer). `summary.json["overall"]` =
  `{baseline, ablated, control}`.
- Cross-check (`tab:goal-ablation`): clean 0.925;
  sufficiency spare-goal 0.806 (122) / 0.913 (250), spare-random 0.103 / 0.087; necessity sink-goal 0.854 / 0.158,
  sink-random 0.925 / 0.925.
- Generator: `readout_sources_and_head_budget.ipynb` cell 9 (`ablation_results.pdf`).
- Smoke: `--limit 12` (round-robin over datatypes).

### 4.3 Temporal persistence: §3.4 (`Text/persistance.tex`, `fig:temporal-retention` = `Figures/generated/temporal_persistence/retention_curves_std-crop.pdf`)

- Commands (no job list):
  ```
  python scripts/evaluation/temporal_persistence_eval.py --cuda 0                          # goal heads
  python scripts/evaluation/temporal_persistence_eval.py --cuda 0 --control --n-trials 64  # control
  ```
  Defaults: `--model Qwen/Qwen2.5-Coder-32B-Instruct --n-trials 256 --seed 1123`; `HEAD_BUDGET = 122` and
  `MAX_NEW = 96` (`src/goals/temporal_persistence.py`, both in every cache key); 4 datatypes x {python, cpp}; C++
  implementation spans are segmented by the Claude oracle (`oracle_value_segment`). `--n-trials 256` enumerates every
  distinct context per (datatype, language) cell (setups saturate at 240).
- Results: `results/temporal_persistence/Qwen2.5-Coder-32B-Instruct/goal/cond_*.pt` (3088 goal records) and
  `.../control/cond_*.pt` (836 files). Conditions already cached are skipped (use `--force` or a scratch
  `RESULTS_DIR`). The notebook computes everything from `cond_*.pt`.
- Generator: `temporal_persistence.ipynb` cell 3 (`retention_curves_std`, impl spans 10-40 tokens), group
  `temporal_persistence`.
- Smoke: `--n-trials 2 --force` into a scratch `RESULTS_DIR`.

### NRGS interventions (shared facts for 4.4-4.6)

Paper: §5 (`Text/06_mechanism.tex`, which inputs `Text/Figures/kv_transfer.tex` and `Text/Figures/q-transfer.tex`), and App. J (`Text/Appendix/pre-commitment.tex`) for 4.6. Model Qwen2.5-Coder-32B-Instruct; `G250` =
`head_sets/Qwen2.5-Coder-32B-Instruct/goal_topk/goal__lam20_top250.json`, `C250` =
`head_sets/Qwen2.5-Coder-32B-Instruct/control/control250__laterhalf.json`. GoalScope readouts inside these scripts use
additive_norm α 0.1, layers 35-45, readout heads = G250. Task banks are inline (`src/nrgs/data.py`,
`src/nrgs/hier_data.py`); no `data/` file is read. `src/nrgs/functional.setup()` only loads the model. Judges:
`src/nrgs/eval.py` (`judge_causal` scores a BOTH verdict as a miss; the stored `causal_trigger_rate` values were
rescored the same way, the old rule survives as `*_incl_both` fields). `KV = results/kv_transfer/Qwen2.5-Coder-32B-Instruct`, `Q = results/q_transfer/Qwen2.5-Coder-32B-Instruct`.
Language pairs: `--lang-transfers py-py py-cpp cpp-py cpp-cpp` (dirs `python-python`, `python-cpp`, ...).

### 4.4 KV transfer of the task span: §5.1.1 (`fig:kv-transfer` = `Figures/transfer/kv-transfer-main-crop.pdf`)

- Commands: `job_list/diff_eval/path_patch_32b_listgap.json` (16 jobs = 4 datatypes x 4 language pairs):
  `python scripts/nrgs/kv_transfer_eval.py --datatypes <dt> --lang-transfers <lt> --n-samples 32 --i3-v-scale 1 2 4`.
  Defaults: `--setup multi --span task --goal-heads G250 --readout-heads G250 --i3-v-scale 4 --n-samples 32 --seed 1123`
  (pivot-onwards KV transfer of SRC's task span = the list `(1)..(3)` + its closing newline up to the user-turn
  `<|im_end|>`; OV α 1; C++ library note in the system prompt). The job list adds the V x1 / x2 arms (appendix only).
- Random-head control ("1/512"): the 16 `ppctrl_*` multi jobs of `job_list/diff_eval/path_patch_control_32b_listgap.json`
  (`--goal-heads C250 --readout-heads C250 --no-readout`, V x4).
- Results: `KV/task_span/I1-off__I2-off__I3-on-V4__win-span__span-list+gap__cppnote-sys__a1__th-goal_250__gh-goal_250__ro-goal_250/<dt>__<src>-<dst>/{meta.json,results.json}`
  (`<dt>` in `int, str, list_int, list_str`; `meta.json` keys `causal_trigger_rate`, `readout_shift_rate`,
  `n_instances`); control `.../...__th-control250_250__gh-control250_250__ro-off/`. V x1 / x2 arms: `I3-on__` / `I3-on-V2__`.
- Cross-check: pooled n=512 causal 0.936, readout 0.900 (paper ~94% / ~90%); `int__python-python` 0.96875 / 0.90625;
  control 0.002 (1/512).
- Generator: `kv_transfer.ipynb` (`KV_RUN` knob) cell 11 (`kv_right_panel` = right panel of `kv-transfer-main`), group
  `<model>/transfer`.

### 4.5 Q transfer with gaze-select-restrict (GSR): §5.1.2 (`fig:q-transfer` = `Figures/transfer/q-transfer-main-crop.pdf`, random-head control "~4%")

- Commands (all `python scripts/nrgs/q_transfer_eval.py --datatypes <dt> --variants <v> --lang-transfers <lt> --tag-suffix __seq`):
  - goal heads, 2 variants x 4 datatypes x 4 language pairs: `job_list/diff_eval/diff_trigger_gsr_eq_crosslang.json`
    (27 jobs) + `diff_trigger_gaze_select_restrict_anc_eq.json` (8 py-py jobs; together they cover the 32 cells).
  - control: `diff_trigger_gsr_eq_control250.json` + `diff_trigger_gsr_eq_control250_rest.json`
    (`--patch-heads C250 --readout-heads C250 --no-readout`).
  Defaults: `--patch-heads G250 --readout-heads G250 --alphas 2 --n-samples 32 --seed 0`, pivot = the frontier's
  `{var} =`, letters naming, closed-loop gaze-select-restrict (no KV transfer). `--tag-suffix __seq` names the reported dirs.
- Results: `Q/ph-goal_250__a2__piv-eq__cl-ppgazerestrict-anceq__seq/{order_matched,order_diff}/<dt>__<src>-<dst>/{summary.json,results.json}`
  (`<dt>` in `int, str, list(int), list(str)`; keys `causal_trigger_rate`, `readout_shift_rate`, `n`); control
  `Q/ph-control250_250__a2__piv-eq__cl-ppgazerestrict-anceq__seq/`.
- Cross-check: pooled causal order-matched 0.871 (paper 87%), order-diff 0.572 (57%); readout 0.668 / 0.412; control
  0.043 / 0.029 (paper ~4%); `order_diff/int__python-python` 0.59375 / 0.53125.
- Generators: `q_transfer.ipynb` (`GSR_ARM`) cell 5 (`q_crosslang_gsr` = right panel of `q-transfer-main`, order-diff),
  group `<model>/transfer`. The left (schema) panel: `python scripts/nrgs/q_transfer_example_attention.py --cuda 0`
  (defaults = the paper's instance: int, py-py, order_diff, key `dst=is_prime-is_positive-last_digit`) ->
  `Q/gsr_example/order_diff__int__python-python__<hash>.json`, then
  `python notebooks/fig_n_table_generators/q_transfer_schema.py` (cairosvg + IBM Plex fonts) ->
  `notebooks/Figures/Qwen2.5-Coder-32B-Instruct/transfer/q_transfer_schema.{svg,pdf,png}` (= the paper's `gsr_schema_clean.pdf`, renamed).

### 4.6 Task span vs code-progress span: pointed to from §5.1.1 (App. `app:task-vs-prog` / `app:pre-com`, `Text/Appendix/pre-commitment.tex`; `fig:task-vs-prog` right panel, `fig:pre-com-kv`)

- Commands: `job_list/diff_eval/task_vs_progress_32b.json` and `task_vs_progress_32b_v2.json` (4 jobs each, one per
  datatype; v2 added configs and resumes from the cache): `python scripts/nrgs/task_vs_progress_eval.py --datatypes <dt>`.
  Defaults: py-py, `--n-samples 32 --seed 1123`, no readout, greedy 64; configs `src/nrgs/task_vs_progress.CONFIGS` =
  none + KV on {G250, all heads} x {task, progress, task+progress} x V {1, 2, 4} + residual x 3 spans.
- Results: `results/task_vs_progress/Qwen2.5-Coder-32B-Instruct/<dt>__python-python__seed1123__mx64__ro0__task+gap{/,.summary.json}`
  (`<dt>` in `int, str, list_int, list_str`; summary keys like `"kv·goal250 · task · V×4"` -> `causal_shift`).
- Cross-check (mean over 4 datatypes): residual task 0.891 / progress 0.062 / task+progress 0.984 (paper 0.89 / 0.06 /
  0.98); KV G250 task V x4 0.969 (the boxed cell), progress V x4 0.258.
- Generator: `task_vs_progress.ipynb` cell 5 (`tvp_residual_grid` = right panel of `task-vs-progress`) and cell 6
  (`tvp_kv_grids`), group `<model>/precommit_ablation`.

### 4.7 Beyond code (ICL, EM): §6 (`Text/05_steerable.tex`, `fig:noncode-icl-em-main`)

The ICL and EM results are reproduced on their own branches: ICL on `icl-eval`, emergent misalignment on `em`.

## 5. Cautions

a. **Judge rubric drift**: GoalScope verdicts depend on the exact system prompts. If you edit
   `FAITH_HARD_SYS` / `CAUSAL_HARD_SYS`, `hard_rubric_tag` changes, the cache misses, and results no longer match the
   paper.
b. **Resumable caches**: GoalScope and ablation smokes get their own `nspec=N` leaf, but the NRGS and
   temporal-persistence scripts skip conditions that are already cached in `RESULTS_DIR` and resume into the full
   run's directories. Use a scratch `RESULTS_DIR` (edit `env.yml`) for test runs, or `--force` where available.
c. **`--cuda` overrides `CUDA_VISIBLE_DEVICES`**: scripts with a `--cuda` flag (temporal persistence, KV / Q transfer)
   set `CUDA_VISIBLE_DEVICES` themselves before importing torch, so an outer `CUDA_VISIBLE_DEVICES=7` is ignored when
   `--cuda` is passed. Pass the physical GPU id to `--cuda`.
d. **Trivial causality read-outs**: `eval_goalscope.is_trivial_causal_readout` scores a read-out as NO without the judge
   when it is a bare sign/zero test or exactly the LM's own uninjected completion of the probe
   (`_UNINJECTED_CAUSAL_DEFAULTS`: Coder-32B + v2 stub -> `return value * 2; }`). Measure the default before adding a
   model or a stub (greedy, no injection).

## 6. Working conventions

### Reports

When generating analysis or reports, ask the user if they want to save it. If yes, save as `.log` or `.md` file in
`agent_reports/claude/` with an appropriate descriptive filename.

### Testing Guidelines

When implementing new features (evals, utils, classes, etc.):

1. **Ask about tests**: After implementing a feature, ask the user if they want a test script created.

2. **Test location**: Place test scripts in the `test/` directory with descriptive names.

3. **Test model** (smoke tests only, i.e. checking that new code runs; reproductions use the paper's models): use a smaller LM
   to save resources:
   - Default test model: `Qwen/Qwen2.5-Coder-14B-Instruct`
   - Initialize with: `ModelandTokenizer(model_key="Qwen/Qwen2.5-Coder-14B-Instruct", dtype=torch.bfloat16, device_map="auto", attn_implementation="eager")`

4. **Conda environment**: Always run tests in the `goals` conda environment:
   ```bash
   conda run -n goals python test/test_<feature>.py
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
  augmentation, src head values recomputed per batch); `--no-augment` precomputes src head values once (used for the other models' masks).
- **Cross-language alignment**: source (C++) and destination (Python) prompts are padded to the same length (one batch
  for o_proj extraction), so both see identical absolute positions.
- **Left padding**: `ModelandTokenizer` left-pads via nnsight (raw Qwen tokenizers pad right).
- **BOTH verdicts**: NRGS judges score a BOTH verdict as a miss (no longer a trigger).

### Known Issues

- **`ask_claude` cache race**: concurrent threads judging the same text can collide on the cache's temp file
  (`FileNotFoundError` on rename); re-run the affected job (finished calls are cached).
