# Goal Encoding in LLMs

Code for *Goal Encoding in LLMs*. Companion site: https://goals.baulab.info (source in the
[goals-web](https://github.com/arnab-api/goals-web) repo).

When an LM writes code for a task with several sub-tasks, does it encode *which sub-task it will implement next*
before writing it? We localize a sparse set of **goal heads** (attention heads that carry this sub-goal), read
them out in natural language with **GoalScope**, test how persistent and how necessary the encoding is, and use
interventions on the task span and the code-progress span to show how a depth-bounded LM pursues its goals
(**NRGS**: it re-reads the goal from the task span at each step).

## Setup

```bash
bash setup_env.sh          # conda env `goals` (Python 3.12, torch>=2.7 / CUDA 12.8, baukit from git)
cp env_demo.yml env.yml    # then fill in API keys (Claude, OpenAI) and cache / results directories
```

Everything runs from the repo root (`sys.path.append(".")`). Results go to `RESULTS_DIR` (`env.yml`,
`src/utils/env_utils.py`). Models: Qwen2.5-Coder-32B-Instruct throughout; Qwen2.5-Coder-14B, Qwen2.5-32B and
OLMo-3.1-32B (Instruct / SFT) for the model comparisons. A 32B model in bf16 needs one 80 GB GPU.

## Layout

```
src/models.py, tokens.py, functional.py, attention.py   model wrapper (nnsight), tokenization, generation + patching
src/goals/                                               goal heads + GoalScope
    head_loc_{data,oracle,pivot,augment,attn}.py         pivot mining (oracle) and head-mask training
    head_sets.py                                         load head sets (head_sets/<model>/*.json)
    goalscope.py, goalscope_viz.py                       GoalScope readout + the paper-style HTML renderer
    temporal_persistence.py                              temporal persistence
    site_data.py                                         data writers for the companion site
src/diff/                                                NRGS interventions (KV / Q transfer, task vs progress)
scripts/                                                 one entry point per experiment (below)
notebooks/fig_n_table_generators/                        figures + tables of the paper, from cached results
notebooks/demos/                                         walk-through demos of the main experiments
job_list/                                                the job lists the reported runs were launched with
run_jobs/                                                multi-GPU job dispatcher (python -m run_jobs.baulab.dispatch --jobs ...)
head_sets/<model>/                                       localized head sets (goal heads, top-k sets, controls)
data/                                                    task corpus, localization pairs, ICL datasets, prefills
```

## Experiments

Paper section → the script that produces the results (with the job list it was launched with) → the notebook that
draws the figure / table.

| Paper | Script | Job list | Figures / tables |
|---|---|---|---|
| Task corpus (App) | `scripts/data/gen_goalscope_eval.py` → `data/goalscope_eval.json` | — | `corpus_stats.ipynb` |
| Localizing goal heads (§3, App) | `scripts/mine_pivot_instances.py` → `scripts/train_head_mask.py --lambda-sparse=20 --lazy --augment --tag=lam20` → `scripts/data/export_head_sets.py` | `head_localization/*.json` | `head_localization.ipynb` |
| GoalScope readout, faithfulness / causality (§3, App) | `scripts/evaluation/eval_goalscope.py --heads goal --strategy additive_norm --alpha 0.1` | `goalscope_eval/goalscope_eval_a_residual.json` | `goalscope_eval.ipynb`, `sparsity_ablation.ipynb` |
| Readout sources and head budget (App) | `eval_goalscope.py --heads <goal_topk json>`, `scripts/evaluation/eval_residual_patchscope.py` | `goalscope_eval_{a_residual*,a_attnout_40-50,b_sparsity}.json` | `sparsity_ablation.ipynb` |
| Across models (App) | `eval_goalscope.py` per model | `goalscope_eval_c_{model_axis,14b,olmo_rematched}.json` | `model_generalization.ipynb` |
| Residual-stream readout, OVReader (App) | `scripts/evaluation/eval_ov_reader.py --locked` | `diff_eval/ov_reader_locked.json` | `ovreader.ipynb` |
| Temporal persistence (§3, App) | `scripts/evaluation/temporal_persistence_eval.py` (+ `--control`) | — | `temporal_persistence.ipynb` |
| Obstacle persistence (App) | `scripts/evaluation/obstacle_robustness_eval.py` | — | `obstacle_robustness.ipynb` |
| Goal-head ablation: sufficiency / necessity (§4, App) | `scripts/evaluation/eval_ablation_{sufficiency,necessity}.py --control_pool later_half` | — | `ablation.ipynb` |
| KV transfer: re-reading the goal from the task span (§5, App) | `scripts/diff/path_patch_eval.py --i3 on --span list+gap --cpp-note system --i3-v-scale 4` | `diff_eval/path_patch_*32b_listgap*.json` | `transfer.ipynb` |
| Q transfer: progress selects the pending sub-goal (§5, App) | `scripts/diff/diff_trigger_eval.py --alphas 2 --pivot eq --naming letters --closed-loop --probe-prompt gaze_select_restrict --gsr-anchors eq` | `diff_eval/diff_trigger_gsr_eq_*.json` | `q_transfer_gsr.ipynb` (+ `gsr_schema_clean.py`, `scripts/diff/gsr_example_attention.py`) |
| Task span vs code-progress span (App) | `scripts/diff/task_vs_progress_eval.py --lang-transfers py-py` | `diff_eval/task_vs_progress_32b*.json` | `precommit_ablation.ipynb` |
| Readouts beyond enumerated tasks (App) | `scripts/generate_well_known_task_completions.py` → `scripts/run_goalscope_web.py`; `scripts/run_goalscope_fig32_stable_marriage.py` | `goalscope_eval/goalscope_web_full_coder32b.json` | rendered with `src.goals.goalscope_viz` |
| ICL verbalization (§6, App) | `scripts/icl/` (our rerun of the paper protocol) | — | companion site, ICL tab |
| Emergent misalignment (§6, App) | `scripts/em/` (our reproduction; the EM model is applied as a LoRA by hand) | — | companion site, EM tab |

Default GoalScope configuration: the 122 goal heads of Qwen2.5-Coder-32B
(`head_sets/Qwen2.5-Coder-32B-Instruct/goal__lam20_thr0.9_band25-56.json`), `additive_norm` with α = 0.1,
injected at layers 35–45 at every decode step, greedy. The NRGS interventions use the 250-head set
(`goal_topk/goal__lam20_top250.json`).

## Companion site

`scripts/run_goalscope_web.py`, `scripts/icl/icl_write_site_data.py` and `scripts/em/em_write_site_data.py` write
the site's data into a goals-web checkout (`GOALS_WEB_SITE`, default `./goals-web/site`); `make demo` in goals-web
renders and publishes it.
