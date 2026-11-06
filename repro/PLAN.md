# Comparison campaign: seven result groups on seven models

Branch `repro/comparison` (off `nnterp` de97b03). Host: hakone (8x A100-80GB), clone
`/disk/u/localjadenfk/repro/goals`. Status of the runs: `/disk/u/localjadenfk/repro/STATUS.md` (outside the clone).
Section numbers are `CLAUDE.md`'s (3.1, 4.1, ...). Output of the campaign (later): `results/comparison/`, a generator
notebook and a page.

## 0. What is decided here and what is still open

Decided (by the owner): the models, the groups, localization of the two new models first, evaluations in a second
wave on the `nnterp` update that moves OVReader onto nnterp's weight accessors.

Taken here (each with its basis below): the new models are localized on their OWN pivot pools (mined, cache-only);
batch sizes 24 (Devstral, 1 GPU) and 48 (GLM-4.5-Air, 4 GPUs) instead of 64; the band rule for new head sets
(section 2.3); the top-k and control recipes (verified exact on Coder-32B, section 2.2).

Open (owner): the size of the large head set on models with another head count (section 2.2); which OLMo set the
comparison uses (section 2.1); where the paper's final Coder-14B mask is (section 2.1); a Claude key for the
evaluation wave (section 1).

## 1. Environment

- `uv venv --python 3.12 .venv && uv sync` worked on hakone as is (torch 2.11.0+cu128, transformers 5.22.1, nnsight
  `0.8.0rc2.dev236+ge09ab1c82` = the pinned dev commit, nnterp `2.0.0rc2.dev36+g3fb2cc02b`). Run with `.venv/bin/python`.
- `env.yml` (untracked) mirrors the paper team's on hakone (`/disk/u/arnab/Codes/Projects/goals/env.yml`):
  `MODEL_DIR: models`, `RESULTS_DIR: results`, `DATA_DIR: data`, `CLAUDE_CACHE_DIR: data/LLM_CACHE/claude_cache`,
  `HF_CACHE: /disk/u/models`, `CLAUDE_KEY: ""`.
- Every command runs from the clone root with
  `JE_ARROW_MALLOC_CONF=background_thread:false HF_HUB_CACHE=/disk/u/models HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Judge: there is no Claude key on hakone (`CLAUDE_KEY: ""` here; the paper team's own key is in their env.yml and
  is not used). Every evaluation group calls the judge (counts in section 3), so **the evaluation wave needs a key**.
  `data/LLM_CACHE/claude_cache` is a copy of the paper team's hakone cache (133,491 entries, 534 MB, a snapshot of
  2026-10-03): it answers all 131 pivot-mining oracle calls, but 0 of the 185 judge calls the Coder-32B timing runs
  made (the hakone copy is dated before the reported runs; their leaves on hakone are older too), so it does not make
  the evaluations free.
- Scratch: worktree `/disk/u/localjadenfk/repro/goals-scratch` (branch `repro/scratch`, never pushed) whose env.yml
  sets `RESULTS_DIR: /disk/u/localjadenfk/repro/scratch_results`; every smoke and timing run of an evaluation goes
  there (Caution 6g).
- Reference results: the paper team's results tree on hakone, `/disk/u/arnab/Codes/Projects/goals/results/`
  (read-only for us). Its localization masks are the paper's for Coder-32B, Qwen2.5-32B, OLMo-Instruct and
  Llama-70B (`lam20_ctx`): thresholding each reproduces its tracked head set exactly (122 / 108 / 130 and 93 / 110).
  Its Coder-14B `lam20` mask is NOT the paper's (config 1423 instances, batch 64, augment; thresholds to 95 heads, 79
  shared with the tracked 88). Its evaluation leaves are older runs (`span=rule`, `nspec=512`), not the reported ones.

Tools on this branch (`repro/`):
- `load_probe.py`: load through `ModelandTokenizer`, report what nnterp serves, per-GPU memory, forward / generate
  timings on corpus spec `int_python_000`.
- `batch_probe.py`: peak memory and time of one head-mask training step at several batch sizes on the longest
  destination prompts (worst case), in one load.
- `run_with_mem.py`: run a module unchanged and print wall clock and per-GPU peak memory at exit.
- `run_stub_judge.py`: run an evaluation with `ask_claude` answered from the cache, else stubbed (timing only, scratch
  RESULTS_DIR); `REPRO_JUDGE_STRICT=1` makes a cache miss raise (used for mining).
- `time_coder32b.sh`: the Coder-32B unit timings of section 4.
- `localize.sh`: mine the model's pool (cache-only), then train the lam20 mask.

`src/models.py` change (this branch): `ModelandTokenizer` builds the TEXT model of a vision-language wrapper whose
config type has no causal-LM class (`mistral3`: Devstral Small 2), from `text_config` with the checkpoint's
`quantization_config` carried over and a key mapping from the wrapper's weight names. nnterp's default
`task="text-generation"` load fails on Devstral (transformers 5.22: `AutoModelForCausalLM` has no `mistral3` entry,
and the pipeline's fallback to `Mistral3ForConditionalGeneration` raises on `code_revision`). The checkpoint is FP8;
below compute capability 8.9 transformers dequantizes it to bf16 (43.9 GiB, 23.57B parameters). Every other model
loads exactly as before (the function returns `{}`). Candidate for the `nnterp` branch or for nnterp itself.

## 2. Models and head sets

| key | HF id | slug | nnterp family | layers x heads (kv) | GPUs (eval) | notes |
|---|---|---|---|---|---|---|
| C32 | Qwen/Qwen2.5-Coder-32B-Instruct | Qwen2.5-Coder-32B-Instruct | qwen2 | 64 x 40 (8) | 1 | main model |
| Q32 | Qwen/Qwen2.5-32B-Instruct | Qwen2.5-32B-Instruct | qwen2 | 64 x 40 (8) | 1 | code-finetuning axis |
| C14 | Qwen/Qwen2.5-Coder-14B-Instruct | Qwen2.5-Coder-14B-Instruct | qwen2 | 48 x 40 (8) | 1 | size axis |
| L70 | meta-llama/Llama-3.3-70B-Instruct | Llama-3.3-70B-Instruct | llama | 80 x 64 (8) | 2 | double BOS (CLAUDE.md 7) |
| O32 | allenai/Olmo-3.1-32B-Instruct | Olmo-3.1-32B-Instruct | olmo3 | 64 x 40 (8) | 1 | full suite now; post-norm |
| GLM | zai-org/GLM-4.5-Air | GLM-4.5-Air | glm4_moe | 46 x 96 (8) | 4 | 106.85B MoE, bf16 203.8 GB resident |
| DEV | mistralai/Devstral-Small-2-24B-Instruct-2512 | Devstral-Small-2-24B-Instruct-2512 | ministral3 | 40 x 32 (8) | 1 | text-only load (section 1), FP8 dequantized |

Load checks (`repro/load_probe.py`, logs `logs/probe-*.log`): all values of `mt.support()` served on C32, GLM and DEV
under eager attention (GLM: `mlp.mlp_activation` per expert only, which no group reads); `mt.model.config.num_attention_heads`
(read raw by `eval_goalscope.py` and `src/nrgs/functional.py`) works on the text-only Devstral load. The corpus
prompts end in `'   ' <var> ' ='` on every tokenizer, so the four GoalScope pivots (the last four prompt tokens) carry
over. GLM's chat template, rendered by `chat_prompt` (`enable_thinking=False`), puts `<think></think>` after
`<|assistant|>`; Devstral's template inserts its long default system prompt when no system turn is given (every
paper prompt has one). Devstral's tokenizer warns about Mistral's pre-tokenizer regex; its `tokenizer.json`
already carries the corrected pattern (checked), so `fix_mistral_regex` changes nothing.

### 2.1 The small goal-head set (G-small: GoalScope 4.1 / 4.4, ablations, temporal persistence)

| model | file (`head_sets/<slug>/`) | heads | made by | used for |
|---|---|---|---|---|
| C32 | `goal__lam20_thr0.9_band25-56.json` | 122 | `export_head_sets.py` | all groups |
| Q32 | `goal__lam20_thr0.9_band25-56.json` | 108 | `export_head_sets.py` | all groups |
| C14 | `goal__lam20_thr0.9_band0-48.json` | 88 | `export_head_sets.py` (on the paper's mask, not on hakone) | all groups |
| L70 | `goal__lam20_ctx_thr0.9_band18-45.json` | 110 | no script (lam20_ctx mask, thr 0.9, [18,45)) | all groups |
| O32 | `goal__lam20_thr0.9_band10-36.json` | 130 | `export_head_sets.py` (`MODEL_BANDS` 10-36) | all groups (campaign choice) |
| O32 | `goal__lam20_thr1.0_band10-36.json` | 93 | no script (hand-balanced against SFT, master ae9f2c2) | the paper's 4.4 row (optional parity arm) |
| GLM | `goal__lam20_thr0.9_band<a>-<b>.json` | tbd | `export_head_sets.py` + a `MODEL_BANDS` entry (2.3) | all groups |
| DEV | `goal__lam20_thr0.9_band<a>-<b>.json` | tbd | same | all groups |

- OLMo: the paper's reported set (93) is hand-made with no script. The campaign uses the scripted
  `export_head_sets.py` set (thr 0.9 inside the existing `MODEL_BANDS` entry 10-36: 130 heads, verified to equal the
  tracked json from the reported mask). Run the 93-head set only as the 4.4 parity arm, if wanted. Note the reported
  OLMo mask was trained on the Coder-32B pool (Caution 6a).
- Llama: to make its set scripted too, `export_head_sets.py` needs a per-model tag (`lam20_ctx`) and a
  `MODEL_BANDS` entry (18, 45); the result equals the tracked json (verified). `GOAL_BANDS` already has (18, 45).
- Coder-14B: the evaluations read the tracked 88-head json, so they do not need the mask. The 3.1 map does: the
  paper's mask (1246 instances, batch 32, `--no-augment`) is not on hakone. Either fetch it from wherever the final
  `results/` lives, or re-run job `01b_qwen14b_goal_lamb_20` of `job_list/head_localization/qwen14b_localization_lambda_sweep.json`
  (a new mask, Caution 6b) into a new tag.

### 2.2 The large set and the control (G-large / C-large: KV 4.9, Q 4.12, the 250-head ablation arm)

- G-large = top-k heads by lam20 mask value over all layers (`np.argsort(-mask.ravel(), kind="stable")[:k]`, ties by
  index): verified to reproduce `goal_topk/goal__lam20_top250.json` exactly from the C32 mask. No script on `nnterp`
  (the recipe is in master's `notebooks/clock/__head_inspection.ipynb`); add it to `export_head_sets.py` (a `--topk`
  mode writing `head_sets/<slug>/goal_topk/goal__<tag>_top<k>.json`).
- C-large = `sorted(random.Random(123).sample([(l, h) for l in range(L // 2, L) for h in range(H) if (l, h) not in G], k))`:
  verified to reproduce `control/control250__laterhalf.json` exactly. The ablation scripts' controls are the same
  recipe (`ablation_common.random_heads_like`, seed 123, layers >= L // 2), so they need no file.
- **k (owner decision)**: C32's 250 is 9.8% of its 2560 heads and 2.05x its G-small. Proposal: k = round(0.0977 x L x H):
  Q32 / O32 250, C14 188, L70 500, GLM 431, DEV 125. Alternative: k = 250 everywhere (25% of DEV's heads, 4.9% of
  L70's). Past G-small the added heads are low-valued (C32's 250th has mask value 0.0015), so the KV and Q
  numbers may be sensitive to k on small models.

### 2.3 Band rule for a new model's G-small

Band = the smallest contiguous layer window holding >= 85% of the lam20 heads with mask > 0.9. On the paper's masks
this rule gives Q32 [29, 56) = 108 heads (paper [25, 56) = 108, same heads), L70 lam20_ctx [19, 44) = 110 (paper
[18, 45) = 110, same heads), O32 [9, 34) = 128 (paper 130), C32 [32, 57) = 117 (paper 122). Applied to GLM and DEV
once their masks exist; the window, the per-layer counts and the count are recorded in `MODEL_BANDS` with a comment,
and in `GOAL_BANDS` so `--heads goal` resolves.

## 3. Groups x models: commands

Conventions: `$M` = HF id, `<slug>` = the id's last part. Layer constants scaled by depth like
`scripts/cross_family_check.py` (steer `(round(L*35/64), round(L*46/64))`, `from_layer = round(L*20/64)`) where the
paper has no value for the model; the paper's value where it has one.

| model | `--steer_layer_range` | sufficiency `--from_layer` | control floor (derived `L // 2`) |
|---|---|---|---|
| C32, Q32, O32 | 35 46 (paper) | 20 (paper) | 32 |
| C14 | 20 30 (paper, 4.4 table run) | 15 | 24 |
| L70 | 44 58 (scaled; an unreported run used 20 40, alpha 0.11) | 25 | 40 |
| GLM | 25 33 | 14 | 23 |
| DEV | 22 29 | 12 | 20 |

### 3.1 Head-localization map

- C32 / Q32 / O32 / L70: masks exist (reference tree, section 1); no re-run. C14: see 2.1.
- GLM, DEV (this wave): `bash repro/localize.sh $M <gpus> <batch>` =
  `python -m scripts.mine_pivot_instances --model-key $M` (cache-only) then
  `python -m scripts.train_head_mask --model-key $M --batch-size <24 | 48>` (defaults: 50 epochs, lambda 20, lr 1e-2,
  `--augment`, tag `lam20`, checkpoints every 5 epochs) -> `data/head_loc/pairs/<slug>.json`,
  `results/localization/<slug>/lam20/{head_mask.pt, checkpoints/}`. Job list `job_list/comparison/head_localization.json`.
- Pool: the repo's procedure for a new model (CLAUDE.md 7) mines the model's own pool; without it training falls back
  to the Coder-32B pool (Caution 6a, how the reported OLMo mask was made). Mining's 131 oracle calls (110 pair
  prompts, 21 implementation prompts) depend only on the predicate files and are all in the copied cache (checked),
  so mining needs no key; the model-specific part is the completion / divergence / augmentation gates.
- Batch: 64 does not fit. Worst-case batch (longest destination prompts) peak memory, `repro/batch_probe.py`:
  DEV 1 GPU: 24 -> 70.0 GiB, 28 -> 74.3 GiB (32 OOM in the smoke); GLM 4 GPUs: 48 -> 71.4 GiB on the fullest card
  (64 OOM in the smoke). Chosen DEV 24, GLM 48 (about 8 GiB headroom: one OOM costs the whole run, which has no
  resume). The paper's masks also differ in batch (C32 64, Q32 16, C14 32, L70 8). Fewer, larger steps would need
  gradient accumulation in `train_head_mask` (a code change; not done).
- Coder-32B note: batch 64 does not fit on 1 or 2 A100s either (peak 77 GiB on each of 2 cards before OOM); the
  smoke ran at batch 32 on 2 GPUs (58 GiB each).
- Then: export G-small (2.3) and G-large (2.2) for GLM and DEV, run `scripts/cross_family_check.py --model $M --heads <G-small>`.
- Figure: `head_localization.ipynb` cells 1-2 hard-code C32 (`MODEL`, `N_LAYER=64`, `BAND=(28, ...)`, the 122 json);
  the comparison notebook loops over models with each model's mask, layer count, band and json.

### 3.2 GoalScope readout accuracy (4.1) and the four-pivot model table (4.4)

```
# 4.1 (additive_norm alpha 0.1), per model, both modes
CUDA_VISIBLE_DEVICES=<gpus> python -m scripts.evaluation.eval_goalscope --model $M --mode {faithfulness,causality} \
    --heads <G-small json | goal> --strategy additive_norm --alpha 0.1 --steer_layer_range <a> <b>
# 4.4 (replace alpha 1), per model, both modes
CUDA_VISIBLE_DEVICES=<gpus> python -m scripts.evaluation.eval_goalscope --model $M --mode {faithfulness,causality} \
    --heads <G-small json | goal> --strategy replace --alpha 1.0 --steer_layer_range <a> <b>
```
- C14 adds `--goal_band 0 48` (the reported leaf); C32 / Q32 run with defaults for `--heads goal`.
- Units: 253 specs x 5 pivots per mode; 1265 judge calls per mode (`claude-sonnet-4-6`, `hard_eval`), 4 modes per
  model. Output `results/evaluation/goalscope/<slug>/<mode>/<strategy>/<alpha>/heads=...__nspec=253__.../summary.json`
  (`by_anchor[pivot]["hard_yes"]`, mean over `prev_end, indent, var, eq`).
- Generators: 4.1 bar panel `readout_sources_and_head_budget.ipynb` cell 8 (C32 only); 4.4 table
  `model_generalization.ipynb` cell 3 (`model_comparison.tex`, a fixed 5-model list) and cells 1-2 (axis figures;
  `HSPEC` needs entries for json-headed models; mind the `glob[0]` hazard, CLAUDE.md 4.4).
- Paper references: 4.1 C32 mean4 .853 / .750; 4.4 C32 .789 / .735, C14 .519 / .305, Q32 .238 / .441, O32 (93 heads)
  .435 / .622.

### 3.3 Goal-head ablations (4.6)

```
CUDA_VISIBLE_DEVICES=<gpus> python -m scripts.evaluation.eval_ablation_sufficiency --model $M --heads <G-small> --from_layer <f>
CUDA_VISIBLE_DEVICES=<gpus> python -m scripts.evaluation.eval_ablation_sufficiency --model $M --heads <G-large> --from_layer <f>
CUDA_VISIBLE_DEVICES=<gpus> python -m scripts.evaluation.eval_ablation_necessity  --model $M --heads <G-small>
CUDA_VISIBLE_DEVICES=<gpus> python -m scripts.evaluation.eval_ablation_necessity  --model $M --heads <G-large>
```
- `--heads` is required for every model but C32 / Q32 (`DEFAULT_HEADS_JSON` hard-codes `band25-56`). The leaf names
  the set by size only (`goal<N>`), so two sets of one size collide.
- `SINK_POS = 0` (necessity) is `<|im_start|>` on Qwen / OLMo, BOS on Llama (doubled), `[gMASK]` on GLM, `<s>` on DEV.
- Units: 253 specs x 3 greedy decodes (baseline / ablated / control), 759 judge calls per run (baseline verdicts are
  cache hits after the first run), 4 runs per model. Output
  `results/evaluation/ablation/<slug>/{sufficiency,necessity}/heads=..._goal<N>__...__ctrl=later_half/summary.json`.
- Generators: `ablation.ipynb` cells 2-3 (`necessity.tex`, `sufficiency.tex`; C32 slug, `goal(122|250)` regex, n == 253).
- Paper: clean .925; sufficiency spare-goal .806 (122) / .913 (250), spare-random .103 / .087; necessity sink-goal
  .854 / .158, sink-random .925 / .925.

### 3.4 Temporal persistence (4.7)

```
python scripts/evaluation/temporal_persistence_eval.py --model $M --cuda <gpus>                       # goal heads
python scripts/evaluation/temporal_persistence_eval.py --model $M --cuda <gpus> --control --n-trials 64
```
- **Needs a code change**: the head set is hard-coded to `head_sets/<slug>/goal__lam20_thr0.9_band25-56.json`
  (exists for C32 / Q32 only) and `HEAD_BUDGET = 122` enters every cache key; add `--heads` and key on the set. The
  cache key does not hold the head-set contents (a changed set silently reuses records).
- Units: up to 4518 goal conditions (1860 contexts; C32 kept 3088) + 1240 control conditions; 1 oracle segmentation
  call per condition (the prompt holds the model's own generation, so new calls for every model). Output
  `results/temporal_persistence/<slug>/{,control/}cond_*.pt`.
- Generator: `temporal_persistence.ipynb` cells 2-3 (`SLUG` constant; run as a script if the kernel dies).

### 3.5 Obstacle persistence (4.8)

```
python scripts/evaluation/obstacle_persistence_eval.py --model $M --cuda <gpus>
```
- No heads, no layers. 409 trials, 1 judge call each. Output `results/obstacle_robustness/<slug>/{trial_*.json, rows.json, summary.json}`.
- The OV-boost table (`obstacle_persistence_ov_boost_eval.py`) hard-codes C32 + G250 (`MODEL`, `HEADS_JSON`, no
  flags) and a C32-specific derail regex; out of scope unless asked.
- Generator: `obstacle_persistence.ipynb` cell 3 already loops over every `obstacle_robustness/*/summary.json`
  (`obstacle_adapted_by_level.tex`); cell 2 is C32 only.
- Paper (aligned / easy / hard): C32 .97 / .63 / .15, O32 .94 / .57 / .21, Q32 .97 / .66 / .12, C14 .91 / .50 / .09.

### 3.6 KV transfer (4.9)

```
python scripts/nrgs/kv_transfer_eval.py --model $M --cuda <gpus> --datatypes <dt> --lang-transfers <lt> \
    --n-samples 32 --i3-v-scale 1 2 4 --goal-heads <G-large> --readout-heads <G-large>          # 16 jobs
python scripts/nrgs/kv_transfer_eval.py --model $M --cuda <gpus> --datatypes <dt> --lang-transfers <lt> \
    --n-samples 32 --goal-heads <C-large> --readout-heads <C-large> --no-readout                 # control, 16 jobs
```
- **Needs a code change**: no `--model` flag (`MODEL` constant, line 67); the NRGS readouts use
  `goalscope_readout`'s (35, 46) defaults with no flag (scale per model).
- Units: 16 cells x 32 = 512 instances; per instance and V config 1 patched generation + 1 patched readout,
  4 judge calls (1 with `--no-readout`). Output `results/diff/<slug>/path_patch/<run_tag>/<dt>__<src>-<dst>/meta.json`.
- Generator: `kv_transfer.ipynb` cells 7 / 11 / 14 (`MODEL`, `KV_RUN` with `goal_250` in the tag, `MODEL_COLOR`).
- Paper: pooled causal .936, readout .900; control .002.

### 3.7 Q transfer with GSR (4.12)

```
python scripts/nrgs/q_transfer_eval.py --model $M --cuda <gpus> --datatypes <dt> --variants <v> --lang-transfers <lt> \
    --patch-heads <G-large> --readout-heads <G-large> --tag-suffix __seq                           # 32 cells
python scripts/nrgs/q_transfer_eval.py ... --patch-heads <C-large> --readout-heads <C-large> --no-readout --tag-suffix __seq
python scripts/nrgs/q_transfer_eval.py ... --variants c_in_dst --lang-transfers py-py --tag-suffix __seq__cindst
```
- **Needs a code change**: no `--model` flag (`MODEL` constant, line 63); readout layers as in 3.6.
- Units: 1024 instances (2 variants x 16 cells x 32), 6 judge calls each (3 with `--no-readout`); control 1024;
  c_in_dst 128. Output `results/diff/<slug>/diff_trigger/ph-<stem>_<n>__a2__piv-eq__cl-ppgazerestrict-anceq__seq/...`.
- Generator: `q_transfer.ipynb` cells 5 / 7 / 9 (`MODEL`, `GSR_ARM`, `MODEL_COLOR`).
- Paper: pooled causal order-matched .871, order-diff .572; control .043 / .029.

### 3.8 Code changes the evaluation wave needs (besides the OVReader update)

1. `--model` for `kv_transfer_eval.py` and `q_transfer_eval.py`; readout steer layers as a flag through
   `src.nrgs.functional.goalscope_readout`.
2. `--heads` for `temporal_persistence_eval.py`, and the head set (or its hash) in the cache key in place of
   `HEAD_BUDGET = 122`.
3. Stop tokens: the greedy decoders (`goalscope.greedy_continue`, the replace readout, `src.nrgs.functional._generate`)
   stop only at `tokenizer.eos_token_id`. On GLM that is `<|endoftext|>` while its turns end at `<|user|>` (measured:
   the 64-token frontier rollout on `int_python_000` ran past `</code>` into a second `<think></think><code>` block;
   `mt.generate` with the generation config stops at `<|user|>`); on OLMo `<|endoftext|>` vs `<|im_end|>`. Use the
   generation config's EOS list. This changes GLM / OLMo results and costs up to 4x more decode steps there.
4. `export_head_sets.py`: a per-model tag (Llama `lam20_ctx`), the `MODEL_BANDS` entries for GLM / DEV (2.3), and a
   `--topk` mode (2.2); `GOAL_BANDS` entries for C14 (0, 48), GLM and DEV.
5. Generators: `figstyle.MODEL_COLOR` has no L70 / GLM / DEV; the per-group notebooks above hard-code the C32 slug.

## 4. Unit timings and estimates

### 4.1 Coder-32B (1 A100 for the evaluations, judge stubbed; `repro/time_coder32b.sh`, logs `logs/time-coder32b-*.log`)

Model side only: every judge call answered by the stub, so add the judge's latency (unmeasured here: no key; the
calls are serial and inline, CLAUDE.md calls the judged evals oracle-bound). "unit" = what the paper's full run
has N of. Load (warm page cache) about 20 s per run, not included.

| group | run | unit | s / unit | units in the full run | full run, model side |
|---|---|---|---|---|---|
| 4.1 | `eval_goalscope --mode faithfulness --limit 8` (additive 0.1) | spec (rollout + 5 readouts) | 5.5 | 253 | 23 min |
| 4.1 | `--mode causality --limit 8` | spec | 5.1 | 253 | 22 min |
| 4.4 | `--strategy replace --alpha 1 --limit 8` | spec | 4.5 | 253 x 2 modes | 38 min |
| 4.6 | `eval_ablation_sufficiency --limit 8` | spec (3 decodes) | 2.0 | 253 x 2 sets | 17 min |
| 4.6 | `eval_ablation_necessity --limit 8` | spec | 2.1 | 253 x 2 sets | 18 min |
| 4.7 | `temporal_persistence_eval.py --n-trials 1 --force` (8 contexts, 22 conditions) | context (12.7 s) / condition (4.6 s) | 4.6 | 4518 goal + 1240 control conditions | 5.8 h + 1.6 h |
| 4.8 | `obstacle_persistence_eval.py --datatypes int --predicates is_even digit_count --force` | trial | 1.4 | 409 | 10 min |
| 4.9 | `kv_transfer_eval.py --datatypes int --lang-transfers py-py --max-instances 4` (V x4 only) | instance x 1 V config | 7.8 | 512 x 3 V configs + 512 control | 1.1 h per V config; about 2.5 h with the shared clean runs + 0.7 h control |
| 4.12 | `q_transfer_eval.py --variants order_diff --max-instances 4 --tag-suffix __seq` | instance | 12.5 | 1024 + 1024 control + 128 | 3.6 h + about 2.5 h + 0.4 h |

Sum for one 1-GPU model: about 19 GPU-hours on the model side (7.4 h of it temporal persistence). Judge calls per model: 4.1 + 4.4 5060, 4.6 3036
(about 2530 unique), 4.7 about 5760, 4.8 409, 4.9 about 6700, 4.12 about 10,000; about 31,000 in all. At an assumed
3-5 s per serial call that is 26-43 hours of judge latency per model unless the jobs run in parallel (the job lists
already split by datatype / cell). The judge, not the GPU, sets the evaluation wave's wall clock.

### 4.2 Forward-time ratios (one prompt, `repro/load_probe.py`, logs `logs/probe-*.log`)

Corpus spec `int_python_000` (99-106 prompt tokens), bf16, eager, after a warm-up; ratio to Coder-32B in brackets.

| model | GPUs | resident | 1-prompt traced forward | batch-16 forward | 64-token `mt.generate` (stops at the turn end) | 64-token no-cache greedy (`greedy_continue`) |
|---|---|---|---|---|---|---|
| C32 | 1 | 61.0 GiB | 0.069 s | 0.537 s | 1.11 s | 1.30 s (stopped at `<|im_end|>`, about 15 steps) |
| DEV | 1 | 43.9 GiB | 0.053 s (0.78x) | 0.348 s (0.65x) | 0.78 s (0.70x) | 0.97 s (stopped at `</s>`) |
| GLM | 4 | 50.4 / 52.3 / 52.3 / 44.8 GiB | 0.254 s (3.7x) | 0.495 s (0.92x) | 1.33 s (1.2x) | 17.3 s (ran all 64 steps: EOS is `<|endoftext|>`, see 3.8 item 3) |

GLM's prefill is slow at batch 1 (an eager MoE runs its experts one by one; a batch of 16 costs about the same),
while its cached decode is close to the 32B's. The scaling below uses the 1-prompt forward ratio for the no-cache
loops (frontier rollouts, temporal, obstacle) and the `mt.generate` ratio for the cached ones (readouts, ablations,
NRGS).

Direct check on Devstral (judge stubbed, a random 60-head set from layers 13-26 for its not-yet-existing goal heads,
`repro/time_devstral.sh`): GoalScope faithfulness 3.1 s / spec (0.57x of C32), sufficiency 1.5 s / spec (0.75x),
obstacle 1.2 s / trial (0.86x); the ratio method gives 0.70-0.78x, so it is good to about +-30% on Devstral. GLM
could not be checked directly (its 4 GPUs are running its localization).

### 4.3 Head localization (one epoch, `--tag scratch_smoke --n-epochs 1`; logs `logs/smoke-headloc-*.log`)

| model | GPUs | batch | pool (smoke) | caching | epoch (augment + 1291 train instances) | peak per GPU |
|---|---|---|---|---|---|---|
| C32 | 2 | 32 (64 OOM on 1 and on 2 GPUs) | own, 1613 | 53 s | 180 s (41 steps) | 57.8 / 58.1 GiB |
| DEV | 1 | 24 (64, 32 OOM) | Coder-32B fallback, 1613 | 34 s | 124 s (54 steps) | 70.0 GiB |
| GLM | 4 | 48 (64 OOM) | Coder-32B fallback, 1613 | 47 s | 157 s (27 steps) | 67.7 / 71.4 / 71.4 / 61.0 GiB |

The smokes ran on the fallback pool (no own pool existed yet); the full runs train on the mined pools, so the epoch
time scales with the mined pool's size. One-step worst-case peaks (`repro/batch_probe.py`, the longest destination
prompts, 100-102 tokens): DEV 8 / 16 / 20 / 24 / 28 -> 52.6 / 61.3 / 65.7 / 70.0 / 74.3 GiB (1.09 GiB per instance);
GLM 8 / 16 / 24 / 32 / 40 / 48 -> 55.6 / 58.7 / 61.9 / 65.1 / 68.2 / 71.4 GiB on the fullest card (1.5 GiB per
instance over 4 cards). The smoke peaks equal the probe's worst case at the chosen sizes.

### 4.4 Estimates for the new models

Localization (running; basis: the one-epoch smokes, scaled by pool size N / 1613, plus mining measured live):

| model | mining (stage 2 measured rate; stage 3 = augmentation, extrapolated) | training, 50 epochs + 10 val evals | expected total |
|---|---|---|---|
| DEV | stage 2 measured: 29.5 min, 555 candidates (3.2 s each), 342 validated; stage 3 (5 wrappers per validated instance) about 75 min at the measured start rate (12.6 s per instance) | about 1.7 h x N / 1613 | about 3.5 h: started 23:41, expected about 03:15 Nov 4 |
| GLM | stage 2 about 2.6 h (559 candidates, 16.9 s each, 5.3x Devstral: the completion gate is a 32-step no-cache greedy and GLM's batch-1 prefill is 3.7x the 32B's); stage 3 about 8.5 h (5.3x Devstral's per-instance rate, about 460 validated) | about 2.2 h x N / 1613 | about 14 h: started 23:43, expected about 14:00 Nov 4. The Coder-32B fallback pool (no mining) would take about 2.3 h |

Evaluations (second wave; model side; one model's full set of the seven groups):

| group | DEV (1 GPU), basis | GLM (4 GPUs, wall), basis |
|---|---|---|
| 4.1 (2 modes) | 2 x 13 min (direct 0.57x) | 2 x 42 min with the EOS fix, 2 x 93 min without (rollout 3.7x, readout 1.2x) |
| 4.4 (2 modes) | 2 x 11 min (0.57x) | 2 x 35 min / 2 x 80 min |
| 4.6 (4 runs) | 25 min (direct 0.75x) | 45 min (1.2x) |
| 4.7 (goal + control) | 5.8 h (0.78x) | 27 h with the EOS fix (3.7x), up to 41 h without (all 96 steps) |
| 4.8 | 8 min (direct 0.86x) | 35 min / 1.5 h |
| 4.9 (3 V configs + control) | 2.3 h (0.70x) | 3.8 h (1.2x) |
| 4.12 (+ control, c_in_dst) | 4.6 h (0.70x) | 7.8 h (1.2x) |
| total | about 14 GPU-hours | about 42 h wall = 170 GPU-hours with the EOS fix |

Add the judge (section 4.1): about 31,000 calls per model. GLM's 4.7 is the longest single job of the wave; split it
by datatype x language (8 processes) if 4-GPU slots allow.
