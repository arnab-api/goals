# Set/count localization v2: two matched readout blocks

This handoff implements the agreed revision in a patch against `arnab-api/goals` commit `8c59544e5d04ed19d3c02c2b2168968c593b28f4`. It does not edit, commit, or push the managed repository. Apply it on a branch and review through a pull request.

**The original source-only H1 is preserved.** Donor D predicts task **a**, not e. No destination/source union rule has been introduced, and `nrgs_prefix.predict` is unchanged.

## Which patch to apply

- **Clean original base:** apply `control_supervised_set_count_v2.patch`. It includes the entire initial implementation plus this revision.
- **The exact earlier v1 patch is already applied:** apply only `set_count_v1_to_v2.patch`.
- Do **not** apply both. If your files have diverged, reconcile the changes rather than forcing a patch or discarding your local work.

```bash
cd /path/to/goals
git switch -c control-supervised-set-count-v2
# Choose exactly one:
git apply --check --whitespace=error /path/to/control_supervised_set_count_v2.patch
git apply /path/to/control_supervised_set_count_v2.patch
# OR, when the original v1 implementation is already present:
# git apply --check --whitespace=error /path/to/set_count_v1_to_v2.patch
# git apply /path/to/set_count_v1_to_v2.patch
```

No dependency pins in the repository are changed. The new standalone trainer avoids the legacy `nnsight`/`baukit` wrappers. Existing callers retain the original query-hook function names through re-exports.

## 1. Source donors and the unchanged hypotheses

All sources list tasks in the same order **[b,a,d,e]**. A prime denotes a completed task in the assistant prefill; letters are task identities, not rendered identifiers.

| Donor | Source configuration | Completed set | Count | First unfinished task |
|---|---|---|---:|---|
| A | b a' d' e | {a,d} | 2 | b |
| B | b a' d e' | {a,e} | 2 | b |
| C | b a' d e | {a} | 1 | b |
| D | b a d' e | {d} | 1 | b |

**Set/H1:** first item in the destination task list that is absent from the source's completed set. Destination-completed tasks are not additionally excluded.

**Count/H1c:** item at zero-based destination position equal to the number of source-completed tasks. These semantic names avoid confusing count with the repository's separate H2/frontier-position hypothesis.

The unfinished variable and completed output variables come from the **same shuffled opaque identifier pool** (`u,v,w,t,z,q`). One distinct identifier is reserved for the unfinished assignment; its sampled value is independent of donor, count, block, and expected target. Names are shared across matched source/destination/reference prompts. No `result =` or descriptive output name is used. Source capture still ends exactly at `=`; recipient prefixes preserve every byte, including meaningful trailing spaces.

Changing names is not proof that the model stops assembling a tuple/list. Actual source continuations still need checking.

## 2. Two binary readout blocks

A single d/e prefix cannot measure D's prediction a. The builder therefore joins two existing directed pivot records from the same datatype:

- **de:** native role d versus role e.
- **da:** native role d versus role a.

The following conditions are produced for each naming variant.

### Block de: A/B/C, nine views

| Recipient configuration | Destination | Native | Set A/B/C | Count A/B/C |
|---|---|---|---|---|
| core | a' d e | d | e / d / d | e / e / d |
| swapped | a' e d | e | e / d / e | d / d / e |
| four | a' d e f | d | e / d / d | e / e / d |

A/B/C share the same destination and d/e prefix within each configuration. The original recipient-order counterbalance remains in this block.

### Block da: C/D, four views

| Recipient configuration | Destination | Native | Set C/D | Count C/D |
|---|---|---|---|---|
| core | a' d e | d | d / a | d / d |
| four | a' d e f | d | d / a | d / d |

C and D share an identical destination and d/a prefix within each configuration. D tests **recomputation of a destination-completed task**, which is the source-only H1 prediction. It is not labeled as skipping forward to e.

The swapped destination is intentionally not used in block da: its native task is e, so it would require a separate e/a readout. We do not force e into a d/a binary instrument.

There are **13 views per family per naming variant**: 9 de + 4 da. The complete family, not a convenient subset of its views, is accepted or rejected during next-token validation.

### Illustrative prefixes, not validated model examples

For a=positive, d=even, e=divisible-by-three, a d/e prefix could be `n % ` (tokens 2 versus 3), while a d/a prefix could be `n` (tokens ` %` versus ` >`). The latter does not globally distinguish evenness from every other modulo predicate. This illustration is not inserted as a fabricated mined record: the actual builder uses existing pool entries and records both of them.

`rendered_family_examples.json` contains constructed examples from actual eligible pool joins. These have not been generated or validated by the pretrained model.

## 3. Pool selection, splitting and weighting

The builder keeps the original pool untouched. Its eligibility view excludes binding-only token contrasts while preserving literals, operators, method names, and builtins such as `any` versus `len`.

Both unordered diagnostic pairs are assigned a split globally before families are formed. A d/e pivot is joined with an actual d/a pivot only if **both pairs have the same assigned split**. This prevents a secondary pair from appearing in training while serving as a primary test pair. Both orientations, all prefixes, and all matched variants of a diagnostic pair remain on one side. Cross-split joins are rejected, not silently moved or merged.

For the recorded pool, seed 0, one role assignment per primary pivot and one naming variant, this produces:

| Measure | Count |
|---|---:|
| Constructed families | 239 |
| Matched views | 3,107 |
| String / integer / integer-list families | 90 / 57 / 92 |
| Train / development / test families | 218 / 7 / 14 |
| Diagnostic unordered pairs used | 45 |

These are unvalidated model inputs. Fifty primary pivot candidates fail lexical eligibility; another 86 have no eligible same-split secondary-pair assignment.

**Coverage limitation:** this seed/pool join has no integer-list development or test families, and no integer development families. The development set is string-only; the test set contains strings and integers. Do not claim across-datatype held-out generalization from this dataset. Expand the paired-readout coverage or choose a prospective split protocol before a scientific run; do not choose a seed after inspecting evaluation performance.

Training uses equal weight per family; within a family, each block gets half the weight, then configurations, donors and naming variants are weighted equally. This avoids the nine-view de block dominating the four-view da block. Native-control multipliers are applied without renormalizing the primary loss scale. Evaluation rows include `view_weight`, block, datatype, prefix, readout roles, and source/destination progress metadata. The optional smoke harness aggregates using those weights rather than a flat mean over 13 rows. Raw counts and whole-family correctness remain available.

## 4. Build fresh v2 data and test the patch

Run from the repository root using a compatible environment:

```bash
uv run python -m pytest -q test/test_set_count.py
uv run python scripts/diff/build_set_count_data.py \
  --model Qwen/Qwen2.5-Coder-32B-Instruct \
  --n-per-pivot 1 --naming-variants 1 --seed 0 \
  --output /absolute/output/path/families-v2.json
```

This builder reads the existing on-disk pivot pool; it does not download a model. New schema `set-count-v2`, renderer `opaque-identifiers-v2`, both pivot identities, and source-only semantics participate in provenance/fingerprints. **Rebuild old datasets; do not relabel v1 JSON.** Old schemas and mixed renderer/semantics metadata are rejected. v1 masks/checkpoints are not resumable as v2 training; keep their old code/data for historical analysis.

The new core tests cover the unchanged predictor, all 13 labels, matched prefixes, opaque names, secondary-pair split isolation, dynamic a/d readouts, block weights, gradients and checkpoint safety. The optional v2 runner has its own CPU tests. **55 tests passed (31 core + 24 runner), with 40 additional subtests.** Counts and logs are recorded in `verification.json`.

### Offline random-model mechanics smoke

```bash
uv run python scripts/diff/train_set_count_mask.py \
  --dataset /absolute/output/path/families-v2.json \
  --output-dir /absolute/output/path/smoke-v2-set-head \
  --hypothesis set --mask-kind head --smoke-random \
  --mask-init 0.5 --device cpu --max-families 1 --max-steps 2
```

Repeat independently with `count` and/or `svd`. The tiny random fixture uses artificial token IDs, not operation labels inferred from a real model. It tests tensor/gradient/checkpoint mechanics only. The normal mask initialization remains 1.0; 0.5 is used in this mechanics test to avoid mistaking a correctly clamped boundary update for a missing gradient.

This revision was checked with two-step CPU mechanics runs for all four combinations. **No new pretrained-model or GPU run was performed.**

## 5. Real-model validation and exploratory training

First validate the new readouts, with a fresh output directory:

```bash
uv run python scripts/diff/train_set_count_mask.py \
  --dataset /absolute/output/path/families-v2.json \
  --output-dir /absolute/output/path/readout-check-v2 \
  --hypothesis set --model Qwen/Qwen2.5-Coder-32B-Instruct \
  --revision 381fc969f78efac66bc87ff7ddeadb7e73c218a7 \
  --device cuda:0 --dtype bfloat16 --max-families 8 --validate-only
```

The validator recomputes token IDs in each reference context, checks a meaningful binary contrast under that view's prefix, and checks that the clean recipient agrees with its native reference. It uses d/e for de and d/a for da. It does not assume stored token strings are correct in the new context, and it never changes D's label to fit the available instrument.

**Full source/recipient/teacher execution validation is still unimplemented.** The explicit `--allow-next-token-only` flag is required for exploratory pretrained optimization. It is an acknowledgement of that limitation, not scientific validation. A successful all-head query transfer is not an enforced prerequisite; selected components may work even when whole-query replacement does not.

After deciding to run an exploratory optimization:

```bash
uv run python scripts/diff/train_set_count_mask.py \
  --dataset /absolute/output/path/families-v2.json \
  --output-dir /absolute/output/path/set-svd-v2-exploratory \
  --hypothesis set --mask-kind svd \
  --model Qwen/Qwen2.5-Coder-32B-Instruct \
  --revision 381fc969f78efac66bc87ff7ddeadb7e73c218a7 \
  --device cuda:0 --dtype bfloat16 --layers 20,21,22 \
  --max-families 8 --max-steps 10 --lr 0.01 \
  --control-weight 1 --kl-weight 0 --sparsity 0 \
  --allow-next-token-only
```

This is a bounded example, not selected scientific hyperparameters. Train count independently with otherwise matched settings and another output directory. There is no joint optimization or disjointness/orthogonality penalty. Base-model weights stay frozen. `--control-weight 0` gives the primary-only ablation. Optional KL remains patched-to-reference. Mask values are clamped to [0,1], AdamW weight decay is zero, and the existing pre-RoPE QK-SVD kernel is unchanged.

`--max-families` caps each split; the current sampler is not datatype-balanced. `--max-steps` is the total step ceiling, including resumed steps. Resume only the same v2 setup with `--resume /path/to/checkpoint.pt`. Keep the test split disabled during development; `--evaluate-test` explicitly exposes it. Only load trusted checkpoints.

The generic trainer saves local JSON, per-view metrics and checkpoints. The separate optional smoke harness supplies W&B and repeated fixed-collection loss evaluation.

## 6. Optional updated 32B smoke harness (not yet run on v2)

The ZIP includes:

```text
control_supervised_set_count_v2.patch   # cumulative patch against original base
set_count_v1_to_v2.patch                # incremental upgrade for exact v1 code
README.md
verification.json
patch_manifest.json
dataset_summary.json
rendered_family_examples.json
set_count_smoke/runner/                 # updated v2 runner and CPU tests
set_count_design/concrete_examples.json # optional actual-pool preferences
smoke_env/pyproject.toml
smoke_env/uv.lock
historical_v1/                         # old results, explicitly not v2 validation
```

The runner imports your patched clone through `--project-root`; it does not need platform code. It checks only a bounded set of train-split candidate families, validates all 13 views before selecting a family, loads the frozen model once, and can run four independent cases on the same selected examples. For three accepted datatypes this means 39 views, not the historical 27. It saves both blocks' references and all four donor continuations for inspection. Check `missing_datatypes` rather than assuming all datatypes pass.

Use the separate accepted Linux Python 3.11 environment if your repository environment is incompatible:

```bash
export HANDOFF="$(realpath /path/to/unpacked-v2-handoff)"
export REPO="$(realpath /path/to/goals)"
export WORK="/absolute/writable/path/outside/repo/set-count-v2"
mkdir -p "$WORK"
export UV_CACHE_DIR="$WORK/uv-cache" HF_HOME="$WORK/hf-cache"
uv sync --project "$HANDOFF/smoke_env" --python 3.11 --locked
export PY="$HANDOFF/smoke_env/.venv/bin/python"
cd "$REPO"
"$PY" -m pytest -q test/test_set_count.py
SET_COUNT_TEST_PROJECT="$REPO" "$PY" -m pytest -q \
  "$HANDOFF/set_count_smoke/runner/test_real_smoke.py"
```

The accepted lock SHA-256 is `b9f1fc7097c19e5daab5381928f3c2106335243df97a61a9f69d2439183ac32a`, originally resolved with uv 0.8.22. It contains PyTorch 2.11.0+cu128, Transformers 5.18.0, NumPy 2.4.6 and W&B 0.30.0. It targets Linux CUDA, not an unchanged macOS installation. The revised real-model workload has not yet been validated in that environment; local CPU regression checks use the available Python 3.12 environment.

Download the pinned public checkpoint once on CPU if it is not already available:

```bash
export MODEL_DIR="$WORK/Qwen2.5-Coder-32B-Instruct"
"$PY" - <<'DOWNLOAD'
import os
from huggingface_hub import snapshot_download
snapshot_download(repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    revision="381fc969f78efac66bc87ff7ddeadb7e73c218a7",
    local_dir=os.environ["MODEL_DIR"],
    allow_patterns=["*.safetensors", "*.json", "vocab.json", "merges.txt", "tokenizer.model"])
DOWNLOAD
```

Configure W&B credentials locally using your normal secret mechanism or `wandb login`; none are bundled. Select one visible CUDA GPU and run only when you intend to spend compute:

```bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
"$PY" "$HANDOFF/set_count_smoke/runner/run_real_smoke.py" \
  --model-dir "$MODEL_DIR" --project-root "$REPO" \
  --examples-json "$HANDOFF/set_count_design/concrete_examples.json" \
  --output-dir "$WORK/real32b-v2-smoke-01" \
  --experiment-id local-set-count-v2-smoke-01 --steps 12 --layers 20..55
```

The proposed default mirrors the historical optimization smoke: mask init 1, learning rate 0.01, control weight 1, KL/sparsity 0, fixed evaluation at 0/3/6/9/12. It now uses equal-block weighting. The runner reports loss descent rather than asserting a pass or changing labels to force one. `--resume` is for compatible v2 runs only, with the same output directory and a total `--steps` ceiling. It is not interchangeable with the generic trainer's checkpoint interface.

The prior full-BF16 run used one H100 80GB and 128 GiB host RAM; weights alone occupy about 65.53 GB. The new prompt collection and runtime need fresh memory/behavior checks. The generic trainer computes full-sequence logits and can use more memory than the last-token-only smoke adapter.

## 7. Historical v1 smoke: retained evidence, not v2 results

On the actual `Qwen2.5-Coder-32B-Instruct`, the **old three-donor, result-stub design** ran 12 optimizer steps per mask on 27 calibration views. Its independently checked cross-entropy was:

| v1 case | Total CE before → after | Intervention CE before → after | Control CE before → after |
|---|---:|---:|---:|
| Set / head | 2.648 → 2.298 | 7.087 → 6.461 | 0.4290 → 0.2167 |
| Count / head | 4.852 → 4.345 | 7.107 → 6.429 | 0.3418 → 0.1777 |
| Set / SVD | 2.656 → 1.566 | 7.099 → 4.510 | 0.4345 → 0.09359 |
| Count / SVD | 4.848 → 2.993 | 7.098 → 4.437 | 0.3492 → 0.1053 |

All three losses decreased at each recorded checkpoint, but final intended non-native top-token success was zero (0/9 set, 0/18 count). Native controls were correct at the top token. This established optimization mechanics, not working selectors.

The three sampled source-A continuations assembled final tuples/lists at `result =`; one list-equality reference also disagreed with the bank on empty inputs. v2 removes the semantic output name, but **does not claim that these behavior issues are solved**. Full execution validation remains necessary before scientific claims.

Historical evidence:
- GPU: `artifact://researcher-david-bau/experiments/exp_01m427ngjbebxa908fyd2kg57z/`
- Independent verification: `artifact://researcher-david-bau/experiments/exp_01m4283tc9epe8xw1xbe91r5jp/`
- W&B: https://wandb.ai/reasoning-iterp/default-exp-exp_01m427ngjbebxa908fyd2kg57z/runs/8wr9jzyv

Small historical summaries/curves are included under `historical_v1/`. Large weights and trained masks are not bundled. The earlier actual-model results must not be relabeled as evidence for these revised prompts, prefixes, splits or loss weights.
