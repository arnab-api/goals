# Set/count control-supervised localization: patch and smoke-test handoff

This handoff is a patch against `arnab-api/goals` commit `8c59544e5d04`. The managed repository could be read but Silico could not create its supported writable editing workspace (`repo-write.lock` permission denied). Therefore **nothing was committed, pushed, or applied to the managed checkout**. Apply the patch in your own writable clone on a branch and review it through a pull request.

## Scope

- Deterministic data builder using the existing contrastive pivot pool and tested predicate implementations for `str`, `int`, and `list[int]`.
- Shared donor families, informative destination expression prefixes, randomized completed-output names, and the same neutral `result =` source stub.
- **Independent** set and count masks. There is no disjointness, orthogonality, or cross-mask penalty.
- Both whole-head query interpolation and the existing QK-SVD-coordinate sub-head interpolation, at the last pre-RoPE query position.
- Explicit expected-token supervision for native-expected controls, rather than merely discouraging the alternative token; `--control-weight 1` enables it, and `0` provides a matched primary-only ablation.
- Pair-grouped train/dev/test splits, dataset fingerprints, raw evaluation rows, and identity-checked checkpoint/resume.

This is a first implementation, not a completed localization experiment. Two different checks have now run: an offline/tiny-random-model mechanics smoke, and a **12-step optimization smoke on the actual pretrained 32B model**. The real-model smoke reduced both intervention and control losses in all four cases, but final non-native top-token success remained zero. See the measured results and reproduction instructions below. The implementation patch is unchanged from the tested version; the exact 32B runner is included separately in the handoff ZIP.

## Data configurations

Source inventory is always `[a,b,d,e]`; the design assigns `b` as the first unfinished source task. This is the intended semantic role, not a validated statement that generation at `result =` begins with a standalone implementation of `b`; the real-model smoke exposed an aggregation behavior at that boundary.

| Donor | Completed tasks | Next source task |
|---|---|---|
| A | `{a,d}` | `b` |
| B | `{a,e}` | `b` |
| C | `{a}` | `b` |

| Recipient configuration | Recipient, with `a` done | Set predictions A/B/C | Count predictions A/B/C |
|---|---|---|---|
| `core` | `[a,d,e]` | e / d / d | e / e / d |
| `swapped` | `[a,e,d]` | e / d / e | d / d / e |
| `four` | `[a,d,e,f]` | e / d / d | e / e / d |

Letters identify semantic tasks, **not** rendered variable names. Names are sampled independently of role/count, remain consistent across matched donors, and all prompts keep the neutral unfinished `result` variable. Each recipient and its references retain the exact expression prefix, including significant trailing whitespace such as `n % `.

With the current pool and `--n-per-pivot 1 --naming-variants 1`, the builder retains 325 families across 55 unordered operation pairs and 2,925 donor/recipient views before model validation. These contain 120 string, 74 integer, and 131 integer-list families.

An unordered diagnostic operation pair is confined to one split. Both orientations, all prefixes for that pair, role assignments, naming variants, and matched donors stay on that side. These splits are fresh for this implementation; they are not evidence that the underlying bank has never been inspected before. Cosmetic variants are not independent samples.

The builder rejects binding-name-only contrasts under a separate eligibility policy while retaining method-name, literal, and operator distinctions, including builtin operations such as `any` versus `len`. It does not alter or delete the raw pivot pool. Naming and data fingerprints are deterministic.

## Files in the patch

- `src/diff/set_count_data.py`: family construction, contextual eligibility, serialization and fingerprints.
- `src/diff/query_mask_ops.py`: shared existing head/SVD query kernels, with standard PyTorch module lookup.
- `src/diff/set_count_train.py`: standalone single-device Qwen2 adapter, validation, independent training, evaluation, checkpoint/resume.
- `scripts/diff/build_set_count_data.py` and `train_set_count_mask.py`: command-line entrypoints.
- `test/test_set_count.py`: focused offline regression tests.
- `src/diff/nrgs_mask.py` and `nrgs_svd_mask.py`: re-export the factored kernels; original public function names and formulas are retained.

The new path does not import the old `nnsight`/`baukit` model wrappers. Existing callers continue using their existing environment. No repository dependency pins were changed.

## Apply and check

From your writable clone:

```bash
git switch -c control-supervised-set-count
# Copy control_supervised_set_count.patch into, or next to, this checkout.
git apply --check /path/to/control_supervised_set_count.patch
git apply /path/to/control_supervised_set_count.patch
uv run python -m pytest -q test/test_set_count.py
```

If your repository environment cannot run this new standalone path, use the separate locked Linux environment included in the ZIP (instructions below). It does not change the repository dependency pins.

The apply check was performed against the named base commit. If your branch has changed these files, inspect/reconcile the diff instead of forcing it. Commit on the branch and open a pull request after your checks.

## Build data (no model download)

Run from the repository root using the repository's Python 3.11 environment:

```bash
uv run python scripts/diff/build_set_count_data.py \
  --model Qwen/Qwen2.5-Coder-32B-Instruct \
  --n-per-pivot 1 --naming-variants 2 --seed 0 \
  --output results/set_count/families-v1.json
```

`--model` selects the existing on-disk pool; it does not download a model. `--n-per-pivot` expands auxiliary task assignments, not the number of independent operation pairs. Increasing `--naming-variants` tests robustness to names, not new semantics. The output preserves unvalidated status and provenance. Do not interpret its size as a count of usable model-validated examples.

## Offline mechanics smoke

```bash
uv run python scripts/diff/train_set_count_mask.py \
  --dataset results/set_count/families-v1.json \
  --output-dir results/set_count/smoke-set-head \
  --hypothesis set --mask-kind head --smoke-random --mask-init 0.5 \
  --device cpu --max-families 1 --max-steps 2
```

Use `--hypothesis count` for the other independently optimized mask and `--mask-kind svd` for sub-head components. Use `--device cuda:0` for the GPU smoke. Random-smoke weights are float32, regardless of the pretrained `--dtype` setting, and runtime metadata records that explicitly. The smoke uses `--mask-init 0.5` to test an optimizer update away from a constraint boundary. The normal training default remains `1.0`, preserving full-transfer initialization; at that boundary a nonzero gradient can legitimately be projected back to an unchanged mask. No pretrained weights or external credentials are needed.

## Actual 32B smoke: completed results

The pretrained smoke used `Qwen/Qwen2.5-Coder-32B-Instruct`, revision `381fc969f78efac66bc87ff7ddeadb7e73c218a7`, on one H100 80GB. All 64 layers ran with frozen BF16 weights and eager attention. Trainable masks covered query layers 20–55 inclusive: 1,440 head coefficients or 184,320 QK-SVD-coordinate coefficients. This is not a parameter-count-matched comparison of mask types.

Fixed settings: independent set/head, count/head, set/SVD, count/SVD cases; mask initialization 1.0; AdamW learning rate 0.01, weight decay 0; native-control multiplier 1; KL and sparsity weights 0. Each case ran 12 steps, cycling one complete family per step. The same 27 matched views (three datatypes × three recipient configurations × three donors) were evaluated at steps 0, 3, 6, 9, and 12. These are **three calibration families, not 27 independent samples or a held-out test**. All three predetermined first candidates passed real-context next-token validation; no alternative candidate was needed.

Cross-entropy in nats, before → after:

| Hypothesis / mask | Total CE | Non-native intervention CE | Native-control CE |
|---|---:|---:|---:|
| Set / head | 2.648 → 2.298 | 7.087 → 6.461 | 0.4290 → 0.2167 |
| Count / head | 4.852 → 4.345 | 7.107 → 6.429 | 0.3418 → 0.1777 |
| Set / SVD sub-head | 2.656 → 1.566 | 7.099 → 4.510 | 0.4345 → 0.09359 |
| Count / SVD sub-head | 4.848 → 2.993 | 7.098 → 4.437 | 0.3492 → 0.1053 |

All three loss columns fell at every recorded evaluation checkpoint; this does not claim monotonic loss on every optimizer step. An independent CPU check recomputed the metrics from per-view records and verified identical example identities. Unpatched total CE was 3.408 for set and 6.815 for count; unpatched control CE was 0.0005364. Thus the trained masks improve the counterfactual objective, but native-control confidence remains worse than the clean baseline.

**Final intended non-native top-token success was 0/9 for set and 0/18 for count, for both mask types.** Native-control top-token accuracy was 100%. This verifies optimization and differentiable wiring, not successful set/count localization, sparse component selection, convergence, or generalization.

Other checks passed: finite nonzero mask gradients on every step; masks remained in [0,1]; zero masks exactly reproduced clean readouts; last-token-only logits exactly matched the ordinary forward's last token on the checked prompt; base parameters remained frozen with no gradients; checkpoints were saved; deployed source hashes matched the reviewed code. Peak allocated GPU memory was about 67.45 GB (decimal) in this small run; longer prompts/batches require a separate memory estimate. The tested environment was Python 3.11.17, PyTorch 2.11.0+cu128, Transformers 5.18.0, NumPy 2.4.6, and W&B 0.30.0.

### Important semantic caveats discovered by this smoke

- All three inspected source-A continuations after the neutral `result =` started assembling a final tuple/list, rather than a standalone expression for one pending task. For example: `result = (w, lst == sorted(lst), t, all(x % 2 == 0 for x in lst))`. The unfinished task appears inside the aggregate, but the captured boundary may also represent output assembly. Validate or revise this donor boundary before interpreting masks as single-operation selectors. **This handoff does not silently fix or change that renderer.**
- The generated list equality reference used `all(x == lst[0] for x in lst) if lst else False`, whereas the bank's empty-list predicate is true. Distinct diagnostic tokens do not establish executable correctness.
- Only representative clean continuations were inspected; there is no full source/recipient/teacher execution-validation suite, no patched free-generation evaluation, and no held-out mechanism claim.
- The fixed-expression recipient prefix remains essential. Fixing donor semantics does not mean replacing the recipient's informative prefix with indentation.

### Reproduce the exact real-model smoke from the ZIP

The ZIP includes the **unchanged eight-file implementation patch**, this README and provenance, plus:

```text
set_count_real_smoke/runner/run_real_smoke.py   # the actual four-case runner
set_count_real_smoke/runner/smoke_helpers.py    # fixed-example aggregation and selection
set_count_real_smoke/runner/test_real_smoke.py  # 12 small CPU tests
set_count_real_smoke/PROTOCOL.md               # recorded smoke protocol
set_count_design/concrete_examples.json       # identities/prefixes from the existing pool
smoke_env/pyproject.toml                      # standalone Python 3.11 Linux environment
smoke_env/uv.lock                             # accepted environment, including W&B
smoke_evidence/verification_summary.json       # independently checked real-model metrics
smoke_evidence/loss_curves.csv                # fixed-example curves
smoke_evidence/representative_continuations.json
```

The runner stays outside your repository unless you choose to add it. It imports the patched training code using `--project-root`. It does **not** require Silico, its job tools, or any platform source. An unavailable Silico telemetry helper only emits a warning; normal W&B logging continues. The generic `train_set_count_mask.py` CLI saves local JSON/checkpoints; the separate exact-smoke runner supplies W&B and fixed-collection evaluation curves.

Use a Linux CUDA machine with one visible GPU large enough for the full BF16 checkpoint. The tested allocation had 128 GiB host RAM and one H100 80GB. Prepare sufficient disk space for approximately 65.53 GB of weights, the environment, scratch, and output checkpoints. Do not expect these GPU instructions or the Linux CUDA lock to work unchanged on macOS.

After applying the patch, set paths and create the separate environment:

```bash
export HANDOFF="$(realpath /path/to/unpacked-handoff)"
export REPO="$(realpath /path/to/goals)"
export WORK="/absolute/path/outside/the/repo/set-count-smoke"
mkdir -p "$WORK"
export UV_CACHE_DIR="$WORK/uv-cache"
export HF_HOME="$WORK/hf-cache"
uv sync --project "$HANDOFF/smoke_env" --python 3.11 --locked
export PY="$HANDOFF/smoke_env/.venv/bin/python"
cd "$REPO"
"$PY" -m pytest -q test/test_set_count.py
SET_COUNT_TEST_PROJECT="$REPO" "$PY" -m pytest -q   "$HANDOFF/set_count_real_smoke/runner/test_real_smoke.py"
```

The accepted lock SHA-256 is `b9f1fc7097c19e5daab5381928f3c2106335243df97a61a9f69d2439183ac32a`. Keep that lock rather than resolving a different stack and calling it the same environment. Use an installed compatible `uv`; the original environment was resolved with uv 0.8.22. A clean rebuild may still require a compatible NVIDIA driver.

Download the pinned public checkpoint once; this can run on a CPU-only machine before transferring the completed directory to the GPU host:

```bash
export MODEL_DIR="$WORK/Qwen2.5-Coder-32B-Instruct"
"$PY" - <<'DOWNLOAD'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    revision="381fc969f78efac66bc87ff7ddeadb7e73c218a7",
    local_dir=os.environ["MODEL_DIR"],
    allow_patterns=["*.safetensors", "*.json", "vocab.json", "merges.txt", "tokenizer.model"],
)
DOWNLOAD
```

Configure your W&B credentials on that machine through your normal secret mechanism or `wandb login`; **the ZIP contains no credentials**. Select the desired W&B account/entity if needed. The run project will be `default-exp-<experiment-id>`. Then, from the repository root:

```bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
"$PY" "$HANDOFF/set_count_real_smoke/runner/run_real_smoke.py"   --model-dir "$MODEL_DIR"   --project-root "$REPO"   --examples-json "$HANDOFF/set_count_design/concrete_examples.json"   --output-dir "$WORK/real32b-smoke-01"   --experiment-id local-set-count-32b-smoke-01   --steps 12 --layers 20..55
```

Use a fresh output directory for a new run. Resume with the same arguments/output directory and `--resume`; `--steps` is the total step ceiling, not an increment. The runner rejects incompatible checkpoint identities. Do not mix these checkpoints with the generic trainer CLI's resume interface. For a local rerun, the protocol records the requested revision and tokenizer/config hashes, but without the original staging manifest it cannot independently attest how the weight files arrived: preserve the pinned download provenance. Local W&B credentials are separate from your Silico Settings.

The exact runner validates authentic next-token readouts and caches source queries before optimization, uses the same selected collection for all four cases, and saves `protocol.json`, selection/rejection records, clean baselines, representative continuations, `summary.json`, W&B identity, and per-case progress/initial and final checkpoints. It requests only the last-token logits, checks equivalence once, and loads the base model once across cases. Changing the pool, tokenizer, naming renderer, or software stack may change accepted families and outcomes; do not claim exact numerical replication without checking the recorded identities.

Evidence from the completed run:
- GPU smoke: `artifact://researcher-david-bau/experiments/exp_01m427ngjbebxa908fyd2kg57z/`
- Independent verification: `artifact://researcher-david-bau/experiments/exp_01m4283tc9epe8xw1xbe91r5jp/`
- Accepted environment: `artifact://researcher-david-bau/experiments/exp_01m427a7cjfsxv37cv6bzrzqtb/environment/`
- W&B: https://wandb.ai/reasoning-iterp/default-exp-exp_01m427ngjbebxa908fyd2kg57z/runs/8wr9jzyv

These `artifact://` references are durable Silico addresses, not local filesystem paths. The ZIP includes the small evidence files and environment lock so you do not need to resolve those addresses to use this handoff. Large model/mask checkpoints are retained in the run outputs, not bundled.

## Validate real-model readouts before exploratory training

```bash
uv run python scripts/diff/train_set_count_mask.py \
  --dataset results/set_count/families-v1.json \
  --output-dir results/set_count/readout-check \
  --hypothesis set --model Qwen/Qwen2.5-Coder-32B-Instruct \
  --revision 381fc969f78efac66bc87ff7ddeadb7e73c218a7 --device cuda:0 --dtype bfloat16 \
  --max-families 8 --validate-only
```

This uses the same pinned revision as the completed real-model smoke. Budget memory/downloads separately: the exact-smoke runner uses a last-token-logits optimization, while the generic CLI computes the full sequence logits and can use more memory. The adapter intentionally supports one fully materialized device and Qwen2, not automatic multi-GPU dispatch.

The validator recomputes d/e token IDs in each new recipient/reference context, rejects coincident or binding-only readouts, verifies the clean recipient agrees with its native reference, and retains/rejects whole matched families. Stored pool tokens are provenance and a diagnostic comparison, not assumed labels.

**It does not execute model completions or verify source/teacher task correctness.** This matters because a distinct next token may still be an incorrect implementation. Non-smoke training is blocked unless `--allow-next-token-only` explicitly acknowledges this limitation. The flag permits an exploratory wiring run; it is not a substitute for full semantic validation.

## Independent exploratory optimization

```bash
uv run python scripts/diff/train_set_count_mask.py \
  --dataset results/set_count/families-v1.json \
  --output-dir results/set_count/set-svd-exploratory \
  --hypothesis set --mask-kind svd \
  --model Qwen/Qwen2.5-Coder-32B-Instruct \
  --revision 381fc969f78efac66bc87ff7ddeadb7e73c218a7 --device cuda:0 --dtype bfloat16 \
  --layers 20,21,22 --max-families 8 --max-steps 10 \
  --lr 0.01 --kl-weight 0 --sparsity 0 \
  --allow-next-token-only
```

The bounded settings above are an example, not selected scientific hyperparameters or a replication of previous training. `--max-families` caps **each split**; defaults deliberately do not train the entire pool. `--max-steps` is a total step ceiling, including resumed steps. One step averages all donor/configuration/name views of one family; families cycle deterministically. Within a family, configurations and donors are equally weighted. The present sampler is not balanced across datatypes: choose that policy before a large scientific run.

Repeat independently with `--hypothesis count`, the same dataset/split/seed/candidate layers, and a distinct output directory. Base-model parameters remain frozen and in evaluation mode. `--control-weight` defaults to `1`; setting it to `0` removes native-expected losses without rescaling primary losses. Compare separate runs with identical data and settings to measure the effect of control supervision. Optional KL uses patched-to-expected-reference orientation. AdamW weight decay is zero; masks are clamped to [0,1]. The SVD basis ignores RoPE, as in the existing implementation; it is generally not an orthogonal query-space projector.

Resume using `--resume PATH/checkpoint.pt` with the same scientific settings and a larger total `--max-steps`. Identity mismatches are rejected. Only load trusted checkpoints; PyTorch checkpoint loading here is not a sandbox for untrusted pickle files.

## Outputs and interpretation

Each training directory contains runtime/config/identity metadata, validation records and rejected families, training events, a checkpoint (including optimizer/RNG and SVD bases when applicable), raw per-view metrics, and split summaries. The test split is not accessed unless `--evaluate-test` is explicitly requested. Keep it disabled during development and smoke testing if you intend to retain it as an untouched final evaluation. The exact 32B smoke uses explicitly designated calibration examples, not an untouched test set.

Look at the full expected pattern across A/B/C and recipient orders, native correctness, target correctness, and log-odds. A drop in alternative-token flips alone is not sufficient: it could mean corrupted output. Whole-family correctness is reported alongside per-view accuracy.

## Remaining work before scientific conclusions

1. Validate full source, recipient, and teacher completions against predicate fixtures in the actual multi-task contexts; implement rejection/provenance without silently relabeling failed cases.
2. Extend the successful real-tokenizer/BF16/32B checks beyond the three smoke families. Resolve the observed source-boundary aggregation and empty-list semantic mismatch; validate the intended operation in full contexts, not only its next token.
3. Add prospective nuisance controls (irrelevant assignments/comments, source-validity checks) and the held-out new-count structural test. These are not implemented by this initial patch.
4. Choose sampling balance, layer range, sparsity and control weights on development data; retain an untouched final evaluation.
5. Add per-datatype/per-condition aggregation and pair-clustered uncertainty before reporting localization claims. The current summaries are basic counts and accuracies, not inferential statistics.

See `verification.json` for the actual tests completed, and `source_manifest.json` for the staged base identity. GPU outputs are separately retained under the execution reference recorded there.

## Earlier mechanics verification (distinct from the pretrained smoke above)

- **26 focused tests passed** locally and again in the Python 3.11 on-demand job.
- All four CUDA combinations passed: set/head, set/sub-head, count/head, count/sub-head. Each executed two optimization steps with finite nonzero mask gradients, bounded mask updates, saved checkpoints, and checkpoint reload.
- A resumed set/head run matched uninterrupted three-step training exactly in the saved mask.
- The successful job's hashes match all eight source files in the delivered patch; its dataset fingerprint matches the local builder output.
- Independently compared the factored kernels with the original repository implementation: bases, forward outputs, and mask gradients matched exactly in the small FP32 GQA check.
- `git apply --check --whitespace=error` passed against `8c59544e5d04`; the original checkout remains unchanged.

Compute evidence: `artifact://researcher-david-bau/experiments/exp_01m425s8eme8n9b33jc14bszyt/`. The successful job used Python 3.11.17, PyTorch 2.11.0+cu128, Transformers 5.18.0, and one H100. It used no pretrained weights. The accepted dependency recipe/lock and full environment manifest are in that output prefix's `environment/` directory.

The first GPU attempt is also retained at `artifact://researcher-david-bau/experiments/exp_01m425daxyfv79288q8w7vqnrs/`. It stopped at a smoke assertion requiring a changed count/head mask. Inspection confirmed every optimizer update direction was outward or zero at the upper mask boundary, so clamping correctly left the mask at 1. The revised smoke tests actual movement from 0.5; it does not alter the production initialization default or any scientific scoring rule.
