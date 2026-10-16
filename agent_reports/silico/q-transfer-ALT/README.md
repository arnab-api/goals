# q-transfer-ALT: opt-in portable package

An additive package for repository base `7ef5101a7a1e28acbc70f0c4f8bf2d73aca6fde7`. It combines the expanded 128-task Python bank with the completed ALT1/ALT2/ALT3 prefill-only gaze protocol. It changes no legacy predicate bank, sampler, script, global dependency pin, `src/attention.py`, or `src/diff/functional.py`.

This is a packaging deliverable, not a new scientific run. Historical full-model checks belong to experiment `exp_01m4f3g9npexwv2rexndg2kcp7`. Fresh verification covers CPU fixtures, imports, panel reproduction, patch application, and CLI behavior only. No model weights, generated outputs, historical results, or benchmark claims are included.

## Apply and verify

Run from a clean, owned checkout. Create your own branch; this package does not push, merge, or create a pull request.

```sh
git switch -c add-q-transfer-alt 7ef5101a7a1e28acbc70f0c4f8bf2d73aca6fde7
git apply --check /path/to/q_transfer_alt.patch
git apply /path/to/q_transfer_alt.patch
python -m unittest discover -s test/q_transfer_alt -v
python scripts/diff/q_transfer_alt_build.py --help
python scripts/diff/q_transfer_alt_run.py --help
python scripts/diff/q_transfer_alt_analyze.py --help
```

Use a Python environment with NumPy and CPU PyTorch to execute all CPU fixture tests. Model-free utilities and CLI help use the standard library alone. The full test suite requires these dependencies; it does not silently substitute model validation for missing packages. The accompanying `verification_manifest.json` and test log identify the actual environment used for this delivery.

The source ZIP contains the same additive repository-relative files as the patch. Extract it into the pinned checkout *instead of* applying the patch. Do not use the ZIP to overwrite an already modified package without inspecting your differences.

## Build the original grid and dry-run

```sh
python scripts/diff/q_transfer_alt_build.py \
  --preset original-grid --output /path/to/panel.json
python scripts/diff/q_transfer_alt_run.py \
  --panel /path/to/panel.json --output /path/to/dry-run --dry-run
```

The builder uses the packaged bank, seed preset, head list, and 16 closed smoke-exclusion signatures. It does not read the frozen panel to construct the new one. It reproduces every original group, task assignment, prompt component, opaque name, prediction, and scientific setting. New bank implementation and packaging provenance are deliberately different. `configs/q_transfer_alt/panel.original.json` retains the exact historical bytes (SHA-256 `c8fdfc60501fcfe5b020684798d81ebe19c0dd6333e0194be586a458dcb6808d`). Use the original only for explicitly historical analysis; either valid panel may be supplied to a new run, which always gets a new runtime recipe identity.

The model-free dry run checks the frozen panel and reports intended coverage. It does **not** tokenize inputs, instantiate a model, allocate a GPU, or verify compatibility with a local checkpoint. Token-map tests use small deterministic fixtures, not the production tokenizer.

## Supported inference environment (not installed by the patch)

The supported historical path is Linux, Python 3.11, Transformers 5.18.0, torch 2.11.0+cu128, and NumPy 2.4.6. The exact historical dependency closure is retained in `configs/q_transfer_alt/environment/uv.lock`. Its project name is retained to keep the lock unchanged; it is not the repository's global environment. Always use the frozen lock rather than resolving its broader project ranges.

On a separately provisioned CUDA machine, if you choose to run inference:

```sh
uv sync --project configs/q_transfer_alt/environment --frozen --python 3.11
PY="$PWD/configs/q_transfer_alt/environment/.venv/bin/python"
"$PY" -m unittest discover -s test/q_transfer_alt -v
```

These setup and inference commands were not executed as part of this CPU-only packaging task. They may download dependencies but the runner never downloads model weights. The package does not promise compatibility with other Transformers cache APIs, model architectures, quantized weights, sliding-window attention, custom position IDs, concurrent patch contexts, or generation/decode workflows. The scoped runtime avoids the repository's legacy setup imports.

## Use a locally staged checkpoint

Stage `Qwen/Qwen2.5-Coder-32B-Instruct`, revision `381fc969f78efac66bc87ff7ddeadb7e73c218a7`, using your own trusted process. Supply its complete local config, tokenizer, index, and safetensor shards. No Hugging Face key or platform environment variable is required by these commands.

```sh
"$PY" scripts/diff/q_transfer_alt_manifest.py \
  --model-path /path/to/local-checkpoint --output /path/to/model-manifest.json
"$PY" scripts/diff/q_transfer_alt_run.py \
  --panel /path/to/panel.json \
  --model-path /path/to/local-checkpoint --model-manifest /path/to/model-manifest.json \
  --output /path/to/grid --pilot
# Omit --pilot for the complete grid, using a separate output directory.
```

The manifest hashes the files in place; it does not load the weights. The intended revision is an explicit declaration, not independent proof that local weights originated from that remote revision. The hashes bind subsequent execution and resume to the exact locally supplied checkpoint and tokenizer. Preserve a trusted download receipt if independent source authentication is needed. Reading all model shards for hashing can take time and requires no GPU memory.

`--pilot` selects index 0 of each datatype: 4 matched groups, 12 configurations. A full run selects 128 groups and 384 configurations. Both use a single CUDA device, BF16 eager attention, batch size one, and no generated tokens. Fresh integration with the full model remains untested in this packaging task.

## Resume safely

```sh
"$PY" scripts/diff/q_transfer_alt_run.py \
  --panel /path/to/panel.json \
  --model-path /path/to/local-checkpoint --model-manifest /path/to/model-manifest.json \
  --resume /path/to/previous-grid --output /path/to/resumed-grid
```

Keep `--pilot` consistent when resuming a pilot. Completed groups require matching panel, group, package/lock recipe, model/tokenizer identity, execution options, and checksums of all referenced products. Corrupt or incompatible completed groups fail; unfinished groups are recomputed without mixing partial files. Paths are not scientific identity, so relocation of identical inputs is supported. Historical `complete.json` recipe hashes are intentionally incompatible with the newly packaged runtime. Analyze historical data offline instead of resuming it as new inference.

## Offline export and matched analysis

No model, tokenizer, CUDA, or PyTorch is needed for saved-data analysis; NumPy is required.

```sh
python scripts/diff/q_transfer_alt_analyze.py \
  --panel /path/to/panel.json --grid /path/to/grid --output /path/to/analysis
# Add --pilot only for a pilot producer.
```

For existing historical data, extract the four `raw_*.zip` archives plus `tokenmaps_checks.zip` into a single grid tree with `scores.json`, `environment.json`, and `groups/`. If a download wrapper adds a directory level, point `--grid` at the directory that actually contains those files. Use the original frozen panel and explicitly label the analysis historical:

```sh
python scripts/diff/q_transfer_alt_analyze.py \
  --historical --panel configs/q_transfer_alt/panel.original.json \
  --grid /path/to/restored-historical-grid --output /path/to/historical-analysis
```

Historical archives are external inputs, not bundled here. Their durable source prefix is `artifact://researcher-david-bau/experiments/exp_01m4f3g9npexwv2rexndg2kcp7/downloads/` (raw_int.zip, raw_str.zip, raw_list_int.zip, raw_list_str.zip, tokenmaps_checks.zip). The original code source is `gaze_grid_code.zip` under that prefix. The bank source is `artifact://researcher-david-bau/experiments/exp_01m4ees8p7ewza1gzqxkp2xxgv/diff_expanded/diff_expanded_source.zip`.

Validation fails closed on missing, duplicate, mismatched, or corrupt records and mixed recipes. The complete original grid requires 32 groups per datatype, not a silently accepted partial subset. Analysis exports all task means/counts/roles/winners, exact ties, configuration rows, matched ALT triplets, and paired summaries. Its original linked-bootstrap procedure resamples matched groups together rather than treating 384 configurations as independent. Raw float64 reaggregation is a diagnostic against saved FP32 scores with absolute tolerance 1e-6, not a new primary score or an outcome filter.

## Protocol and data contracts

- Bank: 128 Python definitions, 32 each for `int`, `str`, `list[int]`, and `list[str]`, with independent reference functions, output types, input domains, operation families, and definition hashes. No C++ implementations are added.
- Grid: 32 seeds per datatype (`31000..31031`, `41000..41031`, `51000..51031`, `61000..61031`), reused in matched ALT1/ALT2/ALT3 groups. Nine task IDs are distinct; lists require nine distinct families; scalars require distinct families for `bcdef` while supporting `axyz` may overlap.
- Roles: source list `abcd`; ALT1/2 written order `abd`; ALT3 written order `abc`. Donor lists `abde`, `adef`, `acbe`, each with `a` written. Original recipient `axyz`, with `a` written. Written order is explicit and is not inferred from list order or variable numbering.
- Predictions across ALT1/2/3: set `eee`, order `dee`, count `efe`, successor `eeb`, KV baseline `bdc`. Off-inventory source-copy and original-recipient-count predictions are not observable donor-task gaze classes.
- Opaque names are independent of count. All three donors and the common recipient are aligned using inserted newline token IDs; sources remain unchanged. Padded strings must **not** be retokenized. Exact input IDs, offsets, description spans, and donor-to-recipient maps are retained.
- Heads: the exact original 250 heads, head-file SHA-256 `6b7c52d21a3cfc75c718a6c618e8f02f15a3372d00434497fbf427923978a96b`.
- Conditions: fresh donor KV for each scenario, then KV-only versus KV plus source query. Replace only selected query-head slices at the final equals token before RoPE. No OV boost, no generation, no GoalScope, no importance filtering.
- Gaze: full final-query causal attention multiplied by the L2 norm of the **effective donor-branch values**. Primary scores are FP32 means across all selected heads and description tokens; sums are diagnostic only. Exact ties are retained. Source and clean-recipient gaze are diagnostics, never exclusion criteria.
- Raw NPZ: `head_ids`, `attention[250,T]`, `value_norm[250,T]`, `head_output[250,head_dim]`. Attention/value arrays retain lossless FP32 representations of the captured values. JSON keeps exact IDs, padded spans, hashes, roles, predictions, and relative raw references. Each group includes `materialized.json`, `scores.json`, `checks.json`, and a checksummed `complete.json`.
- Source-query reuse is limited to identical exact input IDs and runtime recipe; ALT1/ALT2 share a source, ALT3 does not. Hook/forward restoration is scoped to context exit, including exceptions.

The tests and sample prompts are examples of interfaces, not new model measurements. See the accompanying delivery manifest for source identities, additive file hashes, application checks, and checks not run.
