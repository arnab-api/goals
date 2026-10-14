# Context-aware pivot-token filter patch

## Deliverable and status

- Patch: `token_filter_context.patch`.
- Base repository snapshot: `arnab-api/goals`, commit `74f5d2fc57c7`.
- Five changed files: one pure-Python helper, three caller/cache files, and a regression-test file.
- Not applied to the managed repository; no commit or PR has been created.
- The patch passed `git apply --check` against the base checkout.
- Twelve dependency-free regression tests and syntax checks passed locally with Python 3.12. No model, CUDA, API, mining, or training run was performed.

This patch fixes the token-filter issue. It does not change mask losses, head selections, sparsity coefficients, task definitions, or the shared expression prefix. It also does not add the broader multi-task completion-validation work discussed in the audit.

## What changes

1. **Read the candidate in context.** Compare the Python lexical structure of `prefix + next_token`, so a subword fragment following a dot or an opening quote is not mistaken for a free variable.
2. **Preserve meaningful text.** Attribute/method names, operators, numbers, string contents, and unknown identifiers are retained. Whitespace and comments outside literals are ignored for this expression-level filter.
3. **Normalize known variable roles only.** Where already-generated complete code is available, function arguments and simple local/comprehension bindings supply roles. Input arguments and different locals are not all collapsed together. Ambiguous scopes or unknown names are kept conservatively.
4. **Reuse existing completions.** The original completion check optionally returns the full passing code to the filter; this adds no generation calls to that check.
5. **Use the filter consistently.** Raw Case 3b candidates, validated base pivots, wrapper augmentation, and the NRGS next-token check all pass the prefix and known names. Wrapper augmentation previously checked only token inequality.
6. **Do not silently reuse old acceptance decisions.** NRGS gate records carry `TOKEN_FILTER_VERSION`; older records are backed up and rechecked. The version also appears in the generated dataset metadata, changing the existing training-cache hash.
7. **Remove stale accepted pivots when a revalidated pair has no survivors.** Only the selected pairs are replaced; unselected pairs are preserved. Back up the original pool before re-mining.

The old `_tokens_differ_meaningfully` import path remains available as an alias in `head_loc_oracle.py`. Two-token calls still work, but without context/known-variable metadata the helper is intentionally conservative. All inspected production call sites are updated.

## Apply on a branch

Run these commands in your writable repository, not in Silico's read-only resource checkout:

```bash
git switch -c fix/contextual-pivot-token-filter

PATCH=/absolute/path/to/token_filter_context.patch
git apply --check "$PATCH"
git apply "$PATCH"

python -m unittest discover -s test -p 'test_pivot_token_filter.py' -v

git diff --check
git diff --stat
```

If `git apply --check` fails because your repository has changed since the base commit, stop and port the changes against your current version. Do not force the patch through or discard unrelated edits.

Review and commit only these files:

```bash
git add src/goals/pivot_token_filter.py \
        src/goals/head_loc_oracle.py \
        src/diff/nrgs_prefix.py \
        scripts/diff/mine_nrgs_prefix.py \
        test/test_pivot_token_filter.py

git commit -m "Fix contextual semantic filtering of pivot-token contrasts"
git push -u origin fix/contextual-pivot-token-filter
```

Open a pull request using your usual GitHub workflow. Do not push directly to the default branch. No author identity is supplied here; use your configured Git identity.

## Test coverage

The regression suite runs without importing torch, model code, or credential configuration. It tests:

- Figure 9's `.lower` versus `[::-` contrast;
- `.lower` versus `.upper`, and split attribute-name tokens;
- closed, open, raw, and triple-quoted literal fragments, including significant literal spaces;
- known input aliases, local/comprehension alpha-renaming, and preservation of distinct argument/local roles;
- conservative handling of unknown identifiers and ambiguous reference scopes;
- whitespace-only and comment-only differences;
- preservation of operators, numbers, builtins, and keywords;
- prefix propagation at every production filter call site;
- reuse of the two existing completion results without extra generation calls;
- candidate-validator integration;
- clearing a selected pair with zero surviving pivots without changing unselected pairs;
- backing up and invalidating older NRGS gate decisions.

The lexical filter remains a heuristic. Passing it is not proof of operation identity or code correctness. Keep the existing model-token inequality and executable completion checks.

## Revalidate the data deliberately

Previously rejected prefixes are absent from the saved pool. Merely retraining masks on that pool will not recover them.

### 1. Preserve the old pool and results

Before re-mining, copy the current model-specific pool to a separate backup location, and retain historical datasets, masks, and scores. For example:

```bash
MODEL_SHORT=Qwen2.5-Coder-32B-Instruct
BACKUP_DIR=/absolute/path/to/research-backups/pre-contextual-token-filter
mkdir -p "$BACKUP_DIR"
cp -n "data/head_loc/pairs/${MODEL_SHORT}.json" "$BACKUP_DIR/"
```

Check that your backup exists before continuing. Do not add backup data to the code-fix commit. If reusing existing NRGS output directories, preserve their `dataset.json` as well; the automatic backup covers `gate.json`, not every historical product. Prefer a fresh output tag below.

### 2. Re-mine original prefixes in the existing model environment

This step requires the same GPU/model and oracle access as your current mining workflow. It is not part of the local regression suite and has not been run here.

```bash
python scripts/mine_pivot_instances.py \
  --model-key Qwen/Qwen2.5-Coder-32B-Instruct
```

You can start with a selected affected pair using the existing `--pairs SRC,DEST` option, then broaden after inspecting recovered/rejected examples. Do not assume a particular pair is globally approved: the existing pair-selection policy remains unchanged.

Do not delete the oracle-response cache merely to change this filter. Cached implementations/prefix proposals can be processed again through the new lexical and completion checks. Some newly accepted prefixes may still fail completion correctness or the global pair policy; that is expected.

### 3. Rebuild hypothesis datasets under new output tags

For example, rebuild the count hypothesis:

```bash
python scripts/diff/mine_nrgs_prefix.py \
  --model Qwen/Qwen2.5-Coder-32B-Instruct \
  --configs H1c \
  --out-tag h1c-context-v1
```

Use separate tags for the other hypothesis groups you intend to rerun. The new metadata version prevents the subsequent trainer's dataset-hash cache from masquerading as the old dataset. Keep new masks and metrics in separate runs if you choose to train on the rebuilt pools.

### 4. Review the acceptance changes before drawing conclusions

Inspect examples recovered by the fix and examples newly rejected as formatting-only. Report counts by native/target pair and prefix family, and preserve old-versus-new denominators. Compare historical metrics on a matched subset where appropriate; do not present a changed pool as an identical replication.

## Known limits

- Full generated references are reused in original prefix mining, but the NRGS token-only check does not produce those references. It normalizes its known input variable and preserves unknown local names rather than guessing. Broader multi-task completion checks remain separate work.
- The helper is designed for Python expression prefixes, not a language-independent parser.
- It is deliberately conservative for complex scoping and malformed reference code; it may retain uncertain contrasts for later checks rather than discard meaningful ones.
- Real-model BF16/CUDA behavior, pool-size changes, and training effects remain untested.
- Filter versioning fixes stale acceptance reuse for this change. It is not a comprehensive renderer/model/tokenizer provenance system.
