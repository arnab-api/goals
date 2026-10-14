# Pivot re-mining audit

## Scope and provenance

Refreshed private repository `arnab-api/goals` from `80dcba904260` to `dcf1e3dc44d1` (Re-mine Coder-32B pivot pool with the contextual token filter). Compared the committed old/new `data/head_loc/pairs/Qwen2.5-Coder-32B-Instruct.json` files with a standard-library audit. Reused the actual pre-fix normalization functions from `74f5d2fc57c7` and the current contextual lexer. No model, oracle, mining, or training jobs were run; no repository files were changed.

The checkout contains the new pool, the four rebuilding job specifications, and the Stage 0 evaluator/eight job specifications. It does NOT contain the rebuilt NRGS `dataset.json` files, mining logs/oracle caches, or Stage 0 result tables. Their completion and dataset totals remain researcher-reported, not independently verified here.

## Verified pool counts

| Measurement | Old | New |
|---|---:|---:|
| Pivot records | 1,423 | 1,613 |
| Base / wrapper-augmented | 302 / 1,121 | 347 / 1,266 |
| Directed pairs with pivots | 110 | 109 |
| Integer / integer-list / string records | 449 / 569 / 405 | 410 / 701 / 502 |

950 exact records are shared, 663 added and 473 removed. Identity includes direction, datatype, prefix and both next-token strings. Base/augmentation classifications use wrapper-prefix detection, because the saved records do not have explicit augmentation lineage.

275 added records would have been rejected by the old token filter. Their contextual breakdown is **216 bare-identifier contrasts and 59 method/attribute-name contrasts**, not 266 and 9:

- 150 list records: `(x` versus `(lst`, both directions.
- 66 string records: `(c` / `(char` versus `(s`, both directions.
- 9 method cases where the prefix ends at a dot, such as `bool(s.` followed by `ist` versus `isspace`.
- **50 further method cases where the prefix already contains part of the attribute**, such as `s.is` followed by `lower` versus `numeric`, forming `s.islower` versus `s.isnumeric`.

The numerical discrepancy is exactly those 50 partial-attribute cases. Without the local agent's classification script, its exact bug is unverified; checking only whether a prefix ends in `.` would produce this error. Classification must use the concatenated expression prefix and token.

The 216 identifier-only cases comprise 50 base and 166 augmented records. The 59 attribute cases comprise 8 base and 51 augmented records under the same wrapper heuristic.

## Recommended eligibility policy

The current `_variable_roles()` deliberately distinguishes `arg:0` from `local:0`. This is defensible for binding-aware lexical comparison: an input container and its iteration element are not alpha-renamings of the same variable. But it is too permissive if the intended localization instrument must exclude a contrast whose only diagnostic information is which variable/binding is accessed.

For the main operation-localization pool, exclude the 216 identified records through a **versioned eligibility view**, preserving the original mined pool. Keep them as a separately labeled implementation/binding contrast stratum for sensitivity analysis if useful. Retain the 59 method contrasts, subject to the existing and full-context validity checks. A next token separating iteration strategies is not intrinsically meaningless, but it supports a weaker operation-identity claim.

Do not globally declare input arguments and all locals semantically equivalent, and do not apply a blanket 'alphabetic tokens are noise' rule. A lexical-validity check and a task-specific operation-eligibility policy have different purposes. If the stricter rule is integrated into mining, preserve partial attributes, string literals, calls and operators, apply it consistently at candidate/validation/augmentation/NRGS stages, and version the policy/cache identity.

This audit emits a manifest of the 216 candidates in `new_identifier_only_candidates.json`; it does not delete them or create a new training dataset.

## Coverage after proposed exclusion

Running the current `load_pivots(core_only=True)` definition on the three pool views gives:

| View | Raw records | NRGS core records | Unordered predicate pairs |
|---|---:|---:|---:|
| Old | 1,423 | 320 | 55 |
| New | 1,613 | 375 | 55 |
| New minus 216 identifier contrasts | 1,397 | 325 | 55 |

The prospective filtered core has 74 integer, 131 integer-list and 120 string records, versus the old 80 / 132 / 108. This is a modest redistribution, not a broad expansion in predicate-pair coverage. H2 remains integer-only in the static role sampler. These are metadata/symbolic counts, not accepted neural dataset counts.

To quantify the effect on already-built `*-ctx` datasets, join their actual rows to the eligibility classification, preserving their existing splits and recording exclusions. Do not infer accepted-dataset growth directly from raw pivot totals. Fresh rebuilding can also resample role assignments and reassign pair splits; it would obscure a controlled before/after comparison.

## Pool churn is not a controlled filter ablation

388 new records and all 473 lost records pass the old filter when applied to their saved model tokens. This fact does NOT establish that their changes have nothing to do with the filter:

- `get_oracle_pivot_instances()` rebuilds candidates using both pair-level oracle prefixes and cross-products of per-predicate implementations; it does not start by replaying every old accepted prefix (head_loc_oracle.py:1015–1059).
- `_mine_case3b_raw()` applies the filter to **oracle implementation token divergences** before model validation. Those tokens need not be the saved model top tokens (lines 602–628).
- Validation and augmentation run again, and the save step clears every revalidated pair before repopulating it (lines 1133–1143).
- `ask_claude()` caches by system prompt, user prompt and token budget. Model ID and temperature are absent from the key; missing/corrupt cache files trigger a fresh sampled call (oracle_llms.py:16–42,96–112). That permits provenance ambiguity; it does not prove a cache miss occurred in this run.
- The job uses BF16 and model forward/completion checks, but numerical nondeterminism has not been demonstrated as the reason for any loss. GPU completion noise also cannot directly explain a changed Stage-1 list generated solely from fixed oracle text/tokenization/filter inputs.

For a controlled attribution, freeze raw oracle responses, tokenizer and candidates before filtering; save candidate-stage and final-model token pairs/completions; apply old/new/strict eligibility offline wherever possible. For useful lost prefixes, replay the old/new candidate union under one fixed validation configuration rather than immediately sampling fresh oracle outputs. Preserve rejection reasons per stage. No full re-mining is necessary merely to classify and exclude these saved identifier contrasts.

## Stage 0 implementation audit

The committed `stage0_matched_donor_eval.py` reads the **old** `mixed`, `h1c`, `h2`, and `h3` validation datasets, not the new `*-ctx` datasets (DESIGNS and load_design, lines 69–80,167–171). The eight job specifications do not override this. Unless a different local script was executed, these Stage 0 results are not directly affected by the newly recovered identifier contrasts.

Confirmed: destination informative expression prefix is fixed across donors; the existing last-token pre-RoPE query patching functions are reused; head soft/binary and component soft/thresholded variants are kept distinct. The script adds C1 token-length padding and count/stub-letter conflict donors while retaining historical sequential names.

Before interpreting aggregate tables, repair three summary issues:

1. `heldout=None` is included in the `heldout` subset through `is not False` (line 301). Use `is True`, and report unknown provenance separately. Unknown fixed-head baseline provenance is not confirmed pair-disjoint validation.
2. `count_high_stub_low` predicts the target under the count hypothesis, but the generic joint metric still requires that donor to select native (lines 267–301). Use donor-specific expected outcomes, or report count-versus-letter conflict tests separately from native-preserving controls. A genuine count-sensitive mask may appropriately have near-zero primary-minus-control flip difference on this donor.
3. OM1 has no control donor, so its primary rows never enter the paired summary. Add a separate primary-only table; raw rows are still written.

Padding failures are silently omitted; report accepted/dropped counts per padding condition. Clean native destination behavior is checked, but source/control completion validity and a fresh DST* correctness check are not established by this script alone. These are qualifications to interpretability, not evidence that the patching machinery failed.

Expected result location: `results/stage0_matched/Qwen2.5-Coder-32B-Instruct/<job_tag>/{rows.jsonl,prompts.jsonl,summary.json}` with the eight tags in `job_list/nrgs/stage0_matched_donors_32b.json`. The summary fixes above can be computed from saved rows without rerunning GPU inference.

## Audit outputs

- `audit_pool.py`: reproducible committed-pool comparison and old-filter/contextual classification.
- `pool_audit.json`: full counts, token-pair frequencies and all recovered attribute cases.
- `new_identifier_only_candidates.json`: 216-record exclusion-candidate manifest.
- `audit_core_coverage.py` / `core_coverage.json`: actual loader counts and symbolic candidate coverage, no model runs.
