# Compact hypothesis-specific query localization: design and porting guide

## Scope and status

This specification targets `arnab-api/goals` at commit `80dcba904260`. It describes the S1/C1/P1/I1 contrasts, matched donor controls, and the changes required to port them into the existing head- and sub-head-localization pipeline.

The primary contrast configurations already exist. The matched controls and neutral naming described here are proposed extensions, not completed model experiments. This document does not modify the repository or launch mining/training.

**Preserve the Appendix C/Figure 9 measurement principle:** destination and target runs share an implementation prefix such that their next predicted tokens differ and reflect the operation being performed. The symbolic task arrangement and the informative implementation prefix are separate requirements. A bare indentation token is not a substitute for an informative localization position.

The contextual token-filter fix is already present at this commit. Do not reapply the earlier filter patch.

## 1. Hypotheses and notation

A specification such as `a' b c' -> b` has tasks ordered `[a,b,c]`, with a and c already implemented; b is the first pending task. Letters identify operations, not Python result-variable names. Prefilled implementations follow specification order.

| Hypothesis | Source information transferred | Destination prediction |
|---|---|---|
| H0 | No effective transfer | Destination's native first pending task |
| H1: set progress | Which source tasks are complete | First destination task not in the source completed set |
| H1c: count progress | Number of source tasks complete | Destination slot equal to that count, using zero-based indexing |
| H2: frontier position | Source frontier's specification position | Same zero-based slot in destination |
| H3: predecessor | Identity of the source frontier's predecessor | Task immediately following that identity in destination |
| Frontier identity | Source's selected next task itself | Source frontier, even if absent from destination's specification |

H3 is evaluated only for order-matched sources: every task before the source frontier is complete and none after it is complete. There the specification predecessor and last-written task coincide. On order-broken sources, report H3 as N/A, not as a failed prediction.

Keep H1's existing `predict` rule unchanged. The designs below make every destination-completed task complete in the source as well, so H1 does not ask to redo an already-completed destination task.

## 2. Four primary contrasts

Each source and destination contains three or four tasks. The union across a pair contains five identities, except P1, which contains six. Six identities across two prompts does not mean six tasks in either prompt.

| Name | Target hypothesis | Source | Destination | Source / destination length | Existing configuration |
|---|---|---|---|---|---|
| **S1** | H1: completed set | `a' b c' d' -> b` | `a' d e c -> d` | 4 / 4 | `order_diff_2` |
| **C1** | H1c: completed count | `a' b c' -> b` | `a' d e -> d` | 3 / 3 | `h1c_1` |
| **P1** | H2: frontier position | `a' b' c d' -> c` | `a' e f b -> e` | 4 / 4 | `h2_1` |
| **I1** | H3: predecessor | `a' b' c -> c` | `a' e b d -> e` | 3 / 4 | `h3_1` |

### Prediction table

| Design | H0 | H1 | H1c | H2 | H3 | Source frontier |
|---|---|---|---|---|---|---|
| S1 | d | **e** | c | d | N/A | b |
| C1 | d | d | **e** | d | N/A | b |
| P1 | e | e | b | **f** | N/A | c |
| I1 | e | e | b | b | **d** | c |

Every bold target is unique among applicable listed hypotheses, is non-native in the destination, and does not appear in the source specification.

- **S1:** source has completed `{a,c,d}`. Set progress skips d and selects e; count selects destination slot four, c; frontier position selects slot two, d.
- **C1:** source has completed two tasks but its frontier occupies slot two. Count selects destination slot three, e; set and position select d.
- **P1:** source has completed three tasks but its frontier occupies slot three. Position selects f; count selects destination slot four, b; set selects e.
- **I1:** source frontier c follows b. Predecessor transfer selects d, which follows b in the destination. Count and position select b; set progress selects e.

One configuration cannot separate every rule: count and frontier position coincide on an order-matched source, while H3 is not defined on the broken-order sources used here. The four-row battery resolves these equivalences across contexts without five-task prompts.

## 3. Matched source controls

Hold the actual predicate assignment and destination prompt fixed within each comparison. Merely drawing a different random assignment for a control is not sufficient.

### S1: change completed identity while preserving count and frontier

Destination remains `a' d e c -> d`.

| Source variant | Done set | Frontier | H0 | H1 | H1c | H2 | H3 |
|---|---|---|---|---|---|---|---|
| Primary: `a' b c' d' -> b` | `{a,c,d}` | b | d | **e** | c | d | N/A |
| Matched: `a' b c' e' -> b` | `{a,c,e}` | b | d | **d** | c | d | N/A |

Only which of d/e was completed changes. The number completed, source frontier identity, and source frontier position stay fixed.

Counterbalance with recipient `a' e d c -> e`. H1 still selects e for the completed-d donor and d for the completed-e donor, while H0/H2 follow recipient order. This makes a 2×2 source-completed-identity × recipient-order test.

An additional invariance check uses source `a' b d' c' -> b` against the original destination. H1 should remain e even though the last-written completed task changes.

### C1: change count while preserving frontier identity and position

Destination remains `a' d e -> d`.

| Source variant | Count complete | H0 | H1 | H1c | H2 | H3 |
|---|---:|---|---|---|---|---|
| Primary: `a' b c' -> b` | 2 | d | d | **e** | d | N/A |
| Matched: `a' b c -> b` | 1 | d | d | **d** | d | d |

This is the existing SRC′ idea. Neutral result naming is essential because sequential names can reveal the same counter. The code-line count is another plausible shortcut: distinguish semantic completed-task count from irrelevant comments, extra non-task assignments, or equivalent multiline implementations in later checks.

### P1: move the frontier in the specification, not in completed code

Destination remains `a' e f b -> e`.

| Source variant | Done set | Completed-code order | Frontier position, one-based | H0 | H1 | H1c | H2 |
|---|---|---|---:|---|---|---|---|
| Primary: `a' b' c d' -> c` | `{a,b,d}` | a, b, d | 3 | e | e | b | **f** |
| Matched: `a' c b' d' -> c` | `{a,b,d}` | a, b, d | 2 | e | e | b | **e** |

The completed code can be byte-identical. Only the specification position of pending c changes. H3 is N/A for both sources.

### I1: change predecessor, preserving completed set, count, and frontier

Destination remains `a' e b d -> e`.

| Source variant | Done set | Frontier | Predecessor | H0 | H1 | H1c / H2 | H3 |
|---|---|---|---|---|---|---|---|
| Primary: `a' b' c -> c` | `{a,b}` | c | b | e | e | b | **d** |
| Matched: `b' a' c -> c` | `{a,b}` | c | a | e | e | b | **e** |

Counterbalance with recipient `b' e a d -> e`: H3 predicts e for the first donor and d for the second; H0/H1 remain e, and H1c/H2 predict a.

This identifies compatibility with predecessor/last-written identity. It does not by itself prove that the receiving key implements an “I follow x” code. That requires a key-side or path-specific causal test.

### Optional structural confirmations, all within 3–4 tasks per prompt

| Name | Source | Destination | H0 | H1 | H1c | H2 | H3 |
|---|---|---|---|---|---|---|---|
| C2 | `a' b' c d' -> c` | `a' b e f -> b` | b | e | **f** | e | N/A |
| P2 | `a' b' c d' -> c` | `a' e f d -> e` | e | e | d | **f** | N/A |
| I2 | `a' b' c d -> c` | `a' d b e -> d` | d | d | b | b | **e** |

C2 tests a different count, P2 changes the count competitor to the last-written task, and I2 puts a future task after the source frontier. P2 already exists as `h2_2`; C2 and I2 are proposed additions. Do not treat the letter-renamed `h3_2` as an independent structural confirmation of I1.

## 4. Preserve the informative next-token setup

For each primary contrast, choose actual predicates for the native and target roles and an expression prefix valid for both.

| Design | Native operation | Target operation | Pair requiring an informative shared prefix |
|---|---|---|---|
| S1 | d | e | d versus e |
| C1 | d | e | d versus e |
| P1 | e | f | e versus f |
| I1 | e | d | e versus d |

### Required prompts

1. **SRC:** source task specification and completed implementations, stopping at the chosen source capture boundary.
2. **DST:** destination specification and original progress, followed by the common expression prefix.
3. **DST\*:** a reference context where the nominated target is the next task, followed by the same expression prefix.
4. **SRC-control:** the matched source variant, keeping the destination and its prefix fixed.

The current source capture boundary is `=`. The destination measurement position is the **last token of the shared expression prefix**. These positions need not coincide or have the same text.

For example, a native lowercase check and target palindrome check can share `s == s`, with next-token predictions `.lower` and `[::-`. That pair can be reused in C1: the donor query transfers count-associated information even though its own pending operation may be a third predicate.

### Measurement contract

- Both native and target operations must have valid continuations from the prefix.
- Their actual model next-token targets must differ and be representative of the computational distinction.
- A token need not uniquely identify the operation among every possible task.
- If an additional hypothesis predicts another operation with the same target token, do not interpret that one token-flip result as distinguishing those hypotheses. Use another contrast/pivot or operation-level generation to resolve it.
- Do not replace an informative expression prefix with indentation merely to remove naming cues. Remove the naming cue while retaining the expression prefix. Indentation is useful as a separately labeled source-position or free-generation test.

DST\* currently prefills all destination tasks preceding the target. This supplies a natural reference distribution but changes progress and code length. Neutralize result-variable cues and retain that nuisance difference in the interpretation; matching its full distribution is not proof that only the intended causal variable was transferred.

## 5. Porting map for commit 80dcba904260

### 5.1 Reuse the existing positive configuration registry

Relevant file: `src/diff/nrgs_prefix.py`.

- Use `order_diff_2`, `h1c_1`, `h2_1`, and `h3_1` for the primary rows.
- Keep `predict()` as the single source of truth for all prediction tables.
- Optional short aliases `s1`, `c1`, `p1`, `i1` can map to those existing names. If added, define them before constructing `VARIANTS`. Preserve canonical names in stored records for compatibility.
- Do not globally force `strict=True` on the whole H1 group: several historical H1 configurations intentionally have overlapping predictions. Validate S1's distinguishing property separately.
- Do not use `--configs H1` when the intention is specifically S1; it expands to the older six-configuration H1 mixture. Select `order_diff_2` explicitly.

### 5.2 Add explicit donor variants, not ordinary positive configurations

Current `Config.src_ctrl_prefill` only removes completed tasks after the source frontier. It cannot express:

- S1's substitution of one completed task identity for another;
- P1's reordered specification with identical completed code;
- I1's changed predecessor through source-order reversal.

Also, `_cfg()` requires a non-native positive target. The matched controls intentionally predict the native destination operation, so **do not insert them as ordinary `_cfg` rows or weaken that assertion globally**.

A minimal proposed representation is a separate donor-variant structure:

```python
# Proposed interface; not currently implemented.
@dataclass(frozen=True)
class SourceVariant:
    name: str
    order: tuple[str, ...]
    done: frozenset[str]
```

For the core designs, its content would be:

```python
# All letters resolve through the SAME role-to-predicate assignment as the primary.
MATCHED_DONORS = {
    "order_diff_2": SourceVariant("set_identity", tuple("abce"), frozenset("ace")),
    "h1c_1":       SourceVariant("count",        tuple("abc"),  frozenset("a")),
    "h2_1":        SourceVariant("position",     tuple("acbd"), frozenset("abd")),
    "h3_1":        SourceVariant("predecessor",  tuple("bac"),  frozenset("ab")),
}
```

The original role assignment already covers every identity needed by these donor variants. For S1, e exists in the primary destination even though it is absent from the primary source. Do not restrict a control's source task list to the original source's identities.

Compute the variant's frontier and predictions with `predict()` rather than hardcoding them. For all four minimal controls, the target hypothesis predicts the native destination task.

### 5.3 Extend instance rendering and serialization

Relevant symbols: `NRGSPrefixInstance.base`, `parts`, `prompts`, `key`, `to_record`, and `from_record` in `src/diff/nrgs_prefix.py`; `DiffTriggerInstance._parts` in `src/diff/data.py`.

Recommended minimal port:

1. Keep the primary `Config`, native/target mapping, and `Pivot` unchanged.
2. Add an optional matched-donor specification or variant ID to the instance.
3. Render the control by constructing its source order and completed set from the existing complete `roles` map.
4. Reuse the same destination and DST\* prompts. Only the donor changes for the initial paired-control implementation.
5. Return the matched donor under `src_ctrl` for compatibility with the existing single-control cache path, and record a `control_name` so it cannot be confused with the legacy automatically derived SRC′.
6. Preserve the old `src_ctrl_prefill` behavior for legacy instances with no explicit variant.
7. Serialize the variant specification, renderer version, naming seed, and matched-family identifier. Round-trip them through `to_record`/`from_record`; old records should still load with legacy defaults.
8. Include these fields, model/tokenizer identity, and rendered-token provenance in cache identity.

Do not independently resample control predicates. The control and positive donor must be two views of the same role assignment.

### 5.4 Add neutral result names without removing the common prefix

Current rendering hardcodes `naming="letters"`, `letter_stub="var_eq"`, and asserts that DST\* ends with a sequential letter such as `c =`.

Add a renderer mode with:

- random nonsemantic names for completed-task outputs, generated once per matched family and stable for shared tasks;
- one unused neutral output-stub name, fixed across SRC, control, DST, and DST\* within that family;
- the same input argument and expression prefix used to validate the native/target pair;
- the informative prefix appended to DST/DST\*, but not automatically to the source donor.

For example, both reference contexts can end in `result_qx = s == s`, even if different numbers of lines precede them. The stub must not be derived from task identity or completed count.

Replace the hardcoded `chr(ord('a') + len(star_pre))` assertion with a check against the renderer's actual pivot name. Do not change legacy rendering silently; preserve it behind an explicit mode and record the mode in outputs.

Randomizing the input argument also requires rewriting and revalidating the expression prefix. Keep that separate from the initial result-name repair.

### 5.5 Reuse the shared prefix pool, then validate the rendered instrument

Relevant files: `src/goals/head_loc_oracle.py`, `src/goals/pivot_token_filter.py`, `scripts/mine_pivot_instances.py`, `src/diff/nrgs_prefix.py`, and `scripts/diff/mine_nrgs_prefix.py`.

The correct assignment sequence is already supported conceptually:

```text
configuration
  + validated native/target predicate pair and prefix
  + compatible predicates for remaining roles
  -> primary/control source and destination contexts
```

Do not create a separate operation/prefix bank for each hypothesis. Use one shared pool and one hypothesis-conditioned assignment builder.

Retain the corrected contextual token filter. Add or separately run validation for clean source/control frontier correctness and native/target completion correctness in the multi-task context; the current NRGS `gate()` only verifies token divergence and consistency with stored single-predicate targets.

Store the observed native/target token IDs and distributions from the exact rendered prompts. A naming or teacher change requires fresh validation; it is not safe to assume old token IDs remain correct. Keep ordinary binary localization examples distinct from tests claimed to identify one hypothesis against every applicable rival.

Current coverage limitation: the localization bank has seven predicates per type, three data types, and Python implementations only. H2/P1 currently yields integer candidates but no string or integer-list candidates under the existing pool/family constraints. Reassess after recovering prefixes. Expand missing operation/family coverage if necessary, rather than silently turning on `--allow-family-overlap`. Six distinct identities in P1 are spread over two four-task prompts.

### 5.6 Reuse query caching and the head/component trainers

Relevant files: `src/diff/nrgs_mask.py`, `scripts/diff/train_nrgs_mask.py`, `src/diff/nrgs_svd_mask.py`, `scripts/diff/train_nrgs_svd_mask.py`.

The current `NRGSCache` already stores `q_src`, optional `q_ctrl`, target distribution, and native/target token IDs. The minimal one-control-per-primary extension can reuse those tensor fields. Add defaulted provenance fields such as control name and expected operation, and bump the cache schema/identity; do not reuse a legacy `q_ctrl` cache for a different donor.

No change is required to the basic head or component intervention:

\[
q'=q_{dst}+m(q_{src}-q_{dst}),
\]

or

\[
q'=q_{dst}+(q_{src}-q_{dst})M\operatorname{diag}(m)M^{-1}.
\]

Keep the current positive loss: counterfactual KL, directional target-versus-native term, and sparsity. Do not silently add control supervision while describing controls as independent evaluation. Initially retain the existing `w_ctrl=0` setting and evaluate matched controls. A later training condition can explicitly enable a control penalty and report that change.

For the four minimal controls, the expected operation is native, so the current negative target-margin control is directionally relevant. But absence of a target flip is weaker than successful native behavior: report native probability/argmax and, where evaluated, actual first operation. For secondary controls with a different non-native expected operation, the existing generic “do not flip” penalty is insufficient; use an explicit control target distribution or keep those controls evaluation-only.

Multiple controls per primary can later use a dictionary/list of control queries. The first port need not refactor all tensor paths if one named matched control per row suffices.

### 5.7 Keep matched families together in train/validation

Use the existing unordered native/target-pair grouping, and ensure every matched donor/recipient variant and its implementation augmentations stay with that group. Generate matched variants from a shared assignment before splitting, or store controls inside the same primary record.

Do not select a new best mask on these controls and then call them untouched confirmation. Reserve new operation pairs and structural confirmations such as C2/P2/I2 for clearly designated evaluation where appropriate.

### 5.8 Extend the general generation evaluator

Relevant files: `src/diff/hyp_eval.py` and `scripts/diff/q_transfer_hypothesis_eval.py`.

The existing evaluator has useful building blocks: paired clean/patched outputs, shuffled exclusive judge options, all hypothesis predictions, and source-frontier outcomes. Extend it to:

- generate the same matched donor families instead of independently sampling each side;
- render the same neutral naming policy;
- run primary and control query transplants into the same destination;
- report complete first-operation distributions and paired changes;
- keep H3 N/A separate from an incorrect prediction;
- distinguish unknown/invalid behavior from judge parse failures;
- include renderer, donor variant, mask identity, intervention duration, and input hashes in resume keys.

**Component masks:** this general hypothesis evaluator currently accepts head sets but has no `--component-mask` option. The specialized `diff_trigger_eval.py` already supports them. Port that small interface:

1. load `component_projectors()` from `src/diff/nrgs_svd_mask.py`;
2. obtain the selected heads and per-head projector dictionary;
3. pass `component_proj` into `run_q_transfer_intervention`;
4. record whether weights are soft or thresholded.

Expose mutually clear head-set versus component-mask selection; do not accidentally use the default goal-head set when a component mask is intended.

Also, `--patch-heads goal` currently resolves to the model's threshold-selected goal set, not necessarily goal-250. Use the explicit JSON path for a 250-head comparison.

## 6. Adding the recipient counterbalances safely

Implement donor-only controls first. The S1 and I1 recipient counterbalances are useful but require more than editing a destination task string.

For each recipient variant:

- recompute its native frontier;
- recompute each donor's predicted target;
- classify native-coincident donor conditions as controls rather than positive localization examples;
- orient the `Pivot` native/target fields and token IDs accordingly;
- rebuild and validate the corresponding target teacher;
- preserve predicate bindings and matched-family identity across the quartet.

In S1, reversing recipient order can exchange the native and nominated target operations. Do not retain stale `b_id`, `d_id`, or `cfg.native` fields. A common valid prefix may still be reusable with the token roles reversed, but that must be checked in the rendered context.

Never weaken `_cfg`'s positive-target assertions simply to store native-coincident controls. Keep the positive registry and factorial/evaluation conditions distinct.

## 7. Commands supported by the current code

These commands mine the existing primary rows with the **current legacy renderer and legacy automatic controls**. They do not implement the new matched controls or neutral naming. Use the suffixes to keep that distinction visible.

Run from a writable repository on an allocated GPU, in the existing environment. Model/oracle access requirements are the same as the current scripts. These are instructions, not jobs launched by this document.

```bash
MODEL=Qwen/Qwen2.5-Coder-32B-Instruct

# Preserve the existing model-specific pool before re-mining it.
mkdir -p backups
BACKUP=$(mktemp -d backups/compact-localization.XXXXXX)
cp data/head_loc/pairs/Qwen2.5-Coder-32B-Instruct.json "$BACKUP/"

# Recover original pair/prefix candidates using the corrected token filter.
# Do not add --regen-global-pairs unless deliberately changing pair selection.
conda run -n goals python scripts/mine_pivot_instances.py --model-key "$MODEL"

# Build separate primary-row datasets; none mixes all H1 configurations.
conda run -n goals python scripts/diff/mine_nrgs_prefix.py \
  --model "$MODEL" --configs order_diff_2 --out-tag s1-primary-legacy-v1
conda run -n goals python scripts/diff/mine_nrgs_prefix.py \
  --model "$MODEL" --configs h1c_1 --out-tag c1-primary-legacy-v1
conda run -n goals python scripts/diff/mine_nrgs_prefix.py \
  --model "$MODEL" --configs h2_1 --out-tag p1-primary-legacy-v1
conda run -n goals python scripts/diff/mine_nrgs_prefix.py \
  --model "$MODEL" --configs h3_1 --out-tag i1-primary-legacy-v1
```

Both training scripts accept `--data-tag` or an explicit `--dataset` path. Point them at the intended new dataset and retain the chosen historical hyperparameters unless a separate comparison changes them. Do not train before checking accepted counts, native/target validity, and held-out coverage.

After porting the new renderer and controls, use new output tags and schema versions; do not label the existing primary-only outputs as the full matched design. No proposed CLI flag in this document should be treated as already implemented.

## 8. Verification checklist for the port

### Symbolic and rendering tests, no model required

- Verify the four primary prediction rows against `predict()`.
- Check source/destination lengths are 3–4 and task identities do not repeat within a prompt.
- Check nominated positive targets are unique among applicable rules, non-native, and absent from source.
- Verify every minimal matched control's predicted target is native.
- Verify S1 holds count/frontier/position fixed while changing completed identity.
- Verify P1 keeps its completed code identical while moving the source frontier in the specification.
- Verify I1 holds completed set/count/frontier position fixed while changing predecessor.
- Verify completed-task names are stable across matched views and the pivot name carries no task/count cue.
- Verify exact DST/DST* expression-prefix equality and independent source capture positions.
- Round-trip the new record format; load legacy records without changing their prompts.
- Reject reuse of old caches after renderer/control/pool changes.
- Keep matched families in one split.

### Model-side validation before substantive mask fitting

- Confirm clean source/control native behavior, especially on broken-order prefills.
- Verify native and target completions from the shared prefix in the actual multi-task context.
- Record exact next-token IDs and probabilities; reject or explicitly flag stale/ambiguous targets.
- Check zero-mask equals clean destination, full component transfer agrees with the corresponding whole-query transfer, and training/exported generation patches agree within the actual numerical precision.
- Compare primary and matched-control effects on frozen existing masks before changing the objective.
- Report both next-token measures and operation-level generation without treating them as interchangeable.

The current filter tests can be run independently of GPU/model dependencies:

```bash
python -m unittest discover -s test -p 'test_pivot_token_filter.py' -v
```

A separate symbolic check of the primary and control tables accompanied this document. Passing these checks does not establish that the model follows any hypothesis.

## 9. Deliverables for an implementation PR

1. Explicit donor variants and shared-assignment sampler.
2. Versioned neutral renderer that preserves the informative prefix.
3. Updated serialization/cache identity with backward-compatible legacy loading.
4. Full-context validation records and rejection reasons.
5. Matched-control generation evaluation, including component-mask support in the general hypothesis evaluator.
6. Unit tests for the symbolic, rendering, and cache contracts above.
7. Small inspected prompt examples for each primary/control pair and a manifest linking configurations, pool version, model/tokenizer, intervention settings, and outputs.

Preserve existing masks, datasets, and results. Do not overwrite historical conclusions or report the new controls as evaluated until model-side outputs exist.
