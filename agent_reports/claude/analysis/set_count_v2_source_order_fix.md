# Set/count localization v2: the source-order confound, the per-donor fix, and first localization runs

*2026-10-04 · Qwen2.5-Coder-32B-Instruct (rev `381fc969`) · greedy decoding · next-token validation only*

## Context

The silico v2 handoff (`agent_reports/silico/set_count_localization_v2_handoff.zip`) trains independent query masks
for two hypotheses about what a transplanted pivot query (SRC `=` → DST) carries:

- **Set (H1):** DST does the first task in its own list that SRC has *not completed*.
- **Count (H1c):** DST does the task at position |SRC completed|.

Every SRC prompt lists tasks in one fixed order `[b,a,d,e]`. Task `b` is listed first and left unfinished, and it never
appears in DST. The four donors differ only in which other tasks are already written:

| Donor | Completed | Count |
|---|---|---:|
| A | {a,d} | 2 |
| B | {a,e} | 2 |
| C | {a} | 1 |
| D | {d} | 1 |

The design assumes SRC's own next task at `=` is `b`. Under that assumption, a third explanation, **copy** ("DST simply
does whatever task SRC was about to write"), predicts `b`. `b` is not a DST readout, so copy can never be confused with
set or count. **That assumption was never checked.**

## Verification of the patch (before the problem)

- Both patches matched their manifest hashes and applied cleanly on `HEAD`. The cumulative and incremental routes
  produced byte-identical files.
- 31 core tests and 24 runner tests passed.
- The dataset rebuilt from our current pool reproduced their fingerprint exactly: 239 families, 3,107 views.

## The problem: SRC skips `b`

**Check.** 25 families (6 per datatype from train, plus all 7 dev) × 4 donors = 100 SRC prompts, greedy continuation
from the unfinished `=`. Each continuation was **executed** on about 80 inputs per datatype and matched (≥95% agreement)
against the bank implementations of `b, a, d, e`.

No continuation assembled a tuple: the v1 `result =` aggregation is gone. But the model rarely wrote `b`:

| Donor (completed) | Design assumes | Model wrote (`[b,a,d,e]` order) |
|---|---|---|
| A {a,d} | b | **e 17**, b 8 |
| B {a,e} | b | **d 18**, b 7 |
| C {a} | b | b 12, d 9, other 4 |
| D {d} | b | b 13, a 6, e 6 |

Once the code starts at task (2), the model treats task (1) `b` as skipped and fills the next gap. Example
(`list[int]`; a = strictly sorted, b = all even, d = all positive, e = all equal):

```
B:  t = all(lst[i] < lst[i + 1] for i in range(len(lst) - 1))     # a
    v = len(set(lst)) <= 1                                        # e
    q =  →  all(x > 0 for x in lst)                               # d, not b
```

**Why it matters.** With these continuations, copy predicts e / d / d for A / B / C. That is exactly what set predicts
in the `core` and `four` DST configurations. Counted per view over the 25 families (325 views):

| Training target | Views where it differs from native | …and copy predicts the same token |
|---|---:|---:|
| Set | 125 | **64 (51%)** |
| Count | 150 | 52 (35%) |

So half of the set mask's positive targets could be met by a mechanism that transfers SRC's *next-task identity*. No
set (completed-tasks) information is needed. Only `swapped`/C and donor D separated the two, and D's own continuation was `a` (the set
prediction) in 6/25 families.

## The fix: per-donor source order with `b` right after the completed tasks

Each donor lists its completed tasks first, then `b`, then the rest:

| Donor | Old order | New order |
|---|---|---|
| A | `bade` | `adbe` |
| B | `bade` | `aebd` |
| C | `bade` | `abde` |
| D | `bade` | `dbae` |

Reading in order now leads to `b` for every donor. **No label changes:** set depends only on the SRC completed set and
count only on its size, not on SRC order. Rebuilding gives the same 239 families. All **3,107/3,107** set, count and
native labels are identical. Only the random variable names differ, because they are seeded by the family ID, which changed.

Code changes (`src/diff/set_count_data.py`, `test/test_set_count.py`):

- `SOURCE_ORDER = 'bade'` → `SOURCE_ORDERS = {'A':'adbe','B':'aebd','C':'abde','D':'dbae'}`. `verify_dataset`
  checks each view's `src_order` against its donor.
- The renderer tag is bumped to `opaque-identifiers-v2-frontier-b`, so datasets in the old order are rejected rather than
  reused.
- Tests assert the per-donor order and that the first unfinished SRC task is `b`. 31/31 pass.

## Re-check after the fix (same families)

**SRC continuations.** The same 25 families × 4 donors, same executed classifier:

| Donor | `[b,a,d,e]` | Per-donor order |
|---|---|---|
| A | e 17, b 8 | **b 25** |
| B | d 18, b 7 | **b 25** |
| C | b 12, d 9, other 4 | **b 25** |
| D | b 13, a 6, e 6 | **b 25** |

The classifier first labelled 3 of the 100 "other". All three are `b` written as `len(set(lst)) == 1`, which differs
from the bank's `all_same` only on `[]` (the empty-list quirk already noted in the handoff). Same example family:

```
B:  v = all(lst[i] < lst[i + 1] for i in range(len(lst) - 1))     # a
    t = len(set(lst)) <= 1                                        # e
    w =  →  all(x % 2 == 0 for x in lst)                          # b
```

Copy now predicts `b` in every view, and `b` is never a DST readout, so **copy coincides with set in 0% of views**
(51% before) and with count in 0% (35% before).

**Readout validation** (`train_set_count_mask.py --validate-only`; DST and reference prompts are unchanged by the fix):

| | `[b,a,d,e]` (31 families) | Per-donor (same 31 + 2) |
|---|---|---|
| Accepted | 24 (18 train, 6 dev) | 24 (20 train, 4 dev) |
| Rejection reason | reference next tokens coincide (7/7) | reference next tokens coincide (9/9) |
| Clean DST ≠ native reference | 0 | 0 |

21 of the 24 accepted families are the same in both runs. The difference comes from the reseeded names.
In the first run, 90/312 accepted views had a reference top token different from the stored pool token. The validator
recomputes tokens in context, so this is harmless, but stored pool tokens should not be treated as readout labels.

## Localization runs on the fixed data (Coder-32B)

**Setup.** Four independent masks: {set, count} × {head, sub-head (W_QK SVD components)}. Each run used one A100 at
about 66 GB; the base model is frozen and each view is a separate forward/backward pass. Common settings:

- all 64 layers; mask initialised at 1.0; AdamW, learning rate 0.01;
- control weight 1, KL weight 0, sparsity 20 (on the mask mean);
- 1,000 steps, one whole family (13 views) per step, about 6 passes over the training families;
- `--allow-next-token-only`: training and evaluation both score only the first patched token.

Validation accepted 171/218 train, 4/7 dev and 7/14 test families (test: 5 string, 2 integer). Every rejection was
"reference next tokens coincide". Test was evaluated once, after training, by resuming each mask at step 1,000 with
`--evaluate-test` (zero further steps). Train and dev numbers reproduced exactly.

**Mask size (> 0.5):**

| Mask | Active |
|---|---|
| set, head | 141 / 2,560 heads |
| count, head | 135 / 2,560 heads |
| set, sub-head | 20,708 / 327,680 components, touching 1,785 heads |
| count, sub-head | 13,028 / 327,680 components, touching 1,566 heads |

**Views whose expected token is not native** (top-1 = expected):

| Mask | Train | Dev (4 fam.) | Test (7 fam.) |
|---|---|---|---|
| count, sub-head | 1011/1026 = 99% | 19/24 = 79% | **21/42 = 50%** |
| set, sub-head | 745/855 = 87% | 8/20 = 40% | **21/35 = 60%** |
| count, head | 405/1026 = 39% | 9/24 = 38% | 16/42 = 38% |
| set, head | 233/855 = 27% | 3/20 = 15% | 9/35 = 26% |

**Views where set and count predict different tokens:**

| Mask | Train: own / other | Dev: own / other | Test: own / other | Test always-native |
|---|---|---|---|---|
| count, sub-head | 99% / 0% | 85% / 0% | **23/35 = 66%** / 8/35 = 23% | 14/35 = 40% |
| set, sub-head | 89% / 7% | 50% / 30% | **22/35 = 63%** / 12/35 = 34% | **21/35 = 60%** |
| count, head | 55% / 38% | 55% / 40% | 19/35 = 54% / 14/35 = 40% | 14/35 = 40% |
| set, head | 50% / 39% | 45% / 40% | 12/35 = 34% / 17/35 = 49% | 21/35 = 60% |

By datatype on test (sub-head, discriminating views): count gets 16/25 (string) and 7/10 (integer); set gets 16/25 and
6/10.

**Reading:**

- **Head level fails for both hypotheses,** on train as well as held-out. About 140 whole heads can't produce either
  predicted token; the masks mostly leave the output native.
- **Sub-head masks fit both hypotheses in-sample** (set 87%, count 99%), although the two predict different tokens on
  the discriminating views. With 13–21k components spread over 1,500+ of the 2,560 heads, they are neither sparse nor
  localized, and the train fit is not evidence of a mechanism.
- **Held out, only count stays above the always-native baseline:** 66% vs 40% on test, choosing count's answer over
  set's in both datatypes. That is well below dev (85%) and train (99%). Set's 63% is indistinguishable from leaving the
  output native (60%).
- **Consistent with Stage 0** (`stage0_matched_donor_specificity.md`): count is the only hypothesis-specific signal. It
  is not localized at sparsity 20.
- **Small and first-token only:** test is 7 families (5 string, 2 integer, 91 views). Multi-token scoring with executed
  checks would change what counts as a hit, but not the sample size, which is the binding limit.
- **Test has now been used.** Further tuning (sparsity sweep, layer bands) should be selected on dev or a fresh split.

## Still open

- **Scoring is first-token only.** Executed checks covered the SRC continuations only, not patched DST generations.
- **Held-out data is small:** 4 dev and 7 test families were accepted. There are no integer-list families outside
  train.
- **Sub-head masks are not localized at sparsity 20.** Higher sparsity (50–200) is untested.
- **The handoff runner's tests** (`set_count_smoke/runner/test_real_smoke.py`, not in the repo) still hard-code `'bade'`.

## Files

- Data: `results/set_count_v2/families-v2.json` (silico order), `families-v2-frontier-b.json` (fixed),
  `families-v2-frontier-b_checkset.json` (the matched 33-family check subset)
- Checks before the fix: `results/set_count_v2/validation/{readout_check/, source_continuations_labeled.json}`
- Checks after the fix: `results/set_count_v2/validation_frontier_b/{readout_check/, source_continuations_labeled.json}`
- Localization runs: `results/set_count_v2/train/{set,count}_{head,svd}_sp20/`; test evaluation in `…_sp20_test/`;
  job lists `job_list/set_count/set_count_v2_frontier_b_{localization,eval_test}.json`; logs `logs/set_count/`
