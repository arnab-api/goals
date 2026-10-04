# Set/count localization v2: the source-order confound and the per-donor fix

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

## Still open

- **Validation is next-token only.** Only the SRC continuations were executed. DST and teacher completions are not.
- **Nothing has been trained on v2.** The only real-model optimization result is the v1 smoke (0/9 set, 0/18 count),
  which used the old `result =` design.
- **Held-out data is thin:** dev is 7 string families, test is 14 (string and integer).
- **The handoff runner's tests** (`set_count_smoke/runner/test_real_smoke.py`, not in the repo) still hard-code `'bade'`.

## Files

- Data: `results/set_count_v2/families-v2.json` (silico order), `families-v2-frontier-b.json` (fixed),
  `families-v2-frontier-b_checkset.json` (the matched 33-family check subset)
- Checks before the fix: `results/set_count_v2/validation/{readout_check/, source_continuations_labeled.json}`
- Checks after the fix: `results/set_count_v2/validation_frontier_b/{readout_check/, source_continuations_labeled.json}`
