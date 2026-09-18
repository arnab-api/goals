# Predicate-family overlaps in the QTASKS banks (diff-trigger confound audit)

*2026-09-18. Audit of the 4 datatype predicate banks (`src/diff/data.py::QTASKS`) for the
"overlapping family" confound in the diff-trigger q-transfer experiment
(`scripts/diff/diff_trigger_eval.py`, `DiffTriggerInstance`).*

## Why this matters

Each diff-trigger instance samples **4 distinct predicates** → roles **A, B** (shared/prefilled),
**C** (SRC's transplant target), **D** (DST's remaining = the DIFF-predicted outcome). The causal
test asks: does transplanting SRC's query make DST write **D** (its own remaining task) instead of
**B** (native)? This is only clean when **B, C, D are semantically distinct**. If two of them come
from the same *feature family* (e.g. both are "length" ops, or logical complements like even/odd),
the query can't cleanly separate them and the causal verdict is confounded — we saw this repeatedly in
the n=8 smokes (e.g. B=`count_ends_consonant` vs D=`count_starts_vowel`, both vowel/consonant; or
A=`shortest_word_length` ≈ D=`longest_word_length`, both length). The sampler currently draws 4
predicates **uniformly at random with no family constraint** (and does **not** exclude trivial
frontiers), so a large fraction of instances carry a hidden overlap.

## Confound exposure (fraction of random 4-subsets with ≥1 same-family pair)

| datatype | #families | largest family | **P(≥1 same-family pair in a random A/B/C/D)** |
|---|---|---|---|
| int | 6 | decimal_digits (5) | **72.6%** |
| str | 5 | charclass_count (6) | **88.8%** |
| list[int] | 5 | sign (4) | **71.8%** |
| list[str] | 4 | word_length (6) | **88.1%** |

So **~72–89% of randomly sampled instances contain a same-family collision** somewhere in {A,B,C,D}.
`str` and `list[str]` are worst (one dominant 6-member family each); `int`/`list[int]` are milder but
still majority-confounded. This directly explains why hand-picked "clean" bad cases were hard to find
in the smokes.

## Family breakdown per datatype

### int (16 preds; 3 TRIVIAL: is_positive, is_negative, sign)
- **decimal_digits (5):** digit_count, digit_sum, last_digit, is_palindrome, is_two_digit — all read the
  decimal representation `str(abs(n))`.
- **parity (2):** is_even, is_odd — **logical complements** (`n%2==0` vs `n%2!=0`); worst-case overlap.
- **sign_magnitude (4):** is_positive*, is_negative*, sign*, absolute_value (3 trivial).
- **mult_structure (3):** is_prime, is_perfect_square, is_power_of_two.
- divisibility (1): is_multiple_of_3 · bits (1): count_set_bits.

### str (16)
- **charclass_count (6):** vowel_count, count_letters, count_digits, count_spaces, count_uppercase,
  count_lowercase — all `sum(1 for c in s if <class>)`; a large, highly-interchangeable family.
- **charclass_bool (6):** has_digit, has_space, is_alnum, all_lowercase, starts_upper, first_is_letter.
- length_words (2): length, word_count · palindrome (1) · last_char (1).
- **Cross-family same-feature pairs:** `count_digits`↔`has_digit`, `count_spaces`↔`has_space`,
  `count_uppercase`↔`starts_upper`/`all_lowercase`↔`count_lowercase` — the count and bool of the *same*
  character class (in different families above) are still confounding.

### list[int] (16; 4 TRIVIAL: count_positive, count_negative, all_positive, contains_zero)
- **parity (3):** has_even, count_even, count_odd — count_even/count_odd are near-complements.
- **sign (4):** count_positive*, count_negative*, all_positive*, contains_zero* (all trivial).
- **extremes (3):** max_val, min_val, value_range — `value_range = max − min` (derivable from both).
- **aggregate (3):** total_sum, product_all, mean_int — `mean_int = sum // len` (overlaps total_sum).
- **uniq_order (3):** has_duplicate, distinct_count, is_sorted — has_duplicate ↔ distinct_count.

### list[str] (16) — worst
- **word_length (6):** total_length, count_long, count_short, longest_word_length, shortest_word_length,
  all_unique_lengths — all functions of `len(w)`; dominant, heavily self-confounding family.
- **vowel_consonant (3):** count_starts_vowel, count_ends_consonant, total_vowels.
- **case (3):** count_capitalized, count_all_lowercase, total_uppercase_letters.
- content (4): count_with_digit, has_palindrome, any_empty, distinct_count.

## Worst specific offenders (avoid as B/D pairs)
- **Logical complements:** int `is_even`/`is_odd`; list[int] `count_even`/`count_odd`.
- **Derivable:** list[int] `value_range`=`max_val`−`min_val`; `mean_int`≈`total_sum`; `distinct_count`↔`has_duplicate`.
- **Same feature, count-vs-bool:** str `count_digits`/`has_digit`, `count_spaces`/`has_space`.
- **Same operator on a shifted threshold:** list[str] `count_long`(len>5)/`count_short`(len<3);
  `longest_word_length`/`shortest_word_length` (max vs min len).

## Second issue found: trivial frontiers not excluded
`diff_trigger_setups` does **not** filter `TRIVIAL_FRONTIERS` from the C/D roles, so a trivial
sign/zero predicate (int: is_positive/is_negative/sign; list[int]: count_positive/…/contains_zero) can
land as **D** — the causal target — where the no-goal default (`value>0` etc.) inflates/deflates the
score (the same reason OVReader/goalscope sanitize these). This is independent of the family issue but
compounds it for int and list[int].

## Recommendation

Add a **distinct-family + non-trivial-frontier** constraint to `diff_trigger_setups` when sampling
A/B/C/D:
1. Attach a `family` tag to each predicate (the mapping in this report, portable to code — e.g. a
   `PREDICATE_FAMILY[dt]` dict, or a `family` field on `Predicate`).
2. Require **B, C, D to come from three different families** (A may reuse a family — it's only shared
   context; or require all four distinct for maximum cleanliness).
3. Require **C and D non-trivial** (reuse `TRIVIAL_FRONTIERS`); trivial predicates may still appear as
   A/B context.

Feasibility: every datatype has ≥4 non-trivial families, so distinct-family 4-tuples exist —
- int: 6 families (5 non-trivial-heavy) → ample.
- list[int]: 5 families (sign family is all-trivial → 4 usable) → exactly enough for B/C/D distinct + A.
- str: 5 families → ample. · list[str]: 4 families → B/C/D from 3 of the 4; workable but tightest.

With the constraint, the confound exposure drops from ~72–89% to ~0, at the cost of a smaller instance
pool (fine at n=32). **This should be applied before the final cross-datatype / cross-lingual run** so
the causal→D numbers are interpretable.

*(Family map + exposure computed in `scratchpad/families.py`; predicate dump in `scratchpad/dump_preds.py`.)*
