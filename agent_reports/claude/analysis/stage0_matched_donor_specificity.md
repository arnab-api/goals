# Stage 0 — are the query-localization masks hypothesis-specific? (matched-donor test)

*Qwen2.5-Coder-32B-Instruct · 2026-10-03 · script `scripts/diff/stage0_matched_donor_eval.py` ·
results `results/stage0_matched/Qwen2.5-Coder-32B-Instruct/<job>/`*

## TL;DR

We localized query components that make a goal-head query transplant "write" the task each hypothesis predicts
(H1 set progress, H1c count, H2 frontier position, H3 predecessor), and the masks reached 0.8–0.97 next-token
flip (design doc §9a/§9c). Stage 0 asks whether those flips are **specific**: does a mask still flip when the
source is changed *only* in the variable the hypothesis says is being transferred (a **matched donor**)?

- **Count (H1c) is specific and real.**
  - The H1c masks flip with a 2-done source and almost never with a 1-done source: S = +0.83 (all-heads
    sub-head mask), +0.68 (head mask).
  - It is not prompt length: token-length-matched padding behaves like the 1-done source.
  - It is mostly the number of completed tasks, not the stub letter. The sub-head masks do also use the letter.
- **Predecessor (H3) and position (H2) are not.** The H3 mask flips at 0.93 with the primary donor and 0.89 with
  a donor whose predecessor is different (S = +0.03). The H2 mask: S = +0.08. Their high §9 flips do not show
  that these variables are transferred.
- **The "set-progress" (H1) masks behave as count masks.**
  - Swapping *which* task is done (same count) barely moves them: S = +0.10 to +0.31.
  - Removing one done task (count −1) kills them: S up to +0.77.
- **Within the goal heads, every hypothesis is weak** (goal-only sub-head masks S ≤ +0.27; goal-250 ≤ +0.13).
  ctrl-250 is 0 everywhere.

## 1. Background: what we were testing

### The q-transfer setup (NRGS)

At the pivot `=` of a multi-task prompt, transplant the source (SRC) prompt's pre-RoPE goal-head **query** into the
destination (DST) prompt's last token. If the query carries "what to do next", the DST should change its next
operation. Notation: letters are tasks, `'` = already implemented, the arrow gives the native next task; e.g.
`a' b c' -> b` lists tasks a, b, c with a and c done, so the model's next task is b.

### Hypotheses about what the query carries (design doc §2)

| | the query carries | DST should then write |
|---|---|---|
| H0 | nothing usable | DST's own next task (no-op) |
| H1 set progress | *which* tasks are done in SRC | DST's first task not done in SRC |
| H1c count progress | *how many* tasks are done | DST's task at slot = that count |
| H2 frontier position | SRC frontier's position in the spec | DST's task at the same position |
| H3 predecessor ("induction") | the task the SRC frontier follows | the task following that one in DST (order-matched sources only) |
| frontier identity | SRC's next task itself | SRC's frontier (App I.3: the goal heads mostly carry this) |

### What §9 found, and why it was not enough

For each hypothesis we built contrasts where it makes a *unique* prediction, and trained masks over heads (§9a)
or W_QK singular components (§9c) so the transplant flips DST's next token to that prediction. Flips were high
(sub-head all-heads masks 0.82–0.97, H3 highest). The design doc (§9f) read this as an ordering of what the goal
heads carry: frontier identity > H3 ≈ H2 > H1c.

The gap: a mask trained only to *produce the target token* can succeed for reasons unrelated to the hypothesis.
The candidate pool is very large: 184k components in layers 20–55. Only H1c had a working control, the legacy
SRC′, which removes a done task; its count mask did collapse, 0.94 → 0.14. For H2 and H3 the legacy control is
undefined, and for H1 it changes the count, not the set. Without a control that changes only the hypothesized
variable, a high flip rate cannot distinguish "transfers the predecessor" from "pushes toward this target token
from any source".

### Why "Stage 0"

A full implementation of matched controls touches the config registry, renderer, cache identity, trainer and
evaluator (agent_reports/silico/compact_query_localization_design.md). Before that work, Stage 0 asks the
cheaper question: **are the masks we already have specific?** It needs no retraining and no pipeline changes. It
builds donor prompts on the fly with the pipeline's own renderer and scores the frozen masks.

## 2. Design

### Contrasts and matched donors

All donors are built from the **same role → predicate assignment** as the primary instance. The destination and
its informative expression prefix are fixed. Predictions come from `nrgs_prefix.predict`.

| design | config | DST | primary donor → target | matched donor → (hypothesis predicts native) | the only thing changed |
|---|---|---|---|---|---|
| **S1** (H1) | `order_diff_2` | `a' d e c -> d` | `a' b c' d' -> b` → e | `a' b c' e' -> b` | *which* task is done (d vs e); count and frontier fixed |
| **C1** (H1c) | `h1c_1` | `a' d e -> d` | `a' b c' -> b` → e | `a' b c -> b` | number of done tasks (2 vs 1); frontier fixed |
| **P1** (H2) | `h2_1` | `a' e f b -> e` | `a' b' c d' -> c` → f | `a' c b' d' -> c` | frontier's spec position (3rd vs 2nd); written code byte-identical |
| **I1** (H3) | `h3_1` | `a' e b d -> e` | `a' b' c -> c` → d | `b' a' c -> c` | frontier's predecessor (b vs a); done set and count fixed |

The full prediction table for each design is in the design doc, §2 and §3. Every target is that hypothesis's
unique prediction, is not DST's native task, and is absent from SRC. Every matched donor's prediction under the
design's hypothesis is DST's native task.

### Extra controls

- **Legacy SRC′** (S1, OD1): the existing control, with one fewer task done. On S1 it changes the *count*, while
  the matched donor changes the *identity*, so the pair separates count from set.
- **C1 token-length padding:** the 1-done donor gets one padding line so its prompt matches the primary donor's
  token length, and the query is read at the same absolute position. Two forms: a comment (`# x x …`) and an
  unused assignment (`_ = 100…`; valid Python, grown one digit token at a time). Both cover 189/189 instances.
- **Stub letter vs count** (C1, OD1). The pipeline names variables with sequential letters, so the source's
  frontier stub letter equals its done count (2 written → `c =`). For C1 this coincides with the reference
  context's target stub. Two donors put letter and count in opposition, staying one token from the training
  distribution. Names are swapped, never reassigned.

  | donor | written | stub | count reading | letter reading |
  |---|---|---|---|---|
  | `count_low_stub_high` | `a = …` | `c =` | native | target |
  | `count_high_stub_low` | `a = …`, `c = …` | `b =` | target | native |

  A neutral-naming rendering (`result =` stubs, natural names) was considered and rejected. It would move the
  transplanted query off the distribution the masks were trained on, so a drop could not be told apart from
  leak removal.
- **OD1 / OM1** (`order_diff_1` / `order_matched_1`): the main H1 configs. They are a naming-leak check, and OM1 is
  a primary-only replication.
- **Rival check:** for each competing hypothesis whose predicted task exists in DST (S1: c; P1, I1: b), a
  reference DST context with that task next records its top-1 token. An instance is *distinguishing* if no rival
  shares the target's token. The frontier-identity rival is absent from DST in every design.

### Data

- **Instances:** the **validation split** of the dataset each design was mined into:
  `results/nrgs_prefix/<M>/{mixed,h1c,h2,h3}__python-python/`. These are the pre-fix datasets; the pivot
  re-mining did not touch them.
- **Held out:** that split is predicate-pair-disjoint from training, so for the mask trained on that dataset every
  instance is held out. Per (mask, instance), `heldout` records whether the instance's pair is outside that mask's
  training split.
- **Clean check:** only instances whose clean DST predicts the native token are scored. That is all of them:
  S1 185, C1 189, P1 53 (int only), I1 178, OD1 191, OM1 192.

### Masks (32 variants, frozen)

- **Head masks** (§9a), soft and binarized (> 0.9 in band): H1 mix (all layers, L20–55), H1c (all, L20–55), H2 (L20–55),
  H3 (all, L20–55).
- **Sub-head masks** (§9c, L20–55, λ3), soft and thresholded components (> 0.9): H1 mix, H1c, H2, H3, each with all
  heads and goal-only.
- **Fixed sets:** goal-250 and ctrl-250 (whole heads).

The masks are applied with the trainers' own hooks (`nrgs_mask.patched_logprobs`; `nrgs_svd_mask.patched_logprobs`
with `svd_bases`). **Consistency check:** Stage 0's primary flips reproduce each mask's own §9 validation flips on
the same config (H1c on `h1c_1` 0.95 = 0.95; H1 mix on `order_diff_2` 0.80 = 0.80; H2 on `h2_1` 0.83 = 0.83;
H3 on `h3_1` 0.93 vs 0.92).

### Metrics (per instance, then averaged)

- **flip:** DST's top-1 next token becomes the target token.
- **native:** top-1 stays DST's native token.
- **specificity S** = flip(primary) − flip(control). Reported only for controls whose expected outcome is native.
- **control correct:** the control does what its hypothesis reading predicts (flip, or stay native).
- **joint:** the primary flips AND the control is correct, on the same instance.
- **paired log-odds difference:** (target − native log-odds) with the primary donor minus with the control.

## 3. Results

### 3.1 Specificity on each mask's own design

S with the matched donor, soft masks. The held-out (`True`) and rival-distinguishing subsets give the same values
to ±0.07.

| design (hypothesis) | all-heads sub-head mask | goal-only sub-head mask | head mask | goal-250 | ctrl-250 |
|---|---|---|---|---|---|
| **C1 count (H1c)** | **0.95 → 0.12 (+0.83)** | 0.48 → 0.28 (+0.20) | **0.69 → 0.02 (+0.68)** | +0.11 | +0.01 |
| S1 set identity (H1) | 0.80 → 0.49 (+0.31) | 0.48 → 0.29 (+0.19) | 0.36 → 0.25 (+0.10) | +0.06 | 0.00 |
| P1 position (H2), n = 53 | 0.83 → 0.75 (+0.08) | 0.57 → 0.36 (+0.21) | 0.42 → 0.28 (+0.13)¹ | +0.13 | 0.00 |
| **I1 predecessor (H3)** | **0.93 → 0.89 (+0.03)** | 0.73 → 0.69 (+0.04) | 0.69 → 0.61 (+0.08) | +0.13 | 0.00 |

Each cell reads: flip with the primary donor → flip with the matched donor (S). ¹ The H2 head mask is the
layer 20–55 one; there is no all-layer H2 mask.

### 3.2 Every mask × every control (S, soft masks)

| mask | S1 matched (identity) | S1 legacy (count −1) | C1 matched (count) | P1 matched (position) | I1 matched (predecessor) | OD1 legacy (count −1) |
|---|---|---|---|---|---|---|
| sub-head H1 mix, all | +0.31 | **+0.77** | **+0.89** | −0.09 | +0.13 | **+0.85** |
| sub-head H1c, all | +0.04 | **+0.69** | **+0.83** | +0.08 | +0.08 | **+0.81** |
| sub-head H2, all | +0.02 | +0.30 | +0.44 | +0.08 | +0.04 | +0.32 |
| sub-head H3, all | +0.08 | +0.10 | +0.49 | +0.17 | +0.03 | +0.30 |
| sub-head H1 mix, goal-only | +0.19 | +0.27 | +0.26 | +0.13 | +0.06 | +0.28 |
| sub-head H1c, goal-only | +0.08 | +0.10 | +0.20 | +0.13 | +0.07 | +0.15 |
| sub-head H2, goal-only | +0.05 | +0.17 | +0.19 | +0.21 | +0.03 | +0.16 |
| sub-head H3, goal-only | +0.06 | +0.03 | +0.11 | +0.13 | +0.04 | +0.05 |
| head H1 mix, all | +0.10 | +0.34 | +0.58 | +0.09 | +0.04 | +0.64 |
| head H1c, all | +0.10 | +0.32 | +0.68 | +0.04 | +0.03 | +0.76 |
| head H2, L20–55 | +0.02 | +0.12 | +0.25 | +0.13 | +0.04 | +0.27 |
| head H3, all | +0.08 | +0.21 | +0.40 | +0.09 | +0.08 | +0.41 |
| goal-250 | +0.06 | +0.00 | +0.11 | +0.13 | +0.13 | +0.07 |
| ctrl-250 | 0.00 | 0.00 | +0.01 | 0.00 | 0.00 | 0.00 |

The only large specificities are in the count columns (C1 matched, S1 legacy, OD1 legacy). They are large for the
H1 mix and H1c masks, and moderate even for masks trained on H2 or H3. Every set-identity, position and
predecessor column is ≤ +0.31.

### 3.3 C1: count, token length, or stub letter?

Flip rate of each C1 control donor (low = behaves like a 1-done source):

| mask | primary (2 done) | matched (1 done) | comment pad | assignment pad | 1 done, stub `c` | 2 done, stub `b` |
|---|---|---|---|---|---|---|
| head H1c, all, soft | 0.69 | 0.02 | 0.03 | 0.04 | 0.10 | **0.55** |
| head H1c, L20–55, bin | 0.57 | 0.04 | 0.04 | 0.04 | 0.06 | **0.44** |
| sub-head H1c, all, soft | 0.95 | 0.12 | 0.11 | 0.19 | 0.49 | **0.85** |
| sub-head H1 mix, all, soft | 0.92 | 0.03 | 0.03 | 0.08 | 0.22 | **0.79** |
| sub-head H1c, goal-only, soft | 0.48 | 0.28 | 0.22 | 0.18 | 0.23 | 0.40 |
| goal-250 | 0.34 | 0.23 | 0.22 | 0.17 | 0.27 | 0.30 |

- **Not token length or capture position:** both length-matched pads behave like the 1-done source.
- **Not "number of statements":** an unused assignment does not count.
- **Mostly count, partly letter:**
  - With 2 tasks written and the stub forced to `b`, the masks still flip (0.44–0.85).
  - With 1 task written and the stub forced to `c`, the head masks stay near 0 (0.06–0.10).
  - The sub-head masks rise to 0.22–0.49, so part of their effect does come from the stub letter.
- **OD1 shows the same pattern.** Sub-head H1c, all heads: primary 0.94, 1 done with stub `c` 0.54, 2 done with
  stub `b` 0.89.

### 3.4 Set progress vs count (S1)

On S1, the H1 mix masks barely distinguish *which* task is done (matched S +0.10 to +0.31). Removing a done task
(legacy) nearly abolishes them: sub-head H1 mix, all heads, 0.80 → 0.03 (S +0.77); head mask S +0.34. The same holds
for the H1c masks (+0.69 / +0.32). What the "set progress" masks respond to is **how many** tasks are done.

## 4. Interpretation

1. **The flips in §9 measured target-token production, not variable transfer.** For H3 (I1) and H2 (P1), changing
   the hypothesized variable leaves the flip rate almost unchanged, so the masks are not reading it. A large
   component mask can push DST toward the trained target from either donor's query. The design-doc ordering
   "H3 ≈ H2 > H1c" (§9f) rested on these flips and should be withdrawn.
2. **What does survive is a completed-task count.**
   - It is specific: S +0.68 to +0.89 for the count-trained masks, and +0.25 to +0.49 even for masks trained on
     other hypotheses.
   - It is not length or position, and it is not counting statements.
   - It is partly mixed with the stub letter in the sub-head masks.
   - It is what the H1 "set progress" masks carry too.
3. **The goal heads carry little of it.**
   - Goal-only sub-head masks reach S ≤ +0.27, and goal-250 ≤ +0.13.
   - This agrees with §9: the progress signal sits mostly outside the goal heads. Also App I.3: the goal heads
     carry the frontier's identity (c), not progress.
4. **Mask format changes the size of the effects, not the pattern.**
   - Head masks are cleaner than sub-head masks on C1's letter test, but produce smaller flips.
   - Thresholding the sub-head masks (components > 0.9) cuts some effects sharply. For the all-heads H1 mix mask on
     C1, S drops from +0.89 (soft) to +0.34, because its primary flip falls from 0.92 to 0.37.
   - The count-specific / H2-H3-non-specific split holds in every format; see `summary.json` for the `|bin` and
     `|comp` variants.

## 5. Caveats

- **These are frozen masks trained with `w_ctrl = 0`.** This tests whether the masks we *have* are specific, not
  whether a specific H2 or H3 mask *exists*. Positive-only training gives the optimizer no reason to prefer
  hypothesis-specific components, and these results show it did not find them.
- **P1 is small** (53 int-only instances). Its S values have wide uncertainty, about ±0.15.
- **Next-token only.** Free-generation behaviour of the same donors is untested.
- **Clean source validity is not checked.** That the clean source and control donors would write their own
  frontier is not established here; only the clean DST (native) is checked.
- **The S1 matched donor also changes the source's task inventory** ({a,b,c,d} → {a,b,c,e}), not only which task
  is done.
- **Legacy SRC′ and the matched C1 donor are the same prompt** for C1 (`a' b c -> b`).
- **The summary code had three bugs** in the first version, found by an external audit and fixed:
  - `heldout=None` was counted as held out;
  - `count_high_stub_low` was scored as if it should stay native;
  - OM1 was missing.

  Summaries were rebuilt from the saved rows (`--resummarize`). None of the numbers above changed.

## 6. Next steps

1. **Shuffled-donor control (I1, P1).** Transplant a query from an *unrelated* instance. If the H3 and H2 masks still
   flip, they push toward DST's target regardless of the source, which would confirm reading 1 directly.
2. **Control-supervised retraining.** Retrain H2, H3 and the count masks with matched donors in the loss.
   - Use a native-preserving term, not the current anti-target hinge, which does not require native to win.
   - Compare with `w_ctrl = 0` on the same splits, and score on held-out matched families.
3. **Stub-letter-free count.** Repeat C1 with the letter/count conflict donors in training too, to separate the
   count signal from the letter cue in the sub-head masks.
4. **Update design-doc §9.** Mark the §9a/§9c H2/H3 flips as non-specific, and restate §9f around count.
5. **Generation check.** Run the matched donors through the hypothesis generation eval
   (`q_transfer_hypothesis_eval.py`, which needs `--component-mask` support).

## 7. Reproduction and files

- Script: `scripts/diff/stage0_matched_donor_eval.py`
  - `--masks heads | svd:<mix|h1c|h2|h3>`, `--only <regex>`
  - `--limit` (smoke)
  - `--resummarize` (rebuild `summary.json` from the saved rows, no GPU)
- Jobs: `job_list/nrgs/stage0_matched_donors_32b.json` (8 single-GPU jobs, about 2 h each).
- Results, per job in `results/stage0_matched/Qwen2.5-Coder-32B-Instruct/<job>/`:
  - `rows.jsonl`: per (mask, design, donor, instance) flip / native / log-odds / held-out / rival flags;
  - `prompts.jsonl`: every rendered donor, DST and rival prompt with token ids;
  - `summary.json`: `controls` (paired metrics on all / distinguishing / held-out), `primary` (primary-only,
    includes OM1), `donor_coverage`.
- Inputs: masks in `results/nrgs_mask/<M>/` and `results/nrgs_svd_mask/<M>/`; datasets in
  `results/nrgs_prefix/<M>/{mixed,h1c,h2,h3}__python-python/`. All are backed up under
  `results/backups/pre-contextual-token-filter_20261003/`.
- Design and review: `agent_reports/silico/compact_query_localization_design.md`, plus the audit in
  `agent_reports/silico/findings.md`.
