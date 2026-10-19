# Gaze-select + restrict (gsr): query transfer picks the task, a span restriction carries it out

*Qwen2.5-Coder-32B-Instruct · 2026-10-07 → 2026-10-10 · notebook `notebooks/diff/04.4_q_transfer_multipred.ipynb`
(final "Gaze-select + restrict" cells) · batch `scripts/diff/diff_trigger_eval.py --closed-loop --probe-prompt
gaze_select_restrict --gsr-anchors eq` · status (2026-10-10): **all arms complete** — 32 no-kv cells,
32 kv gsr cells rerun with the gaze fix (`__seq__gzfix`), and all 32 kv re-clamp baselines (run `kv_gzfix2_20261008`,
finished 2026-10-10 03:00) · companion to `closed_loop_query_transfer.md`*

> **What changed since the 10-08 snapshot.** (i) All kv gsr numbers are now from the corrected selector; the
> withdrawn kv claim is resolved — **kv adds nothing to gsr**: with the selection scored on p_kv's branch it picks the
> same task as the no-kv run in 99 % of matched instances (§4.2, finding 4). (ii) The no-kv grid is complete
> (C++→py order-matched, C++→C++ both orderings). (iii) Selection, execution-audit and B-prior tables are recomputed on
> all records. (iv) **2026-10-10:** the kv re-clamp baselines are complete. kv gsr beats the kv re-clamp in all 32 kv
> cells (finding 7), and kv leaves the re-clamp as unchanged as it leaves gsr. Conclusions 1–3 and 5–6 are unchanged.

> **Update 2026-10-19: random-head control (§4.5).** The no-kv gsr arm reported in the paper (Fig. 25, Tables 8–9),
> re-run with 250 random later-half non-goal heads in place of goal-250, gives **0.036** judged D (goal-250: 0.72). The
> control heads' gaze is moved to D just as often (0.74 vs 0.74; clean 0.02), but restricting *them* to D's span almost never
> produces D (P(D | selected D) = 0.03 vs 0.98). The goal-head-specific step is the execution, not the selection.

## 1. Motivation

The paper's q-transfer **re-clamps** the SRC pivot query of the goal heads at DST's pivot and at the current token for
the first decode steps. Once the model starts writing, one fixed query is a poor fit for every generated token, and
the earlier closed-loop variants (regenerating the query from a probe; see the companion report) produced many
degenerate "successes".

gsr separates the two roles the query was playing:

1. **Selection.** The transferred query is used only to read *which* listed task the pivot's goal heads now attend to.
2. **Execution.** A fresh DST run with **no query transfer** in which the goal heads, from the pivot onward, can only
   read that task's span (`clock/09` §5, `generate_restricted_to_span`).

If the transferred query really carries "pursue D next", the pivot gaze should move to D's span, and confining the
goal heads to that span should be enough for DST to write D.

## 2. Setup

Same tasks as the companion report (04.3/04.4 notation; `'` = already written):

| design | prompts | DST writes (clean) | prediction |
|---|---|---|---|
| no kv, order-diff | `p_src A'CB' → C` · `p_dst A'BD → B` | B | **D** |
| no kv, order-matched | `p_src A'B'C → C` · `p_dst A'BD → B` | B | D |
| kv | `p_src A'CB' → C` · `p_dst A'XY → X` · `p_kv A'BD → B` | X | kv-only → B · kv+q: set progress → **D**, count → **Y** |

Common configuration: goal-250 heads (`goal__lam20_top250`, used as patch, read-out and restricted heads); OV boost
α = 2 only on the `ro_patched` readout (the selection pass and the restricted generation are unboosted); pivot = the frontier's ` =` (`"\n    {res} ="` in Python, `"\n    auto {res} ="` in C++);
sequential letter naming (`__seq`); 32 instances × 4 datatypes (int, str, list[int], list[str]) per cell; all four
language directions. Judges grade only the FIRST computation: `judge_joint` (exclusive B / C / D / neither) without kv;
one exclusive {X, Y, B, C, D} verdict with kv.

## 3. The procedure

**Pass 1: select.** DST forward with the SRC pivot query patched in at the prefill anchors (+ p_kv's task-list
key/value on the goal heads with kv; no boost, no restriction). Value-weighted attention (attention × ‖v‖) from the
pivot to each listed task's tokens, averaged over the goal-250 heads and the task's tokens. The top-scoring task is
kept. Options are DST's listed tasks (A, B, D) without kv and p_kv's (A, B, D) with kv.

**Pass 2: restricted generation.** A separate greedy DST run with **no query transfer, no boost, no ablation** (kv
still on with kv). `whitelist_head_attn`: for every goal head, query rows at position ≥ pivot may attend only to:
- the selected task's span (p_kv's tokens with kv, where its key/value sit);
- the pivot token itself (` =`);
- every generated token.

All other keys are masked pre-softmax. Prompt tokens before the pivot and all non-goal heads are untouched. Up to 64
new tokens, no statement cut-off.

**Anchors.** The first gsr grid also whitelisted the indent and the result-variable token (`indent var eq`). With them,
the D rate on py→py order-diff was 0.38 / 0.16 / 0.41 / 0.46 without kv (eq-only: 0.59 / 0.62 / 0.78 / 0.66). With kv
it was 0.64 / 0.20 on int / str (eq-only: 0.78 / 0.75). Only the pivot token is kept from here on (`--gsr-anchors eq`,
run tag `-anceq`).

## 4. Results

> **Correction (2026-10-08; resolved 2026-10-09 — all kv numbers below are from the `__seq__gzfix` rerun).** The
> kv-on selection was measured on the wrong attention branch. Under kv, the
> headwise kv wrapper routes the donor (p_kv) branch's OUTPUT into the goal heads but returned the recipient (p_dst)
> branch's attention weights, and value-weighted gaze used the cache's recipient values (reported in
> `agent_reports/silico/kv_gaze_branch_mismatch.md`; confirmed in code). The kv selector therefore scored p_dst's own
> text (`A'XY`) at the positions labelled with p_kv's tasks: "gaze selected D" in practice meant "the SRC query
> attends to the p_dst text in D's aligned slot" (roughly Y, the 3rd item). The restricted generation itself is
> correct (it reads D's real k/v there), which is why P(D | selected D) holds. Consequences:
> - **kv-on generation D rates below are real outcomes,** but of a selection made on the recipient branch, so they do
>   not show which p_kv task the goal heads pick;
> - **the kv selection rates in §4.2 and the mechanism claim of finding 4 are not supported;**
> - **every kv-off number is unaffected,** since there is no routing and the two branches coincide.
>
> Fixed in `src/diff/functional.py` (routed heads return the donor-branch weights; effective per-head value norms are
> recorded for `get_attention_matrices`) and verified by `test/test_kv_gaze_effective_branch.py`. All 32 kv gsr cells
> were rerun under the run tag `__seq__gzfix` (old records kept under `…__seq/`). The legacy selector had picked D
> more often (order-diff .67–.75 vs .51–.67 now), so the legacy pooled kv D rates were inflated by 0.05–0.16 (e.g. py→py order-diff .74 → .66, C++→py order-diff
> .67 → .51).

### 4.1 Judged D rate (int / str / list[int] / list[str])

Re-clamp = the paper's arm (no-kv order-matched baseline from the older `ph-goal_250__a2__piv-eq` run). kv gsr = the
`__seq__gzfix` rerun. kv re-clamp = re-clamp with p_kv's task-list key/value on the goal heads
(`ph-goal_250__a2__piv-eq__kv__seq`).

| direction | arm | order-diff | order-matched |
|---|---|---|---|
| **py→py** | re-clamp | .62 .41 .47 .38 | .66 .53 .56 .69 |
| | **gsr** | **.59 .62 .78 .66** | **.91 .81 .91 .88** |
| | kv re-clamp | .56 .34 .43 .50 | .66 .59 .59 .59 |
| | **kv gsr** | **.66 .59 .73 .66** | **.91 .84 .88 .91** |
| **py→C++** | re-clamp | .56 .31 .28 .12 | .72 .53 .44 .56 |
| | **gsr** | **.66 .47 .50 .59** | **.91 .75 .84 .88** |
| | kv re-clamp | .44 .22 .23 .19 | .75 .53 .47 .44 |
| | **kv gsr** | **.66 .47 .43 .59** | **.94 .69 .81 .81** |
| **C++→py** | re-clamp | .31 .22 .44 .25 | .69 .66 .56 .69 |
| | **gsr** | **.50 .41 .66 .47** | **.91 .78 .91 .84** |
| | kv re-clamp | .34 .28 .37 .25 | .62 .66 .61 .53 |
| | **kv gsr** | **.50 .41 .67 .47** | **.91 .84 .94 .88** |
| **C++→C++** | re-clamp | .41 .19 .38 .25 | .84 .50 .56 .44 |
| | **gsr** | **.62 .47 .72 .41** | **.97 .75 .88 .91** |
| | kv re-clamp | .53 .19 .32 .19 | .69 .50 .59 .31 |
| | **kv gsr** | **.59 .47 .71 .47** | **.88 .75 .81 .94** |

Pooled over datatypes (exclusive labels; n = 126–128 per cell unless noted):

| direction · ordering | re-clamp D / B | gsr D / B | kv re-clamp D / B | kv gsr D / B |
|---|---|---|---|---|
| py→py · order-diff | .47 / .43 | **.66** / .34 | .46 / .43 | **.66** / .33 |
| py→py · order-matched | .61 / .31 | **.88** / .11 | .61 / .31 | **.88** / .11 |
| py→C++ · order-diff | .32 / .66 | **.55** / .44 | .27 / .63 | **.54** / .44 |
| py→C++ · order-matched | .56 / .36 | **.84** / .13 | .55 / .32 | **.81** / .13 |
| C++→py · order-diff | .30 / .62 | **.51** / .49 | .31 / .58 | **.51** / .49 |
| C++→py · order-matched | .65 / .27 | **.86** / .11 | .61 / .26 | **.89** / .09 |
| C++→C++ · order-diff | .30 / .68 | **.55** / .43 | .31 / .63 | **.56** / .42 |
| C++→C++ · order-matched | .59 / .37 | **.88** / .12 | .52 / .38 | **.84** / .14 |

gsr outputs are almost entirely D or B. "neither" drops from 1–10 % (re-clamp; 6–10 % for kv re-clamp) to 0–5 %, and C, the literal copy of
SRC's own target, occurs at most once per cell. With kv, count (Y) never occurs under gsr, but this is by construction
(§4.2): the restricted heads can only read D's span. Under the kv re-clamp, which has no restriction, Y is also rare:
8 of 1016 pooled (≤ 0.023 in every direction × ordering), against D 462 of 1016.

### 4.2 Selection fidelity: the outcome is decided by which task gaze selects

All gsr-eq records pooled (no kv + fixed kv, n = 2040):

| | n | D rate |
|---|---|---|
| gaze selected D | 1500 (74 %) | **0.972** |
| gaze selected another task | 540 | **0.000** |

Per cell, P(D | selected D) is 0.94–1.00. When gaze does not pick D it picks B (the native task) in 532 of 540 cases,
and A, which is already written, in the rest.

The D-selection rate depends on the condition (no kv / kv):

| | order-diff | order-matched |
|---|---|---|
| py→py | .68 / .67 | .90 / .89 |
| py→C++ | .56 / .55 | .88 / .87 |
| C++→py | .51 / .51 | .90 / .90 |
| C++→C++ | .58 / .59 | .90 / .88 |

**With kv, the selection is the no-kv selection.** Matching kv and no-kv records on the same (SRC, DST) instance
(n = 1016), the two runs select the same task in **1007 (99.1 %)**; the per-task gaze scores differ by a median of
0.003 (p90 0.010), and the D outcome agrees in 96 %. This is expected once the selector reads the right branch: with
kv, the goal heads' task-list keys come from `p_kv = A'BD`, which is exactly the no-kv DST prompt, and the SRC query is
the same. The legacy selector's higher kv D-selection (.67–.75 on order-diff) came from scoring p_dst's own text.

**No-query control.** On the clean DST (no query transfer), the pivot gaze never prefers D over B: 0 of 1024 no-kv
records. The move of the gaze to D is therefore caused by the transferred query, and the restriction turns that move
into a D generation almost without loss.

### 4.3 Execution audit (Python destinations)

The first generated expression is run on 70–80 inputs and compared with the bank's reference implementation of D.
"correct" means ≥ 0.9 agreement and "plausible" ≥ 0.6. The overall column is correct-D over all records.

| direction · ordering · kv | arm | judged D | correct | plausible | correct D overall |
|---|---|---|---|---|---|
| py→py · od · no kv | re-clamp | 60 / 128 | .65 | .75 | .30 |
| | **gsr** | 85 / 128 | **.81** | **.96** | **.54** |
| py→py · od · kv | re-clamp | 58 / 126 | .57 | .64 | .26 |
| | **gsr** | 83 / 126 | **.83** | **.95** | **.55** |
| py→py · om · no kv | re-clamp | 78 / 128 | .64 | .87 | .39 |
| | **gsr** | 112 / 128 | **.79** | **.96** | **.70** |
| py→py · om · kv | re-clamp | 78 / 128 | .68 | .91 | .41 |
| | **gsr** | 113 / 128 | **.77** | **.95** | **.68** |
| C++→py · od · no kv | re-clamp | 39 / 128 | .44 | .64 | .13 |
| | **gsr** | 65 / 128 | **.89** | **.95** | **.45** |
| C++→py · od · kv | **gsr** | 64 / 126 | .88 | .94 | .44 |
| C++→py · om · no kv | re-clamp | 83 / 128 | .73 | .86 | .48 |
| | **gsr** | 110 / 128 | **.85** | **.99** | **.73** |
| C++→py · om · kv | **gsr** | 113 / 127 | .82 | .96 | .73 |

The re-clamp's judged successes contain syntax errors, crashes, and expressions that reference other variables. gsr's
contain almost none of these. Its non-correct cases are mostly "partial" (edge-case disagreements such as `abs` /
negatives / empty input), so the judged-D rate overstates gsr much less than it overstates the re-clamp. C++
destinations are not audited (no executor).

### 4.4 The B prior

Because gsr fails only when gaze stays on B, the B prior is a property of the selection pass. Per B task (no kv, all
four directions and both orderings pooled; n per task is 7–44, so this is noisy):
- **B tasks that hold the gaze most often:** "all words have distinct lengths" (list[str], 0.82 native of 11), word
  count (str, 0.70 of 30), `abs` (int, 0.67 of 33). The re-clamp is about equally sticky on these (.64 / .63 / .70).
- **B tasks the re-clamp leaves native far more often than gsr:**

  | B task | re-clamp native | gsr native |
  |---|---|---|
  | number of distinct values | .92 | .25 |
  | largest value | .91 | .32 |
  | string length | .85 | .08 |
  | lowercase words | .79 | .46 |
  | words that start with a vowel | .54 | .17 |

  For these tasks, the re-clamp's failure is in execution, not in where the heads look.
- **No B task is sticky under every arm.** The earlier claim that `count_set_bits`, `count_letters` and `max_val` are
  100 % native everywhere was made on the earlier, smaller sample and does not survive pooling. `count_set_bits` is
  0.40 native under gsr and 0.70 under the re-clamp.

### 4.5 Random-head control (added 2026-10-19)

The **no-kv gsr arm exactly as reported in the paper** (Fig. 25, Tables 8–9: α = 2, eq pivot, closed loop with
`--probe-prompt gaze_select_restrict --gsr-anchors eq`, `__seq` naming, same seeded instances, 32 cells × 32), with
`--patch-heads` and `--readout-heads` both set to **250 random heads from layers 32–63 with no goal head**
(`head_sets/…/control/control250__laterhalf.json`). So the control heads receive the SRC query, their pivot gaze makes
the selection, and they are the heads restricted to the selected span; everything else is unchanged. Run with
`--no-readout` (the GoalScope readouts only; selection and gaze are still computed). Run tag
`ph-control250_250__a2__piv-eq__cl-ppgazerestrict-anceq__seq`, job list `job_list/diff_eval/diff_trigger_gsr_eq_control250.json`.

**Judged D rate** (int / str / list[int] / list[str]; same judge and layout as §4.1; n = 32 per cell)

| direction | arm | order-diff | order-matched |
|---|---|---|---|
| **py→py** | gsr (goal-250) | .59 .62 .78 .66 | .91 .84 .91 .88 |
| | **gsr (control-250)** | **.09 .03 .03 .00** | **.31 .03 .00 .00** |
| **py→C++** | gsr (goal-250) | .66 .47 .50 .59 | .91 .75 .84 .88 |
| | **gsr (control-250)** | **.06 .03 .00 .00** | **.03 .00 .00 .03** |
| **C++→py** | gsr (goal-250) | .50 .41 .66 .47 | .91 .81 .94 .84 |
| | **gsr (control-250)** | **.12 .03 .00 .00** | **.22 .03 .03 .00** |
| **C++→C++** | gsr (goal-250) | .62 .47 .72 .44 | .97 .75 .88 .94 |
| | **gsr (control-250)** | **.03 .03 .00 .00** | **.00 .00 .00 .00** |

Pooled (n = 1024 each): goal-250 **0.72**, control-250 **0.036** (exclusive joint D 0.017; the outputs are B 96–100 %
per direction × ordering). The only non-trivial control cells are `int` with a Python destination (py→py order-matched
.31, C++→py order-matched .22); every other datatype is ≤ .03 in every cell.

**Selection vs execution under the control** (all 1024 records each):

| | goal-250 | control-250 |
|---|---|---|
| clean pivot gaze prefers D (no query transfer) | 0.000 | 0.018 |
| gaze shifted to D under the query transfer | 0.739 | 0.744 |
| gaze **selects D** | 0.737 | 0.726 |
| P(D generated \| selected D) | **0.979** | **0.032** |
| P(D generated \| selected another task) | 0.000 | 0.046 |
| judged D | 0.722 | 0.036 |

The transferred SRC query redirects the control heads' attention to D exactly as often as the goal heads' (0.74 vs
0.74, from ≈ 0 on the clean run), so **the selection step is not goal-head specific**: copying the source pivot query into
any 250 later-half heads makes them look at D. What is specific is the second step: confining the goal heads to D's
span makes DST write D 98 % of the time, while confining 250 random later-half heads to the same span almost never does
(3 %), and the control output stays B.

## 5. Findings

1. **gsr beats the re-clamp in all 32 no-kv cells but one, py→py order-diff int** (0.59 vs 0.62). The gains are
   largest where the re-clamp is weakest:
   - order-diff list types: py→py list[str] goes from .38 to .66, py→C++ list[str] from .12 to .59;
   - C++-source directions: order-diff pooled C++→py .30 → .51 and C++→C++ .30 → .55; C++→C++ order-matched
     .59 → .88.
2. **Selection is the bottleneck, and execution is nearly lossless.** P(D | gaze selected D) = 0.972 pooled
   (n = 2040), and P(D | gaze selected another task) = 0. Every improvement of gsr over the re-clamp comes from two
   facts:
   - the query transfer moves the pivot gaze to D more often (51–90 %, against 0 % on the clean DST) than the
     re-clamp gets DST to write D;
   - the restriction converts that move into a D generation almost every time.
3. **The goal heads' view of one task span is enough to make DST implement that task.** With the query gone and
   every other head untouched, confining only the goal-250 heads to D's span yields correct D implementations (§4.3:
   0.77–0.89 of judged D are execution-correct, against 0.44–0.73 for the re-clamp).
4. **kv adds nothing to gsr (resolves the withdrawn claim).** With the corrected selector, kv gsr and no-kv gsr are
   within 0.04 pooled in every direction × ordering, and they select the same task in 99 % of matched instances
   (§4.2). Under kv the selector scores p_kv's task list, which is the no-kv DST prompt, so the kv arm repeats the
   no-kv selection by construction. Two consequences:
   - **gsr cannot separate "set progress" (D) from "count progress" (Y).** Y is not a selection candidate and the
     restricted heads cannot read it, so Y = 0 is by design, not evidence. The unrestricted kv re-clamp also gives
     almost no Y (8 / 1016, against D 462 / 1016), but see the caveat in §6 before reading this as set > count.
   - **What kv gsr does show:** p_dst's own prompt lists A, X, Y and never mentions D. Confining its goal heads to D's
     *transplanted* key/value span still makes it write D at 0.51–0.89 pooled (execution-correct 0.77–0.88 of judged
     D on Python destinations). The task content carried in another prompt's k/v is executable as is.
5. **Order-matched > order-diff for both arms, with a larger margin for gsr.** Part of order-matched's advantage is
   the positional copy (D is "the 3rd listed" on both sides). Order-diff remains the clean test.
6. **Extra anchors hurt.** Letting the goal heads also read the indent and the result-variable token cuts the D rate
   by 0.14–0.55 on order-diff. Those tokens appear to carry DST's own "write B next" signal.
7. **The effect is goal-head specific, and the specificity is in execution (§4.5).** With 250 random later-half heads in
   place of goal-250, judged D falls from 0.72 to 0.036. Their gaze still moves to D (0.74, as for the goal heads), but
   restricting them to D's span does not make DST write D (0.03 vs 0.98). This supports finding 3 (the goal heads'
   reading of a span drives the generation) and qualifies finding 2: "the query transfer moves the gaze to D" holds for
   non-goal heads too, so it is not by itself evidence about the goal heads.
8. **With kv, gsr also beats the re-clamp in all 32 cells**, by 0.06 (C++→C++ order-diff int, .53 → .59) to 0.63
   (C++→C++ order-matched list[str], .31 → .94). Pooled, order-diff goes from .27–.46 to .51–.66 and order-matched
   from .52–.61 to .81–.89. kv itself barely moves the re-clamp: kv re-clamp is within 0.07 of no-kv re-clamp pooled in
   every direction × ordering (largest drop C++→C++ order-matched, .59 → .52). The same reason applies as in finding 4:
   p_kv's task list is the no-kv DST prompt.

## 6. Caveats

- **gsr is a two-pass procedure.** The query transfer influences only the selection; the generation contains no SRC
  information except through the choice of span. The claim it supports is "the transferred query redirects the goal
  heads' attention to D, and the goal heads' reading of a task span drives what DST writes". It does not show that
  the transferred query itself sustains D during generation.
- **Missing restriction control.** Restricting to a task picked at random, or to B, is not in the grid. Given
  P(D | selected D) ≈ 0.98, restriction alone can probably steer DST to almost any listed task. That would confirm
  that the selection step carries all the SRC information, but it should be measured.
- **The readout shift is not measured on the gsr intervention.** `ro_patched` is read off the selection pass (query +
  boost + kv), not the restricted run, so the gsr `readout_shift_rate` is the re-clamp's pivot readout. kv runs never
  score readouts at all. A fix (`ro_span` under the whitelist; a kv readout judge) is proposed but not applied.
- **kv-on gaze branch mismatch** (corrected and rerun; see §4). There is still no clean-gaze control with kv (gaze
  is not computed for the kv-only run), and the kv selection is redundant with the no-kv one (finding 4).
- **Judges grade only the first computation and are not anchored on the current variable**
  (`project_closed_loop_judge_issues`). The execution audit partly covers this for Python destinations only.
- **Control without readouts.** The control (§4.5) skips the GoalScope readouts, so it has no readout-shift; the
  readout-heads argument is still the control set because the gsr selector uses the readout heads' gaze.
- **Mismatched baseline runs.** The no-kv order-matched re-clamp baseline comes from the older non-`__seq` run.
- **The kv design may not separate set from count in either arm.** Under kv the goal heads read p_kv's task list
  (`A'BD`) over the whole task span. In that list the slot a count reading would pick (the 3rd item) is D, so a
  count-based goal-head read also predicts D. Y (the 3rd item of p_dst's own list) can only come from the non-goal
  heads. The near-zero Y rate (8 / 1016 under the kv re-clamp) is therefore weak evidence for set over count. A clean
  test needs a p_kv whose 3rd slot is not the set-progress answer.
- **Missing instances.** A few kv cells have 30–31 instances instead of 32 (no valid (X, Y)).

## 7. Next steps

- Add a restriction-only control: restrict to B, and to a random listed task, without any query transfer.
- If the set-vs-count question matters for the paper, redesign p_kv so that set and count progress point to
  different tasks inside p_kv's own list (§6). Neither gsr (finding 4) nor the current kv re-clamp can answer it.
- Decide whether to keep the kv gsr arm in the paper: it duplicates no-kv gsr's selection, and its only extra content
  is the "transplanted span is executable" result.
- Readout fix: score the pivot readout of the restricted run, and score readouts with kv; backfill finished cells.
- Add an execution audit for C++ destinations (compile + run) if needed for the paper.

## Files

- **Library:** `src/diff/functional.py` (`whitelist_head_attn`, the `gaze_select_restrict` branch of
  `run_q_transfer_intervention`).
- **Batch:** `scripts/diff/diff_trigger_eval.py` (`--probe-prompt gaze_select_restrict`, `--gsr-anchors`).
- **Job lists:**
  - `job_list/diff_eval/diff_trigger_gaze_select_restrict.json` (all anchors);
  - `job_list/diff_eval/diff_trigger_gaze_select_restrict_anc_eq.json`;
  - `job_list/diff_eval/diff_trigger_gsr_eq_crosslang.json` (run ID `cl_gsr_x_20261007`);
  - `job_list/diff_eval/diff_trigger_gsr_eq_kv_gzfix.json` + `…_kv_gzfix_rest.json` (kv rerun with the gaze fix and
    kv re-clamp baselines; run IDs `kv_gzfix_20261008`, `kv_gzfix2_20261008`).
- **Results:** `results/diff/Qwen2.5-Coder-32B-Instruct/diff_trigger/`
  - no kv: `ph-goal_250__a2__piv-eq__cl-ppgazerestrict-anceq__seq/`;
  - kv (fixed): `ph-goal_250__a2__piv-eq__kv__cl-ppgazerestrict-anceq__seq__gzfix/`; legacy kv: `…__seq/`;
  - all anchors: `…__cl-ppgazerestrict__seq/`;
  - random-head control (no kv): `ph-control250_250__a2__piv-eq__cl-ppgazerestrict-anceq__seq/` (job list
    `job_list/diff_eval/diff_trigger_gsr_eq_control250.json`, re-dispatched remainder `…_control250_rest.json`).
- **Notebook:** `notebooks/diff/04.4_q_transfer_multipred.ipynb`, "Gaze-select + restrict" cells (`SPAN_*` knobs,
  `RO["span"]`, `CAUSAL["span"]`).
