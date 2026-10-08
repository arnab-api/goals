# Gaze-select + restrict (gsr): query transfer picks the task, a span restriction carries it out

*Qwen2.5-Coder-32B-Instruct · 2026-10-07 → 2026-10-08 · notebook `notebooks/diff/04.4_q_transfer_multipred.ipynb`
(final "Gaze-select + restrict" cells) · batch `scripts/diff/diff_trigger_eval.py --closed-loop --probe-prompt
gaze_select_restrict --gsr-anchors eq` · status: grid in progress (snapshot 2026-10-08 14:30, 37 of 83 jobs done) ·
companion to `closed_loop_query_transfer.md`*

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
α = 2 on the selection pass; pivot = the frontier's ` =` (`"\n    {res} ="` in Python, `"\n    auto {res} ="` in C++);
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

### 4.1 Judged D rate (int / str / list[int] / list[str])

Re-clamp = the paper's arm (no-kv order-matched baseline from the older `ph-goal_250__a2__piv-eq` run). kv re-clamp
baselines exist only for py→py; the cross-language ones are queued at the end of the grid.

| direction | arm | order-diff | order-matched |
|---|---|---|---|
| **py→py** | re-clamp | .62 .41 .47 .38 | .66 .53 .56 .69 |
| | **gsr** | **.59 .62 .78 .66** | **.91 .81 .91 .88** |
| | kv re-clamp | .56 .34 .43 .50 | .66 .59 .59 .59 |
| | **kv gsr** | **.78 .75 .77 .66** | **.91 .94 .94 .94** |
| **py→C++** | re-clamp | .56 .31 .28 .12 | .72 .53 .44 .56 |
| | **gsr** | **.66 .47 .50 .59** | **.91 .75 .84 .88** |
| | **kv gsr** | **.75 .62 .60 .72** | **.88 .75 .78 .81** |
| **C++→py** | re-clamp | .31 .22 .44 .25 | .69 .66 .56 .69 |
| | **gsr** | **.50 .41 .66 .47** | queued |
| | **kv gsr** | **.69 .69 .73 .59** | **.91 .78** .92 1.00 (26 and 11 of 32, running) |
| **C++→C++** | re-clamp | .41 .19 .38 .25 | .84 .50 .56 .44 |
| | **kv gsr** | **.78** (int) | queued |

Pooled over datatypes (exclusive labels):

| direction · ordering · kv | re-clamp D / B | gsr D / B |
|---|---|---|
| py→py · order-diff · no kv | .47 / .43 | **.66** / .34 |
| py→py · order-diff · kv | .46 / .43 | **.74** / .24 |
| py→py · order-matched · no kv | .61 / .31 | **.88** / .11 |
| py→py · order-matched · kv | .61 / .31 | **.93** / .06 |
| py→C++ · order-diff · no kv | .32 / .66 | **.55** / .44 |
| py→C++ · order-matched · no kv | .56 / .36 | **.84** / .13 |
| C++→py · order-diff · no kv | .30 / .62 | **.51** / .49 |
| C++→py · order-diff · kv | — | **.67** / .32 |
| C++→py · order-matched · kv (n = 101) | — | **.88** / .11 |

gsr outputs are almost entirely D or B. "neither" drops from 5–10 % (re-clamp) to 0–5 %, and C, the literal copy of
SRC's own target, does not occur. With kv, count (Y) is 0 in every gsr cell.

### 4.2 Selection fidelity: the outcome is decided by which task gaze selects

All gsr-eq records so far pooled (n = 1407):

| | n | D rate |
|---|---|---|
| gaze selected D | 1062 (75 %) | **0.976** |
| gaze selected another task | 345 | 0.009 |

Per cell, P(D | selected D) is 0.94–1.00. When gaze does not pick D it picks B (the native task) in 335 of 345 cases,
and A, which is already written, in the rest.

The D-selection rate depends on the condition:

| | order-diff | order-matched |
|---|---|---|
| py→py | .68 (kv .75) | .90 (kv .93) |
| py→C++ | .56 (kv .71) | .88 (kv .86) |
| C++→py | .51 (kv .67) | — (kv .88, partial) |
| C++→C++ | — (kv .78, int only) | — |

**No-query control.** On the clean DST (no query transfer), the pivot gaze never prefers D over B: 0 of 640 no-kv
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
| | **gsr** | 93 / 126 | **.82** | **.95** | **.60** |
| py→py · om · no kv | re-clamp | 78 / 128 | .64 | .87 | .39 |
| | **gsr** | 112 / 128 | **.79** | **.96** | **.70** |
| py→py · om · kv | re-clamp | 78 / 128 | .68 | .91 | .41 |
| | **gsr** | 119 / 128 | **.76** | **.93** | **.71** |
| C++→py · od · no kv | re-clamp | 39 / 128 | .44 | .64 | .13 |
| | **gsr** | 65 / 128 | **.89** | **.95** | **.45** |
| C++→py · od · kv | **gsr** | 85 / 126 | .81 | .94 | .55 |
| C++→py · om · kv (n = 101) | **gsr** | 89 / 101 | .81 | .97 | .71 |

The re-clamp's judged successes contain syntax errors, crashes, and expressions that reference other variables. gsr's
contain almost none of these. Its non-correct cases are mostly "partial" (edge-case disagreements such as `abs` /
negatives / empty input), so the judged-D rate overstates gsr much less than it overstates the re-clamp. C++
destinations are not audited (no executor).

### 4.4 The B prior

Because gsr fails only when gaze stays on B, the B prior is a property of the selection pass. Per B task (no kv, all
directions and orderings pooled; n per task is 4–34, so this is noisy):
- **B tasks that hold the gaze most often:** `abs` (int, 0.71 native), word count (str, 0.69), "all words have
  distinct lengths" (list[str], 0.86 of 7). The re-clamp is about equally sticky on these.
- **B tasks the re-clamp leaves native far more often than gsr:**

  | B task | re-clamp native | gsr native |
  |---|---|---|
  | largest value | .91 | .25 |
  | number of distinct values | .92 | .40 |
  | string length | .85 | 0 |
  | lowercase words | .79 | .31 |
  | words that start with a vowel | .54 | .13 |

  For these tasks, the re-clamp's failure is in execution, not in where the heads look.
- **No B task is sticky under every arm.** The earlier claim that `count_set_bits`, `count_letters` and `max_val` are
  100 % native everywhere was made on the earlier, smaller sample and does not survive pooling. `count_set_bits` is
  0.43 native under gsr and 0.70 under the re-clamp.

## 5. Findings

1. **gsr beats the re-clamp in every comparable cell except py→py order-diff int** (0.59 vs 0.62). The gains are
   largest where the re-clamp is weakest:
   - order-diff list types: py→py list[str] goes from .38 to .66, py→C++ list[str] from .12 to .59;
   - C++-source directions: C++→py order-diff gains 0.19–0.22 in every datatype (pooled .30 → .51).
2. **Selection is the bottleneck, and execution is nearly lossless.** P(D | gaze selected D) = 0.975 pooled. Every
   improvement of gsr over the re-clamp comes from two facts:
   - the query transfer moves the pivot gaze to D more often (51–93 %, against 0 % on the clean DST) than the
     re-clamp gets DST to write D;
   - the restriction converts that move into a D generation almost every time.
3. **The goal heads' view of one task span is enough to make DST implement that task.** With the query gone and
   every other head untouched, confining only the goal-250 heads to D's span yields correct D implementations (§4.3:
   0.76–0.89 of judged D are execution-correct, against 0.44–0.68 for the re-clamp).
4. **kv helps gsr on order-diff in every direction** (pooled: py→py +0.08, py→C++ +0.12, C++→py +0.16); on
   order-matched it is flat to slightly positive (py→py +0.05, py→C++ −0.04). Count (Y) never appears. Under kv+q the goal
   heads pick D from p_kv's list (set progress), not the third position.
5. **Order-matched > order-diff for both arms, with a larger margin for gsr.** Part of order-matched's advantage is
   the positional copy (D is "the 3rd listed" on both sides). Order-diff remains the clean test.
6. **Extra anchors hurt.** Letting the goal heads also read the indent and the result-variable token cuts the D rate
   by 0.14–0.55 on order-diff. Those tokens appear to carry DST's own "write B next" signal.

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
- **No clean-gaze control with kv.** Gaze is not computed for the kv-only run.
- **Judges grade only the first computation and are not anchored on the current variable**
  (`project_closed_loop_judge_issues`). The execution audit partly covers this for Python destinations only.
- **Mismatched baseline runs.** The no-kv order-matched re-clamp baseline comes from the older non-`__seq` run, and
  kv re-clamp exists only for py→py.
- **Missing instances.** A few cells have 30 instances instead of 32 (no valid (X, Y) for kv), and some cross-language
  cells are still running.

## 7. Next steps

- Finish the grid: C++→py order-matched (no kv), C++→C++, then the 24 cross-language kv re-clamp jobs, which may be cut.
- Add a restriction-only control: restrict to B, and to a random listed task, without any query transfer.
- Readout fix: score the pivot readout of the restricted run, and score readouts with kv; backfill finished cells.
- Add an execution audit for C++ destinations (compile + run) if needed for the paper.

## Files

- **Library:** `src/diff/functional.py` (`whitelist_head_attn`, the `gaze_select_restrict` branch of
  `run_q_transfer_intervention`).
- **Batch:** `scripts/diff/diff_trigger_eval.py` (`--probe-prompt gaze_select_restrict`, `--gsr-anchors`).
- **Job lists:**
  - `job_list/diff_eval/diff_trigger_gaze_select_restrict.json` (all anchors);
  - `job_list/diff_eval/diff_trigger_gaze_select_restrict_anc_eq.json`;
  - `job_list/diff_eval/diff_trigger_gsr_eq_crosslang.json` (current queue, run ID `cl_gsr_x_20261007`).
- **Results:** `results/diff/Qwen2.5-Coder-32B-Instruct/diff_trigger/ph-goal_250__a2__piv-eq{,__kv}__cl-ppgazerestrict-anceq__seq/`
  (all anchors: `…__cl-ppgazerestrict__seq/`).
- **Notebook:** `notebooks/diff/04.4_q_transfer_multipred.ipynb`, "Gaze-select + restrict" cells (`SPAN_*` knobs,
  `RO["span"]`, `CAUSAL["span"]`).
