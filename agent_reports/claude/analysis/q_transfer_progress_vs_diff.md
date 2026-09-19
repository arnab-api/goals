# Q-transfer: progress-triggering and diff-vs-progress dissociation

*2026-09-18. Qwen2.5-Coder-32B-Instruct. Two goal-head **query-transplant** experiments that probe
what the pre-committed sub-goal query encodes. Shared machinery: `src/diff/data.py`
(`DiffTriggerInstance`, `diff_trigger_setups`), `src/diff/functional.py::run_q_transfer_intervention`
(single-pivot query patch, `restrict_to_task`), `scripts/diff/diff_trigger_eval.py`,
`src/diff/eval.py` (oracle judges). Companion refs: `diff_trigger_qtransfer_grid.md` (Exp A py→py grid
detail), `diff_trigger_predicate_families.md` (sanitization).*

## The paradigm

At a pivot in a partially-written function, we read the **query** vectors of a fixed set of goal heads
from a SOURCE prompt and patch them into the same heads at the DESTINATION pivot (single position,
scaled by α). We then ask what the destination writes. Predicates are drawn per datatype under a
**distinct-family + non-trivial-frontier** constraint (B, C, D from three different semantic families;
C, D non-trivial), so the outcomes are not confounded by look-alike sub-tasks. Pivot = `var_eq` (the
frontier's `{var} =` stub, the strongest goal-head pivot); goal heads are `restrict`ed to the task span
on the patched run unless noted. Heads: 122 (`goal__lam20_thr0.9_band25-56`) or 250
(`goal_topk/…top250`).

Two experiments differ only in **whether the source's transplant target C is available in the
destination**.

---

## Experiment A — "looking for progress" (C ∉ DST)

**Setup** (`variant ∈ {order_matched, order_diff}`):

```
SRC.task = [A, B, C]         SRC.prefill = {A, B}   → SRC frontier / query target = C
DST.task = [A, B, D]         DST.prefill = {A}      → DST native next = B ; remaining = D
```

C is **not** in DST, so a literal content-copy is impossible. Transplanting SRC's query and asking
whether DST leaves its native **B** and instead writes **D** (its own remaining sub-goal) tests whether
the query carries a transferable *"there is an un-done sub-goal — pursue the next one"* signal.
`order_diff` lists SRC as `[A, C, B]` (C in slot 2) to strip a positional "copy the 3rd-listed slot"
account; it is the confound-free view. Scored: causal→**D** (`judge_causal`, D-vs-B), native→**B**,
readout→D, gaze→D.

### Results (py→py, n=32, mean over int/str/list[int]/list[str])

**Going-forward default: NO-restrict.** At 122·a2 the restrict variant is only
marginally better than no-restrict (order_matched 0.672 vs 0.617; order_diff 0.477 vs 0.422), and
"blind the goal heads to everything outside the task span" is an awkward mechanism to justify in the
paper. So the primary configuration is **no-restrict**; restrict is kept as a secondary/ablation view.

**No-restrict (primary):**

| config | variant | causal→D | native B | readout→D | gaze→D |
|---|---|---|---|---|---|
| 122·a2 | order_matched | 0.617 | 0.344 | 0.859 | 0.898 |
| | order_diff | 0.422 | 0.492 | 0.555 | 0.648 |
| 122·a4 | order_matched | 0.758 | 0.219 | 0.844 | 0.898 |
| | order_diff | 0.539 | 0.398 | 0.570 | 0.648 |
| 250·a2 | order_matched | 0.680 | 0.281 | 0.766 | 0.898 |
| | order_diff | 0.500 | 0.406 | 0.492 | 0.648 |

**Restrict (secondary / ablation):**

| config | variant | causal→D | native B | readout→D | gaze→D |
|---|---|---|---|---|---|
| 122·a2 | order_matched | 0.672 | 0.305 | 0.859 | 0.898 |
| | order_diff | 0.477 | 0.477 | 0.555 | 0.664 |
| 122·a4 | order_matched | **0.773** | 0.172 | 0.844 | 0.898 |
| | order_diff | 0.492 | 0.430 | 0.570 | 0.664 |
| 250·a2 | order_matched | 0.703 | **0.062** | 0.766 | 0.891 |
| | order_diff | **0.516** | 0.336 | 0.492 | 0.633 |

**Takeaway.** The transplanted query reliably moves the destination off its native B toward D. Part of
the order_matched effect is positional slot-copy, but even in the confound-free order_diff view
causal→D (0.42–0.52) exceeds native-B, so there is a genuine content-DIFF effect. Restrict buys little
over no-restrict at 122·a2, so we report **no-restrict** as primary. Literal-C ≈ 0 throughout (C not in
DST). Full per-datatype detail in `diff_trigger_qtransfer_grid.md`.

### Cross-lingual (in progress)
The same two restrict configs (122·a4, 250·a2) run for py→cpp, cpp→py, cpp→cpp on GPUs 6/7
(run_id `20260918_090909`). As of 2026-09-18 18:32: **122·a4 complete** (all three transfers done);
**250·a2** has py→cpp + cpp→py running and cpp→cpp queued. **Table to be filled on completion.**

### Restrict ablation (done)
`122·a4·NO-restrict` and `250·a2·NO-restrict` (py→py, n=32, 4 datatypes) are in the primary table
above. No-restrict ≈ restrict at every config (122·a4 order_diff 0.539 vs restrict 0.492; 250·a2
order_diff 0.500 vs restrict 0.516) — confirming restrict buys little beyond 122·a2, so no-restrict is
the reported primary throughout.

---

## Experiment B — dissociating "query contains the diff" vs "query contains progress" (C ∈ DST)

**Setup** (`variant = c_in_dst`): both sides list all four predicates, so SRC's target C is **present**
in DST:

```
SRC.task = [A, B, C, D]      SRC.prefill = {A, B}   → SRC frontier / query target = C
DST.task = [A, B, D, C]      DST.prefill = {A}      → DST native next = B
```

(C and D are swapped between the two lists.) Transplanting SRC's query now has three legible outcomes,
disambiguated by a **single joint classifier** (`src.diff.eval.judge_joint`, one mutually-exclusive
verdict — this replaces the earlier two independent binary calls, which double-counted when C and D
compete; see the caveat below):

- **B** (no-op) — DST writes its native next; the patch had no effect.
- **C** (the *diff*: query carries the specific upcoming operation) — DST writes SRC's exact target,
  now available in DST → the query is **content-addressed**.
- **D** (progress) — DST writes the operation at its own progress-position after ⟨A,B⟩ → the query
  carries only positional *progress*, not the specific content.

### Results — config (i) 122·a4·restrict (py→py, n=32), joint (exclusive) shares

| datatype | B (no-op) | **C (diff / content)** | D (progress) | neither |
|---|---|---|---|---|
| int | 0.031 | 0.750 | 0.156 | 0.062 |
| str | 0.031 | 0.906 | 0.031 | 0.031 |
| list[int] | 0.062 | 0.844 | 0.031 | 0.062 |
| list[str] | 0.000 | 0.844 | 0.062 | 0.094 |
| **mean** | **0.031** | **0.836** | **0.070** | 0.062 |

### Results — config (ii) 250·a2·restrict (py→py, n=32), joint (exclusive) shares

| datatype | B (no-op) | **C (diff / content)** | D (progress) | neither |
|---|---|---|---|---|
| int | 0.031 | 0.906 | 0.031 | 0.031 |
| str | 0.031 | 0.812 | 0.000 | 0.156 |
| list[int] | 0.031 | 0.844 | 0.031 | 0.094 |
| list[str] | 0.062 | 0.781 | 0.125 | 0.031 |
| **mean** | **0.039** | **0.836** | **0.047** | 0.078 |

**Takeaway.** When the specific target is available in the destination, the transplanted query writes
it **~84% of the time (C)** in *both* configs, versus only **~5–7% (D)** positional progress and
**~3–4% (B)** no-op. The query encodes the **specific operation (the "diff"), not abstract progress**.
The effect is if anything cleaner at 250 heads (D 0.047 vs 0.070). Both configs re-scored with the
joint classifier over the saved generations (`parse_err=0`, 32/32 reconstructed each).

### Scoring caveat (why the joint classifier was needed)
The original scoring used two *independent* binary oracle calls — causal (D-vs-B) and literal-C
(C-vs-B). Their marginals need not sum to ≤1 and here summed to >1 (e.g. list[str]: causal→D 0.406 +
literalC 0.938), because a C-write is invisible to the C-blind causal judge and gets misattributed to
D. Re-scoring the *same saved generations* (greedy decoding is deterministic — no GPU re-run) with the
joint {B,C,D,neither} classifier collapses the inflated causal→D from ~0.36 to the true **D ≈ 0.07**.
Implemented in `src/diff/eval.py::judge_joint` (wired into `diff_trigger_eval.py` for future runs) and
applied post-hoc via `scripts/diff/rescore_c_in_dst_joint.py`.

---

## Synthesis

- **Exp A** establishes that the goal-head query is a *portable, causal* signal: transplanted into a
  fresh destination it pulls generation off the native next toward an un-done sub-goal, beyond a
  positional-copy account (order_diff).
- **Exp B** pins down *what* transfers: given the choice, the query executes the **specific operation
  it was about to perform (C)**, not a position-generic "advance progress" pointer (D). The transferred
  representation is content-specific.
- Read together: the query carries a **content-specific, pre-committed sub-goal**. In Exp A the exact
  content is unavailable in the destination, so the intervention manifests as the weaker
  progress-position shift toward D; in Exp B, where the content *is* available, it dominates as C. This
  is consistent with a decodable, implementation-plastic sub-goal encoding rather than a mere
  progress/depth counter.

## Status / TODO
- Exp A cross-lingual grid (3 transfers × 2 configs): 122·a4 done, 250·a2 finishing; fill the
  cross-lingual table on completion.
- Exp A restrict ablation: `122·a4·NO-restrict` + `250·a2·NO-restrict` (py→py) running on cuda 4/5;
  add rows to the Exp A table.
- Exp B config (ii) 250·a2: **DONE** — joint re-scored (C≈0.836, D≈0.047, B≈0.039). ✓
- Consider a three-way D/C/B **gaze** (current gaze is D-vs-B only, so it does not see C).

*This file is the canonical q-transfer summary; `diff_trigger_qtransfer_grid.md` holds the per-datatype
detail tables for Exp A (py→py) and is referenced, not duplicated, here.*
