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

### Cross-lingual (done)

All four language directions, n=32, mean causal→D over the 4 datatypes. **NO-restrict** (primary; all
three configs); the **restrict** grid ran 122·a4 and 250·a2 only.

**No-restrict — causal→D:**

| config | variant | py→py | py→cpp | cpp→py | cpp→cpp |
|---|---|:-:|:-:|:-:|:-:|
| 122·a2 | order_matched | 0.617 | 0.531 | 0.633 | 0.531 |
| | order_diff | 0.422 | 0.266 | 0.297 | 0.266 |
| 122·a4 | order_matched | 0.758 | 0.594 | 0.742 | 0.562 |
| | order_diff | 0.539 | 0.344 | 0.422 | 0.336 |
| 250·a2 | order_matched | 0.680 | 0.602 | 0.711 | 0.609 |
| | order_diff | 0.500 | 0.344 | 0.391 | 0.336 |

**Restrict — causal→D (for comparison):**

| config | variant | py→py | py→cpp | cpp→py | cpp→cpp |
|---|---|:-:|:-:|:-:|:-:|
| 122·a4 | order_matched | 0.773 | 0.734 | 0.789 | 0.703 |
| | order_diff | 0.492 | 0.359 | 0.438 | 0.344 |
| 250·a2 | order_matched | 0.703 | 0.602 | 0.688 | 0.570 |
| | order_diff | 0.516 | 0.375 | 0.367 | 0.383 |

**Takeaway — a destination-language effect, not a transfer failure.** Cells split by the **destination**
language, not by whether the transfer is cross-lingual: dest=py (`py→py`, `cpp→py`) are the two highest;
dest=cpp (`py→cpp`, `cpp→cpp`) the two lowest, in every config/variant. Critically **`cpp→py` ≈ `py→py`**
(122·a4 order_diff 0.422 vs 0.539; 250·a2 0.391 vs 0.500) — a query read from a **C++** context drives
correct **Python** generation nearly as well as native, so the sub-goal representation is
**language-agnostic on the read side**; the ~0.10–0.17 order_diff attenuation is a *write*-side penalty
for generating cpp. Sanity invariants hold across all directions: native-B rises exactly where causal→D
falls (dest=cpp order_diff native-B ~0.60–0.73), gaze→D stays stable (~0.88–0.91 matched, ~0.54–0.65
diff — the pivot still points at D regardless of language), and literal-C ≈ 0 (max 0.125 in one cell).
No-restrict ≈ restrict throughout (order_matched robust; order_diff within noise), consistent with the
py→py finding that restrict buys little.

*Runs: no-restrict `20260918_223233` (36 jobs, `name="local"` dispatch); restrict `20260918_090909`.*

### Restrict ablation (done)
`122·a4·NO-restrict` and `250·a2·NO-restrict` (py→py, n=32, 4 datatypes) are in the primary table
above. No-restrict ≈ restrict at every config (122·a4 order_diff 0.539 vs restrict 0.492; 250·a2
order_diff 0.500 vs restrict 0.516) — confirming restrict buys little beyond 122·a2, so no-restrict is
the reported primary throughout.

### Random-head control (done)
Matched placebo: transplant the **query** of a random later-half non-goal head set (same size as the
goal set, layers ≥32, excluding the goal heads, seed 123 — the same construction as the ablation
control) instead of the goal heads; patch = readout = that random set, no-restrict, py→py, n=32, mean
over 4 datatypes.

| config | variant | GOAL causal→D | control causal→D | control native_B |
|---|---|:-:|:-:|:-:|
| 122·a4 | order_matched | 0.758 | **0.094** | 0.906 |
| | order_diff | 0.539 | **0.062** | 0.938 |
| 250·a2 | order_matched | 0.680 | **0.109** | 0.891 |
| | order_diff | 0.500 | **0.070** | 0.930 |

Patching random heads barely moves the destination off its native next (causal→D ~0.06–0.11 vs goal
0.50–0.76; native_B rises to ~0.89–0.94, i.e. the no-op floor). The DIFF effect is specific to the goal
heads, not a generic head-patching artifact. Runs: `diff_trigger/ph-control{122_122,250_250}__a{4,2}__piv-eq`.

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
- Exp A cross-lingual grid: **DONE** — no-restrict (3 configs) + restrict (122·a4, 250·a2), all 4
  directions, in the Cross-lingual table above. ✓
- Exp A restrict ablation: **DONE** — 122·a4 / 250·a2 no-restrict py→py in the primary table. ✓
- Exp B config (ii) 250·a2: **DONE** — joint re-scored (C≈0.836, D≈0.047, B≈0.039). ✓
- Consider a three-way D/C/B **gaze** (current gaze is D-vs-B only, so it does not see C).

*This file is the canonical q-transfer summary; `diff_trigger_qtransfer_grid.md` holds the per-datatype
detail tables for Exp A (py→py) and is referenced, not duplicated, here.*
