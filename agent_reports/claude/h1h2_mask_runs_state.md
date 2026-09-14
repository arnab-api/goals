# H1-vs-H2 query-mask runs (`train_h1h2_mask.py`) — current state

_Last updated: 2026-09-13. Numbers pulled from the 28 trained `mask.pt` under
`results/evaluation/h1h2_mask/Qwen2.5-Coder-32B-Instruct/`. Two runs still training (marked ▶)._

## Question & why
By layer ~40 the model already "knows" its next sub-task. **Two mechanisms could produce that selection:**
- **H1 — induction ordering**: "what token/task comes right after the last-done one" (positional / induction copy).
- **H2 — set-difference**: "what's left to do" (the remaining task in the spec minus what's done).

We want to know which one drives the goal-head gaze. The test: a contrastive SRC/DST setup where the H1
target and the H2 target are **different spans**, and we train a **sparse query mask** over attention heads
(layers `< L_READ`) that tries to flip the collapsed goal-head gaze onto **one** hypothesis's target span.
If a hypothesis's target is reachable with a sparse, sensible mask, that mechanism is available in the query
states.

## Method
- **Objective**: per-position gaze loss `−log(p_target) + comp_weight·(1−p_target)`, optimized over a
  `HeadMask` via projected GD. Per run outputs `mask.pt` with final gaze **mass** (fraction of goal-head
  gaze on the hypothesis target span) and **n_selected** (heads with mask>0.5). `mask.pt` is a rolling
  checkpoint (written every 5 epochs) — a run is *done* only when its train process exits (ep31 = final of
  32 epochs).
- **Knobs**: `L_READ ∈ {40, 44}` (depth the collapse reads from); `patch_last_k ∈ {1, 3}` (patch the pivot
  query only, vs the last 3 tokens `[indent, var, " ="]`); `λ` (sparsity weight).
- **Three interventions** (nested by how much they free up):
  1. **block_head_communication (bhc)** — direct per-head OV deltas, linear in the mask, no grad forward
     (cheapest). Isolates the direct path.
  2. **mlps_frozen** — full grad forward but MLP outputs frozen to clean-DST (blocks MLP recompute).
  3. **free_form** — full grad forward, everything live (most expressive; needs 3 GPUs).

## Run matrix
`{L40, L44} × {bhc, mlps, free_form} × {k1, k3} × {h1, h2}`, plus an initial **λ-tuning sweep**
(L40, k1, λ∈{0.1,1,5,10,50}) and a **first λ=5/k1** run. free_form is k3-only; k1 was added for the cheap
interventions (bhc, mlps) to separate the **λ-effect** (λ5 vs λ1) from the **k-effect** (k1 vs k3).

## Results (final masses; ▶ = still training)

**λ tuning (L40, k1, bhc, h2)** — sparsity / mass tradeoff:

| λ    | 0.1  | 1    | 5    | 10   | 50   |
|------|------|------|------|------|------|
| mass | 0.317| 0.305| 0.283| 0.274| 0.238|
| n_sel| 381  | 106  | 31   | 22   | 9    |

**λ=5, k1, L40** (first run): bhc h1 .272/27, h2 .279/32 · mlps h1 .241/26, h2 .241/35 ·
free_form h1 .277/37, h2 .281/39.

**λ=1 grid:**

| L  | interv     | k | h1 mass / nsel      | h2 mass / nsel        |
|----|------------|---|---------------------|-----------------------|
| 40 | bhc        | 1 | .288 / 75           | .300 / 97             |
| 40 | bhc        | 3 | .288 / 75           | .300 / 97             |
| 40 | mlps       | 1 | .251 / 76           | ▶ .261 / 125 (ep16)   |
| 40 | mlps       | 3 | .252 / 95           | .309 / 195            |
| 40 | free_form  | 3 | .313 / 123          | .342 / 175            |
| 44 | bhc        | 1 | .433 / 119          | .378 / 127            |
| 44 | bhc        | 3 | .434 / 118          | .378 / 128            |
| 44 | mlps       | 1 | ▶ .420 / 219 (ep11) | — (queued)            |
| 44 | mlps       | 3 | .474 / 252          | .529 / 327            |

_Pending: S4 = L44 free_form k3 (h1+h2), not yet deployed._

## What worked / findings
1. **Depth dominates (L44 ≫ L40).** Reading from L44 nearly doubles achievable gaze mass
   (bhc h1 .288→.433; mlps k3 h2 .309→.529). The next-task query is far more steerable deeper in — but L44
   also selects **more heads** (bhc h1 75→119; mlps k3 195→327), so it's partly a head-budget effect, not
   pure depth.
2. **λ trades heads for mass, mildly.** λ5→λ1 roughly triples heads (bhc h2 32→97) for a small mass gain
   (.279→.300). Very sparse (λ=50, 9 heads) still reaches .238 → a *small* head set already carries most of
   the steerability; extra heads add little.
3. **k1 vs k3 — the encoding is at the pivot, not distributed (for bhc).** bhc masks are **byte-identical**
   between k1 and k3 (h1 .288/75, h2 .300/97 both). Patching the extra `[indent, var]` queries adds nothing
   on the direct path → the H1/H2 query signal sits on the pivot `" ="` token. mlps shows a mild k3>k1 lift
   (h1 76→95 heads), so MLP-mediated routing spreads a little across the 3 tokens.
   **TODO/verify:** the exact bhc k1==k3 equality (mass AND nsel identical) is striking — sanity-check the
   k3 cache wasn't reused for k1.
4. **Intervention ranking**: bhc ≈ free_form > mlps at L40; the direct OV path suffices, and freezing MLPs
   *hurts* (breaks gaze) rather than helps — mlps is consistently weakest at L40.
5. **H1 vs H2 — neither cleanly wins.** Both targets are reachable to similar (weak-to-moderate) degree; at
   L44 bhc favors **H1** (.433 vs .378) but mlps favors **H2** (.529 vs .474). No decisive dissociation yet —
   the gaze is not purely one mechanism.
6. **All masses are weak-to-moderate** (0.24–0.53 gaze mass). Even the best (L44 mlps k3 h2, 0.53) leaves
   half the gaze off-target, so "steer the gaze to a chosen hypothesis" only partially succeeds.

## Current status
- **Running** (S3b tail, 2-GPU mlps): L40 mlps k1 h2 (ep16), L44 mlps k1 h1 (ep11) — slow (~13–15 min/epoch
  under host contention). L44 mlps k1 is the long pole (~overnight).
- **Pending**: S4 = **L44 free_form k3** (h1+h2) — waits for the 2→3-GPU CONFIG flip after S3b, handled by
  the `/loop` driver.

## Caveats
- Head-budget confound on the depth effect (L44 selects more heads than L40).
- The bhc k1==k3 identity should be verified (cache reuse vs real finding).
- mass baseline / chance depends on span sizes; treat absolute values as relative comparisons.
