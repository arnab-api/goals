# H1 vs H2 — how the LM decides the next sub-task (dataset + query-mask design)

*Design reference for `src/diff/h1h2.py` (dataset) and `src/diff/h1h2_mask.py` +
`scripts/evaluation/train_h1h2_mask.py` (query-mask training). Created 2026-09-12.*

## Question

By ~layer 40 the LM already "knows" which sub-task it will implement next (collapse eval,
`eval_collapse_layerwise.py`: goal-head gaze + read-out lock onto the next condition by L≈40). **How is
that next sub-task decided?** Two hypotheses:

- **H1 — induction ordering.** Each task's description latent stores its predecessor ("I come after C").
  At the pivot the LM, having just finished the last sub-task, looks for "what comes after the last-done
  task" and the matching description forms the intent.
- **H2 — set difference (DIFF).** The LM carries a representation of *what it has done* and computes the
  next task as the set-complement of the done-set against the full spec.

## Dataset — the contrastive pair (`src/diff/h1h2.py`)

Five DISTINCT predicates fill five roles (letters A/C/D/B/E):

| role | letter | SRC list pos | DST list pos | purpose |
|---|---|---|---|---|
| anchor   | A | 1 | 1 | shared; DST's only prefilled task |
| frontier | C | 2 | 2 | the linchpin: its successor DIVERGES (→D in SRC, →B in DST) |
| src_targ | D | 3 | 5 (if `SRC_TARG_IN_DST`) | SRC target; the COPY outcome |
| dst_h1   | B | 4 | 3 | SRC gap-filler (done); DST successor-of-C → the H1 outcome |
| dst_targ | E | — | 4 | DST-only; the H2 (set-complement) outcome |

```
SRC.tasks = [A, C, D, B]      prefill written A→B→C (done={A,B,C})   target = D
DST.tasks = [A, C, B, E(, D)] prefill = {A}                          native target = C
```

**Why SRC is neutral (D reachable by BOTH):**
- H1: prefill is written **A → B → C so the LAST-done task is the frontier C**; D immediately follows C
  in the task list → "successor of C" = D. *(This is why prefill order matters: if B were written last,
  "successor of B" = nothing, and only H2 could reach D — SRC would be biased.)*
- H2: done-set {A,B,C}, full {A,B,C,D} → complement = {D}.
- **Counting ruled out:** the gap lives in the task list (D at pos 3 undone while B at pos 4 done). Naive
  "3 done → do the 4th slot" = B (already done), so SRC→D cannot be positional counting.

**Patching SRC → DST splits the hypotheses** (the four graded outcomes are distinct DST conditions):

| mechanism carried by the patched SRC latent | resolves in DST | outcome |
|---|---|---|
| H1: relational "successor-of-C" | B follows C in DST | **B** |
| H2: progress "done {A,B,C}" | complement in DST | **E** |
| COPY: already-resolved goal "D" | D verbatim | **D** (needs `SRC_TARG_IN_DST=True`) |
| NO-OP | DST native | **C** |

`SRC_TARG_IN_DST=True` appends D to DST (`[A,C,B,E,D]`) so COPY has a gaze target and the complement
splits into {E,D}: **gaze on E = recomputed set-difference (H2); gaze on D = decision already committed
in `h_src` (COPY)**.

**Prompt conventions:** plain `computes:` (NOT "in sequential order:" — the out-of-order SRC done-set
would contradict it); sequential-by-appearance letter naming so prefilled lines read `a,b,c` and the
target is the tail `d =`, independent of task-list position.

**Distinct pairs available** (B/C/D/E non-trivial; A any): int 205,920 · str 524,160 · list[int] 142,560
· list[str] 524,160 → **~1.4M total**. Effectively unbounded; the cap is compute/oracle budget, so we
subsample with `sample_pairs(..., seed=)`.

## What we optimize — query masking (à la `filter.pdf`)

At the pivot the LM can only pool information by attention (query = pivot, keys/values = context). The
filter-paper finding is that a head's **query** carries the reusable "what am I looking for." So we mask
**queries**: for a candidate head `h` at layer `< L_READ`, blend in the SRC pivot query
`q_h = m_h·q_src_h + (1−m_h)·q_dst_h` (`q_src` RoPE-aligned at the **DST pivot position**). A soft mask
`m ∈ [0,1]^{heads}` (`HeadMask`, projected GD, `λ·mean(m)` sparsity — same as `train_head_mask`, but on
queries) is trained.

**Objective — per hypothesis.** The loss is the **collapsed goal-head gaze at `h^{L_READ}`** (Tool 1 of
`eval_collapse_layerwise`: force all goal heads to read `h^{L_READ}`, take value-weighted pivot→span
attention, aggregate over goal heads) pushed toward **one hypothesis's target span**:
`target = D` (COPY), `B` (H1), or `E` (H2). We train a separate mask per hypothesis and compare: which
target is achievable, how sparse the mask, and which heads/layers it selects. That is the readout of
*which mechanism the query carries*.

**`L_READ` (default 40) is a single knob** controlling BOTH (a) the candidate-head cutoff (maskable
heads are at layers `[0, L_READ)`) and (b) the collapse read-layer (`h^{L_READ}`, the residual AFTER
block `L_READ−1` that the goal heads are forced to read, and where the #3 delta is injected). Goal heads
below `L_READ` are negligibly few and not special-cased.

**Why the collapse readout makes "no patching after `L_READ`" coherent:** the collapse forces *every*
goal head (including those above `L_READ`) to read `h^{L_READ}`, so the entire readout is a function of
`h^{L_READ}` alone. We shape `h^{L_READ}` with pre-`L_READ` query patches and never touch anything after.

## Three interventions (restrictiveness ladder: #1 ⊃ #2 ⊃ #3)

All patch queries at masked heads (layers `< L_READ`); they differ in which downstream pathways carry
the effect to `h^{L_READ}`.

1. **`free_form`** — patch queries; everything below `L_READ` recomputes (masked heads' writes → MLPs →
   later heads' queries *and* keys/values). Full mediation; = the filter-paper query patch.
   **1 forward/step** (destination, under the mask), grad through to `h^{L_READ}`.

2. **`mlps_frozen`** — as #1 but every MLP output below `L_READ` is repatched from the clean-DST run, so
   the effect flows through attention→attention paths only, never through MLPs. **3 forward passes:**
   (i) SRC → cache queries; (ii) clean DST → cache MLP outputs; (iii) patched DST (queries from (i),
   MLPs frozen from (ii)) → `h^{L_READ}`.

3. **`block_head_communication`** — masked heads cannot influence each other or intermediate
   computation; only their aggregated write reaches the goal heads. Per masked head: compute
   `OV(q_src, k_dst, v_dst)` (query from SRC, keys/values from **clean DST**),
   `Δ_h = OV_patched − OV_clean`; inject `Σ_h m_h·Δ_h` into the residual at `L_READ` (the exact
   `h^{L_READ}` the collapse reads). Each `Δ_h` uses clean-DST k/v ⇒ no inter-head communication; the
   sum reaches only the goal heads. **The `Δ_h` are constants** (fixed q_src, fixed clean k/v), so
   `Σ m_h·Δ_h` is **linear in the mask** and precomputes once → training is a **partial forward from
   `L_READ` onward** per step. Cheapest and most interpretable (pure direct query→goal-head path).

**Interpretation:** survives #3 → direct query→goal-head effect; needs #2 but not #3 → routes through
inter-head attention; needs #1 → routes through MLPs.

## Loss detail

Collapsed goal-head gaze = value-weighted pivot→position attention, mean over goal heads, summed over
the target predicate's description span, normalized over the T predicate spans (a fraction in [0,1]).
Loss per hypothesis `= −log(mass_on_target) + λ·mean(m)` (cross-entropy-style pull toward the target
span + sparsity). Trained with projected GD (`clamp_` to [0,1] each step), reusing `HeadMask`. Gaze is
differentiable via eager `output_attentions` under grad; value-norm weighting matches the eval (a plain
softmax-mass variant is available as a knob).

## Files

- `src/diff/h1h2.py` — dataset (`H1H2Spec`, `src_instance`/`dst_instance`, `build_prompts`,
  `sample_pairs`, `count_pairs`). `SRC_TARG_IN_DST` flag.
- `src/diff/h1h2_mask.py` — caching, differentiable collapse-gaze loss, the three intervention forwards,
  training loop.
- `scripts/evaluation/train_h1h2_mask.py` — driver (per-hypothesis, per-intervention, `--l_read`).
- `notebooks/diff/11_induction_vs_diff.ipynb` — SRC/clean-DST inspection + layer-wise collapse tools.
