# How is the DIFF computed? Where "TASK − PROGRESS" comes from

*Brainstorm note, 2026-08-27. Not polished; sketches hypotheses + a discriminating experiment battery. Cross-refs: [`2_clock_state_machine.md`](2_clock_state_machine.md) (g/c two-signal), [`3_clock_gaze_transition_progress.md`](3_clock_gaze_transition_progress.md) (gaze/transition/order-ID), [`design/goalscope.md`](design/goalscope.md).*

---

## 0. The phenomenon we are trying to explain

At a mid-generation pivot the model must encode **what is still left to do**. Concretely, in the `checklist.ipynb` DIFF-CMA setup:

- **TASK** = the full spec in the prompt (e.g. *"compute (1) words with len>3, (2) True iff longest word is a palindrome, (3) word count"*).
- **PROGRESS** = the code already written in the prefill (`a = [w for w in words if len(w)>3]`, and we cut at `b = `).
- **DIFF = TASK − PROGRESS** = *"I still owe condition 2 (and 3)"* — ideally sharpened to the immediate next item, "condition 2".

**What we know (the empirical hook).** The **goal heads** (~122 heads, the trained goal `head_mask`) carry a representation of DIFF at the pivot that is:
1. **Verbalizable** — goalscope readout at the pivot (`apply_goalscope`, additive_norm, L35–46) emits text describing the *remaining* item, not the completed one. (`checklist.ipynb` cell 27.)
2. **Transplantable** — caching the goal heads' write at the *source* pivot and injecting it at the *destination* pivot (`replace` over `GOAL_MASK`, or additive over the aggregated write) carries the source's pending item *across contexts* into the dest generation. (`checklist.ipynb` cell 29, "DIFF CMA".)
3. **Gated by the boundary signal** — `BOUNDARY_PATCH` blocks the leftover DIFF from both being read out *and* being executed (commit `3def99c`). This is the single most informative existing datapoint and it bears directly on the question (see §4).

The question: **who actually computes DIFF, and where/when?** Goal heads clearly *carry and expose* it. Do they *compute* it?

---

## 1. Sharpening the question: DIFF is two operations, not one

DIFF = TASK − PROGRESS decomposes into three sub-computations, and the hypotheses really disagree about *which component the goal heads own*:

| # | sub-op | nature | plausible locus |
|---|--------|--------|-----------------|
| **R** | **retrieve TASK** — parse/lift the full spec from the prompt | attention read (retrieval) | attention heads (goal heads are natural candidates) |
| **B** | **bookkeeping** — mark which spec items PROGRESS has already satisfied ("`a=` filter ⇒ item 1 done") | alignment / set-membership over generated code — the hard, nonlinear part | MLPs + boundary/decision heads (the `c` clock) |
| **S** | **subtract/select** — expose the *frontier* undone item in a generation-driving form | gating of R by B | ? — this is the crux |

**Set subtraction over a variable-size set of completed items is not a natural one-head operation.** A single attention head outputs an attention-weighted mix of value vectors + an OV map — it routes and linearly transforms; it does not natively compute "the set of spec items minus the set of done items." That argues subtraction is done by a *circuit* (many heads across layers, and/or MLPs), not by any single goal head. It does **not** rule out the goal-head *population* implementing S across its layer span. Keep R/B/S separate when reading the hypotheses below.

**Frontier vs. full remainder (important reframing).** The transplant evidence is consistent with DIFF being just the **next pending item** (the frontier), not the entire tail of the plan. This distinction dissolves much of the layer-budget worry in §3 and is itself testable (§5, exp. 5).

---

## 2. H1 — the goal heads themselves compute the DIFF

**Claim.** The goal heads own S (and read R directly): at the pivot they attend to *both* the spec tokens and the already-written code, and their OV write *is* the subtraction result.

### H1.1 — recomputed from scratch at every position
At each generated position, the goal heads re-derive DIFF by re-reading spec + progress. 

- **Prediction:** goal-head write drifts continuously token-to-token (it is a fresh function of an ever-growing prefix), with no privileged "compute-here" positions.
- **Depth cost:** *bounded* — per-token compute is capped by model depth regardless of how many steps remain. **This is the version the "runs out of layers" critique does *not* touch** (see §3). The cost is width/attention-span (must re-scan all done lines each token), not depth.
- **Weakness:** wasteful, and predicts DIFF should be equally sharp at every position — but empirically the readout is cleanest at structural pivots (the `:` / newline), which fits H1.2/H3 better.

### H1.2 — precomputed at specific positions, reused when needed
DIFF is computed at structural boundaries (line-end, `=`, `:` after a variable) and then copied forward by later positions until the next boundary.

- **Prediction:** goal-head write is ~piecewise-constant — step-changes at boundaries, flat between them.
- **User's critique — and its correct scope.** "Runs out of layers for deep sequential tasks." This bites *one specific strawman*: that the model computes the *entire remaining plan* at a single position and stores it (needs depth ∝ #remaining steps). It does **not** bite the version where each boundary recomputes only the *incremental* update (drop the just-finished item) — that is O(1) depth per boundary and is really a hybrid of H1.1 (recompute) and H1.2 (at boundaries). See §3.

**Net on H1:** viable only if "goal heads" means the whole cross-layer goal-head *circuit*, and only in the incremental-at-boundaries form. A single head, or a "compute the whole tail once" reading, is implausible for op B/S.

---

## 3. The layer-budget argument, made precise

Fixed depth `D`. Let `k` = number of sub-steps already done, `m` = number remaining.

- **"Compute the full remaining plan at one position and store it"** → needs depth roughly ∝ `m` (must unroll/serialize the tail). **Runs out of layers.** Correctly rejected.
- **"Recompute DIFF from scratch every token" (H1.1)** → depth is O(1) in `m` (one subtraction against the current done-set). **Does not run out of layers.** Cost is attention width (re-scan done items), which is cheap and parallel.
- **"Maintain the done-set incrementally; at each boundary do one update" (H3 below)** → O(1) depth *per token*, and the expensive bookkeeping B is amortized across the sequence as a scan, exactly the "associative scan / depth-bounded recurrence storing intermediate results at selected token positions" mechanism reported for state-tracking and multi-step reasoning (Li–Guo–Andreas 2025; Brinkmann et al. 2024; see `2_clock_state_machine.md` §2).

**Conclusion:** the layer budget rules out *storing an unrolled plan*, not *incremental recomputation of a frontier*. So the live contest is **H1.1 (per-token recompute) vs. H2/H3 (incremental clock computed elsewhere, goal heads gate+expose it)** — not "recompute vs. precompute-once".

---

## 4. H2 — DIFF is computed elsewhere; goal heads retain + verbalize it

**Claim.** Ops B (and much of S) live in other components — MLPs and the **boundary/decision heads** that carry the completion clock `c`. The goal heads perform R (retrieve the spec) and read the *result* of the subtraction off the residual stream, holding it in a linearly-decodable, generation-driving, transplantable form. Goal heads are the **readout + router**, not the calculator.

This is the natural continuation of the **g + c two-signal hypothesis** (`2_clock_state_machine.md`):
- `g` (goal heads) = *which* item — the identity/content of the frontier.
- `c` (boundary/decision heads, the "move-on" signal, `16_move_on_signal.ipynb`) = *how far along / when to transition* — the bookkeeping that determines *which* item is frontier.
- **DIFF ≈ gate(TASK-spec by c).** The spec is always in context; `c` selects which slice of it is still owed.

**Why the existing evidence leans H2/H3:**
- **`BOUNDARY_PATCH` blocks leftover readout *and* execution** (`3def99c`). If the goal heads computed DIFF autonomously, corrupting the *boundary* signal should not erase the goal-head DIFF readout. That it does erase both the verbalization and the behavior is direct evidence that **the goal-head DIFF is downstream of, and gated by, a separately-carried completion signal `c`.** Goal heads expose DIFF; they don't manufacture it from nothing.
- **Transplantability** is also more parsimonious under H2: a retained/routed vector is exactly the kind of thing that survives being cached and re-injected verbatim into a different context. A live computation reading *dest*'s spec+progress would be expected to partly overwrite the injected src signal — yet the src frontier carries across.

**H2's own soft spot:** if goal heads were *pure* passive carriers, ablating them should leave DIFF recoverable elsewhere in the residual stream (other reads, or behavior). Whether that holds is exp. 7 below — and if DIFF vanishes on goal-head ablation, we're pushed back toward H3 (goal heads own the final selection step S).

---

## 5. H3 — the hybrid (current best guess)

> **B is an incremental clock (`c`) maintained by boundary/decision heads + MLPs, updated O(1) per boundary. Goal heads retrieve the spec (R) and apply a shallow gate/selection (S) of the spec by `c`, exposing the frontier item in verbalizable, transplantable form.**

- Reconciles everything: layer budget (B is amortized scan, S is shallow), `BOUNDARY_PATCH` (kill `c` ⇒ S has nothing to select ⇒ no DIFF), transplant (the post-gate frontier vector is a clean, movable object), and the goalscope readout (S's output is what goalscope decodes).
- Splits the labor the way the two circuits are already localized separately: goal `head_mask` (g) vs. `completion_mask` / move-on heads (c).
- Under H3 the goal heads *do* compute something (the gate S), but the "minus PROGRESS" content — the hard part — is imported from `c`. So "H1 vs H2" is a false binary; the real content is **how much of S the goal heads own vs. how much is already resolved in `c` before they read it.**

---

## 6. What would distinguish them

| observable | H1.1 (heads recompute) | H1.2 (precompute+reuse) | H2 (retain only) | H3 (gate) |
|---|---|---|---|---|
| goal-head write across positions | drifts every token | piecewise-const, steps at boundaries | tracks whatever `c` says | steps at boundaries (follows `c`) |
| layer where DIFF first linearly decodable | ≈ goal-head band | ≈ goal-head band | **before** goal-head band | `c` before band; frontier at band |
| goal-head attention keys at pivot | spec **and** done-code | spec **and** done-code | a local summary slot | spec items (R); little on done-code |
| corrupt `c` (BOUNDARY_PATCH) | DIFF survives | DIFF survives | **DIFF dies** ✓(observed) | **DIFF dies** ✓(observed) |
| ablate goal heads, probe DIFF elsewhere | gone | gone | **still recoverable** | gone (S is theirs) |
| patch goal-head *inputs* only (src→dest) | little effect (they recompute) | little effect | **output flips to src** | output flips to src |

The observed `BOUNDARY_PATCH` result already crosses out the two H1 rows on that line. The remaining discriminators are the ablation row (H2 vs H3) and the input-patch / attention-key rows (retain vs gate vs recompute).

---

## 7. Experiment battery (cheapest → most diagnostic)

All reuse existing tooling (`apply_goalscope`, `headpatched_generate`, `collect_head_steering_vecs`, value-weighted `get_attention_matrices`, the `completion_mask` / move-on heads).

1. **Layer-emergence probe.** Goalscope-decode / linear-probe DIFF at the pivot layer-by-layer. Is the frontier decodable *before* the goal-head band (⇒ retained, H2) or only within it (⇒ computed there, H1/H3)? Do the same for `c` — expect `c` to precede the frontier. *(cheap; one forward + sweep.)*

2. **Position-trajectory of the goal-head write.** Cosine of the aggregated goal-head write across consecutive positions across two boundaries. Continuous drift ⇒ H1.1; step-at-boundary + flat ⇒ H1.2/H3. *(cheap.)*

3. **Attention-source analysis at the pivot.** For the goal heads, value-weighted attention: do they read the *undone spec item's* `id_substrs`, the *done-code lines*, or only a local summary slot? Reading spec **and** done-code ⇒ heads compute (H1). Reading spec but not done-code ⇒ R-only, subtraction imported (H2/H3). *(cheap; `attn.npz` machinery exists.)*

4. **`c`-gating dose-response (extends BOUNDARY_PATCH).** Graded ablation/patching of the boundary/decision heads; measure (a) goalscope DIFF-readout fidelity and (b) whether the correct next item is generated. Monotone collapse ⇒ DIFF is gated by `c` (H2/H3). We already have the on/off endpoint. *(moderate.)*

5. **Frontier vs. full-remainder (depth-stress).** Tasks with 3/5/8/12 sequential items; read out DIFF at a fixed pivot. If fidelity depends only on the *next* undone item and is flat in `m` (total remaining) ⇒ frontier/incremental (bounded depth, kills the layer-budget worry). If it degrades with `m` ⇒ something is being unrolled/stored. Also: does the transplant carry *one* item or several? *(moderate; needs a small graded dataset — condition-check clock already supports variable `n`.)*

6. **Input-path-patch on goal heads.** Patch only the residual *feeding* the goal heads' Q/K/V from a different-progress run, leaving their OV intact. Output flips to the patched input ⇒ pass-through selector (H2/H3). Output unchanged ⇒ they recompute internally (H1). *(moderate; contrast with the existing OV-output CMA.)*

7. **Goal-head ablation, DIFF probed off-stream.** Ablate goal heads; try to recover DIFF via (a) a linear probe on the residual and (b) behavior (does it still do the right next item?). Recoverable ⇒ pure retention, goal heads are one of several carriers (H2). Vanishes ⇒ goal heads own the selection S (H3). **This is the H2-vs-H3 decider.** *(moderate.)*

---

## 8. Bottom line

- The **"compute the whole remaining plan at one position and store it"** reading is the only thing the layer budget actually forbids — reject it, but note it is a strawman for both H1.1 and H2/H3.
- The observed **`BOUNDARY_PATCH`** result already argues the goal heads **do not compute DIFF autonomously**: the leftover DIFF is **gated by the boundary/completion signal `c`**. That pushes against pure H1 and toward **H2/H3**.
- Best current guess: **H3** — bookkeeping ("which items are done") is an incremental clock `c` carried by the boundary/decision heads; the goal heads retrieve the spec and apply a shallow gate to expose the **frontier** undone item, which is what goalscope verbalizes and what the CMA transplants. "Goal heads compute vs. retain the DIFF" is a false binary; the sharp question is **how much of the final selection `S` the goal heads own vs. how much `c` has already resolved before they read it** — which exp. 4, 6, 7 target directly.
- Cheapest next step that moves the needle: **exp. 1 + 3** (layer-emergence of DIFF vs. `c`, and whether goal heads even *look at* the done-code). If goal heads never attend to the completed lines, they cannot be doing the subtraction — and H2/H3 is essentially settled.
