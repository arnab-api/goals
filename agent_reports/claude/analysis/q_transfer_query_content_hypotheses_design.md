# What does the goal-head query carry? Contrastive designs for H1–H3

*2026-09-30. Design doc (no runs yet). Follows App I.2 / I.3 (`5.1_appI.2_appI.3__q_transfer_progress_vs_diff.md`).
All prediction tables below are **computed** by the rule-based predictor in §7, not hand-derived.*

## 0. Question

> **How does the LM know which operation is pending?** What abstract information about "where I am" does
> the goal-head pivot query carry, and which rule turns it into the next sub-goal?

In every design below, SRC's own target is **absent from DST**. So DST can only move toward a letter that
one of the rules in §2 derives from the transplanted query.

## 1. Notation (updated)

$p: \mathrm{a'b'c \rightarrow c}$

- **Letters are sub-task (predicate) identities**, shared across $p_\mathrm{src}$ and $p_\mathrm{dst}$: the
  same letter means the same predicate. Each string lists that prompt's sub-tasks **in the order they are
  specified** in its task description.
- $\mathrm{x'}$ means x's implementation is **prefilled** (already written, in spec order).
- $\rightarrow \mathrm{y}$ is the clean model's **native next** sub-goal (the first unprimed letter).
- Code variable names are **sequential** (named in writing order, never by spec position). These are
  separate from the identity letters above; see §4.
- $D_\mathrm{src}$ is SRC's done set, $t_\mathrm{src}$ its target (native next), $\mathrm{pos}(\cdot)$ the
  0-based index in a prompt's spec order, and $\mathrm{pred}(t)$ the item right before $t$ in the spec.
- A source is **order-matched** if everything before $t_\mathrm{src}$ in its spec is done and nothing after
  it is (e.g. $\mathrm{a'b'c}$, $\mathrm{a'b'c'd}$). Otherwise it is **order-broken** (e.g.
  $\mathrm{a'cb'}$, $\mathrm{a'b'cd'}$).

**Intervention (unchanged):** cache the goal heads' pre-RoPE query at SRC's pivot, replace DST's goal-head
query at DST's last prefix token, then decode greedily. **Readout:** which DST sub-task the patched run
implements first.

## 2. Hypotheses: what the transplanted query carries, and what DST should then write

| id | query content | DST prediction rule | defined for |
|---|---|---|---|
| **H0** | nothing usable (transfer fails) | DST native | all |
| **H1** set progress | the done set $D_\mathrm{src}$ | first item in DST spec order $\notin D_\mathrm{src}$ | all |
| **H1c** count progress | how many are done, $\lvert D_\mathrm{src}\rvert$ | DST spec slot $\lvert D_\mathrm{src}\rvert$ (the $(\lvert D\rvert{+}1)$-th item) | all |
| **H2** order id | the spec position of the pending task, $\mathrm{pos}_\mathrm{src}(t_\mathrm{src})$ | DST spec slot $\mathrm{pos}_\mathrm{src}(t_\mathrm{src})$ | all |
| **H3** induction | "I follow x", with x = $\mathrm{pred}_\mathrm{src}(t_\mathrm{src})$: the query matches the key of the spec item whose predecessor is x | the item right after x in DST spec | **order-matched sources only** |

Also tracked, but as a control rather than a letter:
- **H6 semantic similarity.** The query may carry features of $t_\mathrm{src}$, such as its return type or
  predicate family, and DST may pick its most similar pending item. See §4.

The query need not carry exactly one of these. Section 5 scores each design as a distribution over outcomes,
so mixtures show up as split mass.

### Why H3 needs an order-matched source

Under induction, $t_\mathrm{src}$ is selected because **its key encodes its spec predecessor** ("c: I follow
b"), and the pivot query carries that predecessor's identity. In an order-matched source, the predecessor is
also the last-written item, so "I just did x" and "the target follows x" are the same statement.

In an order-broken source they come apart. In $\mathrm{a'b'cd'\to c}$, the last-written item is d, but c's key
says "I follow b". The source's own next step is therefore **not** an induction step from what it just did,
so there's no well-defined induction target to transfer. So:
- H3 is tested only in order-matched sources (D1, D2).
- In order-broken sources (D3, and the earlier `order_diff_*` configs), H3 makes no prediction.

**Positional induction dropped.** "Slot of the last-written item + 1" equals $\mathrm{pos}(t_\mathrm{src})$
and $\lvert D_\mathrm{src}\rvert$ in every order-matched source. It is therefore indistinguishable from
H1c/H2 wherever induction is meaningful, and it isn't motivated where the order is broken.

## 3. The design battery

In an order-matched source, $\mathrm{pos}(t_\mathrm{src}) = \lvert D_\mathrm{src}\rvert$, so **H1c ≡ H2**. The
two can only be split by an order-broken source, where H3 is undefined. So the battery has two order-matched
designs for H1 vs {H1c, H2} vs H3, and one order-broken design for H1 vs H1c vs H2.

| design | source | $p_\mathrm{src}$ | $p_\mathrm{dst}$ | H0 | H1 | H1c | H2 | H3 | what it isolates |
|---|---|---|---|---|---|---|---|---|---|
| **D1** | matched | $\mathrm{a'b'c\to c}$ | $\mathrm{a'ebd\to e}$ | e | e | b | b | **d** | H3 vs {H1c, H2}. H1 = H0 here, so H1 is invisible |
| **D2** | matched | $\mathrm{a'b'c'd\to d}$ | $\mathrm{a'becf\to b}$ | b | **e** | c | c | **f** | H1 vs {H1c, H2} vs H3: one non-native letter each |
| **D3** | broken | $\mathrm{a'b'c'de'\to d}$ | $\mathrm{a'bfcg\to b}$ | b | **f** | **g** | **c** | n/a | H1 vs H1c vs H2 |

Signatures over (D1, D2, D3), all pairwise distinct:

```
H0   e b b      H1c  b c g      H3   d f –
H1   e e f      H2   b c c
```

Notes on the designs:
- **D3 is the smallest valid order-broken design.** An exhaustive search over order-broken sources of ≤ 4
  items (with DST ≤ 5 items) finds none that gives H0, H1, H1c and H2 four distinct, pending letters. The
  smallest has 5 + 5 items.
- **D3 source details:** $\mathrm{a'b'c'de'}$ is done on {a, b, c, e} with target d.
  - H1c: 4 done → slot 4 → **g**.
  - H2: d sits in slot 3 → **c**.
  - H1: the first DST item not done in SRC is **f**.
- **n per design:** H1c vs H2 is separated **only by D3**, so D3 needs the largest n.

**What existing data already says.** `order_diff_1` ($\mathrm{a'cb'\to c}$ / $\mathrm{a'bd\to b}$) is
order-broken, so H3 doesn't apply there. H2 predicts b, **which is the native answer**, while H1 and H1c
predict d. So goal-250's py→py order-diff causal-shift of 0.56 toward d counts against order id as the *sole*
content of the query. `order_matched_1` and `om2` give all the non-null hypotheses the same letter, so they
only reject H0.

## 4. Confounds to control

1. **Variable names leak count or order id.** With sequential naming, the pivot stub's variable letter
   (`c =` after two lines) literally encodes $\lvert D\rvert{+}1$. That gives H1c a surface route, and
   positional naming would do the same for H2.
   - **Primary fix:** use the bare-indent pivot (`--pivot indent`), so there's no stub name at the pivot.
   - **Secondary fix:** use a *random-name* variant (fresh random 2-letter names per line, consistent between
     SRC and DST for shared items), so the names in the earlier lines don't reveal the count either.
   - Report the `=`-pivot + sequential condition as a comparison.
2. **H6 semantic similarity.** Draw roles from **distinct predicate families**
   (`appI.2__diff_trigger_predicate_families.md`). In each design, check that the letter each hypothesis
   predicts isn't the one most similar to $t_\mathrm{src}$. Run a companion design where a
   $t_\mathrm{src}$-lookalike (same family and return type) sits at a slot that no hypothesis predicts. A
   jump to the lookalike means H6.
3. **"Done in DST" items.** No prediction in D1–D3 points at an item already done in DST, so no rule asks
   the model to re-implement a written line. Keep that property when adding designs.
4. **Native coincidence.** Only a predicted letter that differs from DST native counts as evidence for that
   hypothesis. This is why D1 can't test H1: H1 predicts e there, which is also the native answer.
5. **Completeness bias.** A model may prefer an item that was *done in SRC but pending in DST* (b and c in
   D2; b, c and e in D3) simply because the transplanted query has seen it. H1 predicts it *skips* those
   items. If the patched run implements them anyway, that's evidence against H1, not noise.

## 5. Scoring and analysis

- **Judge:** an exclusive **letter judge**: "which of DST's pending sub-tasks does the first new computation
  implement: {each pending letter} / something else / broken?" This extends the joint b/c/d/neither judge
  from App I.3 to arbitrary label sets. The binary causal judge can't be used here, because each design has
  3–4 competing targets.
- **Per design:** report the distribution over DST letters for the clean DST, the patched run (goal-250,
  α = 2), and the size-matched random-head control. Report the shift in each letter from clean to patched
  (analogous to IE).
- **Across designs:** fit a non-negative weight per hypothesis (H1, H1c, H2, H3) that reconstructs the
  per-design patched-minus-clean letter shifts from the signatures in §3. H3 enters only D1 and D2. The
  weights answer "what the query carries" directly. Unexplained mass goes to H0 or "other".
- **n:** 32 per design × datatype (int / str / list[int]; there are no list[str] pivots) × py→py first, with
  D3 doubled to 64. That's about 390 instances: roughly 1.6 GPU-hours at about 15 s per instance with
  `--no-readout`. D3 needs 5-predicate specs, so check that the pivot pool has enough 5-item tasks per
  datatype.
- **Head sets:** goal-250 as the primary; the union-387 and mix-λ20 sets as secondary.

## 6. Beyond query transfer: locating *where* the pending operation is decided

1. **Layer sweep of residual patching at the pivot.** Patch SRC→DST residuals at the pivot at each layer
   below the goal-head band (about L25–L40), using the same designs. The layer at which DST's letter
   distribution starts following one of H1/H1c/H2/H3 shows where that code is formed.
2. **Linear probes on queries and keys.**
   - Probe pivot **queries** for $\lvert D\rvert$ (H1c), $\mathrm{pos}(t_\mathrm{src})$ (H2) and the identity
     of $\mathrm{pred}(t_\mathrm{src})$ (H3), with shuffled-label controls. Use order-broken prompts to
     separate the first two.
   - Probe goal-head **keys** over each task-span item for the identity of its spec predecessor. That's the
     key-side half of H3's "I follow x" match.
3. **Attention knockout below the goal-head band.** Block the pivot's attention to (i) the written code
   lines and (ii) the task spec at layers < 40. If progress is read from the code, (i) should break H1/H1c;
   (ii) should mainly break H2/H3, which need spec structure.

## 6b. Implemented configs (mining / mask training, 2026-09-30)

The localization pipeline (`mine_nrgs_prefix.py` → `train_nrgs_mask.py`) trains **one mask per hypothesis**,
with that hypothesis's letter as the DST* target. The 7-predicate head-loc bank has only 5 predicate families
for str / list[int] (6 for int), so D2 and D3 above (6 and 7 tasks) can't be filled. The registry therefore
uses per-hypothesis designs of ≤ 5 tasks (6 for H2). In each, the target is that hypothesis's **unique**
prediction, never appears in SRC, and SRC's frontier is absent from DST (`_cfg(..., strict=True)` asserts
this).

| config | hypothesis | $p_\mathrm{src}$ | $p_\mathrm{dst}$ | target | other predictions |
|---|---|---|---|---|---|
| `h1c_1` | H1c | $\mathrm{a'bc'\to b}$ | $\mathrm{a'de\to d}$ | e | H1 = H2 = native d |
| `h1c_2` | H1c | $\mathrm{a'bc'\to b}$ | $\mathrm{a'dec\to d}$ | e | H1 = H2 = native d |
| `h2_1` | H2 | $\mathrm{a'b'cd'\to c}$ | $\mathrm{a'efb\to e}$ | f | H1 = native e, H1c b |
| `h2_2` | H2 | $\mathrm{a'b'cd'\to c}$ | $\mathrm{a'efd\to e}$ | f | H1 = native e, H1c d |
| `h3_1` | H3 | $\mathrm{a'b'c\to c}$ | $\mathrm{a'ebd\to e}$ | d | H1 = native e, H1c = H2 b |
| `h3_2` | H3 | $\mathrm{a'b'c\to c}$ | $\mathrm{a'dbe\to d}$ | e | H1 = native d, H1c = H2 b |

- **H2 needs 6 tasks.** With distinct families, it's therefore int-only. `--allow-family-overlap` fills
  str / list[int] too, but lets near-duplicate roles in.
- **SRC′ control.** The control query is dropped where the hypothesis would also predict the target under
  SRC′. That covers H2 (same frontier slot) and the order-matched H3 sources, which have no SRC′ anyway.
- **Stub-name leak (§4.1) is present in this pipeline.** It uses the `=` stub pivot with sequential names.
  In the H1c configs, SRC's stub and DST*'s target stub are **both `c =`**. The same holds for
  `order_diff_1` / `order_matched_1` in H1. So a mask can partly be satisfied by heads that copy the stub
  variable name. A neutral-stub or indent-pivot variant of the renderer would remove this.

## 7. Implementation notes

- **Instances:** add D1–D3 to the `src/diff/nrgs_prefix.py` config registry. The registry's single `target`
  field needs to become a `{hypothesis: letter}` map (with H3 omitted for order-broken sources); `native`
  already exists.
- **Eval:** reuse `diff_trigger_eval.py`'s intervention (`run_q_transfer_intervention`, `--no-readout`) with
  the new letter judge, and log the full label distribution per instance.
- **Mining:** the existing pivot pool plus the gate (both SRC and DST continuations must match their native
  letters; the divergence gate) apply unchanged.

### 7a. Prediction rules (reference implementation)

```python
def predictions(src, dst):                        # e.g. "a'b'c", "a'ebd"
    Ss, Ds = parse(src); Sd, Dd = parse(dst)      # spec order, prefilled set
    t = next(x for x in Ss if x not in Ds)        # SRC target
    i = Ss.index(t)
    matched = all(x in Ds for x in Ss[:i]) and not any(x in Ds for x in Ss[i:])
    after = lambda S, x: S[S.index(x) + 1] if x in S and S.index(x) + 1 < len(S) else None
    at = lambda S, k: S[k] if 0 <= k < len(S) else None
    return {"H0":  next(x for x in Sd if x not in Dd),
            "H1":  next((x for x in Sd if x not in Ds), None),
            "H1c": at(Sd, len(Ds)),
            "H2":  at(Sd, i),
            "H3":  after(Sd, Ss[i - 1]) if matched else None}   # induction: order-matched only
```

## 8. Open choices

1. **Pivot.** Indent-only (cleaner) vs `=` stub (comparable to App I.2 / I.3). My recommendation is to run
   indent as the primary.
2. **Cross-language.** H3 vs H1/H1c is most interesting cross-language (py→cpp), where the surface form of
   the written lines changes but the spec doesn't.

## 9. Results to date (updated 2026-10-02, Qwen2.5-Coder-32B-Instruct)

### 9a. Head-level mask localization (`train_nrgs_mask.py`, next-token flip on val)

Each run trains one head mask whose DST* target is that hypothesis's letter (§6b). Columns:
- **soft:** the trained mask, every head weighted by m ∈ [0, 1];
- **bin:** m = 1 for heads > 0.9 inside layers 25–56, 0 elsewhere;
- **goal / ctrl:** the fixed goal-250 and later-half control-250 sets under the same transplant;
- **′:** the SRC′ query (same frontier, one fewer task done). It's defined only for configs that have a valid control.

| run (target) | val n | opt. steps | soft | soft′ | bin | bin′ | goal | goal′ | ctrl | heads > 0.9 (all layers) | Σm |
|---|---|---|---|---|---|---|---|---|---|---|---|
| H1, `order_diff_1` only (50 ep) | 172 | 600 | **0.80** | 0.16 | 0.66 | 0.12 | 0.37 | 0.30 | 0.02 | 265 | 392 |
| H1 mix (6 configs) λ20 | 1,126 | 710 | 0.63 | 0.10 | 0.40 | 0.07 | 0.43 | 0.33 | 0.004 | 225 | 351 |
| H1 mix λ50 | 1,126 | 710 | 0.54 | 0.10 | 0.35 | 0.06 | 0.43 | 0.33 | 0.004 | 140 | 219 |
| **H1c** count | 375 | 240 | 0.62 | **0.02** | 0.43 | 0.03 | 0.33 | 0.25 | 0.005 | 267 | 488 |
| **H2** order id ⚠ | 108 | **40** | 0.39 | — | 0.37 | — | 0.32 | — | 0.00 | 554 | 1,891 |
| **H3** induction | 354 | 240 | **0.69** | — | 0.45 | — | **0.49** | — | 0.01 | 247 | 465 |
| *layers 20–55 (`--layers 20-56`, λ20; goal = the 204 goal-250 heads inside the band):* | | | | | | | | | | | |
| H1 mix | 1,126 | — | 0.56 | 0.08 | 0.44 | 0.10 | 0.39 | 0.31 | 0.00 | 187 | — |
| H1c | 375 | — | 0.56 | **0.02** | 0.47 | 0.03 | 0.30 | 0.23 | 0.00 | 201 | — |
| H2 (re-run, `--batch-size 16 --epochs 30`) | 108 | ~400 | 0.45 | — | 0.31 | — | 0.31 | — | 0.00 | 177 | — |
| H3 | 354 | — | 0.62 | — | 0.44 | — | 0.44 | — | 0.01 | 198 | — |

Soft-mask flip by config:

| run | configs |
|---|---|
| H1 mix λ20 | om1 0.82, om2 0.74, od1 0.68, dvp1 0.62, od3 0.58, od2 0.35 |
| H1c | h1c_1 0.70, h1c_2 0.54 |
| H2 | h2_1 0.36, h2_2 0.42 |
| H3 | h3_1 0.68, h3_2 0.70 |

Readings:
1. **H3 is the strongest learned signal, and the one the goal heads already carry best.** The soft mask reaches 0.69, and goal-250 alone reaches 0.49, the only hypothesis where goal-250 beats the learned binarized set (0.45). Both H3 designs agree (0.68 / 0.70).
2. **H1c is learnable and done-count-specific.** The learned heads collapse under SRC′ (0.62 → 0.02 soft, 0.43 → 0.03 bin). Goal-250 barely moves (0.33 → 0.25), so its H1c flips are mostly *not* count transfer; the count signal sits in heads largely outside goal-250 (only 88 of the 267 H1c heads are goal heads).
3. **The H1 results don't isolate set progress.** In om1, om2 and dvp1, every hypothesis predicts the target, and in od1 and od3 H1 = H1c. `od2` is the only config that separates H1 (e) from H1c (c), and it's the weakest config (0.35). That's weakly in favour of a count over a done *set*.
4. **H2 is the weakest hypothesis at the head level.** The first run never sparsified (every head > 0.5 after 40 steps). The layer-20–55 re-run with ~400 steps does sparsify (177 heads > 0.9), but only reaches 0.45 soft / 0.31 binarized, which is exactly goal-250's 0.31: the learned H2 heads add nothing over the goal heads.
5. **Thresholding loses about a third of the effect** (H3 0.69 → 0.45, H1c 0.62 → 0.43, H1 mix 0.63 → 0.40). The soft masks spread weight over many partially-weighted heads: about 800 heads have m > 0 for H3. A top-250 cut keeps only 51–53% of the mask mass for H1c / H3, and top-400 about 75%. H2's binarized set loses little only because its mask is near-uniform.
6. **Head-set overlaps (0.9 sets, all layers):**
   - between hypotheses: H1c ∩ H3 = 161, H1c ∩ H2 = 134, H3 ∩ H2 = 129;
   - with the H1 mix λ20 set: H1c 189, H3 170;
   - with goal-250: H1c 88, H3 101.

   The learned sets share a layer-40–47 core that goal-250 mostly misses.

   Notebook `notebooks/diff/17_nrgs_hypothesis_heads.ipynb` has the landscapes, ranked mask curves, soft-mask cosine / Pearson, and overlaps against a layer-matched null.
7. **Restricting to layers 20–55 costs little.** Soft flips drop 0.03–0.07 (H3 0.69 → 0.62, H1c 0.62 → 0.56, H1 mix 0.63 → 0.56). The binarized sets hold or improve (H1c 0.43 → 0.47, H1 mix 0.40 → 0.44) with fewer heads (187–201). The H1c SRC′ collapse is unchanged (0.56 → 0.02).

### 9b. Q-transfer generation evals on H1-style heads (order-matched / order-diff, py→py, causal-shift, n = 32 × 4 datatypes)

| patch heads | α | order-matched | order-diff | diff-vs-prog: c rate |
|---|---|---|---|---|
| goal-250 | 2 | **0.68** | **0.56** | — |
| goal-250 | 1 | 0.61 | 0.46 | — |
| H1 progress heads, od1 only (265) | 2 | 0.56 | 0.53 | 0.10 |
| H1 mix λ20 (225) | 2 | 0.56 | 0.42 | 0.13 |
| H1 mix λ20 (225) | 1 | 0.49 | 0.38 | 0.02 |
| H1 mix λ50 (140) | 2 | 0.55 | 0.41 | 0.23 |
| H1 mix λ50 (140) | 1 | 0.45 | 0.32 | 0.02 |
| **union goal-250 ∪ mix λ20 (387)** | 1 | **0.76** | 0.54 | 0.83 |
| **union (387)** | 2 | 0.73 | **0.59** | **0.87** |

- **Learned H1 head sets alone don't beat goal-250 in generation** at either α. The OV boost (α = 2) helps every set, and the ranking holds at α = 1. So the gap isn't an α artefact.
- **The union beats both of its parts.** At α = 1 it reaches 0.76 / 0.54 vs goal-250's 0.61 / 0.46. On diff-vs-prog it writes SRC's frontier c 83–87% of the time, against about 2% for mix λ20 alone.
- **Open confound:** the union has more heads (387 vs 250). A size-matched control (goal-250 + 137 random non-goal heads) hasn't been run yet.

### 9c. Sub-head (W_QK SVD-component) masks (`train_nrgs_svd_mask.py`, filter paper App. I)

Each candidate head's query transplant is split into its 128 W_QK singular components (q_patch = q_dst + (q_src − q_dst)·M·diag(m)·M⁻¹), and a mask over the components is trained with the same next-token objective as §9a.
- **Math checks pass on 32B:** U·S·Vᵀ reproduces W_QK to about 1e-5, and the all-ones mask matches the head-level transplant (max |Δlogp| = 0.000).
- **Sparsity** is the mean over *all* candidate components, so goal-only runs aren't pushed 10× harder.
- **Columns** (next-token flip on the same val sets as §9a):
  - **soft**: the trained component mask;
  - **comps > 0.9**: the binarized component set;
  - **same heads, whole**: every head that owns a selected component, patched as a whole head;
  - **goal**: goal-250, or the 204 in-band goal heads for the layer 20–55 runs;
  - **′**: the SRC′ control.

| run | candidates | soft | soft′ | comps > 0.9 | comps′ | same heads, whole | goal |
|---|---|---|---|---|---|---|---|
| **H3**, all heads, all layers, λ1 | 327,680 | 0.91 | — | 0.89 (90,874) | — | 0.15 (2,448 h) | 0.49 |
| H3, goal-only, all layers, λ1 | 32,000 | 0.74 | — | 0.73 (16,121) | — | 0.49 (250 h) | 0.49 |
| **H3**, all heads, L20–55, λ3 | 184,320 | 0.92 | — | **0.94** (33,618) | — | 0.35 (1,311 h) | 0.44 |
| H3, goal-only, L20–55, λ3 | 26,112 | 0.73 | — | **0.74** (12,772) | — | 0.44 (204 h) | 0.44 |
| **H1c**, all heads, all layers, λ1 | 327,680 | 0.93 | 0.20 | 0.82 (87,988) | 0.27 | 0.12 (2,440 h) | 0.33 |
| H1c, goal-only, all layers, λ1 | 32,000 | 0.51 | 0.32 | 0.52 (16,428) | 0.30 | 0.33 (250 h) | 0.33 |
| **H1c**, all heads, L20–55, λ3 | 184,320 | 0.94 | 0.14 | 0.88 (34,141) | 0.31 | 0.30 (1,308 h) | 0.30 |
| H1c, goal-only, L20–55, λ3 | 26,112 | 0.47 | 0.28 | 0.47 (13,067) | 0.25 | 0.30 (204 h) | 0.30 |
| **H2**, all heads, L20–55, λ3 | 184,320 | 0.82 | — | 0.74 (26,572) | — | 0.22 (1,270 h) | 0.31 |
| H2, goal-only, L20–55, λ3 | 26,112 | 0.54 | — | 0.56 (11,507) | — | 0.31 (204 h) | 0.31 |
| **H1 mix**, all heads, L20–55, λ3 | 184,320 | 0.91 | 0.14 | 0.29 (9,610) | 0.04 | 0.36 (1,035 h) | 0.39 |
| H1 mix, goal-only, L20–55, λ3 | 26,112 | 0.58 | 0.32 | 0.55 (12,179) | 0.27 | 0.39 (204 h) | 0.39 |
| **dvp1** (`diff_vs_prog_1` only, target d), all heads, L20–55, λ3 | 184,320 | **0.97** | — | 0.72 (21,673) | — | 0.25 (1,248 h) | 0.30 |
| dvp1, goal-only, L20–55, λ3 | 26,112 | 0.49 | — | 0.44 (11,892) | — | 0.30 (204 h) | 0.30 |

Readings:
1. **Sub-head masks are far stronger than head masks, and the selection is in the components, not the heads.** Every all-heads run reaches 0.82–0.97 soft (head masks: 0.45–0.69). Patching the *same heads whole* collapses to 0.12–0.36, below goal-250. A head-level transplant drags along many components that oppose the target; only a component subset carries it.
2. **Components binarize cleanly** (comps > 0.9 within 0.0–0.1 of soft) for H3, H1c, H2 and goal-only runs. The two exceptions are H1 mix (0.91 → 0.29; only 18% of its mask mass is above 0.9, so the solution is diffuse) and dvp1 (0.97 → 0.72).
3. **Within the goal heads, every hypothesis is weaker but present**: goal-only binarized 0.74 (H3) > 0.56 (H2) > 0.55 (H1 mix) > 0.47 (H1c) > 0.44 (dvp1), each above the same heads patched whole (0.30–0.44). H3 is again the best-carried hypothesis inside the goal heads.
4. **The count signal (H1c) is the most specific to SRC′**: all-heads soft 0.94 → 0.14, but the binarized components only drop to 0.31, and the goal-only runs barely move (0.47 → 0.28). The done-count lives mostly in components *outside* the goal heads.
5. **H3 vs H1c component overlap is 2.0× chance**, so the two hypotheses' components are related but far from identical.

### 9d. Hypothesis-config generation evals (`q_transfer_hypothesis_eval.py`, α = 2, `=` pivot, multi-letter judge, QTASKS banks, n = 64 × 4 datatypes per hypothesis group)

Each group pools the strict configs of §6b (H1: od1–3, om1–2, dvp1; H1c: h1c_1–2; H2: h2_1–2; H3: h3_1–2). Clean DST writes the native letter (H0) ≥ 99% in every group, so every patched rate below is a shift. Entries are the rate at which the patched DST writes **that group's unique target**, followed by its strongest competitor.

| patch heads (n) | H1c configs: H1c | H2 configs: H2 / H1c | H3 configs: H3 / (H1c = H2) | H1 configs: H1 · src_target |
|---|---|---|---|---|
| goal-250 (250) | **0.25** | **0.30** / 0.10 | **0.33** / 0.09 | 0.33 · 0.15 |
| control-250 (250) | 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 · 0.00 |
| H1c heads, all layers (267) | 0.23 | 0.06 / 0.14 | 0.10 / 0.20 | 0.22 · 0.00 |
| H1c heads, L20–55 (201) | 0.21 | 0.05 / 0.15 | 0.09 / 0.23 | 0.23 · 0.01 |
| H2 heads, L20–55 (177) | 0.14 | 0.18 / 0.08 | 0.19 / 0.13 | 0.25 · 0.12 |
| H3 heads, all layers (247) | 0.12 | 0.12 / 0.09 | 0.18 / 0.17 | 0.27 · 0.08 |
| H3 heads, L20–55 (198) | 0.16 | 0.13 / 0.12 | 0.20 / 0.15 | 0.32 · 0.09 |
| H1 mix heads, L20–55 (187) | 0.20 | 0.10 / 0.12 | 0.17 / 0.18 | 0.28 · 0.03 |

Readings:
1. **The goal heads prefer order-id / induction over count by about 3×.** In H2 configs goal-250 writes the H2 target 0.30 vs H1c 0.10; in H3 configs H3 0.33 vs 0.09.
2. **The H1c heads are the only count-specific set.** They flip the preference: H2 configs H1c 0.14–0.15 vs H2 0.05–0.06; H3 configs 0.20–0.23 vs 0.09–0.10.
3. **The learned H2 / H3 / mix head sets are weak and barely selective in generation** (own target 0.17–0.20, close to their competitor), well below goal-250, even though H3 is the strongest hypothesis at the next-token level (§9a). Head masks trained on next-token flips don't carry over to free generation.
4. **control-250 is exactly 0** in every group, so the shifts are specific to the patched heads.

### 9e. Component-mask generation evals (α = 1, py→py, `=` pivot, n = 32 × 4 datatypes; out-of-distribution for the masks)

The §9c masks are patched as partial query transplants (`--component-mask`, soft values) in the existing generation evals.

| patch | order-diff (causal D) | order-matched (causal D) | diff-vs-prog, old judge: **c** rate | `c_in_dst` joint: **d** / c / b |
|---|---|---|---|---|
| goal-250, whole heads | 0.46 | 0.61 | 0.86 | 0.05 / 0.87 / 0.05 |
| H1 mix, all heads | **0.66** | **0.80** | 0.08 | 0.28 / 0.12 / 0.59 |
| H1 mix, goal-only | 0.40 | 0.48 | 0.76 | 0.12 / 0.58 / 0.26 |
| H2, all heads | 0.63 | 0.66 | 0.58 | 0.12 / 0.55 / 0.21 |
| H2, goal-only | 0.31 | 0.46 | 0.83 | 0.05 / 0.69 / 0.23 |
| **dvp1, all heads** (trained on target d) | — | — | — | **0.46** / 0.12 / 0.35 |
| dvp1, goal-only | — | — | — | 0.15 / 0.52 / 0.28 |

The diff-vs-prog configuration is $p_\mathrm{src}: \mathrm{a'b'cd \to c}$, $p_\mathrm{dst}: \mathrm{a'bdc \to b}$. Both H1 (set progress) and H2 (order id) predict **d**; **c** is SRC's literal frontier (paper App I.3); **b** is the no-op. The "old judge" column is `q_transfer_diff_vs_prog_eval.py` (c vs b only, d falls in "neither"). The `c_in_dst` column is the paper's joint b/c/d judge (`diff_trigger_eval.py --variants c_in_dst`), i.e. the setup reported in App I.3 (goal-250 there: c 0.84, d 0.05 at α=2 with restrict).

Readings:
1. **Out of distribution, the all-heads component masks beat goal-250 on the H1 generation evals** (order-diff 0.66 vs 0.46, order-matched 0.80 vs 0.61 for H1 mix). Restricted to the goal heads they fall below it (0.40 / 0.48).
2. **The goal heads carry c, not d.** Whole goal-250 writes c 0.87 / d 0.05; every goal-only mask also lands mostly on c (0.52–0.69), with d at 0.05–0.15.
3. **d is reachable, but only with components outside the goal heads.** The dvp1 all-heads mask reaches d = 0.46 in free generation (9× goal-250, 0.41–0.53 across datatypes), and H1 mix all-heads 0.28. Restricting either to the goal heads drops d to 0.15 / 0.12 and brings c back.
4. **Most misses fall back to b (no-op), not c**, for the all-heads masks (dvp1 b 0.35, H1 mix 0.59). The progress-carrying components can suppress c without always committing to d.

### 9f. Synthesis

- **What the goal-head query carries, in order of strength:** SRC's frontier identity (c; App I.3), then induction / order-id (H3 ≈ H2), then count (H1c), with set progress (H1) not separable from count in these designs (§9a reading 3).
- **The progress / count signal is mostly outside the goal heads.** At the head level (H1c learned set ∩ goal-250 = 88 / 267), at the component level (goal-only H1c 0.47 vs all-heads 0.88), and in generation (d only appears when non-goal components are allowed).
- **Components, not heads, are the right unit.** The same heads patched whole fall below goal-250 while their selected components reach 0.7–0.9 next-token flip and the best out-of-distribution generation numbers we have.

### 9g. Known caveats

- **Stub-name leak:** in the H1c configs and in `od1` / `om1`, SRC's pivot stub and DST*'s target stub are both `c =`. The SRC′ drop fits count *or* stub-name copying; a neutral-stub variant would separate them.
- **Bank mismatch:** mining uses the 7-predicate head-loc bank (3 datatypes), while the generation evals use the 16-predicate QTASKS banks (4 datatypes). Mask flips and generation rates are on different predicate sets.
- **H2 is int-only at mining time:** with distinct families, the head-loc bank can't fill 6 tasks for str / list[int].
- **In- vs out-of-distribution:** §9a / §9c flips are validation sets of the training configs (pair-disjoint predicates, same config). Only §9d / §9e are out of distribution.
- **Size confound** for the union set in §9b (387 vs 250 heads): no size-matched control yet.
- **The dvp1 masks were not run on order-diff / order-matched**, and the component-mask evals use α = 1 without restrict, while App I.3 used α = 2 with restrict.

Nothing in this battery is still queued.
