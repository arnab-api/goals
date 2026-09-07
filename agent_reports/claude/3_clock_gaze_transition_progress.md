# Pursuing the Clock `c`: Gaze, Transition, and Order-ID — Two-Week Progress

*Written 2026-06-10. Covers work from ~2026-05-28 to 2026-06-10. Progress/brainstorm note; experiments here include several inconclusive or failed attempts, reported deliberately.*

---

## 0. Where this sits

Report `2_clock_state_machine.md` laid out the **two-signal hypothesis**: during code generation the residual stream may carry separable signals
- **`g`** — *which* instrumental goal (predicate) is being pursued (context-stable, implementation-invariant), and
- **`c`** — a *completion/progress clock* that drives the transition from one goal to the next.

The last two weeks were almost entirely about making `c` (and its companion machinery) **empirically tractable**: building a clean dataset where the "clock" is legible, localizing the attention heads that read the current instruction, and trying to isolate the moment/representation of *transition* between instructions. Along the way the framing sharpened from a single "commitment-region" probe into three more specific objects:

1. **Gaze** — the attention pattern by which generated tokens look *back* at the instruction they are implementing.
2. **Transition** — the property that makes a token stop looking at the current instruction and move on.
3. **Order-ID** — whether the model indexes "the next instruction" by *position/order* (instruction #2) rather than by *content*.

---

## 1. Datasets: from `multistep_clock` (v2/v3) to condition-check

**Multistep clock (v1–v3).** The first substrate (`scripts/clock/build_multistep_clock_data.py`, `src/goals/multistep_data.py`, `src/goals/multistep_label.py`, nb `17_clock_multi_attn`). Functions implement several sequential sub-steps; generations were annotated with `# Start 'STEP X'` / `# End 'STEP X'` markers (v2 targeted the comment `#`; v3 targeted line-ends). An oracle labeled each generation into per-step spans (`prompt_token_range`, `gen_token_range`, `gen_line_range`). This gave us the contrastive positions for the commitment probe, but the comment markers were an artificial scaffold and a confound magnet.

**Condition-check clock (the current substrate).** `5f63a50` introduced a cleaner, more controllable dataset (`src/goals/condition_data.py`, `scripts/clock/build_condition_data.py`, nb `01_data`). The task: `check_conditions(n)` evaluates an **ordered list of independent integer predicates** ("n is odd", "n is a perfect square", "divisible by 12", …), assigning each result to a single-letter variable (`a`, `b`, `c`, …) in order, then a final print. Key properties:
- **Independent, order-free predicates** → the contrast between "implementing predicate i" vs "predicate j" is clean (all orderings are valid; only *content* and *order* differ).
- **Two decoding styles**: `monolithic` (each predicate = one line, `a = <expr>`) and `modular` (each predicate spans multiple lines). This duality became the lever for the transition probe (§7).
- **Precomputed `id_substrs` key ranges** (`key_ranges.json`): per-predicate token ranges of the *informative* keywords ("perfect", "square", "divisible", "{k}") — the keys that carry the predicate's identity, with stopwords removed. This is what the head-scoring and gaze-attractor code reads.
- 1000 samples generated for Qwen2.5-Coder-32B-Instruct, both modalities, with cached generations + oracle labels + value-weighted attention (`attn.npz`).

---

## 2. Gaze-head localization (nb `02_condition_gaze`, `score_condition_heads.py`)

**Step-lookback score.** For each attention head `(ℓ,h)` we measure how much, while generating predicate *k*'s code, the head attends *back* to predicate *k*'s `id_substrs` versus the other predicates'. Formally (per cached sample):
```
score[k'] = mean_{q ∈ query_window}  mean_{s ∈ keys(k')}  attn[ℓ,h,q,s]
head_score = score[k] / AGG_{k'} score[k']
```
- `AGG = mean` (target included) or `max` (other predicates only, target excluded) — the `--compare_with` knob.
- The query window length cancels in the ratio (positive homogeneity), so query-span normalization is cosmetic — confirmed for both `mean` and `max`.
- Scoring knobs (`--q_span`, `--compare_with`, `--key_agg`) were added (`b2937e2`) and made **config-tagged** in the cache (`b4bb536`) after a bug where re-running with a different `--compare_with` silently reused stale `mean` sums (`[skip cached]`). Results now live under `head_scores/qspan-<n>_cmp-<mode>/`.

**Two head sets, and their overlap.** We ended up with two candidate localizations:
- **`gaze_heads`** — top-K by the step-lookback score above (e.g. top-100/200; threshold variant too).
- **`goalscope_heads`** — heads from a *separately trained* differentiable head mask (`results/localization/.../head_mask.pt`), zeroed outside layers 25–55, threshold 0.9 → ~122 heads.

A visualization cell (`ff87407`) shows their layer×head distribution and overlap (Jaccard). They overlap partially but are not identical — gaze (attention-pattern-derived) vs goalscope (causally-trained) capture related but distinct head populations.

---

## 3. Causal test of the gaze heads — what failed and what worked

The hypothesis chain: gaze heads look back at the predicate being implemented; if we **redirect** that gaze to a *different* predicate's spec at the assignment pivot (`b =`), the model should implement *that* predicate instead.

**Attempt 1 — knockout-redirect (FAILED).** Knock out the gaze heads' attention to the *current* predicate's span and hope the freed mass lands on the target. Even with 500 heads, **no effect**. Diagnostics revealed why: knockout removes the competitor span's mass, but the freed attention flows to **code tokens**, not to the intended target span. Conclusion: **knockout alone cannot test sufficiency** — it can suppress but not steer.

**Attempt 2 — restrict-to-target (WORKS).** Instead, restrict the selected gaze heads to attend *only* to the target predicate's spec span (mask everything else). This is a positive lever, not a subtractive one. Findings:
- The intervention does redirect generation toward the target predicate — but the effect is **distributed**: it needs ~300 heads (~11% of all heads) to show qualitatively.
- A **random-head control** of the same size shows no change from baseline → the effect is gaze-head-specific, not a generic "scramble attention" artifact.
- Implementation note: the masked, per-head, non-cached decode requires declaring nnsight interventions in ascending layer order (`OutOfOrderError` otherwise); softmax-input editing via the eager `nn_functional_softmax_0` hook.

**Lesson.** Sufficiency tests need an *attractive* intervention (restrict-to-target), not a *repulsive* one (knockout). The gaze signal is causal but diffuse — no small set of heads is the bottleneck.

---

## 4. Gaze-attractor score (nb `02` §2b)

To view gaze *continuously* (not just an aggregate head score), we built a per-query-token **gaze-attractor** score: for each gaze head, take its per-head query `q` at each generated token and key `k` at each predicate's `id_substrs`, score `q·k` (knob: raw dot vs cosine; knob: include RoPE or not), mean over the span's keys, mean over gaze heads. Plotted as one line per predicate over the generated tokens, with each predicate's own code segment shaded — the hypothesis being the diagonal: predicate *i*'s line peaks while *i*'s code is generated.

Design decisions worth recording:
- **Affinity, not softmax weights** — deliberately use `q·k` (the value-weighted attention isn't in [0,1], and cosine gives cross-head comparability). RoPE optional because we want the *content* component; the residual carries no absolute position.
- **GQA** handled explicitly (query head → kv group).
- **Baseline line** = mean affinity to non-`id_substr` prompt tokens ("rest of prompt"), so a high score is read above chance.
- **Cross-sample aggregation** is *bucketed*: the 4 generation spans (3 predicates + print/rest) are each split into N equal buckets, giving a fixed-width axis so different-length generations are averageable; `NORM_MODE ∈ {diff, ratio, none}` with `diff` (score − baseline) the default (ratio is fragile when the baseline ≈ 0 or negative). Status: built; aggregate run pending.

---

## 5. Order-ID interchange (nb `03_oid`) — inconclusive

**Hypothesis.** The model assigns each instruction an *order id*; at `b =` it looks up "the instruction with order id 2," **independent of that instruction's content**. If so, the order-id representation should be transportable across prompts.

**Setup.** Cross-prompt activation patching: capture the source prompt's latent at its `c =` (order id 3) and patch it into a destination prompt's `b =` (order id 2), over a 5-layer window swept across depth, at residual / mlp / attn outputs. Readout = first generated token: a *late*-layer patch should copy the source's content (`s3`); an *early/mid* patch, if order-id is disentangled, should yield the destination's *own* 3rd predicate (`d3`). Non-collision source/destination pairs were filtered from cached generations (no forward needed).

**Refinement — content-averaged order-id vector.** Single source latents are content-entangled, so we built a **content-averaged** vector: average the module latent over the first k tokens of the order-*i* instruction span across many prompts, hoping predicate content cancels and the shared "I am the i-th instruction" factor survives. Tested both `replace(μ_target)` and `add(μ_target − μ_current)`; cached to disk. Cosine sanity showed the order means are *highly* similar (cos > 0.9 in residual), with a small but non-trivial difference vector.

**Status: inconclusive.** Neither the single-latent interchange nor the averaged-vector injection cleanly transported the order id. Multi-pair sweeps with softer metrics (argmax-fraction + mean±std logit/prob with error bars) were added (`cf44216`) to aggregate over many pairs rather than eyeball single examples — those aggregate runs are the next thing to inspect. The high-cosine, small-delta structure of the order means is a warning sign that "order id" may not be a clean linear factor separable from the shared instruction scaffolding.

---

## 6. Transition probe (nb `04_newline_probe`) — the conceptual core

This is where the two-week arc converges. The question from `draft.md`: **what makes the final token of a code segment shift its gaze away from the current instruction?** That shift is the *transition* — the visible footprint of `c` ticking over.

### 6.1 The Δq brainstorm (and its critique)

The initial idea (draft.md): the transition lives in the difference between consecutive query states. For a gaze head, project the residual difference `h(n) − h(n−1)` through the head's Q-matrix, then back via pseudo-inverse, sum over heads → a residual-space "transition direction" to ablate / inject / logit-lens.

Critique that reshaped the plan:
- **`Q†Q` is just an orthogonal projector** onto the head's read-subspace; the whole pipeline collapses to "mean residual delta, filtered to the gaze-head read subspace." All the risk lives in the mean delta.
- **Token-identity dominates** `h(n) − h(n−1)`: token *n* is almost always `\n`, so averaging extracts "newline-ness," not "time to move on." This is the project's competence-vs-goal confound in a new costume.
- **Summing across heads double-counts** (overlapping subspaces); **RoPE** is a positional, non-residual contributor to the attention shift (though for a purely *semantic* residual direction it's fine to ignore).

**Resolution: make it contrastive.** Define the signal by contrasting *transition* newlines against *non-transition* newlines, holding token identity as fixed as possible, and use Q as a *test* ("do the gaze heads read this direction?") rather than the *definition*.

### 6.2 The cross-modality contrast (and the irreducible confound)

The clean contrast requires non-transition newlines, which only exist in **modular** (multi-line predicate bodies), while transition newlines that are a *single* `\n` (token id 198) exist in **monolithic** (each predicate = one line, each line-end = a transition). Tokenizer check confirmed `\n`=198 and `\n\n`=271 are distinct, but **monolithic line-ends and modular interior line-ends are the same token 198** — so the token-matched contrast exists, with the remaining nuisance being *decoding style* (mono vs modular), not the newline token.

- **D+ (transition)** = monolithic condition line-ends.
- **D- (no transition)** = modular interior line-ends.
- The style confound is irreducible in training (D+ is all-mono, D- is all-modular). It's addressed by a **held-out disentangling test**: evaluate the trained probe on **modular *transition* newlines** (segment-ends, never trained). If the probe calls them D+, it learned *transition*; if D-, it learned *style*. (A diff-of-diffs across the 2×2 of {style}×{token} was also sketched as an alternative.)

### 6.3 Implementation

`04_newline_probe` ports the probe machinery from `18.1` (TokenRow position sets, HS caching with per-token-id means, token fixed-effects demeaning, sample-level split, hinge + mean-diff directions, layer sweep, raw-vs-demeaned sweep). Newlines absorb `)\n`, `]\n` (tokens whose decoded text ends in `\n`). The probe reads at the `\n` token (LAG=0).

**Confound caught immediately:** a large fraction of D- newlines end in `:\n` (`if …:`, `for …:`, `def …:`), giving the *embedding* layer ~0.88 separability — pure token identity. Fixed by **blacklisting `:\n`-terminated newlines** from all roles. (Status: blacklist applied; re-run of the HS pass + sweeps pending to confirm emb collapses toward chance.)

---

## 7. The commitment-region probe lineage (nb `18`, `18.1`)

Worth recording because the transition probe inherits its methodology and its hard-won confound list. The commitment probe (`18`, then `18.1` on prompted-to-label v3 generations) asked: is there a hidden-state direction at a step's boundary (d+ = last code line-end of a step) vs mid-step (d- = interior line-end)? Confounds chased and partially defused:
- **Whitespace-only / `\n\n` detector confound** — the probe may just detect the `\n\n` token. Diagnosed via per-token-string row counts and per-layer same-token "sat"; attacked with **augmentation** (re-tokenized/rewritten copies) to break the `\n\n`↔boundary correlation.
- **Token fixed-effects demeaning** (`59b9ced`) — subtract the per-(layer, token-id) mean before probing; at the embedding layer this should collapse the probe to chance (a sanity check that the signal isn't token identity). The d+/d- token-id overlap (Jaccard) tells whether demeaning risks ablating real signal.
- **LAG sweep** — is the signal bound to the newline token, or present a token or two earlier?

These are exactly the diagnostics the transition probe now runs by default.

---

## 8. Cross-cutting methodological lessons

1. **Token identity is the recurring adversary.** Every probe/score that reads at a structurally-special token (`\n`, `\n\n`, `:\n`, `#`) risks learning the token, not the concept. Defenses now standard: token fixed-effects demeaning, raw-vs-demeaned sweep, `:\n` blacklist, embedding-layer collapse as a sanity check.
2. **Sufficiency needs an attractive intervention.** Knockout suppresses but cannot steer (freed mass goes to code). Restrict-to-target works but is diffuse (~11% of heads).
3. **The gaze signal is real but distributed** — no small head bottleneck; consistent with `g` being a population-level direction rather than a few heads.
4. **Order-id may not be a clean linear factor** — order means are near-collinear (cos > 0.9); interchange did not cleanly transport it.
5. **Cache hygiene matters** — config-tagged caches after the `[skip cached]` bug; cached greedy tokens reused for collision filtering (no forwards).
6. **Use the real thing over a proxy when it's one hook away** — e.g. softmax weights vs cosine — but the user deliberately chose `q·k`/cosine for cross-head comparability; documented as a choice, not an oversight.

---

## 9. Open questions / next steps

- **Transition probe** (nb 04): re-run after the `:\n` blacklist; does emb collapse? Does a mid-layer probe beat chance on held-out **modular transitions** (the transition-vs-style verdict)? If style still leaks, run the diff-of-diffs.
- **Gaze-attractor aggregate** (nb 02): does the bucketed cross-sample matrix show the diagonal (predicate *i* attracts predicate *i*'s keys during *i*'s code)?
- **Order-id** (nb 03): inspect the multi-pair aggregate sweeps; if the averaged-vector injection is flat across layers, the order-id-as-linear-factor hypothesis is likely wrong — pivot to "order id is computed, not stored."
- **Tie `g` and `c` together:** the gaze heads (`g`-adjacent) and the transition signal (`c`-adjacent) live in overlapping but distinct head/layer populations. A clean next experiment: does the transition direction *gate* the gaze heads (turn their look-back off) at the boundary?
- **Verifier caveat** (from draft.md): if the model also carries a "did I do this right so far?" signal, it will fight transition ablation/injection. Testable by conditioning on whether the current line is already complete & correct.

---

## Appendix — commit trail (2026-05-28 → 2026-06-10)

| commit | date | what |
|---|---|---|
| `8133b37` | 05-28 | recursion-direction baselines, V_LAYER sweep, Schut supervision note (prior track) |
| `8764b07` | 05-29 | clock nb17 + interactive attention viz + batched `get_attention_matrices` |
| `bcfb6d3`/`1d25272` | 06-01 | multistep clock dataset + build script + attention viz; `multistep_data`/`multistep_label` |
| `726b4e9` | 06-02 | commitment-region probe (nb18); nb17 rename |
| `a5725c6` | 06-04 | nb18 iteration: lag-shift, ws-only confound, augmentation, lag-sweep |
| `b2937e2` | 06-05 | `score_multistep_heads`: q_span / compare_with / key_agg knobs |
| `59b9ced` | 06-07 | nb18.1 prompted-to-label probe (v2/v3) + token fixed-effects demeaning |
| `5f63a50` | 06-08 | **condition-check dataset**: generator, build/score scripts, 1000 samples, nb01 |
| `b4bb536` | 06-09 | condition gaze: config-tagged head scores + causal nb (02) |
| `ff87407` | 06-09 | oid interchange nb (03) + gaze/goalscope overlap + query-span norm |
| `cf44216` | 06-10 | gaze-attractor score + oid multi-pair sweeps + newline transition probe (04) |
