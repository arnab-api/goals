# Schut et al. — Dynamic Concepts and the Supervision Question

**Paper.** Schut et al., *"Bridging the Human-AI Knowledge Gap: Concept Discovery and Transfer in AlphaZero"* (`papers/bridging_human_AI.pdf`).
**Status.** Working notes from a discussion thread. Iteratively revisit — open questions are marked **[TODO]**. Companion to [`../schut_et_al_adaptation_analysis.md`](../schut_et_al_adaptation_analysis.md), which covered the original adaptation plan; this note focuses specifically on the supervision structure of §4.1.2 and what it implies for the recursion-direction setup.

---

## 0. Why this thread

Our recursion-direction pipeline (notebook 16) cites Schut §4.1.1 as its methodological ancestor. While re-reading the paper we kept coming back to §4.1.2 (dynamic concepts) because that's the genuinely novel contribution — and because the paper's framing suggests they extract concept directions *without* ground-truth concept labels, which our pipeline does assume. This note pins down what level of supervision §4.1.2 actually needs, and where our setup sits relative to it.

---

## 1. Questions we asked, in order

1. **Q1.** In §4.1.2, do they mine one concept from one board state (one root position), or from many?
2. **Q2.** A single strategic concept (e.g. "queenside redeployment") might be optimal in many different board states — this is exactly the *plasticity* property we care about. Do they cluster the per-position directions into a "general" strategy direction?
3. **Q3.** If they're not labeling positions by concept, by what criterion do they decide which positions go into `X⁺`? You can't sample a thematic set without some notion of the theme.

These three questions are the spine of this note.

---

## 2. What the paper says

### 2.1. Q1 — one position vs. a set

**Eq. 5 (textbook formulation, §4.1.2).** Reads as one root position, one optimal rollout `X⁺_{≤T}`, multiple subpar rollouts `X⁻_{≤T,j}` indexed by branching depth `j`. The constraint is `v · z⁺_{t} ≥ v · z⁻_{t,j}` for `t ≤ T, j ≤ T̃`. Standalone, this is **one chess position, many within-tree forks**.

**Practical use (§8.4 appendix).** They quietly extend the constraint set across many positions:

> *"For each opening, we use the encyclopedia of chess openings to find relevant starting chess positions to construct `X⁺`. For each chess position, we ran MCTS to obtain the search statistics and used the formula in Equation 5 to find a concept for each opening."* (§8.4 Openings, line 2351–2354)

> *"…we constructed a dataset with 3,974 chess positions and used the formulation provided in Equation 5 to find concept vectors."* (§8.4 AZ Games, line 2367)

So `X⁺` in practice is a **set of positions**, and Eq. 5 is run with the union of constraints over all of them. One `v` is fit to satisfy `v · z⁺_{i,t} ≥ v · z⁻_{i,t,j}` jointly across positions `i`, depths `t`, and within-tree branching `j`.

The paper itself flags single-position mining as a failure mode:
> *"Our approach […] provides many concept vectors, some or many of which represent known concepts or **non-generalisable concepts (i.e., only applicable to a single chess position)**."* (§4.2, line 525, emphasis added)

**Bottom line on Q1.** The mathematics permits one-position mining; the experimental pipeline never does it. Generality is required from the start.

### 2.2. Q2 — do they cluster directions?

**No.** There is no post-hoc clustering or averaging of `v` vectors anywhere in the paper. Generality is achieved by three earlier-pipeline mechanisms instead:

1. **Pre-aggregation in the constraint set.** Curate `X⁺` as a multi-position thematic group before optimization runs. The resulting `v` must satisfy all positions' constraints jointly.
2. **Teachability filter (§4.2.1).** Any candidate `v` is taken as a curriculum: AZ teaches a student network on prototypes scoring high on `v · z`, the student is tested on held-out prototypes. Position-specific `v`'s don't generalize and get dropped.
3. **Prototype identification (§4.2, line 551 ff.).** After `v` is fit, score every position in a corpus by `c(x) = v · f_l(x)`, take the top 2.5%, call those the prototype set. This is clustering of *positions* by the direction — not of directions with each other.

The §6.2.1 case study (queenside redeployment via `Qc1`, prophylactic `Qd2 → b4`, bishop-driven rook displacement `Bb3`, …) shows the resulting plasticity: one mined direction fires on a prototype set whose concrete realizations look entirely different from one another.

**Bottom line on Q2.** Plasticity is an output property of the mined direction (visible in the prototype set), not produced by combining many directions. The direction itself is fit *once* against a multi-position constraint set.

### 2.3. Q3 — what's the criterion for `X⁺`?

This is the most important point. Schut don't escape supervision; they decouple two kinds of it that we usually conflate:

| Supervision kind | What it labels | §4.1.1 (static) | §4.1.2 (dynamic) |
|---|---|---|---|
| **Membership** | which positions go into `X⁺` | from concept label (`is king on e1?`) | from external proxy structure |
| **Semantics** | what the concept is named / means | from concept label | post-hoc, by inspecting prototypes |

§4.1.2 sheds the *semantics* supervision but still needs *membership* supervision. The latter comes from a cheaper, model- or structure-internal proxy. Three concrete proxies appear in §8.4:

1. **Opening taxonomy (ECO codes).** "Positions arising in the Sicilian Defense" — a syntactic grouping over opening move sequences. No claim about what the Sicilian-concept *is*; whatever `v_{Sicilian}` ends up encoding is discovered, not declared.
2. **STS themes.** Strategic Test Suite — positions grouped by human-curated themes ("square vacancy", …). Used mostly as a benchmark; weak semantic labels but not strong concept supervision.
3. **AZ-internal disagreement.** Positions where current-AZ and a 75-Elo-weaker checkpoint pick different moves. Pure model-internal proxy for "interesting position", zero human labels.

The **within-position contrast** (`X⁺_i` optimal vs. `X⁻_i` subpar rollout) replaces concept labels with AZ's own value/visit signal — the model tells you which trajectory it prefers, no labels needed.

So the full pipeline reads:

```
(structural / model-internal proxy → groups positions into X⁺)
        ×
(model value signal → picks X⁺_i vs. X⁻_i within each position)
        →
one v fit jointly over the constraint set
        →
teachability filter
        →
prototype inspection
        →
human names what survives
```

**Bottom line on Q3.** You don't sample `X⁺` from nothing. You sample it from a proxy grouping that you *suspect* carries strategic content — without having to name the content. The criterion is much cheaper than a concept label, but it isn't free.

---

## 3. What the paper leaves out / open tensions

- **No formal account of "good proxy".** The choice of ECO codes / AZ disagreement / STS is justified empirically (it works) but not theoretically. There's no characterization of which proxies *would* yield a coherent concept and which wouldn't. This is the missing rigorous link between "grouping criterion" and "coherent direction emerges".
- **No baseline against random `X⁺`.** Would mining `v` from a random set of positions still yield *something* that passes teachability filtering? The paper doesn't run this. Without it, we can't quantify how much of the work the proxy structure does vs. how much teachability filtering does after the fact.
- **Filtering rate is large.** §4.2 mentions filtering out the majority of candidate concepts as either known or non-generalisable. This suggests a lot of the heavy lifting is done downstream by teachability — the convex optimization is noisy and most directions don't survive. Quantitatively, how much survives?
- **No identifiability claim.** A given prototype set could be consistent with multiple semantically distinct `v`'s. The paper doesn't address whether the mined `v` is unique up to scaling, or whether different random initializations / different `X⁻` choices give different `v`'s that all "explain" the same prototypes.
- **"Concept" semantics are post-hoc and human-labelled.** Step 6 of the pipeline ("human names what survives") is the same epistemic move as in any unsupervised representation-learning paper — you can always find a story. The chess grandmaster validation in §6 is the paper's main hedge against this, but for non-game domains there's no grandmaster equivalent.

---

## 4. Mapping to our setting

### 4.1. Where the recursion direction (nb 16) actually sits

We have:
- **Membership** supervision: the binary label "recursive vs. iterative" from `label_strategy`. Every prompt-completion pair in `D⁺` / `D⁻` is bucketed by this label.
- **Semantics** supervision: we *named* the concept "recursion" before mining started.

That's both kinds of supervision — i.e., our setup is a **§4.1.1 static concept**, not a §4.1.2 dynamic one. Calling our pipeline "Schut §4.1.1" is accurate; calling it "Schut §4.1.2" would not be. We pay for full supervision and get interpretability and a clean steering target as a result.

### 4.2. What an unsupervised (§4.1.2-style) analog would look like for us

To shed the *semantics* supervision while keeping the experiment tractable:

- **Membership proxy** (cheap structural grouping, replacing ECO/STS/disagreement):
  - "all C++ tasks whose docstring says 'from scratch without using built-in functions'"
  - "all tasks with `std::vector<int>` input returning a scalar"
  - "all tasks where the greedy completion has > N tokens before the first `return`"
  - "all tasks where two different LM checkpoints' greedy completions diverge" *(the most Schut-faithful: model-internal disagreement, zero hand-labelling)*
- **Within-prompt value proxy** (replacing AZ's MCTS value): the LM's own log-probability over candidate continuations. `X⁺_i` = greedy continuation of prompt `i`; `X⁻_i` = a lower-log-prob branch from the same prompt (sampled, or from an MCTS tree on the token-tree we already have infrastructure for).
- **Mining.** Single `v` fit jointly across the membership group via dynamic-concept convex program (Eq. 5).
- **Filter.** Test that `v` causally influences generation on held-out prompts in the same membership group (steering causality, analogous to teachability).
- **Inspect.** Score the full task corpus by `v · z`, take the top-`k` prototypes, ask "what do they have in common?" — *only then* propose a semantic label.

We might rediscover the recursion direction. We might find something we hadn't pre-specified — "uses helper function direction", "early-return-on-base-case direction", "comprehension-vs-loop direction". The point of running it that way is to put the *discovery* claim back on the table rather than just confirming a chosen concept.

### 4.3. Specific things to revisit later **[TODO]**

- **[TODO] Pick a membership proxy and run a §4.1.2-faithful mining pass** on top of the Track-1 MCTS infrastructure. Don't pre-label by recursion. See whether `v` and its prototype set fall out as something nameable.
- **[TODO] Random-`X⁺` baseline.** Run the full pipeline with a random grouping of prompts as `X⁺` and see if anything survives the causality / teachability filter. This is the baseline Schut don't run; we should.
- **[TODO] Compare unsupervised-mined `v` against our supervised recursion `v`.** Cosine similarity, prototype-set overlap, steering-effect correlation. If they're highly aligned, that's strong evidence the concept is "real" and not an artifact of either pipeline.
- **[TODO] Membership proxies worth trying.**
  - Two checkpoints disagree (Qwen-Coder-32B vs. 14B greedy divergence at the docstring-end token).
  - High-entropy fork point in greedy generation (the LM itself flags "I don't know which way to go here").
  - Same task across two languages (Python vs C++) — already partly in nb 16 but used differently.
- **[TODO] Identifiability check.** Re-fit `v` with different `X⁻` choices / random seeds / different layers and see how stable the resulting prototype set is. Schut don't do this; we should know how much our mined direction depends on the contrast set we happened to pick.
- **[TODO] How does the §4.1.2 framing interact with Bratman's *partial plan* picture?** A dynamic concept fired across a rollout is the operational signature of a future-directed commitment that has not yet been elaborated into specific moves — the placeholder is still open. The §6.2.1 plasticity example is a chess-side instance of this. Worth integrating into [`bratman_1987_intention_plans_practical_reason.md`](bratman_1987_intention_plans_practical_reason.md) when we revise that note. See [[bratman-1987-intention-plans-practical-reason]].

---

## 5. One-line takeaway

Schut §4.1.2 doesn't escape supervision — it **moves supervision from "name the concept" to "group the positions by some proxy"**, and replaces "label the trajectory" with "let the model's own value rank it". That trade is what enables genuine concept *discovery*. Our recursion-direction setup currently sits on the §4.1.1 (fully supervised) side of that trade. The interesting next experiment is to take a step toward §4.1.2 and see what falls out when we don't pre-name what we're looking for.
