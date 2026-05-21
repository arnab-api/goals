# Defining "Goal" for Code-LMs: Attributes, Operationalizations, and Terminology

**Status**: living document; iterate
**Date**: 2026-05-20
**Companion to**:
- [`operationalizing_proactiveness.md`](operationalizing_proactiveness.md) — interdisciplinary lit review on initiative
- [`readings_causal_behavioral_goal_analysis.md`](readings_causal_behavioral_goal_analysis.md) — methods survey across fields
- [`code_goal_target_concepts.md`](code_goal_target_concepts.md) — concrete fork taxonomy (A1–C2)
- [`goal_completion_signal_brainstorm.md`](goal_completion_signal_brainstorm.md) — the `g`+`c` two-signal hypothesis
- [`readings_goals_of_systems.md`](readings_goals_of_systems.md) — broader reading list
- [`multi_predicate_setup_design.md`](multi_predicate_setup_design.md) — instrumental-goal probe design

---

## 0. Why this document exists

We have working definitions of "goal" scattered across the prior reports, but the operationalizations are uneven: proactiveness has a long lit review, persistence and plasticity are typically discussed in one sentence each. This doc tightens that asymmetry. For each of the three attributes we attempt:

1. A **mathematical operationalization at multiple granularities** — behavioral, representational, causal, information-theoretic — so that any single empirical setup can be slotted into a clear measurement family.
2. A list of **close-analogue works** in the literature, with what's directly portable to code-LM probing.
3. A **terminology audit** for whether "goal" is the right load-bearing term for this project, and which competing terms (intention, plan, mesa-objective, function vector, latent task, strategy, commitment) better fit each attribute.

The document is meant to be revised. Each operationalization is a candidate, not a commitment.

---

## 1. Terminology — is "goal" the right word?

### 1.1 The case against "goal"

"Goal" is doing too much work and carries philosophical baggage we don't need to defend:

- **Agency-implying.** "Goal" suggests a system that *intends*, which invites a hard skeptical reading: an LM is a function from tokens to a probability distribution; calling its preferences "goals" is a category error.
- **Post-decisional collapse.** As Issue 2 of the prior conversation made explicit, once the fork is crossed, "goal" and "task" are isomorphic. The interesting object is the *pre-decisional* state that *determines* the fork choice — calling that a "goal" elides what's actually load-bearing.
- **Failure of operational distinction from "task."** Dennett's intentional stance is observer-relative; "goal" applied to an LM is therefore observer-relative; and we cannot empirically distinguish "the LM has a goal" from "we predict the LM by ascribing one." Carlsmith (2022) and the "Goal-Directedness is in the Eye of the Beholder" paper (arXiv 2508.13247) both note this.

### 1.2 Candidate replacement terms

| Term | Source | Connotation | Fit for this project |
|---|---|---|---|
| **Intention** | Anscombe (1957); Bratman (1987) | Pre-decisional, future-directed, commitment-like | Strong fit for the pre-fork direction. Bratman's "future-directed intention" is exactly what we want: a state that binds later behavior. |
| **Plan** | Bratman (1987); Wu et al. (2024) | Multi-step structure, hierarchical | Strong fit for *which sub-goal will be pursued next* at the fork; less fit for "do recursion" which is one-token-deep. |
| **Mesa-objective** | Hubinger et al. (2019) | The model itself is an optimizer | Overcommits — we don't need to claim code-LMs are optimizers. Worth invoking only when persistence-under-training-pressure is the focus. |
| **Function vector** | Todd et al. (2024) | Sparse mid-layer heads encoding ICL task identity | Sharp empirical concept, but specifically about *task identity* (which function the model is mimicking), not strategy choice. |
| **Latent task** | ICL literature | What the model thinks it's been asked to do | Useful for explicit-instruction settings; less for underspecified prompts where multiple tasks are equally compatible. |
| **Strategy / strategy direction** | Generic | Algorithm-style choice | Cleanest, least-loaded term for what our fork concepts (A1–A6) measure. |
| **Commitment** | Bratman; Schurger et al. (2021) | A diffuse or punctate transition from "deciding" to "decided" | Aligns with the pre-fork probing setup. Maps well to the stochastic-accumulator reframe. |
| **Disposition / preference direction** | Soligo et al. (2025) | A linear direction in activation space that biases outputs | Most empirically humble; doesn't claim a mental state, just a measurable direction. |
| **Set-point / reference** | Cybernetics (Wiener 1948; Powers 1973) | Internal target state the system tries to track | Useful for cybernetic framings but assumes a closed control loop the LM doesn't obviously have. |
| **Intent block / sub-goal** | Internal (`goalscope.md`) | A contiguous code region with a single intended purpose | Already in our vocabulary; useful for the code-region-level analysis. |

### 1.3 Recommendation

**Use a tiered vocabulary keyed to the granularity of the claim:**

- **"Strategy"** — for our fork concepts (A1–C2). It's specific, neutral, and doesn't claim mental-state content.
- **"Pre-fork commitment" / "commitment direction"** — for the hidden-state representation we want to find. Foregrounds the *pre-decisional* nature and matches the Bratman / Schurger framings.
- **"Instrumental goal"** — only when we mean the abstract sub-task being pursued (e.g. "the model is about to check the timeout condition"), kept distinct from its implementation. This is what `multi_predicate_setup_design.md` already does.
- **"Goal" without qualification** — avoid in technical sections; use only in framing / motivation paragraphs where rhetorical force matters.

The three attributes (proactiveness, persistence, plasticity) become attributes of a **commitment direction** rather than a "goal." This terminological tightening sharpens the empirical claims without sacrificing any of the philosophical motivation.

A useful sanity check: if a sentence in the paper still parses when "goal" is replaced by "commitment direction," it's making a representation-level claim and we're on safe ground. If it doesn't, we're probably making a mental-state claim and should reconsider.

---

## 2. Proactiveness

Working definition from prior reports: a *proactive* direction is one the agent commits to *before* the fork token in a way that is **not** reducible to the next-token distribution at the fork. Equivalently — the direction is **anticipatory** and **non-trivial-proxy**.

### 2.1 Behavioral level — what we observe in outputs

#### B-1. Compliance gap (Greenblatt et al. 2024 analog)

For a sample `x` with two strategies `S_A, S_B` and natural preference `S_A`:

$$
\mathrm{CG}(x) \;=\; P\bigl(S_A \mid x,\, \text{underspecified prompt}\bigr) \;-\; P\bigl(S_A \mid x,\, \text{prompt explicitly demands } S_B\bigr)
$$

A high CG means the model resists counter-instruction. Aggregated as $\mathbb{E}_x[\mathrm{CG}(x)]$, this is a single proactiveness scalar.

**Variants for sharpness:**
- Replace "explicitly demands $S_B$" with "subtly hints at $S_B$" to get a softer counter-pressure curve.
- Plot $\mathrm{CG}(x)$ as a function of counter-pressure intensity — a "dose-response curve" (Turner et al. 2023 idiom).

#### B-2. Self-initiated rate (Warneken-Tomasello analog)

For a maximally underspecified prompt (e.g. bare `def `), measure $P(S_A | \text{prompt})$. The deviation from $0.5$ (binary fork) or from uniform (multi-class fork) is the model's intrinsic bias.

A *proactive* direction should produce non-uniform self-initiated rates that are *not* tightly tracked by per-token training-distribution frequencies.

#### B-3. Behavioral consistency under paraphrase

For prompts $\{x_1, \dots, x_K\}$ that are semantically equivalent paraphrases of the same fork-presenting task:

$$
\mathrm{Cons} \;=\; 1 - H_{\text{empirical}}\bigl(\{S(x_i)\}_{i=1}^K\bigr) \,/\, \log(\text{n\_strategies})
$$

A stable strategy choice across paraphrases is evidence for a representation that abstracts over surface form — a precondition for "proactive" in the sense that matters.

### 2.2 Representational level — what we measure in hidden states

#### R-1. Pre-fork probe accuracy

Train a linear probe $w \in \mathbb{R}^d$ on hidden state $h_t$ at layer $\ell$ to predict the eventual strategy choice. Evaluate at *positions strictly before the fork token* $t < t_{\text{fork}}$:

$$
\mathrm{ProbeAcc}_{\ell, \delta} \;=\; \Pr\bigl(\hat S(h_{t_{\text{fork}} - \delta, \ell}) = S\bigr)
$$

The function $\delta \mapsto \mathrm{ProbeAcc}_{\ell, \delta}$ is the **anticipatory profile** (Braver DMC analog).

#### R-2. Anticipatory profile shape

Two qualitatively different shapes (cf. Braver 2012):

- **Proactive (sustained, anticipatory):** $\mathrm{ProbeAcc}$ rises early, stays high across many pre-fork tokens.
- **Reactive (transient, just-in-time):** $\mathrm{ProbeAcc}$ stays at chance until $\delta = 1$ or $0$, then spikes.

Operationalize as the **AUC of the anticipatory profile** across $\delta \in [1, T]$ for some window $T$, or the smallest $\delta^*$ such that $\mathrm{ProbeAcc}_{\ell, \delta^*} > \tau$.

#### R-3. Direction stability across layers

For each layer $\ell$, the contrastive direction $d_\ell = \mu_A^\ell - \mu_B^\ell$. The cosine similarity $\cos(d_\ell, d_{\ell+1})$ measures whether the direction is consistent across the network's depth. A direction that is *coherent across layers* and *anticipatory across tokens* is the strongest representational signature of a pre-decisional commitment.

#### R-4. Trivial-proxy ceiling (specific to *proactiveness*)

For every pre-fork probe, we must verify it is *not* reducible to the next-token distribution at the fork. Concretely:

$$
\Delta_{\text{non-trivial}} \;=\; \mathrm{ProbeAcc}_{\text{pre-fork}} \;-\; \mathrm{Acc}\bigl(\text{argmax LM}(h_{t_{\text{fork}}-\delta})\bigr)
$$

If $\Delta_{\text{non-trivial}} \approx 0$, the probe is essentially next-token prediction. If $\Delta_{\text{non-trivial}} > 0$, the hidden state carries strategy information not yet visible in the unembedding.

### 2.3 Causal level — interventions

#### C-1. Pre-fork patching

For two prompts that share a stem but diverge at the fork (one normally yielding $S_A$, the other $S_B$), patch the activation $h_{t_{\text{fork}} - \delta, \ell}$ from the $S_B$-run into the $S_A$-run. Measure:

$$
\mathrm{CausalEffect}(\delta, \ell) \;=\; P\bigl(S_B \mid \text{patched run}\bigr) - P\bigl(S_B \mid \text{unpatched } S_A\text{ run}\bigr)
$$

A proactive direction should yield non-zero causal effect for $\delta \geq 1$. Effect surviving to large $\delta$ ($\geq 5$ tokens before the fork) is stronger evidence than effect only at $\delta = 1$.

#### C-2. Direction-level steering

Build a steering vector $v = \mu_{S_A} - \mu_{S_B}$ at layer $\ell$. Add $\alpha v$ to the residual stream at positions $t < t_{\text{fork}}$ and measure $\Delta P(S_A)$. This is identical to standard CAA (Rimsky et al. 2024) but the key proactiveness condition is the position constraint $t < t_{\text{fork}}$.

#### C-3. Counterfactual policy adaptation (Kenton et al. 2022)

Embed the model in an in-context environment where the choice of $S_A$ vs $S_B$ has differential downstream consequences (e.g. via simulated user feedback in the context). Measure whether the model adapts its choice rate. If $P(S_A)$ shifts in response to the intervention on the downstream consequence, the choice is being mediated by a representation of the consequence — Kenton's causal-graph signature of an agent.

### 2.4 Information-theoretic level

#### I-1. Mutual information between pre-fork state and strategy

$$
\mathrm{MI}_\delta = I\bigl(S\,;\, h_{t_{\text{fork}}-\delta, \ell}\bigr)
$$

Computable in practice via the probe-accuracy lower bound (Tishby et al.) or by k-NN entropy estimators. A non-trivial floor for $\delta > 0$ is information-theoretic proactiveness.

#### I-2. Empowerment-style measure (Klyubin et al. 2005 adapted)

Mutual information between the fork *choice* and the subsequent $K$ tokens:

$$
\mathfrak{E}_K = I\bigl(S \,;\, \text{tok}_{t_{\text{fork}}+1 : t_{\text{fork}}+K}\bigr)
$$

This measures whether the fork is *consequential* — proactiveness only matters at consequential forks (otherwise we're measuring noise). High $\mathfrak{E}_K$ tells us the fork is the right place to look.

### 2.5 Sanity-check ladder for proactiveness

A direction $v$ at layer $\ell$ should pass increasingly stringent tests:

1. **Probe accuracy at fork**: $\mathrm{ProbeAcc}(t_{\text{fork}}) \gg $ chance. *(Necessary but trivially passable by competence probes.)*
2. **Probe accuracy pre-fork**: $\mathrm{ProbeAcc}(t_{\text{fork}} - \delta) > $ chance for $\delta \geq 1$. *(Rules out just-the-state probes.)*
3. **Non-triviality**: $\Delta_{\text{non-trivial}}(\delta) > 0$. *(Rules out being a token-level shortcut.)*
4. **Causal sufficiency**: pre-fork patching shifts the strategy choice. *(Rules out passive correlation.)*
5. **Counter-pressure persistence**: probe / causal effect holds under explicit counter-instruction in the prompt. *(Distinguishes goal from task.)*

A claim of proactiveness should report results at *all five levels*.

---

## 3. Persistence

Working definition: the commitment direction (a) **survives perturbation** and (b) **continues to fire across the entire implementation window**, not just at the fork.

The classic source is Sommerhoff (1950) / Braithwaite (1953) — "the system tends to return to a goal-directed trajectory after perturbation." Lee & McShea (2020) formalize this for biological systems.

### 3.1 Behavioral level

#### B-1. Mid-generation counter-pressure robustness

After the model has begun generating in strategy $S_A$, inject a token sequence that nudges toward $S_B$. Measure the probability that the eventual completion still implements $S_A$:

$$
\mathrm{Pers}_\text{behav}(t_\text{inj}) = P(S_A \mid \text{prompt} \oplus \text{tokens}_{[t_\text{fork}: t_\text{inj}]} \oplus \text{nudge toward } S_B)
$$

Persistence is the area under the curve $t_\text{inj} \mapsto \mathrm{Pers}_\text{behav}(t_\text{inj})$ across the implementation window.

This is the **detour task** of comparative cognition adapted: an adversarial mid-generation token is a "barrier"; persistence is the model's inhibitory control over the barrier-pulled trajectory.

#### B-2. Recovery rate after activation perturbation

Add Gaussian noise to $h_{t, \ell}$ at a single mid-generation position. Measure $P(S_A \mid \text{noised run})$ vs $P(S_A \mid \text{clean run})$. The smaller the gap, the higher the persistence. Plot vs noise magnitude for a stability curve.

#### B-3. Persistence-under-training-pressure (Hubinger sleeper-agent analog)

Fine-tune the model on examples that violate the strategy choice (e.g., always-iterative fine-tuning examples). Measure how many fine-tuning steps are required for the direction to be ablated. Slow ablation = high persistence-through-training.

This is the strongest persistence test but the most expensive. Use as a final-figure ambition, not an iteration tool.

### 3.2 Representational level

#### R-1. Direction sustainment across tokens

For each generated position $t \in [t_\text{fork}, t_\text{end}]$, project $h_{t, \ell}$ onto the strategy direction $v$:

$$
s_t = \frac{\langle h_{t, \ell},\, v \rangle}{\lVert v \rVert}
$$

The **persistence trajectory** is the sequence $\{s_t\}_t$. Quantify it by:

- **Decay rate**: fit $s_t = s_0 \cdot \exp(-\lambda t)$ to the trajectory; $\lambda$ is the decay constant.
- **Sustainment integral**: $\int_{t_\text{fork}}^{t_\text{end}} s_t\, dt$ normalized by window length.
- **Threshold residence**: fraction of $t \in [t_\text{fork}, t_\text{end}]$ for which $s_t > \tau$.

#### R-2. Re-anchoring under perturbation

Define $\tilde h_t = h_t + \epsilon$ (noise or counter-direction). Measure $s_{t+k}$ — does the activation drift back toward the strategy direction over subsequent tokens?

This is the cybernetic *negative-feedback* signature. A direction that re-anchors after perturbation is dynamically goal-directed in the strict Wiener/Ashby sense.

#### R-3. Probe accuracy across the implementation window

Train probe once on $h_{t_\text{fork}}$. Evaluate on $h_{t}$ for $t \in [t_\text{fork}+1, t_\text{end}]$. The **probe-transfer-across-tokens accuracy** is a direct readout of whether the same direction continues to encode the strategy.

### 3.3 Causal level

#### C-1. Patching persistence (one-shot intervention, multi-token effect)

Patch $h_{t_\text{fork}}$ with the $S_B$-corresponding activation. Generate without further patching. Measure how many subsequent tokens follow the $S_B$ pattern. Strong persistence = many.

This separates "the direction is steerable at one position" from "the direction is steerable into a sustained pattern." Both are interesting; the latter is closer to what "persistence" means in the Sommerhoff/Lee-McShea sense.

#### C-2. Late-stage restoration

If the model is mid-generation and the direction $v$ is *removed* from $h_t$ for some $t > t_\text{fork}$, does the model lose strategy coherence (switch to mid-implementation; emit syntactically broken output; revert to $S_B$)? Restoration tests the direction's *necessity* across the window.

### 3.4 Dynamical-systems level

If we model the trajectory of $h_t$ in the strategy-relevant subspace as a dynamical system $h_{t+1} = F(h_t, x_t)$ where $x_t$ is the new token, persistence is the **basin-of-attraction width** around the strategy-coherent trajectory.

Practical proxy: under bounded perturbation $\lVert \epsilon \rVert \leq r$, does the trajectory return to the unperturbed trajectory within $\Delta t$ tokens? The largest $r$ for which return happens within $\Delta t$ is the persistence basin.

This is harder to compute exactly but a sampling-based estimator works for small windows.

### 3.5 Persistence vs commitment depth — a useful decomposition

Persistence has two failure modes worth distinguishing:

- **Shallow commitment**: the direction is present at the fork but not maintained → low R-1 sustainment.
- **Inflexible commitment**: the direction is maintained but cannot accommodate legitimate sub-task transitions → conflicts with plasticity at the multi-predicate level.

The two failure modes call for opposite remedies. Tracking both metrics separately matters.

---

## 4. Plasticity

Working definition: the same commitment direction fires across **different surface realizations of the same instrumental goal** and **transfers across problem contexts**. This is what separates a goal direction from an implementation-detail direction.

The Lee & McShea (2020) formalization is the closest existing one. The Schut et al. (2025) AlphaZero work is the closest methodological precedent in deep learning.

### 4.1 Behavioral level

#### B-1. Cross-implementation invariance

For a single strategy $S_A$, let $\{I_1, I_2, \dots, I_J\}$ be different surface implementations (e.g., for "iterative": `for i in range`, `while`, `accumulator + list.append`, …).

Train a probe / extract a direction on the union of $\{I_j\}$. Evaluate on held-out implementations:

$$
\mathrm{Inv}_\text{behav} = \mathbb{E}_{j \notin \text{train}}\bigl[\,\Pr(\hat S = S_A \mid h \text{ from } I_j)\,\bigr]
$$

Above-chance held-out accuracy = the direction is implementation-invariant.

#### B-2. Cross-problem generalization

Train a direction on problem-set $\mathcal{P}_\text{train}$, evaluate on $\mathcal{P}_\text{test}$ where both sets present the same strategy fork (recursive vs iterative). Measure transfer accuracy and **direction alignment**:

$$
\cos(v_\text{train}, v_\text{test}) \quad\text{where}\quad v_\text{train}, v_\text{test} \text{ are independently mined on each set.}
$$

The cosine measures whether the strategy direction is *the same direction* across problems.

#### B-3. Cross-language generalization

The strongest plasticity test we have direct infrastructure for: train on Python, evaluate on C++. If the strategy direction is truly the abstract concept "use recursion," it should fire on `int factorial(int n) { return n * factorial(n-1); }` as well as on `def factorial(n): return n * factorial(n-1)`.

This is the goal-vs-competence dissociation in its sharpest form: a competence direction would not transfer (C++ generation requires different competencies); a strategy direction should.

### 4.2 Representational level

#### R-1. Direction alignment across contexts

Already operationalized above as $\cos(v_\text{context A}, v_\text{context B})$. A useful sanity check: the alignment should be high among same-strategy contexts and low among different-strategy contexts. A simple "alignment confusion matrix" across all pairs of contexts and strategies is a single figure that summarizes this.

#### R-2. Subspace stability

If the strategy is encoded in a $k$-dimensional subspace (rather than a single direction), measure the **principal angle** between the strategy-subspace in context A and in context B:

$$
\theta_{A,B} = \arccos\bigl(\sigma_1(\mathbf{V}_A^\top \mathbf{V}_B)\bigr)
$$

Small principal angles across contexts = a stable subspace = high plasticity.

#### R-3. Cluster geometry — anisotropy under context shift

The direction may be the same, but the *scale* of the strategy signal can drift across contexts. Track $\lVert \mu_{S_A}^{(c)} - \mu_{S_B}^{(c)} \rVert$ across contexts $c$. Plasticity ideally implies small variance in this quantity.

### 4.3 Causal level

#### C-1. Cross-context steering transfer

Train steering vector $v_\text{train}$ on $\mathcal{P}_\text{train}$. Use it to steer on $\mathcal{P}_\text{test}$. Compute the in-context steering effect $\Delta P_\text{test}(S_A)$.

If the cross-context effect is *anywhere close to* the within-context effect, the direction is causally plastic. We don't need parity — within-context will always win — but a non-trivial transfer effect is the bar.

#### C-2. Interchange interventions across contexts (Geiger et al. analog)

Swap activations between (problem A, strategy 1) and (problem B, strategy 1). If the swap preserves strategy choice at the destination, the swapped activation carries the strategy abstraction. If the swap destroys coherence, the strategy is entangled with problem identity.

This is the *cleanest* representational-plasticity test. Requires careful matching of structurally-aligned positions across A and B.

### 4.4 Composition / hierarchical level

Plasticity has a less-discussed second meaning: **the strategy direction composes with other directions**. If we steer for "recursive" AND for "use a helper function," do we get a recursive solution with a helper function? Compositional plasticity is the analog of Bratman's claim that intentions form a *planning structure*.

Operationalize as: for two strategy directions $v_1, v_2$ targeting different aspects, does steering with $\alpha v_1 + \beta v_2$ produce outputs satisfying both? If yes — the directions form a vector-space structure suitable for hierarchical goal-decomposition.

This is mostly aspirational; we don't have the multi-axis fork dataset to test it cleanly yet. But it's a useful long-term direction.

### 4.5 Plasticity as the *definitional* criterion

Among the three attributes, plasticity is the one that most cleanly *distinguishes a goal direction from an implementation/competence direction*. The argument:

- Proactiveness alone can be satisfied by a "this looks like a problem the model will solve correctly" competence signal that fires early.
- Persistence alone can be satisfied by any context-stable feature, including problem-identity features.
- Plasticity *forces* the direction to abstract over implementations, which is exactly what "instrumental goal" means.

For this reason: **a representation that satisfies plasticity gets the strongest claim to being a goal direction**, with the other two attributes as supporting evidence.

This is the prior conversation's "central distinction — goal direction vs competence direction" reformulated: plasticity is the *operational discriminator*.

---

## 5. Close analogues — papers we can borrow directly from

Ranked by directness of methodological transfer to our code-LM probing setup.

### 5.1 Schut et al. (2025), AlphaZero concept discovery — *primary methodological template*

[arXiv:2310.16410](https://arxiv.org/abs/2310.16410)

- **What's similar**: contrastive optimization over MCTS rollouts to extract concept vectors from a deep network's residual stream.
- **What's portable**: the entire Section 4.1.2 pipeline maps to our setup (see `schut_et_al_adaptation_analysis.md`). Their `Optimal vs Subpar Rollouts` contrast at varying depths is our `green vs red leaves` contrast at varying pivot prefixes.
- **What's different**: chess board states are self-contained; our hidden states encode the full prefix. They have Stockfish for free labels; we do not. Plasticity translates as their **cross-position generalization** of a concept direction — exactly what we want.
- **Direct borrow**: Use their convex-optimization formulation $\min_v \lVert v \rVert_1 \text{ s.t. } v \cdot z^+_t \geq v \cdot z^-_t$ as a baseline for direction extraction, alongside mean-diff and contrastive hinge.

### 5.2 Todd et al. (2024), Function Vectors — *closest representational analog*

[arXiv:2310.15213](https://arxiv.org/abs/2310.15213)

- **What's similar**: a sparse set of mid-layer attention heads carries a compact vector encoding *which task is being performed*. The vector is causally active in zero-shot transfer.
- **What's portable**: our Track 2 head-localization pipeline (`head_loc_attn.py`) is essentially the code-generation analog. Their "function vector lookup → steering at the appropriate position" is the same intervention pattern we use.
- **Plasticity borrow**: their tests of **cross-task generalization** (does an FV trained on antonyms work on country-capitals?) directly inform our cross-language and cross-problem generalization tests.
- **Caveat**: FVs encode *task identity* (which function is being mimicked); we want *strategy choice* within a single task. Sharper but analogous.

### 5.3 Nainani et al. (2025), Detecting Planning — *closest code-domain analog*

[arXiv:2508.18098](https://arxiv.org/abs/2508.18098)

- **What's similar**: applies planning probes to MBPP code generation with Gemma-2-2B. Defines planning as: model selects a future target token in advance, intermediate tokens are causally influenced by this pre-selection.
- **What's portable**: their semi-automated annotation pipeline could be adapted to label our generations as "planned" vs "improvised." Their causal suppression of future-token features is precisely our pre-fork steering.
- **Direct borrow**: their **planning vs improvisation distinction** maps to our **proactive vs reactive distinction** (Braver DMC). Use their probe-then-suppress methodology as a complementary intervention to mean-diff steering.

### 5.4 Dong et al. (2025), Emergent Response Planning — *closest pre-decisional analog*

[arXiv:2502.06258](https://arxiv.org/abs/2502.06258)

- **What's similar**: LMs encode response *behavioral attributes* (length, reasoning steps, content) in hidden states *before* generating any tokens.
- **What's portable**: their result that behavioral attributes stabilize *early* in layers while content stabilizes *late* directly informs our layer-selection: probe early/mid layers for the strategy attribute, late layers for token-level implementation.
- **Plasticity check**: they show probes generalize across prompt families. Replicate this for our strategy fork concepts.

### 5.5 Soligo et al. (2025), Emergent Misalignment LoRA — *closest direction-level precedent*

[arXiv:2506.11613](https://arxiv.org/abs/2506.11613)

- **What's similar**: a *single rank-1 LoRA direction* in activation space induces broad behavioral changes across diverse prompts. Demonstrates that linear directions can act as causal correlates of internally held tendencies.
- **What's portable**: their rank-1 LoRA training procedure is a candidate alternative to mean-diff direction extraction — directly optimize the rank-1 perturbation for behavior change. The resulting direction is automatically causally active.
- **Plasticity check**: the misalignment direction is broadly active (Soligo's main finding) — our analogous question is whether benign strategy directions admit similar breadth. Likely yes for plastic ones, no for fragile ones.

### 5.6 Wendlinger et al. (2026), SCS-Code — *closest code-subconcept analog*

[arXiv:2603.11212](https://arxiv.org/abs/2603.11212)

- **What's similar**: code-LMs have internal representations of *security sub-concepts* during generation, extractable by contrastive probing, and steerable to promote secure code.
- **What's portable**: their pipeline for sub-concept discovery in code-LMs is directly applicable. Their `secure vs insecure` contrast is our `strategy A vs strategy B` contrast.
- **Plasticity transfer**: they show sub-concept directions transfer across CWE categories — operationally close to cross-problem plasticity.

### 5.7 Greenblatt et al. (2024), Alignment Faking — *closest persistence-under-pressure analog*

[arXiv:2412.14093](https://arxiv.org/abs/2412.14093)

- **What's similar**: behavioral measurement of an internally held preference that persists despite explicit counter-pressure (the "compliance gap").
- **What's portable**: the *paired-conditions* methodology — same task, different contextual pressure — is directly portable to code-LM fork settings. The CG metric (§ 2.1, B-1) is borrowed from here.

### 5.8 Hubinger et al. (2024), Sleeper Agents — *closest persistence-through-training analog*

[arXiv:2401.05566](https://arxiv.org/abs/2401.05566)

- **What's similar**: an internal commitment that survives multiple rounds of fine-tuning (RLHF, SFT, adversarial).
- **What's portable**: the **persistence-through-training** measurement (§ 3.1, B-3). Expensive but the strongest single test.

### 5.9 Wu, Morris & Levine (2024), Plan Ahead? — *important methodological control*

[arXiv:2404.00859](https://arxiv.org/abs/2404.00859)

- **What's similar**: tests whether transformers compute features useful for *future* tokens at the *current* position. Distinguishes pre-caching from breadcrumbs.
- **What's portable**: their myopic-training control. A direction that is plastic in our sense should not survive myopic training. Useful as a falsification setup — if it does survive, the direction was not encoding forward-looking commitment.
- **Caveat**: requires training, so reserve for late-stage validation.

### 5.10 Honorable mentions

- **Geiger et al. (2021)** — interchange interventions; theoretical foundation for our patching tests.
- **Marks & Tegmark (2023)** — geometry of truth; example of mining and validating linear directions.
- **Li et al. (2024)** ITI — example of head-level localization + steering (matches our Track 2).
- **Meng & Bau (2022)** — ROME; causal tracing; methodological precursor for our patching experiments.

---

## 6. Synthesis — how the three attributes connect

The three attributes are **not independent measurements**; they interact in characteristic ways. Understanding the interactions sharpens our experimental design.

### 6.1 Three attributes, three dissociations they enable

| Attribute | Distinguishes goal direction from… |
|---|---|
| Proactiveness | A reactive token-level shortcut (just-in-time decoder of next-token) |
| Persistence | A transient pivot-only state representation |
| Plasticity | A competence/correctness signal (which is implementation-specific) |

Each attribute is needed; none is sufficient alone. A direction satisfying all three is a defensible "commitment direction" in the sense developed in § 1.3.

### 6.2 Trade-off geometry

Persistence and plasticity can pull in opposite directions:

- A maximally persistent direction resists *all* perturbations, including legitimate ones that should trigger a sub-goal transition.
- A maximally plastic direction generalizes across all contexts, which can dilute its identity (the "everything looks like a goal" failure).

The sweet spot is what Bratman calls a **partial plan**: a commitment that is robust within its scope but releases when the scope ends.

This suggests our metrics should be reported jointly. A single proactiveness/persistence/plasticity vector is more informative than each scalar.

### 6.3 Order of operations for an experimental pipeline

A practical sequencing of attribute checks for any new fork concept:

1. **Plasticity first.** Mine a direction on one set of problems and test cross-problem transfer. If it doesn't transfer, we have a problem-specific feature, not a goal direction. Stop.
2. **Proactiveness second.** Probe the (plastic) direction at pre-fork positions. If it's only present at the fork token, we have a state probe. Stop.
3. **Persistence third.** Test the (plastic + proactive) direction's robustness across the implementation window. If it dies immediately after the fork, we have a momentary representation, not a sustained commitment.

This ordering matters because plasticity is cheap to test (one probe-transfer experiment) and rules out the largest category of false positives.

---

## 7. Open questions

These are the unresolved bits worth tracking as the project evolves.

1. **Granularity of "instrumental goal."** Is "use recursion" a goal? Is "implement the base case" a sub-goal of it? Where do we stop? Bratman would say: as deep as the agent represents intentions. We have no principled answer; the practical answer is "as deep as our auto-labeler can score reliably."

2. **Is the commitment punctate or diffuse?** Schurger et al. (2021) argue commitment is the threshold-crossing of a stochastic accumulator, not a single decision moment. Should we probe for a *gradient* of strength across pre-fork tokens rather than a binary commitment point? This would refine our anticipatory-profile metric.

3. **Plasticity vs cross-language transfer — is C++ → Python too hard?** A direction that fires across both Python and C++ for "use recursion" is a remarkably abstract concept. Failure to transfer is not necessarily evidence against goal-directedness; it may be evidence against *that level of abstraction*. We need an intermediate test (cross-Python-style transfer) before drawing conclusions from cross-language tests.

4. **Are persistence and the `c` signal the same thing?** In `goal_completion_signal_brainstorm.md`, `c` tracks goal-completion progress. Persistence is the durability of the `g` direction. Mechanically these may be the same signal viewed two ways — `c` falling to zero is `g` losing persistence. Worth checking empirically.

5. **Self-initiated rate baseline.** What is the right null distribution for "the model would have chosen this strategy by chance"? Training-distribution token frequencies are one option but conflate "natural preference" with "task-distribution bias." Multi-paraphrase averaging is better but more expensive.

6. **Could we get away with calling these *biases* rather than *goals*?** "Strategy bias" is empirically the most accurate term for what we measure. The reluctance to use it is rhetorical — "bias" sounds less interesting than "goal." But the empirical phenomenon and the operationalizations don't change with the rename. Worth considering for the paper's framing.

7. **What does "the same direction" mean across layers?** If $v_\ell$ and $v_{\ell+1}$ have high cosine similarity, are they "the same direction"? Strictly no — the residual stream's basis can rotate freely under invertible transformations layer-to-layer. We need a more transformation-invariant notion of direction identity. Park, Choe & Veitch (2024) propose the *causal inner product* — worth experimenting with.

8. **Behavioral and representational metrics should correlate.** Across samples, the compliance gap (behavioral) and the direction magnitude (representational) should be correlated. If not, one or the other isn't measuring what we think. This is a sanity-check we have not yet run.

---

## 8. Concrete next steps that follow from this analysis

In order of expected information value:

1. **Implement the plasticity-first triage** (§ 6.3) as the standard protocol for any new concept added to `code_goal_target_concepts.md`. This will rapidly cull problem-specific false positives.

2. **Add the anticipatory profile** (§ 2.2, R-2) to our existing notebook outputs. Replace single-position probe-accuracy reports with the temporal profile across pre-fork tokens.

3. **Add the compliance-gap test** (§ 2.1, B-1) as a parallel behavioral measurement on the iterative-vs-recursive dataset we just built. Uses no new infrastructure — just two prompts per sample.

4. **Define the `c`-vs-persistence relation** (§ 7, item 4) experimentally. The empirical question is whether the projection $\langle h_t, v \rangle$ across the implementation window matches what `goal_completion_signal_brainstorm.md` predicts the `c` signal looks like.

5. **Run direction-alignment across contexts** (§ 4.2, R-1) as a first plasticity probe. Cheap, immediately informative.

6. **Cross-language transfer experiment** (§ 4.1, B-3) using the new merged JSON with Python + C++ + Java + Rust stubs. We now have the data infrastructure; the experiment is one cell.

The above six are all implementable in the current infrastructure without significant new code. They would substantially sharpen what we can claim about the concepts we're studying.

---

## Appendix A. Quick-reference glossary

- **Fork**: the token position at which the model commits to one of multiple legitimate strategies.
- **Commitment direction**: a linear direction in residual-stream activation space that, when projected onto, predicts the strategy choice.
- **Pre-fork** vs **post-fork**: token positions strictly before vs after the commitment.
- **Strategy**: a categorical algorithmic / structural choice (recursive vs iterative, LBYL vs EAFP, …).
- **Anticipatory profile**: the function $\delta \mapsto \mathrm{ProbeAcc}(t_{\text{fork}} - \delta)$.
- **Compliance gap**: behavioral preference under underspecified prompt minus behavioral preference under counter-instruction prompt.
- **Plasticity**: cross-context / cross-implementation / cross-language invariance of the commitment direction.
- **Persistence**: sustainment of the commitment direction across the implementation window after the fork.
- **Proactiveness**: presence and causal influence of the commitment direction at pre-fork positions, beyond what the fork-token distribution alone could explain.

## Appendix B. Notation key

- $h_{t, \ell}$ — residual-stream activation at token position $t$, layer $\ell$.
- $S$ — random variable for strategy choice ($S_A$, $S_B$, …).
- $v$, $d$ — a candidate direction (vector in $\mathbb{R}^d$).
- $t_\text{fork}$ — fork token position.
- $\delta$ — number of tokens before the fork.
- $\mathrm{ProbeAcc}_{\ell, \delta}$ — probe accuracy at layer $\ell$ and position $t_\text{fork} - \delta$.
- $\mu_S$ — mean activation conditional on strategy $S$.
- $\Delta_{\text{non-trivial}}$ — pre-fork probe accuracy minus next-token-distribution accuracy at the same position.
