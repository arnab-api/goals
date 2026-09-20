# Introduction reframing — plan, literature, and recommendations

*2026-09-20. Brainstorm + lit review for rewriting the Introduction around a functional, three-attribute
definition of goals (pre-commitment · plasticity · persistence). Aligns with the updated title/abstract
("precede and guide action, adaptive plasticity, persist"). Companion: `1_defining_goal.md` (the deep
attribute/operationalization doc this builds on), `abstract_intro_draft.md` (prior 4-property draft this
supersedes), `section5_2_rewrite.md`.*

**Decisions locked (with the user):**
- Persistence framing = **two strong + one partial**: lead with pre-commitment + plasticity
  (representational, strong); persistence third, partially established (temporal shown; robustness =
  behavioral now + encoding-level test as the air-tight upgrade).
- Opening = **open on the observer problem (Dennett), then resolve** into a functional definition +
  mechanistic correlate.

---

## 1. The intro arc (spine)

Three beats:

1. **"Do LLMs have goals?" is, by default, an observer question.** Ascribing goals is Dennett's
   *intentional stance* — a predictive posture, not a mechanistic claim. A 2025 NeurIPS position paper
   makes this explicit for AI: goal-directedness "cannot be measured objectively" and intentional
   attribution is pluralistic/context-dependent (Rajcic & Søgaard). This is the tension we open on, and
   the reason we must be careful not to anthropomorphize an LLM's words.
2. **So define "goal" functionally — by what the system does, not what it is made of.** Bratman's planning
   theory: an intention is a *conduct-controlling pro-attitude* that precedes and constrains action,
   resists reconsideration (is stable), and feeds means-end reasoning as part of a *partial plan*. This
   licenses defining a (sub-)goal by its functional role, then testing for it. We specialize to the *next
   instrumental sub-goal* during multi-step code generation — concrete and checkable.
3. **Exhibit the mechanistic correlate.** The contribution is the pivot from "eye of the beholder" to
   "here is the mechanism": the intentional stance one takes toward the model has an explicit mechanistic
   correlate — a small set of *goal heads* whose encoding satisfies the functional attributes. No claim
   about subjective experience.

Live gap to cite: existing "measuring goal-directedness" work is behavioral / agent-level (MacDermott et
al. 2024; the 2026 Behavioural-and-Representational eval); we do it **representationally, inside the
forward pass**. Greenblatt's call-to-action (high-level interpretability of how LLMs pursue goals is
promising-but-neglected) motivates the concreteness.

## 2. The three-attribute framework

Unifying backbone: goal-directedness in the dynamical-systems/cybernetics tradition = **equifinality +
persistence under perturbation** (Lee & McShea 2022; von Bertalanffy). Equifinality (same end state via
different paths) grounds *plasticity*; persistence-under-perturbation (negative feedback / basin of
attraction) grounds *persistence(ii)*. One classical concept covers two of our three attributes.

| Attribute | Grounding | Our evidence | Strength |
|---|---|---|---|
| **1. Pre-commitment** — the goal is an intermediate state that *precedes* action | Bratman (intention precedes/controls conduct); Dong 2025 (emergent response planning); Nainani 2025 (planning probes) | GoalScope reads the next sub-goal *before* it is written; OVReader shows it is committed by ~layer 40, before implementation begins | **Strong** |
| **2. Plasticity** — one goal, many realizations; adapts to environment | Equifinality; Bratman partial plans; Todd 2024 (function vectors) | causal/transfer metric: transplant the encoding → the *same* sub-goal realized in a *different token sequence*, even across languages (Py↔C++) | **Strong** — the goal-vs-competence discriminator |
| **3a. Persistence (temporal)** — encoding holds over a token span | Bratman (stability/inertia of intention) | same readout across 4 pre-implementation pivots; goal-encoding affinity across the implementation span (`REF/goal-enc-affinity.png`) | **Medium** (affinity ≈ done → appendix; readout-through-implementation TODO) |
| **3b. Persistence (robustness to obstacles)** — stays on the goal despite interference | Equifinality/negative feedback; comparative-cognition detour behavior | behavioral: adversarial-prefill robustness (Qwen2.5-Coder-32B 74%, `0_lm_performance.md`) | **Partial** — behavioral only; encoding-level test proposed (§4) |

**Where decodability and the causal bottleneck fit (they are not attributes):**
- *Decodability* (GoalScope / OVReader) is the **instrument** that makes pre-commitment observable — it is
  how we read the encoding, not a separate goal property.
- The *ablation causal bottleneck* is **evidence the encoding is load-bearing, not epiphenomenal** — it
  underwrites all three attributes (esp. pre-commitment: the read goal is actually used).

This is cleaner than the prior draft's flat 4-property list (committed / decodable / causal / portable),
which mixed *claims* (committed, portable) with *methods/evidence* (decodable, causal). New structure:
**three attributes = the claims; decodability + causality = the evidence types.**

## 3. Paper-result → attribute mapping (for the contributions list)

- **Goal heads** (causal-mediation localization) → the substrate for all three.
- **GoalScope readout** → pre-commitment (read before action) + temporal persistence (read across span).
- **OVReader** (committed by ~layer 40) → pre-commitment, sharpened (decided *before* the goal heads act).
- **Transplant / causal-shift, cross-language** → plasticity (equifinality).
- **Ablation necessity/sufficiency + later-half random-head control** → causal bottleneck (evidence the
  encoding is used; effects are goal-head-specific).
- **Affinity across span; adversarial-prefill robustness** → persistence (temporal; robustness).

## 4. Measuring persistence with the encoding (the air-tight upgrade)

Temporal persistence:
- **Readout persistence** — GoalScope/OVReader at every position from pivot through end of `g_k`; metric =
  fraction still naming `g_k` (threshold residence). Extends the 4-pivot result.
- **Affinity trajectory** *(partly done)* — `s_t = cos(hₜ / goal-head output, v_{g_k})` across the span;
  decay λ / sustainment integral. → **appendix now**, promotable.

Robustness to obstacles (pairs behavior + encoding + cause):
- **① Detour test (headline).** Inject an adversarial prefill/distractor (reuse `0_lm_performance`
  obstacles) that pulls toward the wrong op. Measure **(a)** behavioral correctness of the final
  implementation and **(b)** whether the goal-head readout *still names `g_k`* at/after the obstacle and
  affinity **re-anchors**. Decisive analysis: does encoding-persistence *separate* recover-vs-fail trials
  (does it mediate the behavioral robustness)? Converts the 74% adversarial number into a representational
  claim.
- **② Representational re-anchoring (pure encoding).** Add noise or a competing-goal vector to the
  goal-head output/residual at one position; measure whether `s_{t+k}` recovers toward `v_{g_k}` over
  subsequent tokens — the negative-feedback / basin-of-attraction signature (`1_defining_goal.md` §3.2-R2).
- **③ Causal restoration (necessity capstone).** After the obstacle, patch the clean goal-head
  query/encoding back and measure behavioral recovery vs not — tests whether the encoding *causally*
  underwrites robustness.

Recommendation: **① is the air-tight test** (behavioral + encoding in one design); **②** is a cheap
pure-representational supplement; affinity → appendix now; **③** as the causal capstone if wanted. All
reuse existing infra (goal heads, GoalScope/OVReader, adversarial-prefill set, affinity projection).

## 5. Proposed introduction skeleton

1. **Hook** — the human analogue (deciding *what to check first* in `validate_config(...)` before typing
   `timeout > 0`): the sub-goal precedes the code. Does an LLM do likewise?
2. **The observer problem** — ascribing goals is the intentional stance (Dennett); goal-directedness is
   "in the eye of the beholder" (Rajcic & Søgaard); LLMs mimic human words, so apparent ≠ actual. Need a
   test that pulls "goal" apart from "prediction."
3. **A functional definition** — Bratman: intention = pre-committed, conduct-controlling, stable,
   plan-structured. We define a *sub-goal* by three functional attributes it must exhibit —
   **pre-commitment, plasticity, persistence** — and ask whether a code LLM's internals satisfy them.
   (Note: resource-boundedness of a fixed-depth forward pass is *why* pre-committing is useful.)
4. **What we find** — goal heads (localization) → pre-commitment (GoalScope/OVReader read before action) →
   plasticity (transplant, cross-language: same goal, different tokens) → persistence (temporal shown;
   robustness behavioral + encoding test). Decodability = instrument; ablation = load-bearing evidence.
5. **Contributions** — functional definition + evidence; goal-head localization; GoalScope decoding;
   plasticity/transplant; causal-bottleneck + replication + non-code readouts.

## 6. Citation audit — do the three legs have proper backing?

**Verdict: yes.** Two of the three legs rest on the *textbook* definition of goal-directedness; the third
has strong recent empirical precedent. The one soft spot is temporal persistence (our own
operationalization, grounded in intention-stability).

### Backbone: persistence + plasticity ARE the canonical definition
In philosophy of biology, teleological/goal-directed behavior is classically defined by **exactly two
signatures — persistence and plasticity** (Sommerhoff 1950; Nagel 1977/79), operationalized with explicit
metrics by Lee & McShea (2020). Their definitions map directly onto our legs:
- **persistence** = "returning to a goal trajectory following deviations / maintaining it despite
  perturbation" → leg (iii), incl. the **obstacle** sub-leg verbatim;
- **plasticity** = "reaching a goal-directed end from a variety of starting points / paths" → leg (ii)
  (equifinality; von Bertalanffy).
So legs (ii)+(iii) are not an invented framing — they are the standard two-criterion definition.

### Leg-by-leg
| Leg | Strength | Backing |
|---|---|---|
| **(i) pre-commitment** | **solid** | Bratman 1987 (future-directed, conduct-controlling intention prior to action); empirical: Dong et al. 2025 (emergent response planning), Anthropic *Biology of an LLM* 2025 (causal planned-rhyme evidence), Look-Ahead Planning Mech-Interp 2024 (2406.16033), Wu-Morris-Levine 2024 (plan-ahead/pre-caching), Nainani 2025 (planning in code) |
| **(ii) plasticity** | **solid** | canonical plasticity signature (Sommerhoff/Nagel/Lee-McShea) + equifinality (von Bertalanffy); DL portability analog: Todd et al. 2024 (function vectors); our cross-language transplant is the instance |
| **(iii-obstacle) robustness** | **solid** | canonical persistence signature (Sommerhoff/Nagel/Lee-McShea); cybernetic negative feedback (Wiener 1948; Ashby 1956); comparative-cognition detour behavior (Köhler; Tolman). Our encoding-level obstacle test = novel empirical instantiation |
| **(iii-temporal) held across a span** | **thinnest** | Bratman's *stability/inertia of intention* (resistance to reconsideration). "Representation persists across token positions" is **our** operationalization — present as such, not as a cited result. This is the leg the persistence notebooks probe |

### Two cautions
- **"Pre-commitment" term collision.** In decision theory "precommitment" (Strotz 1955; Elster) means
  *binding one's future self* — a different sense. Lead with Bratman's "future-directed intention" + the
  planning-interp cluster; define "pre-commitment" as our label explicitly so it isn't read as Strotz.
- **Framing vs. legs.** Dennett + Rajcic & Søgaard motivate *why* a functional definition is needed (the
  observer problem); they are the intro's opening, not evidence for a leg. Keep the separation clean.

### Grouped references (reconcile keys with `.bib`)
**Observer problem / measuring goal-directedness (framing):**
- Dennett, D. (1987). *The Intentional Stance.* MIT Press.
- Rajcic & Søgaard (2025). *Goal-Directedness is in the Eye of the Beholder.* arXiv:2508.13247 (NeurIPS position).
- MacDermott et al. (2024). *Towards Measuring Goal-Directedness in AI Systems.* arXiv:2410.04683.
- *A Behavioural and Representational Evaluation of Goal-Directedness in LM Agents* (2026). arXiv:2602.08964.

**Functional definition of intention (leg i + temporal persistence):**
- Bratman, M. (1987). *Intention, Plans, and Practical Reason.* Harvard UP. (+ *Planning and the Stability of Intention*.)
- Anscombe (1957), *Intention* (secondary).

**Goal-directedness = persistence + plasticity (backbone; legs ii + iii):**
- Sommerhoff, G. (1950). *Analytical Biology.* — persistence & plasticity as hallmarks.
- Nagel, E. (1977/79). teleology / goal-directedness.
- Lee, J. & McShea, D. (2020). *Operationalizing Goal Directedness.* Phil. Theory Pract. Biol. 12(5). — metrics for persistence & plasticity.
- McShea/Lee (2022). *The meaning and origin of goal-directedness: a dynamical systems perspective.* Biol. J. Linn. Soc. 139(4):370.
- von Bertalanffy (general systems / equifinality). · Wiener (1948) *Cybernetics*; Ashby (1956) — negative feedback / persistence. · Köhler; Tolman — detour behavior (obstacle robustness).

**Mechanistic / representational precedents (pre-commitment leg):**
- Dong et al. (2025). *Emergent Response Planning.* arXiv:2502.06258.
- Lindsey et al. (2025). *On the Biology of a Large Language Model* (Anthropic) — causal planned-token evidence.
- *Look-Ahead Planning Mechanistic Interpretability* (2024). arXiv:2406.16033.
- Wu, Morris & Levine (2024). *Do LMs plan ahead?* (pre-caching vs breadcrumbs).
- Nainani et al. (2025). *Detecting Planning.* arXiv:2508.18098.
- Todd et al. (2024). *Function Vectors.* arXiv:2310.15213. (plasticity/portability)
- Schut et al. (2025). *AlphaZero concept discovery.* arXiv:2310.16410. (direction extraction)
- Greenblatt (2026 commentary). (call-to-action / motivation)

## 7. Open decisions / to confirm
- Run the **detour encoding test (①)** before submission, or ship "two strong + one partial" and mark ①
  as the robustness upgrade? (User leans toward running it to make the claim air-tight.)
- Keep "goal" (motivational) vs "sub-goal" (technical) split consistent (per `1_defining_goal.md` §1.3).
- Whether to name the equifinality backbone explicitly in the intro or reserve for a framing paragraph.
