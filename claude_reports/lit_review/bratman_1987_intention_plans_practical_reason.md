# Bratman (1987) — *Intention, Plans, and Practical Reason*

**Citation.**
Bratman, M. E. (1987). *Intention, Plans, and Practical Reason.* Harvard University Press, Cambridge, MA. (Reprinted: CSLI Publications, Stanford, 1999.)

**Status.** Summary for project use. Cited repeatedly in [`1_defining_goal.md`](../1_defining_goal.md) (§1.2, §4.4, §6.2). This note distils the big ideas and flags which ones load-bear for the "commitment direction" framing.

---

## 1. The thesis in one paragraph

Intentions are not reducible to belief–desire pairs (contra Davidson's "pro-attitude + belief" theory). They are a **distinct mental state** with their own functional role: they let resource-bounded agents act coherently across time. The book defends a *planning theory* of intention — intentions are the elements of partial future-directed plans that we form, hold onto, elaborate, and execute. The whole apparatus (filter, stability, means-end coherence, partial structure) is what bounded rationality requires of a temporally extended agent.

---

## 2. Future-directed intentions are the central object

Most prior philosophy treated intention as **intention-in-action** ("the intention with which I am now $\phi$-ing"). Bratman's move is to foreground **future-directed intention** — "I intend to do $\phi$ tomorrow." These have their own functional profile:

- They persist between formation and execution.
- They guide further deliberation between now and then.
- They are revisable, but not casually so.

The two kinds of intention (in-action vs future-directed) are continuous, but the future-directed case reveals structure the present-tense case hides — which is why it's where the action is for a theory of practical reason.

This is the Bratman concept that maps most directly onto the project's notion of a *pre-fork commitment direction*: a state held before the choice is acted on.

---

## 3. The three functional roles of intention

Bratman attributes three jobs to future-directed intentions:

1. **They pose problems for further reasoning.** Once I intend to go to the dentist, I now have a sub-problem: how to get there. Intentions set the agenda for means-end deliberation.
2. **They filter options.** Settled intentions screen off alternatives from reconsideration. Without this, an agent would deliberate endlessly. Filtering is what makes intention a *commitment*, not just a current preference.
3. **They coordinate action over time and across agents.** Holding an intention now lets future-me act on plans present-me made, and lets others coordinate with me. Coordination is impossible without stable cross-time commitments.

These three roles are what distinguishes intention from desire and from belief.

---

## 4. Plans are *partial* — the central technical innovation

Plans are not complete blueprints. They are **hierarchical, partial structures** with intentions at their nodes and "placeholders" for details to be filled in later.

- I can intend to fly to Boston next month without yet intending a particular flight.
- The placeholder ("some flight") is filled as the deadline approaches.
- The hierarchy nests: sub-plans for sub-intentions, recursively.

Plans are elaborated under the pressure of approaching action. This **incrementalism** is essential — it's what lets bounded agents commit early without committing to everything.

For our project: this is the philosophical underwriting of why "instrumental goal" and "implementation strategy" can come apart. An LM that has committed to *use recursion* but not yet committed to *which base case* is, in Bratman's terms, holding a partial plan with an open placeholder.

---

## 5. The norms of plans

Plans are evaluable. Bratman argues for several normative constraints:

- **Internal consistency.** The set of one's intentions should be jointly executable in some possible world (one cannot rationally intend both $\phi$ and $\neg\phi$).
- **Means-end coherence.** If I intend the end and believe a means is necessary, I am rationally pressured to intend the means (or to reconsider the end).
- **Strong consistency / agglomerativity (qualified).** Under appropriate conditions individual intentions agglomerate into a consistent joint plan. Bratman is careful: full agglomerativity faces counterexamples (lottery-style cases), so the constraint is suitably weakened.
- **Stability.** Intentions should resist reconsideration. The stronger version: the agent should have *habits of non-reconsideration* that aren't themselves continuously up for review.

The stability norm is what protects against the regress of always asking "should I still want this?" — a regress that would defeat the purpose of forming an intention in the first place.

---

## 6. Habits of reconsideration & bounded rationality

A resource-bounded agent cannot afford to deliberate at every step. So agents develop **policies of (non-)reconsideration** — rough heuristics that tell them when to leave a plan alone and when to revisit it. These policies are themselves rationally evaluable: an agent with bad reconsideration habits (always changing course, never adapting to news) is irrational at a meta-level.

This puts Bratman's theory squarely in the **bounded-rationality** tradition: planning is an *adaptation* to limited cognitive resources, not an approximation to ideal Bayesian decision-making. Without bounded resources, the planning theory would be unmotivated — an ideal agent could just maximize expected utility one act at a time.

This is also the bridge to AI. Bratman's framework was taken up by Rao & Georgeff (1991) and the broader BDI (Belief-Desire-Intention) agent-architecture tradition precisely because real AI agents face the same resource constraints.

---

## 7. Two-tier structure of practical reasoning

Bratman's picture has a two-level architecture:

- **Tier 1 — prior intentions** act as a *frame* for deliberation. They filter the option set, fix sub-goals, and set the agenda.
- **Tier 2 — deliberation within the frame** does means-end reasoning, fills in placeholders, and updates the plan locally.

This two-tier picture is what saves the agent from re-litigating everything at every moment. It's also what gives intentions their *distinctive role* — they aren't just inputs to Tier 2; they constitute Tier 1.

---

## 8. Intention is not reducible to desire + belief

Bratman directly attacks the dominant Davidsonian view. Key contrasts:

- **Desires** are gradable, can be incompatible, and do not commit. I can desire $\phi$ and $\neg\phi$ simultaneously.
- **Beliefs** can be partial / probabilistic; they aim at truth.
- **Intentions** are committal (you can't rationally intend both $\phi$ and $\neg\phi$), partial-but-extendable, and aim at *making true*.

The functional roles in §3 are what make this irreducibility plausible: nothing in the belief-desire calculus by itself does the work of filtering and coordinating.

---

## 9. The Toxin Puzzle and the limits of the theory

Bratman engages Kavka's toxin puzzle — can you intend to drink a toxin tomorrow purely because you've been promised a reward today for *forming* the intention, even though you'll have no reason to drink it when tomorrow arrives? His treatment defends the planning theory by distinguishing the question "is this intention rationally formable?" from "is this intention rationally executable?" and shows the cases where these come apart don't break the theory.

This matters because it forces clarity on what counts as having an intention vs. merely planning-to-pretend-to-have-one. (Echoes in our project: a probe that fires on the *appearance* of a strategy commitment without the downstream behavioral commitment is the analog confound.)

---

## 10. Influence and downstream lineage

- **Cognitive science / philosophy of action.** The planning theory became the default account of intention. Search "Bratman + intention" — the citation network is enormous.
- **AI agents.** Rao & Georgeff's BDI architecture (1991), the PRS (Procedural Reasoning System), and most "deliberative agent" architectures of the 1990s explicitly trace to Bratman.
- **Shared intentionality.** Bratman's later work (*Shared Agency*, 2014) extends the framework to joint action — relevant for multi-agent systems but tangential here.
- **Modern echoes.** Recent "agentic" LLM work that talks about plans, sub-goals, and commitments inherits Bratman-shaped vocabulary, usually without citation.

---

## 11. Why this matters for *our* project

Three reasons the planning theory is the right philosophical anchor for the "commitment direction" framing:

### 11.1. Pre-decisional state is the load-bearing concept

The whole point of §3's filter-and-coordinate roles is that the intention exists **before** the action is taken and **shapes** what happens afterwards. This is exactly the structure of the pre-fork direction we're trying to mine: a state at $t < t_{\text{fork}}$ that constrains generation at $t \geq t_{\text{fork}}$. Calling it a "commitment direction" rather than a "goal" foregrounds the Bratman lineage cleanly.

### 11.2. Partial plans validate the goal/implementation split

The three-level hierarchy in [`CLAUDE.md`](../../CLAUDE.md) (broader goal → instrumental goal → implementation) is exactly Bratman's plan-elaboration picture: the instrumental goal is committed, the implementation is the placeholder filled in later. **Plasticity** in our sense (cross-implementation invariance of the direction) is the operational signature of Bratman's claim that placeholders can vary while the plan stays the same.

### 11.3. Stability ↔ persistence

Bratman's stability norm — intentions resist reconsideration absent significant new information — is the philosophical version of our *persistence* attribute. The trade-off geometry in [`1_defining_goal.md`](../1_defining_goal.md) §6.2 ("maximally persistent direction resists all perturbations, including legitimate ones") is essentially Bratman's caveat that stability is rational only relative to good *habits of reconsideration* — you want to resist noise but update on real news. This is what makes persistence and plasticity not contradictory but co-required.

### 11.4. Bounded rationality as the implicit premise

LMs are resource-bounded in a strong sense: a fixed-depth forward pass at each token. Bratman's argument that *planning is an adaptation to bounded resources* gives a principled reason to expect commitment-like structure to emerge in LMs at all — not because LMs are agents, but because the same computational pressures that make plans useful for humans make pre-decisional commitments useful for any system doing sequential generation under bounded compute.

The empirical question this leaves open: do LMs actually exhibit Bratman-style commitments, or do they re-deliberate token-by-token? Our proactiveness, persistence, plasticity ladder is one way to answer.

---

## 12. Where to go next

- **Bratman, M. E., Israel, D. J., & Pollack, M. E. (1988).** "Plans and Resource-Bounded Practical Reasoning." *Computational Intelligence*, 4(3), 349–355. — the canonical AI-side companion piece; lays out the BDI link explicitly.
- **Rao, A. S. & Georgeff, M. P. (1991).** "Modeling Rational Agents within a BDI-Architecture." *KR*. — the BDI formalization that grew out of Bratman + Pollack.
- **Bratman, M. E. (2014).** *Shared Agency: A Planning Theory of Acting Together.* OUP. — extends the framework to joint intention; only relevant if we go multi-agent.
- **Anscombe, G. E. M. (1957).** *Intention.* — the philosophical backdrop Bratman responds to; useful for context but the planning theory subsumes most of what we'd want from Anscombe.

---

## Appendix — terminology cheat-sheet for cross-referencing the docs

| Bratman term | Our project's term | Where in our docs |
|---|---|---|
| Future-directed intention | Pre-fork commitment direction | `1_defining_goal.md` §1.2, §1.3 |
| Partial plan | Instrumental goal (placeholder for implementation) | `CLAUDE.md` hierarchy; `1_defining_goal.md` §4.5 |
| Filter on options | Proactiveness (constraint on later tokens) | `1_defining_goal.md` §2 |
| Stability / non-reconsideration | Persistence | `1_defining_goal.md` §3 |
| Plan elaboration with placeholders | Implementation plasticity | `1_defining_goal.md` §4 |
| Habits of reconsideration | (no current analog — possible future direction) | — |
| Means-end coherence | Hierarchical decomposition of broader → instrumental → implementation | `CLAUDE.md` |
