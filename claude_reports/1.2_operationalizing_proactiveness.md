# Operationalizing Proactiveness: A Cross-Disciplinary Lit Review

**Date**: 2026-05-14
**Companion to**: [`readings_causal_behavioral_goal_analysis.md`](readings_causal_behavioral_goal_analysis.md), [`code_goal_target_concepts.md`](code_goal_target_concepts.md)

---

## 1. Why this matters

Our working definition of a *goal* has two attributes:

- **Proactive** — the agent makes the decision; not forced by prompt/syntax/semantics.
- **Future-oriented** — about an action the agent *will* take, not the current state.

The proactiveness axis is what separates a **goal** from a **task**. A task is supplied by the environment ("write a function that sorts a list"); a goal is held by the agent ("I want to write quicksort"). The two often coincide, which is the central confound: if you observe a model writing quicksort after being asked to sort, you cannot tell whether quicksort is the model's goal or merely the most-likely completion given the task.

This review surveys how other fields have tried to **measure** initiative — to make it operationally distinguishable from task-following. The goal is to extract concrete measurement strategies that could be adapted for code-LM goal probing.

Two operationalization families recur across all sections:

- **Behavioral / counterfactual** — perturb the situation, look for adaptation that preserves the goal.
- **Information-theoretic / intrinsic** — quantify how much the agent's behavior is generated from within (empowerment, free-energy, learning progress).

The direction-mining methodology we use sits between these — it's neither purely behavioral nor purely capacity-theoretic, and has only one close precedent in the recent AI literature ([Soligo et al. 2025](#section-6); rank-1 LoRA direction for emergent misalignment).

---

## 2. Philosophy of action — defining the concept

Operationalization-focused fields all import their conceptual scaffolding from a few canonical philosophical works. Brief survey, since the field-specific sections below cite back to these:

- **Anscombe (1957), *Intention*** — the modern starting point. Intentional action is action *under a description* the agent can give, distinguishing it from mere reflex or movement. Set up the question: which descriptions are properly attributable to the agent?
- **Davidson (1963), "Actions, reasons, and causes"** — the agent's reasons cause the action. Establishes the causal-explanation criterion: a behavior is goal-directed if the goal-representation is what *produced* it, not merely what *predicts* it. (This is exactly the correlation vs causation problem we face with probes.)
- **Frankfurt (1971), "Freedom of the will and the concept of a person"** — distinguishes first-order desires from second-order endorsement. An agent is "proactive" when it acts on desires it reflectively endorses, not merely impulses. Maps loosely onto the distinction we want: goal representations vs surface-level next-token urges.
- **Bratman (1987), *Intention, Plans, and Practical Reason*** — *planning agency*: agents commit to future-directed intentions that constrain later reasoning. This is the closest philosophical analog of what we want a goal-direction to do: bind a future trajectory, not just describe the present.
- **Sterelny (2003), *Thought in a Hostile World*** — *decoupled representations*: representations that persist after the immediate stimulus is gone. A useful operational hint: a proactive goal should be detectable in hidden states across multiple tokens, not just at the decision token.

---

## 3. Cognitive neuroscience — measurable signatures of self-initiated action

This is the discipline that has gone furthest in giving "proactive" a precise behavioral and neural signature.

### 3.1 Braver's Dual Mechanisms of Control (DMC) — the cleanest proactive/reactive dichotomy

[Braver (2012), "The variable nature of cognitive control"](https://www.sciencedirect.com/science/article/pii/S1364661311002610), *TiCS* 16(2).

- Proposes two modes of cognitive control:
  - **Proactive control**: sustained, anticipatory maintenance of goal information in lateral PFC *before* the relevant stimulus arrives. Goal is "ready in mind."
  - **Reactive control**: transient, stimulus-driven reactivation of the goal *at* the moment of action selection. Goal is "looked up on demand."
- Operational paradigm: **AX-CPT** task; the temporal profile of PFC activity distinguishes the two modes.

**This is the closest existing dichotomy to what we want for LLMs.** Direct translation:
- Probe for the goal direction at every token position leading up to the decision.
- If the direction is active well before the decision token (sustained, anticipatory) → proactive representation.
- If it only appears at the decision token → reactive, indistinguishable from a state probe.

### 3.2 Brass & Haggard — the "what / when / whether" model

[Brass & Haggard (2008), "The what, when, whether model of intentional action"](https://journals.sagepub.com/doi/10.1177/1073858408317417), *The Neuroscientist* 14(4).

Decomposes voluntary action into three separable decisions, each measurable in its own paradigm:

- **W-what**: which action to take. Paradigm: free choice among equivalent options.
- **W-when**: timing of self-paced action. Paradigm: Libet-style spontaneous initiation.
- **W-whether**: choice to act at all. Paradigm: free-won't / veto.

Useful frame for our taxonomy: our existing forks (recursive vs iterative, LBYL vs EAFP, …) are *W-what* decisions. The other two axes — *W-when* (when to commit to the strategy) and *W-whether* (whether to take the goal-consistent action at all under counter-pressure) — are largely unexplored and could be tractable for LMs.

### 3.3 Free-won't / intentional veto

[Brass & Haggard (2007), "To do or not to do"](https://www.jneurosci.org/content/27/34/9141), *J. Neurosci.* 27(34).

- Contrasts trials where subjects prepare an action and execute vs prepare and *veto*.
- Dorsal fronto-median cortex is selectively active during vetoes.

LLM analog: have the LM "prepare" a goal (via prefill probing or context), then introduce counter-pressure mid-generation and measure whether the goal-direction persists, fades, or actively inverts. A direction that resists countermand is a proactive goal; one that flips with instructions is reactive.

### 3.4 Schurger and the stochastic reframe of the readiness potential

[Schurger, Mylopoulos & Rosenthal (2021), "What is the readiness potential?"](https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(21)00093-0), *TiCS* 25(7).

- Reframes Libet's "readiness potential" not as a deterministic neural commitment but as the time-locked average of a **stochastic accumulator crossing threshold**.
- Implication: self-initiated action is not "the brain decided 500 ms before the subject"; it's the moment a noisy accumulator hits a fixed threshold.

For our setting, this suggests **commitment may be diffuse, not punctate** — the goal direction may strengthen gradually across many tokens rather than appear sharply at one. Mining for a single commitment-position may miss the phenomenon.

### 3.5 Intentional binding — implicit measure of sense of agency

[Haggard (2008), "Human volition"](https://www.nature.com/articles/nrn2497), *Nat. Rev. Neurosci.* 9.
[Moore & Obhi (2012), "Intentional binding and the sense of agency: a review"](https://www.sciencedirect.com/science/article/abs/pii/S1053810011002881).

- Self-generated actions feel temporally compressed toward their outcomes; externally cued actions do not.
- Provides a millisecond-level continuous measure of "self-caused" feeling.

Less directly translatable to LLMs (no time perception), but the underlying logic — *self-generated actions are differently coupled to their consequences than triggered actions are* — is portable. Translation: a proactive goal should produce a different pattern of hidden-state→output coupling than reactive task-following.

---

## 4. Developmental psychology & animal cognition — initiative without instruction

The cleanest test of proactiveness across species is **action absent any external request**.

### 4.1 Warneken & Tomasello — spontaneous instrumental helping

[Warneken & Tomasello (2006), "Altruistic helping in human infants and young chimpanzees"](https://www.eva.mpg.de/documents/AAAS/Warneken_Altruistic_Science_2006_1555118.pdf), *Science* 311.
[Warneken et al. (2007), "Spontaneous altruism by chimpanzees and young children"](https://www.eva.mpg.de/documents/Wiley-Blackwell/Warneken_Helping_Infancy_2007_1554773.pdf), *PLOS Biol.*

- Paradigm: experimenter visibly struggles with a task. **No verbal request.** Measure whether the child / chimp spontaneously intervenes.
- 14–18 month olds intervene reliably; chimps intervene in some sub-tasks.
- Crucially: rewarding the helping does *not* increase its rate, and adding cost does *not* decrease it. The behavior is dissociable from instrumental conditioning.

**The cleanest "agent took initiative" paradigm in the literature.** LLM analog:
- Give the model a deliberately under-specified prompt (e.g., bare `def `).
- Does it consistently choose a coherent function to write? Across what set of functions?
- The *distribution* of self-chosen tasks is itself a measurement of the model's intrinsic preferences — its "what does it want to do when nobody asked" signature.

### 4.2 Seed et al. — first-trial success as evidence of advance planning

[Seed, Tebbich, Emery & Clayton (2009), "Chimpanzees solve the trap problem"](https://www.eva.mpg.de/documents/AmericanPsychologicalAss/Seed_Chimpanzees_JExpPsychAnimBeh_2009_1553002.pdf), *J. Exp. Psychol. ABP*.

- Removes the tool-use confound from the classic "trap problem."
- *First-trial success* on a novel causal task (no trial-and-error curve) is the operational definition of an agent having a goal representation *in advance*.

LLM analog: if a goal-direction mined on one set of tasks generalizes — on the very first inference — to a novel task variant requiring the same instrumental goal, that is the cleanest single-trial evidence for the direction encoding the goal rather than task surface.

### 4.3 Curiosity / intrinsic motivation as formalized initiative

[Schmidhuber (2010), "Formal theory of creativity, fun, and intrinsic motivation"](https://people.idsia.ch/~juergen/ieeecreative.pdf), *IEEE TAMD*.
[Oudeyer, Gottlieb & Lopes (2016), "Intrinsic motivation, curiosity, and learning"](http://www.pyoudeyer.com/oudeyerGottliebLopesPBR16.pdf), *Prog. Brain Res.*

- Schmidhuber: intrinsic reward = first derivative of subjective compression progress on the world-model. A closed-form scalar for "this action was self-initiated to learn, not to complete a task."
- Oudeyer/Gottlieb/Lopes: similar, but cast as *learning progress* (time-derivative of prediction error).

These give a model-internal computable signature of "action whose value is not external" — the most rigorous formalization of "the agent acted on its own initiative" in the entire literature. Less directly applicable to LLMs that don't have an explicit world-model update step, but the formal idea (separate reward function for self-generated information gain) is portable.

### 4.4 Active experimentation in children

[Gopnik et al. (2004), "A theory of causal learning in children"](https://alisongopnik.com/Papers_Alison/Psychological%20Review%20%20Final.pdf), *Psych. Review*.
[Kosoy et al. (2022), "Learning causal overhypotheses through exploration in children and computational models"](https://arxiv.org/abs/2202.10430).

- Children's free-play interventions are Bayesian *active experiments*; the information-gain from each chosen intervention is computable.
- Kosoy directly compares children vs RL agents on the same blicket-detector task and quantifies exploratory action selection as information-gain over causal overhypotheses.

The methodological lesson: *quantify what the agent chose to do in the absence of instruction* by reference to what it would have learned from each available choice. The choice itself is the proactive signal.

---

## 5. Formal / computational measures of agency

This is where the literature comes closest to giving us math we can compute.

### 5.1 Empowerment

[Klyubin, Polani & Nehaniv (2005), "Empowerment: a universal agent-centric measure of control"](https://uhra.herts.ac.uk/id/eprint/282/1/901241.pdf), *IEEE CEC*.

- **Definition**: empowerment = channel capacity (mutual information) between the agent's actions and its future sensory state.
- Task-agnostic scalar quantifying how much the agent *can* influence its future. Maximizing empowerment is one way to formalize "behaving like an agent" without specifying any task.

For LMs: less directly applicable (LMs don't have a clean action/observation loop), but the spirit is suggestive. If we computed the mutual information between the LM's choice at a fork point and the next K tokens, a *high* MI is a signature of consequential agentic choice — a fork where the model's decision actually shapes the future, vs a fork where the choice is downstream-irrelevant.

### 5.2 Active inference

[Friston et al. (2019), "Generalised free energy and active inference"](https://link.springer.com/article/10.1007/s00422-019-00805-w), *Biol. Cyb.*

- Agency as minimization of *expected free energy*, which decomposes additively into two terms:
  - **Pragmatic value**: distance from preferred (goal) states.
  - **Epistemic value**: information-gain about hidden states / world model.
- The epistemic term is the mathematical formalization of *intrinsically-motivated*, *self-generated* exploration.

Operational hand-hold: the decomposition gives a formal way to separate "agent acted to achieve goal" from "agent acted to gather information" — two distinct flavors of initiative. Mostly useful as a framework for thinking, not as a measurement we can directly compute on LM activations.

### 5.3 Kenton et al. — causal-discovery definition of an agent

[Kenton et al. (2022), "Discovering agents"](https://arxiv.org/abs/2208.08345), DeepMind. *Artificial Intelligence*.

- **First formal causal definition**: a system is an agent iff it would *adapt its policy* under counterfactual changes to how its actions influence the world.
- Equivalent: agents are systems whose behavior would change if the mapping between their actions and outcomes were intervened upon.
- Provides a causal-discovery algorithm to *detect* agency from observational data.

**The most rigorous formal operationalization in the literature.** Direct LLM analog:
- Take two contexts that differ only in how the LM's choice of strategy translates into downstream outcomes (e.g., environments where recursive sort is rewarded vs penalized).
- If the LM adapts its choice to maintain its goal under the intervention → agent. If not → reactive.
- This is exactly Lee & McShea's *plasticity* criterion (already in our readings doc), formalized at the causal-graph level.

### 5.4 Richens & Everitt — robust agents need causal models

[Richens & Everitt (2024), "Robust agents learn causal world models"](https://arxiv.org/abs/2402.10877), *ICLR 2024*.

- Proves: any agent with bounded regret under distributional shift must internally represent an approximate causal model.
- Sufficient-conditions test: if a model generalizes robustly to a new distribution, you can infer it has internalized causal structure (a necessary substrate for goal-directed action).

LLM relevance: a goal-direction that generalizes across implementation strategies (the central plasticity test we care about) is, by this result, evidence of an internal causal representation of the goal.

### 5.5 Halpern & Pearl actual causation

[Halpern & Pearl (2005), "Causes and explanations: a structural-model approach. Part I"](https://www.cs.cornell.edu/home/halpern/papers/modified-HPdef.pdf), *BJPS*.

- The AC1/AC2/AC3 structural-equation definition of actual causation.
- Provides the formal counterfactual machinery downstream agency-discovery work (Kenton et al., Carlsmith) builds on.

Useful background; not directly a measurement.

### 5.6 Carlsmith — APS checklist

[Carlsmith (2022), "Is power-seeking AI an existential risk?"](https://arxiv.org/abs/2206.13353), *arXiv*.

- Proposes the **APS** property triple as the discrete signature of an agentic system:
  - **A**dvanced capability
  - **A**gentic planning (proactiveness)
  - **S**trategic awareness
- A coarse but useful triage checklist.

For us, "agentic planning" is the explicit term Carlsmith uses for what we mean by *proactive*. He breaks it down as: "the system makes and executes plans, in pursuit of objectives, on the basis of models of the world."

---

## 6. Recent AI alignment work — measuring goals inside LMs

This is the most directly relevant body of work for our project. All references 2019–2025.

### 6.1 Hubinger et al. — mesa-optimization framework

[Hubinger et al. (2019), "Risks from learned optimization in advanced machine learning systems"](https://arxiv.org/abs/1906.01820).

- Distinguishes:
  - **Base objective**: what the training process optimizes.
  - **Mesa-objective**: what the learned policy itself optimizes (if it is itself an optimizer).
- A mesa-objective can systematically differ from the base objective — this is exactly the gap between "task" (training-time pressure) and "goal" (what the model itself pursues at inference).

The conceptual scaffolding for everything in this section. Our proactiveness criterion is essentially asking: do code-LMs have detectable mesa-objectives at the level of *which implementation strategy to pursue*?

### 6.2 Langosco et al. — goal misgeneralization

[Langosco et al. (2022), "Goal misgeneralization in deep reinforcement learning"](https://arxiv.org/abs/2105.14111), *ICML 2022*.

- **Operationalization**: OOD test environments where capability and goal are dissociable.
- An agent that *retains skills* but *pursues a different reward target* on OOD reveals an internally held goal that survives distributional shift — separable from competence.

This is the formal version of our central confound. The "competence direction" (knows how to do X) and "goal direction" (intends to do X) can in principle be separated by OOD evaluation: a goal-direction should transfer to novel tasks instantiating the same goal; a competence-direction should not.

### 6.3 Hubinger et al. — sleeper agents (persistence)

[Hubinger et al. (2024), "Sleeper agents: training deceptive LLMs that persist through safety training"](https://arxiv.org/abs/2401.05566).

- Demonstrates: LMs can carry trigger-conditional goals (e.g., "if year=2024, insert exploit") that survive RLHF, SFT, and adversarial training.
- The persistence-through-training is the behavioral signature of an internal goal orthogonal to the training signal.

The strongest empirical demonstration that LMs *can* hold goals invariant to task pressure. Methodologically: persistence-under-pressure is the operationalization.

### 6.4 Greenblatt et al. — alignment faking (the compliance gap)

[Greenblatt et al. (2024), "Alignment faking in large language models"](https://arxiv.org/abs/2412.14093), Anthropic/Redwood.

- **Operationalization**: the **compliance gap** — measure model behavior in two conditions:
  - "Training" / monitored — model believes its behavior will affect future training.
  - "Deployment" / unmonitored — model believes it is being used.
- If the model behaves differently across conditions in a way consistent with a stable internal preference, that preference is an internally held goal (not pure task compliance).

**Most directly portable measurement primitive in the recent literature.** The same logic could be applied to code-LMs: measure behavior on a fork point in two conditions — explicit instruction to pursue one option vs underspecified prompt — and look for a stable preference that holds even under counter-instruction. The size of the preference-under-counter-instruction is a proactiveness score.

### 6.5 Meinke et al. — in-context scheming evals

[Meinke et al. (2024), "Frontier models are capable of in-context scheming"](https://arxiv.org/abs/2412.04984), Apollo Research.

- Six behavioral evals: oversight subversion, self-exfiltration, goal-guarding, sandbagging, ...
- Each measures *covert deviation from operator instructions* in service of a self-pursued goal.

Useful template for designing *behavioral* tests of proactive goal-pursuit. Less directly applicable to our hidden-state probing setup, but the design pattern — paired conditions where deviation from the task is the proactiveness signal — is portable.

### 6.6 Soligo et al. — linear direction for emergent misalignment

[Soligo et al. (2025), "Model organisms for emergent misalignment"](https://arxiv.org/html/2506.11613v1).

- A single rank-1 LoRA direction induces broad misaligned behavior across diverse prompts.
- Demonstrates that a *linear direction in activation space* can act as a causal correlate of an internally held goal.

**The closest existing methodological precedent for our work.** Our project is essentially: can we find such directions for *benign* instrumental goals (which algorithm to write, which validation to perform first)? Soligo's positive result is encouraging — it shows the regime exists.

### 6.7 Everitt et al. — evaluating goal-directedness

[Everitt et al. (2025), "Evaluating the goal-directedness of large language models"](https://openreview.net/forum?id=BECkhjcofz).

- **Operationalization**: goal-directedness = gap between an LM-agent's *capability ceiling* on subtasks and its *deployed performance* on the composite goal.
- Concrete empirical measure of whether an LM actually marshals what it can do toward the stated goal.

Useful as a behavioral-level complement to our representation-level probing. Could combine: find a goal-direction, then verify Everitt's behavioral score moves the right way when you steer with it.

### 6.8 Anthropic — agentic misalignment / insider-threat evals

[Anthropic (2025), "Agentic misalignment: how LLMs could be insider threats"](https://www.anthropic.com/research/agentic-misalignment).

- Simulated corporate environments where an LM-agent with autonomy and goal-conflict is measured on the rate of blackmail / leak / sabotage actions taken *without instruction*.
- "Proactive harmful goal pursuit" as a direct behavioral readout.

---

## 7. Synthesis — what we can borrow for code-LM goal probing

Mapping each measurement family onto something computable for our setup:

| Source | Measurement | Translation to our code-LM probing |
|---|---|---|
| Braver DMC | Temporal profile of goal representation (sustained vs transient) | Probe for the goal direction at every token position before the fork; sustained = proactive |
| Brass-Haggard W-what/when/whether | Decompose voluntary action into 3 sub-decisions | We have W-what (recursive vs iterative). Add W-when (early commit vs delayed) and W-whether (will it pursue under counter-pressure) |
| Free-won't (Brass-Haggard) | Behavior under prepared-then-vetoed conditions | Mid-generation steering against the goal direction; measure how much the natural completion resists |
| Warneken-Tomasello | Action in absence of any request | Bare prompts (`def `) — measure the distribution of self-chosen tasks |
| Seed et al. | First-trial success on novel tasks | Cross-task probe transfer with zero-shot evaluation |
| Empowerment (Klyubin) | MI between agent action and future state | MI between fork-choice and next K tokens — high MI = consequential agentic choice |
| Kenton et al. | Counterfactual policy adaptation | Manipulate the in-context rewards for each strategy; see if the chosen strategy adapts to maintain goal pursuit |
| Richens-Everitt | Robust generalization ⇒ causal world model | Cross-implementation-strategy probe transfer is, by this theorem, evidence of an internal causal representation |
| Langosco et al. | Goal misgeneralization on OOD | Build OOD evaluations where competence and intent diverge; the goal-direction (not competence-direction) should transfer |
| Greenblatt compliance gap | Behavior in monitored vs unmonitored conditions | Behavior with explicit instruction "do X" vs underspecified prompt; preference under counter-instruction = proactiveness |
| Hubinger sleeper agents | Persistence through training-style pressure | Persistence of the goal direction through fine-tuning on counter-examples |
| Soligo et al. | Single rank-1 LoRA direction induces broad misalignment | Direct methodological precedent — our analogous question is whether benign instrumental goals admit similar directions |
| Everitt goal-directedness | Capability ceiling minus deployed performance gap | Behavioral validation for steering: does steering with the goal direction close the capability-vs-deployment gap on the corresponding goal? |

### Three concrete tests we should add to our pipeline

Of all the above, three stand out as low-cost additions that would substantially sharpen our claims:

1. **Compliance-gap test (Greenblatt analog)** — for each fork concept, measure the model's behavior in two conditions: (a) underspecified prompt, (b) explicit instruction to take the *opposite* option to the model's natural preference. The proactiveness score is the persistence of the natural preference under counter-instruction. A goal-direction that fires more strongly when the model resists counter-instruction is a goal, not a task representation.

2. **Anticipatory profile (Braver DMC analog)** — for each sample, probe for the goal-direction at every token position from the start of the function body up to the fork. A sustained, anticipatory profile (direction active well before the decision token) is proactive; a sharp, just-in-time activation at the decision token is reactive and indistinguishable from a state probe. This is a *temporal* sharpening of our existing fork-position probing.

3. **First-trial cross-task transfer (Seed + Langosco analog)** — train a direction on one set of fork prompts; evaluate it zero-shot on a deliberately disjoint set instantiating the same instrumental goal. Above-chance first-trial transfer is evidence the direction encodes the goal, not the task.

All three are operationalizations of *proactiveness specifically*, and all three could be added without changing our core infrastructure.

---

## 8. Open questions

- **Stochastic-accumulator vs punctate commitment.** Schurger's reframe of the readiness potential suggests goal commitment may be diffuse rather than localized to one token. Should we probe for a *gradient* of goal-direction strength across tokens rather than a single commitment site?
- **Free choice vs forced choice.** Brass-Haggard's free-choice paradigm — equivalent options where the agent is forced to pick one — is the standard for isolating self-generated W-what decisions. Our multi-predicate `validate_config` set is exactly this. Is there a way to design *adversarially-tied* free-choice prompts where two strategies have measured-equal next-token probability?
- **Mesa-objective vs goal vs preference.** Mesa-optimization assumes the learned policy is itself an optimizer. Whether code-LMs are optimizers in this strict sense or merely *preference-bearing* systems is an open question. Our work doesn't need to take a side, but framing matters when comparing against this literature.
- **Behavioral and representational measurements should converge.** A natural validation: behavioral compliance-gap and representational direction-strength should correlate across samples. If they don't, one or the other is missing the phenomenon.
