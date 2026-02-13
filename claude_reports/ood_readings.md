# Readings: Analyzing Goal-Directed Behavior in Systems

Curated list focused on **analyzing/detecting/understanding goal-directed behavior in systems**, spanning AI, biology, philosophy, cognitive science, and control theory.

---

## Philosophy & Foundations

**1. Dennett (1987), "The Intentional Stance"**
The foundational framework for attributing goals to systems. Dennett argues you can predict a system's behavior by treating it *as if* it has beliefs and goals — the "intentional stance." The key question he raises is yours exactly: is goal-attribution a real property of the system, or just a useful description? Your linear probes are essentially testing whether the intentional stance has a mechanistic correlate in the residual stream.

**2. Bratman (1987), "Intention, Plans, and Practical Reason"**
Distinguishes between *desire* (wanting an outcome) and *intention* (committing to a plan to achieve it). A system can "want" correct code without having committed to a specific strategy. Your prefill probing tests exactly this: does the model *commit* to a plan (intention) or just stumble toward correctness (desire)? The key diagnostic: intentions constrain future deliberation, which is what you see when the model adapts its generation to different prefills.

**3. Frankfurt (1978), "The Problem of Action"**
Asks: what distinguishes genuine *action* (goal-directed behavior) from mere *happening* (things that occur without an agent's guidance)? Frankfurt's criterion: an action is guided by the agent throughout its execution, with ongoing sensitivity to feedback. Your MCTS tree analysis is essentially Frankfurt's test — do the model's intermediate states show ongoing guidance toward the goal, or does it just happen to arrive at correct code?

**4. Woodward (2003), "Making Things Happen: A Theory of Causal Explanation"**
Interventionist account of causation. Relevant because your activation steering experiments are literally interventionist tests: you intervene on the "goal direction" and check if it *makes* the model produce correct code. Woodward's framework gives you the philosophical grounding for claiming your probes are *causal* rather than merely correlational.

---

## Cybernetics & Control Theory

**5. Wiener (1948), "Cybernetics: Or Control and Communication in the Animal and the Machine"**
The original formulation of goal-directed behavior in terms of negative feedback loops. A system is goal-directed if it: (a) has a reference state, (b) detects deviations, (c) acts to reduce them. Your contrastive probes can be read through this lens: the "goal direction" is the reference signal, and the model's hidden state trajectory should converge toward it across different prefills.

**6. Rosenblueth, Wiener & Bigelow (1943), "Behavior, Purpose, and Teleology"**
The paper that launched the cybernetic view of goal-directedness. They define purposeful behavior as behavior directed toward a goal with continuous feedback correction. Their taxonomy (predictive vs. non-predictive, feedback vs. non-feedback) gives you a framework for classifying what *kind* of goal-directedness your LLM exhibits. Critically: they argue you can identify goal-directedness from behavior alone, without knowing the mechanism — but your work goes further by looking *inside* the mechanism.

**7. Powers (1973), "Behavior: The Control of Perception"** (Perceptual Control Theory)
Argues that organisms don't control their *behavior* — they control their *perceptions* (internal representations of the world). The organism acts to keep its perceptual signal matching a reference signal. This maps remarkably well onto your setup: the LLM's hidden state (perception) is steered to match the "goal direction" (reference), and the generated tokens (behavior) are just the means to maintain that match.

---

## Biology & Neuroscience

**8. Cisek (2007), "Cortical Mechanisms of Action Selection: The Affordance Competition Hypothesis"**
Proposes that the brain simultaneously prepares multiple actions and selects among them through competition. This is strikingly similar to your MCTS tree: multiple possible continuations compete, and the model's hidden state encodes the "winning" plan. Cisek's key finding — that neural populations encode multiple potential actions before committing — maps to your question of whether the model represents the goal before committing to specific tokens.

**9. Friston (2010), "The Free-Energy Principle: A Unified Brain Theory?"**
Argues all biological behavior is goal-directed in the sense of minimizing surprise (free energy). Organisms have a generative model of the world and act to confirm their predictions. The connection: your LLM has learned a generative model of code, and the "goal direction" you extract might be the model's internal prediction of what the completed code should look like — the reference state it's trying to make true through generation.

**10. Gallistel (1980), "The Organization of Action: A New Synthesis"**
Classic book on how animals organize hierarchical, goal-directed behavior. Key concept: **hierarchical goal decomposition** — animals break complex goals into sub-goals with a specific temporal organization. Relevant question for your work: does the model's goal direction decompose into sub-goal directions at different layers? (Early layers: "return a boolean"; middle layers: "use modulo"; late layers: "write `% 2 != 0`".)

**11. Shadlen & Shohamy (2016), "Decision Making and Sequential Sampling from Memory"**
Shows that neural decision-making involves sequential accumulation of evidence toward a threshold (drift-diffusion models). Analogous to your token-by-token generation: does the model accumulate "evidence" for the correct goal in its hidden state across tokens? Your depth-wise contrastive pairs could test whether the goal signal strengthens monotonically along the generation path (evidence accumulation) or appears suddenly (abrupt commitment).

---

## Cognitive Science & Psychology

**12. Miller, Galanter & Pribram (1960), "Plans and the Structure of Behavior"**
Introduced the TOTE (Test-Operate-Test-Exit) model of goal-directed behavior: an agent tests if the goal is achieved, operates if not, tests again, exits when done. This is the cognitive science version of the cybernetic feedback loop. Your different prefills create different "operate" conditions — the model must adapt its operations while the goal (test condition) remains fixed.

**13. Tolman (1948), "Cognitive Maps in Rats and Men"**
Classic paper showing rats build internal representations of mazes (cognitive maps) rather than just learning stimulus-response chains. The decisive experiment: rats that learned a maze took novel shortcuts when the usual path was blocked — proving they had a *goal representation*, not just a behavioral habit. Your prefill probing is Tolman's experiment for LLMs: blocking the "usual path" (with adversarial prefills) and testing if the model finds an alternative route to the same goal.

**14. Newell & Simon (1972), "Human Problem Solving"**
Defined problem solving as search through a "problem space" guided by goals and sub-goals. Their General Problem Solver (GPS) used means-ends analysis: detect the difference between current state and goal state, then apply an operator to reduce that difference. Your "goal direction" $v_l$ is literally the means-ends difference vector: it points from the current state toward the goal in representation space.

**15. Tomasello et al. (2005), "Understanding and Sharing Intentions: The Origins of Cultural Cognition"**
Studies how human infants learn to attribute goals to other agents. Key diagnostic: infants distinguish between *successful* and *failed* goal pursuit (they find failed attempts surprising if they've inferred the goal). Your external evaluation (unit tests) is doing the same thing — you're the "infant" judging whether the LM's generation successfully pursued its inferred goal.

---

## AI & Machine Learning (Goal-Directedness Specifically)

**16. Kenton, Kumar, Farquhar & Richens (2023), "Discovering Agents"**
Directly tackles: how do you determine if a system is an agent (goal-directed) vs. just a process? They propose formal criteria based on counterfactual interventions — a system is an agent if intervening on its "goals" changes its behavior in the expected direction. This is *exactly* what your steering experiments test: you intervene on the goal direction and check if behavior shifts toward correctness.

**17. Orseau, McGill, & Legg (2018), "Agents and Devices: A Relative Definition of Agency"**
Proposes that agency (goal-directedness) is relative to a model of the environment. A thermostat is goal-directed with respect to temperature but not with respect to anything else. For your work: the model might be goal-directed with respect to code correctness but not with respect to code style — and your probes should be specific to what you're measuring.

**18. Turner, Smith, Shah & Tadepalli (2021), "Optimal Policies Tend to Seek Power"**
Proves that optimal agents in MDPs tend to acquire resources and keep options open — a formal consequence of goal-directedness. Relevant background: goal-directed systems exhibit characteristic behavioral signatures beyond just achieving the goal. You could look for these signatures in your MCTS trees (e.g., do models prefer paths that keep more options open early on?).

**19. Ngo, Chan & Mindermann (2024), "The Alignment Problem from a Deep Learning Perspective"**
Section 2 specifically discusses conditions under which neural networks develop goal-directed behavior. They argue that mesa-optimization (internal goal pursuit) becomes more likely as tasks require sequential, adaptive decisions — which is exactly what code generation requires.

**20. Cohen, Hutter & Osborne (2022), "Advanced Artificial Agents Intervene in the Provision of Reward"**
Analyzes goal-directed behavior in RL agents through the lens of reward hacking. While your setting is different (no explicit reward), the analytical framework — distinguishing between "pursuing the intended goal" and "pursuing a proxy" — applies. Your contrastive probes might be picking up on either genuine goal pursuit or superficial correlates.

---

## Developmental Biology / Basal Cognition

**21. Levin (2019), "The Computational Boundary of a 'Self': Developmental Bioelectricity Drives Multicellularity and Scale-Free Cognition"**
Michael Levin's work shows that **biological cells** exhibit goal-directed behavior at multiple scales — individual cells navigate chemical gradients, cell collectives regenerate complex morphologies after damage. His key argument: goal-directedness is not unique to brains; it's a property of *any* system that maintains a setpoint against perturbation. Your prefill perturbation experiments have the same structure as his regeneration experiments: perturb the system and see if it returns to the same target state.

**22. Levin (2022), "Technological Approach to Mind Everywhere: An Experimentally-Grounded Framework for Understanding Diverse Bodies and Minds"**
Extends the above into a general framework for detecting goal-directedness across substrates (cells, organisms, AI). Proposes operational tests: (1) the system pursues the same outcome via different paths when perturbed, (2) the system adjusts its behavior to novel obstacles. Both tests map directly onto your prefill probing methodology.

---

## Summary: The Core Diagnostics of Goal-Directedness

Across all these fields, the same **three diagnostics** keep appearing:

| Diagnostic | Your Experiment |
|---|---|
| **Equifinality**: Multiple paths converge on the same outcome | Different prefills → same correct code |
| **Perturbation robustness**: System corrects after disruption | Adversarial prefills → model adapts and still solves |
| **Internal reference signal**: System has a representation of the goal state *before* achieving it | Contrastive probes detect goal direction in hidden states *during* generation |

The first two are behavioral (observable from outputs). The third — which is what your probing work uniquely contributes — requires looking *inside* the system. This is what separates your work from most of the literature above: Tolman could only infer cognitive maps from behavior; you can actually measure the "map" in the residual stream.
