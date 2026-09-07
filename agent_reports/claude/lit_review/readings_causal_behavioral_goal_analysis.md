# Causal and Behavioral Analysis of Goal-Directed Behavior: A Cross-Disciplinary Methods Survey

**Purpose**: Methodological ideas for studying whether LLMs pursue consistent intermediate goals during code generation, using prefill probing, contrastive directions, and activation steering.

**Compiled**: 2026-02-13

---

## Table of Contents

1. [AI/ML: Causal Interventions on Goal-Directed Behavior in Neural Networks](#1-aiml-causal-interventions-on-goal-directed-behavior-in-neural-networks)
2. [Neuroscience: Lesion, Optogenetic, and Stimulation Studies](#2-neuroscience-lesion-optogenetic-and-stimulation-studies)
3. [Developmental Psychology: Goal Attribution in Infants](#3-developmental-psychology-goal-attribution-in-infants)
4. [Animal Behavior: Purposive Behaviorism and Detour Experiments](#4-animal-behavior-purposive-behaviorism-and-detour-experiments)
5. [Biology Beyond Brains: Bioelectric Goal-Directedness](#5-biology-beyond-brains-bioelectric-goal-directedness)
6. [Philosophy of Action: Operational Definitions of Goal-Directedness](#6-philosophy-of-action-operational-definitions-of-goal-directedness)
7. [Complex Systems / Cybernetics](#7-complex-systems--cybernetics)
8. [Synthesis: Borrowable Methods for LLM Goal Analysis](#8-synthesis-borrowable-methods-for-llm-goal-analysis)

---

## 1. AI/ML: Causal Interventions on Goal-Directed Behavior in Neural Networks

### 1.1 Causal Tracing and Activation Patching (Meng, Bau et al., NeurIPS 2022)

- **System studied**: GPT-2 and GPT-J autoregressive transformers processing factual statements (e.g., "The Eiffel Tower is in [Paris]").
- **Method**: *Causal tracing* -- three forward passes: (1) clean prompt with cached activations, (2) corrupted prompt (subject tokens noised), (3) corrupted prompt with individual hidden states restored from the clean run. If restoring a specific state recovers the correct output, that state is causally responsible.
- **Findings**: Factual knowledge retrieval is localized in MLP modules at early-to-mid layers at the subject token position. Attention heads at later positions propagate this information to the final prediction site. Led to ROME (Rank-One Model Editing) for targeted fact editing.
- **Borrowable ideas**:
  - **Corrupt-and-restore paradigm directly maps to prefill probing**. Your adversarial prefills are analogous to "corrupted" inputs. You could patch activations from aligned-prefill runs into adversarial-prefill runs at specific layers/positions to see which components carry the "goal" information that allows recovery despite the adversarial prefill.
  - **Layer-by-layer causal tracing** can identify *when* (at which layer) the model "commits" to the goal, complementing your existing hidden-state probing approach.

**Sources**: [Meng et al., Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262), [ROME project page](https://rome.baulab.info/)

### 1.2 Causal Abstraction and Interchange Interventions (Geiger et al., NeurIPS 2021; JMLR 2025)

- **System studied**: BERT models trained on natural language inference (MQNLI corpus).
- **Method**: *Interchange interventions* -- align variables in an interpretable high-level causal model (e.g., a tree-structured natural logic model) with neural representations in the low-level model. Then swap activations between inputs that differ on one high-level variable. If the network's output changes exactly as the high-level model predicts, the neural representation faithfully implements that high-level variable.
- **Findings**: BERT partially realizes the causal structure of a compositional natural logic model. Some neurons cleanly align with logical operations while others are more distributed.
- **Borrowable ideas**:
  - **Define a high-level causal model of goal pursuit**: e.g., a variable "intended goal" that causally determines "token choice given prefill." Then use interchange interventions to test whether specific neural representations implement this variable. Concretely: take two samples with different goals, swap the "goal representation" at a candidate layer, and check if the model now generates code consistent with the swapped goal.
  - **Graded faithfulness**: Geiger et al. formalize that causal abstractions can be approximate (not all-or-nothing), which is useful since goal representations in LLMs may be distributed or imperfect.

**Sources**: [Geiger et al., Causal Abstractions of Neural Networks (NeurIPS 2021)](https://arxiv.org/abs/2106.02997), [Geiger et al., Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability (JMLR 2025)](https://arxiv.org/abs/2301.04709), [SAIL Blog on Causal Abstraction](https://ai.stanford.edu/blog/causal-abstraction/)

### 1.3 Do Language Models Plan Ahead? (Wu, Morris & Levine, 2024)

- **System studied**: Pythia model family (70M to 2.8B parameters) on autoregressive language modeling.
- **Method**: *Myopic training* -- train language models without propagating gradients to past timesteps. If a myopically-trained model performs the same as a normally-trained model, then the normal model was not "pre-caching" future-relevant information. Compare hidden states from both training regimes to measure the extent of look-ahead.
- **Findings**: Two hypotheses tested: (a) *pre-caching* (model deliberately stores future-relevant features at current timestep) vs. (b) *breadcrumbs* (features useful now happen to also be useful later). On natural language, breadcrumbs dominate, but pre-caching increases with model scale (non-negligible at 2.8B parameters). On synthetic tasks, clear pre-caching exists.
- **Borrowable ideas**:
  - **Myopic vs. normal training as a diagnostic**: If you trained a code model myopically and it could no longer adapt to diverse prefills (i.e., lost goal-consistency), that would be evidence that goal-pursuit relies on pre-cached future representations rather than just immediate next-token features.
  - **Scale-dependent planning**: The finding that planning increases with scale motivates testing your prefill probing on models of different sizes.

**Sources**: [Wu et al., Do language models plan ahead for future tokens?](https://arxiv.org/abs/2404.00859), [LessWrong distillation](https://www.lesswrong.com/posts/KoCS6LqJjKhCqDqnX/distillation-of-do-language-models-plan-for-future-tokens)

### 1.4 Detecting and Characterizing Planning in Language Models (Nainani et al., 2025)

- **System studied**: Gemma-2-2B (base and instruction-tuned) on MBPP code generation and poem generation tasks.
- **Method**: Formalize *planning* as: the model selects a future target token in advance, and intermediate tokens are causally influenced by this pre-selected target. The authors build a semi-automated annotation pipeline with causal criteria -- they probe for representations of future tokens at earlier positions and test whether suppressing these representations changes what gets generated.
- **Findings**: In poetry generation (Claude 3.5 Haiku), the model plans ahead for rhyming words -- internal representations of "rabbit" and "habit" are active at the position right after the prompt, and suppressing "rabbit" features causes a different rhyming word to be selected. Applied to code generation on MBPP with Gemma-2-2B.
- **Borrowable ideas**:
  - **This is the closest methodological relative to your work.** Their code generation experiments on MBPP directly parallel your prefill probing on `incomplete_code.json`. Their causal suppression of future-token features is analogous to your activation steering.
  - **Semi-automated annotation pipeline**: Their pipeline for labeling whether a generation exhibits "planning" vs. "improvisation" could be adapted to classify your prefill probing results.
  - **Future-token probing at earlier positions**: Train probes to predict the final goal-relevant tokens (e.g., the function's return value or the key algorithmic operation) from hidden states at the prefill position.

**Sources**: [Nainani et al., Detecting and Characterizing Planning in Language Models](https://arxiv.org/abs/2508.18098)

### 1.5 Activation Steering and Representation Engineering (Turner et al., 2023; Rimsky et al., 2023; Li et al., 2024)

- **System studied**: LLaMA-2 (7B, 13B) and other open-weight LLMs.
- **Method**: *Contrastive Activation Addition (CAA)* -- compute "steering vectors" by averaging the difference in residual stream activations between pairs of positive and negative examples of a behavior (e.g., sycophantic vs. non-sycophantic responses). During inference, add this vector at all token positions with a positive or negative coefficient to amplify or suppress the behavior. *Representation Engineering* (RepE) generalizes this: (1) *Reading*: derive a linear direction capturing a concept, (2) *Steering*: add/subtract this direction during the forward pass. *Inference-Time Intervention (ITI)*: Li et al. identify attention heads with high probing accuracy for truthfulness, then shift activations in those heads during inference.
- **Findings**: Steering vectors reliably control behaviors like sycophancy, honesty, corrigibility, and hallucination. The key finding is that many human-interpretable behaviors are linearly encoded in activation space.
- **Borrowable ideas**:
  - **Contrastive goal vectors**: Compute a "goal direction" by contrasting activations from runs where the model successfully achieves the goal (across diverse prefills) vs. runs where it fails. Then steer with this vector to test if you can rescue failing generations or derail succeeding ones.
  - **Layer-specific intervention**: ITI's approach of identifying specific heads/layers that encode a concept (via probing accuracy) before intervening is directly applicable. First probe for goal-encoding across layers, then steer only at the layers where the goal is most linearly represented.
  - **Dose-response curves**: Varying the steering coefficient lets you measure how strongly the goal representation controls behavior, analogous to pharmacological dose-response studies.

**Sources**: [Turner et al., Activation Addition: Steering Language Models Without Optimization](https://arxiv.org/abs/2308.10248), [Rimsky et al., Steering Llama 2 via Contrastive Activation Addition](https://arxiv.org/abs/2312.06681), [Representation Engineering survey](https://arxiv.org/html/2502.17601v1)

### 1.6 Causal Scrubbing (Chan et al., Redwood Research, 2022)

- **System studied**: Small transformer circuits (e.g., induction heads).
- **Method**: Formalize a mechanistic interpretability hypothesis as a mapping between an interpretable computational graph and the model's computational graph. Then *resample* activations at all nodes that the hypothesis claims are irrelevant. If behavior is preserved after resampling, the hypothesis is validated; if behavior degrades, the "irrelevant" nodes actually matter and the hypothesis is incomplete.
- **Findings**: Provides a rigorous method for testing whether a proposed circuit/mechanism is a sufficient explanation of behavior. Stricter than activation patching alone because it simultaneously tests all "irrelevant" components.
- **Borrowable ideas**:
  - **Hypothesis-driven resampling**: Formulate a specific hypothesis like "the goal is encoded in a low-dimensional subspace at layer L, and everything else is goal-irrelevant." Then resample all non-goal activations and check if goal-consistent behavior is preserved. This is stronger evidence than probing alone.
  - **Distinguishing necessary vs. sufficient**: Causal scrubbing complements your probing by testing sufficiency (are the identified representations *enough* to drive goal-directed behavior?), not just correlation.

**Sources**: [Chan et al., Causal Scrubbing](https://www.alignmentforum.org/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing), [Redwood Research sequence](https://www.alignmentforum.org/s/h95ayYYwMebGEYN5y)

### 1.7 Evaluating Goal-Directedness of LLMs (Everitt, Garbacea et al., 2025)

- **System studied**: LLMs from DeepMind, OpenAI, and Anthropic, evaluated as agents.
- **Method**: Measure goal-directedness as the extent to which an LLM uses its capabilities toward a given goal. Evaluate on tasks requiring (a) information gathering, (b) cognitive effort, and (c) plan execution. Use subtasks to infer each model's relevant capabilities, then measure whether it deploys those capabilities toward the goal. Crucially, goal-directedness is separated from raw task performance.
- **Findings**: Goal-directedness is relatively consistent across tasks, differs from task performance, is only moderately sensitive to motivational prompts, and most models are not fully goal-directed.
- **Borrowable ideas**:
  - **Separating capability from goal-pursuit**: Your prefill probing inherently separates these -- the model may *know* how to write correct code (capability) but fail to *adapt* to a prefill (goal-pursuit). Everitt's framework validates this distinction.
  - **Subtask-based capability inference**: Before testing goal-directed adaptation, first verify the model can solve each subtask in isolation. This controls for cases where failure reflects lack of knowledge rather than lack of goal-pursuit.

**Sources**: [Everitt et al., Evaluating the Goal-Directedness of Large Language Models](https://arxiv.org/abs/2504.11844)

### 1.8 Best Practices in Activation Patching (Heimersheim et al., 2024; Zhang et al., 2024)

- **System studied**: Various transformer LMs.
- **Method**: Systematic comparison of activation patching variants. Key dimensions: (a) *denoising* (restore clean activations into corrupted run -- tests sufficiency) vs. *noising* (inject corrupted activations into clean run -- tests necessity); (b) evaluation metrics (logit difference, KL divergence, probability of correct answer); (c) corruption methods (Gaussian noise, token replacement, mean ablation); (d) *path patching* to isolate causal mediation along specific edges.
- **Findings**: Methodological choices (metric, corruption type) significantly affect results. Different metrics can lead to different conclusions about which components are important. Recommendations: use both noising and denoising; report multiple metrics; be explicit about what corruption method means in context.
- **Borrowable ideas**:
  - **Both necessity and sufficiency tests**: When patching goal-related activations, always test both directions: (a) Can restoring "goal" activations rescue a corrupted/adversarial-prefill run? (sufficiency) (b) Can destroying "goal" activations in a clean run break goal-directed behavior? (necessity).
  - **Path patching**: Trace the specific circuits through which goal information flows from the prefill position to the generation positions.
  - **Report multiple metrics**: For your goal-directed code generation, use both binary metrics (does the code pass unit tests?) and continuous metrics (probability assigned to goal-consistent tokens, logit difference between goal-consistent and goal-inconsistent continuations).

**Sources**: [Heimersheim, How to use and interpret activation patching](https://arxiv.org/abs/2404.15255), [Zhang et al., Towards Best Practices of Activation Patching](https://arxiv.org/abs/2309.16042)

---

## 2. Neuroscience: Lesion, Optogenetic, and Stimulation Studies

### 2.1 Outcome Devaluation and the Goal-Directed / Habitual Dissociation

- **System studied**: Rodent cortico-striatal circuits (dorsomedial striatum / DMS for goal-directed; dorsolateral striatum / DLS for habitual).
- **Method**: *Outcome devaluation* -- the gold standard test for goal-directedness in neuroscience. Protocol: (1) Train an animal to press a lever for food reward. (2) Devalue the reward (via satiation or taste-aversion pairing). (3) Test lever-pressing in extinction (no reward delivered). If the animal reduces lever-pressing after devaluation, behavior was goal-directed (sensitive to outcome value). If it persists, behavior is habitual. Combined with lesion studies: (a) DMS lesions abolish sensitivity to devaluation (loss of goal-directed control), (b) DLS lesions prevent habit formation, (c) infralimbic cortex lesions prevent habitual behavior from developing even with extensive training.
- **Findings**: Goal-directed and habitual behaviors are neurally dissociable. Different brain regions compute outcome value (BLA, insular cortex) vs. response-outcome associations (DMS) vs. stimulus-response associations (DLS). Lesioning insular cortex impairs sensitivity to devaluation but not to contingency degradation, showing that different aspects of goal-directedness have distinct neural substrates.
- **Borrowable ideas**:
  - **Outcome devaluation as a paradigm for LLMs**: This is conceptually close to your setup. "Devalue" a code completion goal by changing the unit test (the "reward") after the model has started generating, and check if it adapts. Or more precisely: the prefill *is* a kind of perturbation, and goal-directed behavior is defined as behavior that adapts to the perturbation while preserving sensitivity to the goal.
  - **Lesion + devaluation**: Ablate candidate "goal-encoding" layers/heads (the LLM equivalent of lesioning DMS) and test if the model loses its ability to adapt across prefills. This directly parallels the DMS-lesion finding.
  - **Dissociating mechanisms**: Just as neuroscience distinguishes goal-directed vs. habitual systems, you could test whether some model components implement "goal-directed" adaptation (flexible response to prefills) while others implement "habitual" completion (rigid continuation regardless of prefill).

**Sources**: [Fractionating goal-directed and habitual decision-making (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7176551/), [Outcome devaluation as a method for identifying goal-directed behaviors in rats (Nature Protocols)](https://www.nature.com/articles/s41596-024-01054-3)

### 2.2 Optogenetic Dissection of Prefrontal Cortex Circuits

- **System studied**: Infralimbic (IL) and prelimbic (PL) cortex in rodents.
- **Method**: *Optogenetics* -- express light-sensitive ion channels (e.g., channelrhodopsin, halorhodopsin) in specific neuron populations, then use fiber-optic light delivery to activate or silence those neurons with millisecond temporal precision during specific behavioral epochs.
- **Findings**: Brief photoinhibition of infralimbic pyramidal cells during habit formation blocks both formation and expression of habitual behavior -- animals continue goal-directed behavior when IL is silenced. The medial PFC D1 system regulates temporal control of goal-directed behavior. Optogenetic experiments have mapped how prefrontal-striatal circuits control the flexible switching between goal-directed and habitual strategies.
- **Borrowable ideas**:
  - **Temporally precise intervention**: Optogenetics intervenes at specific moments during behavior. The LLM analogue is intervening at specific token positions during generation. Your prefill probing already does this implicitly (the prefill changes the "initial conditions"). You could extend this by patching/steering at specific token positions *during* generation (not just at the start) to see when the model "decides" its goal.
  - **Cell-type specificity maps to component specificity**: Just as optogenetics targets specific neuron types, you can target specific attention heads, MLP layers, or even specific neurons/features identified by SAEs. The precision of the intervention determines what conclusions you can draw.
  - **Online vs. offline**: Optogenetic silencing *during* behavior (online) is analogous to activation steering during generation. This is distinct from pre-training manipulations (offline).

**Sources**: [Using Optogenetics to Study Habits (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3654045/), [Optogenetic dissection of medial prefrontal cortex circuitry](https://www.frontiersin.org/articles/10.3389/fnsys.2014.00230/full)

---

## 3. Developmental Psychology: Goal Attribution in Infants

### 3.1 Woodward's Habituation Paradigm for Goal Attribution (1998, 1999)

- **System studied**: 5- to 12-month-old human infants.
- **Method**: *Visual habituation / looking-time paradigm*. Protocol: (1) Habituate infants to an actor reaching for Object A (not Object B) in a fixed spatial arrangement. (2) Switch the positions of the two objects. (3) Show two test events: (a) "new-goal" -- actor reaches to old location but grasps Object B, (b) "new-path" -- actor takes a different path but grasps Object A. Measure looking time. Longer looking indicates a "surprising" event (violation of expectation).
- **Findings**: 9-month-olds (and to some degree 5-month-olds) looked longer at the "new-goal" event, indicating they encoded the actor's *goal* (the target object) rather than the *path* of the reach. Infants attributed goal-directedness to human hands but not to inanimate rods performing the same motion, unless the rod had been previously "introduced" as an agent.
- **Borrowable ideas**:
  - **The habituation paradigm IS your prefill probing, conceptually.** Woodward "habituates" the infant to a goal (reaching for Object A), then perturbs the setup (swaps positions) and checks if the infant tracks the goal vs. the path. You "habituate" the model to a code task (the prompt), then perturb the setup (different prefills) and check if the model tracks the goal (correct code) vs. the surface form (continuing the prefill literally).
  - **Varying the agent**: Woodward tested whether infants attribute goals differently to hands vs. rods. You could test whether goal-directed behavior differs between (a) instruction-tuned vs. base models, (b) code-specialized vs. general models, (c) models of different scales.
  - **"Surprise" as a measure**: Woodward uses looking time as a proxy for surprise. Your analogue is the model's probability distribution -- you could measure the KL divergence between the model's next-token distribution after an aligned prefill vs. after an adversarial prefill. High divergence = the model is "surprised" by the adversarial prefill, suggesting it had a prior goal.

**Sources**: [Woodward (1998), Infants selectively encode the goal object of an actor's reach](https://pubmed.ncbi.nlm.nih.gov/9871370/), [Woodward & Sommerville (2000)](https://journals.sagepub.com/doi/10.1111/1467-9280.00218), [Frontiers review of infant goal attribution](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2016.00111/full)

---

## 4. Animal Behavior: Purposive Behaviorism and Detour Experiments

### 4.1 Tolman's Latent Learning Experiments (1930, 1932, 1948)

- **System studied**: Rats navigating complex mazes.
- **Method**: Three groups of hungry rats in a maze: (1) always rewarded at end, (2) never rewarded, (3) no reward for 10 days, then reward introduced on day 11. Measured errors (wrong turns) per trial.
- **Findings**: Group 3 showed dramatic improvement immediately after reward introduction on day 11 -- by day 12, they matched or surpassed the always-rewarded group. This demonstrated *latent learning*: rats had formed "cognitive maps" of the maze even without reinforcement. The maps were only expressed in behavior once a goal was introduced.
- **Borrowable ideas**:
  - **Latent representation vs. behavioral expression**: The distinction between having a goal representation and expressing it behaviorally is central. Your code LLMs may have latent knowledge of the goal (the correct algorithm) stored in their representations but may not always express it through correct generation, especially under adversarial prefills. Probing hidden states (like your probe approach) is the LLM analogue of "looking at the cognitive map" rather than just observing the maze behavior.
  - **Introducing goals after learning**: Tolman introduced food *after* the rats had learned the maze layout. Analogously, you could study models that were pre-trained on code (learned the "maze") but not instruction-tuned (no explicit "goal" to solve tasks), then fine-tune on goal-directed tasks and see if goal representations emerge in the same locations as the pre-existing code representations.

**Sources**: [Psychology Fanatic on Tolman's experiments](https://psychologyfanatic.com/tolmans-rat-experiments/), [Simply Psychology on latent learning](https://www.simplypsychology.org/tolman.html), [Latent learning, cognitive maps, and curiosity (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8031467/)

### 4.2 Detour Experiments in Comparative Cognition

- **System studied**: Broad range of species -- apes, monkeys, dogs, birds (corvids, parrots), reptiles, fish, octopuses, spiders.
- **Method**: Present an animal with a visible goal (food) behind a transparent or opaque barrier. The direct path is blocked; the animal must take a detour (move away from the goal initially) to reach it. Variants include: cylinder task (reach around transparent cylinder), V-shaped barrier (walk around), swing-door (push through a non-obvious door). Measure success rate, latency, and whether the animal can inhibit the prepotent direct approach.
- **Findings**: Success varies widely across species and correlates with executive function / inhibitory control. The critical cognitive challenge is *inhibiting the direct approach* in favor of a goal-consistent detour. Performance improves with sensorimotor experience and with the opaqueness of the barrier (transparent barriers are harder because the goal is visible, tempting a direct approach).
- **Borrowable ideas**:
  - **Adversarial prefills as "transparent barriers"**: Your adversarial prefills (e.g., a misleading partial expression) are like transparent barriers -- the model can "see" a simple continuation (the direct approach) but must take a "detour" (a more complex continuation) to achieve the goal. Goal-directed models should inhibit the direct continuation.
  - **Graded difficulty**: Detour experiments systematically vary barrier type and difficulty. You could create a graduated difficulty scale for prefills: from "slightly misaligned" to "maximally adversarial," creating a psychometric curve of goal-directed behavior.
  - **Cross-species comparison as cross-model comparison**: The detour paradigm's strength is its applicability across species. Similarly, your prefill probing paradigm can be applied unchanged across model architectures and sizes, enabling fair comparison.

**Sources**: [The detour paradigm in animal cognition (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5756264/), [Kabadayi et al. (2018), Animal Cognition](https://link.springer.com/article/10.1007/s10071-017-1152-0)

---

## 5. Biology Beyond Brains: Bioelectric Goal-Directedness

### 5.1 Levin's Bioelectric Pattern Memory in Planaria

- **System studied**: Planarian flatworms (Dugesia japonica) -- organisms that regenerate any body part, including the head and brain.
- **Method**: *Bioelectric manipulation via gap junction disruption*. Protocol: (1) Amputate planarian, (2) During the first 3 hours of regeneration, briefly expose the fragment to octanol (a gap junction blocker) that disrupts the bioelectric network, (3) Allow regeneration to complete. The gap junction blocker partially fragments the long-range bioelectric circuit, creating smaller isopotential domains with depolarized voltage on both ends.
- **Findings**: Brief disruption of gap junctions caused planaria to regenerate with *two heads* instead of one. Remarkably, this altered "target morphology" is *permanent* -- in subsequent rounds of amputation and regeneration (without further treatment), the worms continue to regenerate as two-headed. The altered target is encoded not in the genome but in the bioelectric pattern stored across the gap-junction-coupled cell network. This is a form of "somatic pattern memory."
- **Borrowable ideas**:
  - **Transient perturbation with permanent effect**: Levin's brief octanol treatment permanently rewrites the organism's "goal state" (target morphology). The LLM analogue would be: can a brief activation steering intervention at early layers permanently alter what goal the model pursues for the rest of generation? Test whether steering at layer 0-5 is sufficient to change the goal of an entire 50-token generation.
  - **Goal state as a stored pattern**: Levin shows the goal (target morphology) is encoded as a distributed bioelectric pattern, not as a localized instruction. This supports the hypothesis that LLM goals might similarly be distributed across layers as a pattern in activation space, detectable via linear probes but not localized to a single component.
  - **Hierarchical goal encoding**: The bioelectric "prepattern" specifies the high-level goal (one head vs. two heads) while leaving details (cell differentiation, specific tissue patterns) to lower-level processes. Your probe might similarly find that early layers encode high-level goals (algorithm class) while later layers encode implementation details (specific variable names).
  - **Error minimization framework**: Levin's lab models morphogenesis as continuous error minimization between current morphology and target morphology. This maps directly to your hypothesis that LLMs maintain an internal "target" and minimize divergence from it during generation.

**Sources**: [Levin Lab, Planaria publications](https://drmichaellevin.org/publications/planaria.html), [Long-Term Stochastic Editing of Regenerative Anatomy via Targeting Endogenous Bioelectric Gradients (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5443973/), [Bistability of somatic pattern memories (Phil Trans B)](https://royalsocietypublishing.org/doi/10.1098/rstb.2019.0765), [Wyss Institute profile](https://wyss.harvard.edu/news/mike-levin-on-electrifying-insights-into-how-bodies-form/)

---

## 6. Philosophy of Action: Operational Definitions of Goal-Directedness

### 6.1 Persistence and Plasticity (Sommerhoff, 1950; Braithwaite, 1953; Lee & McShea, 2020)

- **System studied**: Abstract definition applicable to any system.
- **Method**: Two key behavioral criteria for goal-directedness:
  1. **Persistence**: The system tends to return to a goal-directed trajectory after perturbation. If you knock it off course, it corrects.
  2. **Plasticity**: The system can reach the goal from a variety of different starting states or through different means. It is not locked into a single stereotyped behavior.
- **Operationalization by Lee & McShea (2020)**: Proposed quantitative measures for persistence and plasticity that can be applied empirically to any system, from bacteria to robots.
- **Borrowable ideas**:
  - **Your prefill probing is literally a test of plasticity.** You provide different "starting states" (prefills) and check if the model reaches the same goal (correct code). Models that succeed across diverse prefills exhibit high plasticity.
  - **Persistence can be measured via intermediate perturbation**: Beyond just initial prefills, you could perturb the model *mid-generation* (via activation steering at intermediate token positions) and check if it recovers and still produces correct code. High recovery rate = high persistence.
  - **Quantitative goal-directedness score**: Lee & McShea's framework suggests computing a single scalar for how goal-directed a system is. You could define: `goal_directedness = (fraction of prefill types that produce correct code) * (average adaptation quality across prefills)`.

**Sources**: [Lee & McShea, Operationalizing Goal Directedness (2020)](https://quod.lib.umich.edu/p/ptpbio/16039257.0012.005/--operationalizing-goal-directedness-an-empirical-route?c=ptb;c=ptpbio;g=ptpbiog;rgn=main;view=fulltext;xc=1;q1=Article)

### 6.2 Dennett's Intentional Stance (1971, 1987)

- **System studied**: Any system whose behavior can be predicted.
- **Framework**: To adopt the *intentional stance* toward a system: (1) treat it as a rational agent, (2) ascribe beliefs based on its access to information, (3) ascribe desires/goals based on its design or situation, (4) predict it will act to further its goals in light of its beliefs. Goal-directedness is a *predictive strategy*, not necessarily a claim about internal mechanisms. A system is goal-directed insofar as the intentional stance yields reliable predictions.
- **Borrowable ideas**:
  - **The intentional stance as a framing device**: Your work can be positioned as testing whether the intentional stance is *causally grounded* for code LLMs -- i.e., whether the behavioral predictions it makes (the model will adapt to achieve its goal) correspond to actual internal representations (goal-encoding directions in activation space). This bridges behavioral and mechanistic analysis.
  - **Dennett's key criterion -- behavioral predictability**: If treating the LLM as having the goal "write correct code for this task" allows you to predict its behavior across diverse prefills, that is evidence for goal-directedness *at the behavioral level*, independent of any mechanistic claims. Your probing and steering then test whether this behavioral pattern has a mechanistic basis.

**Sources**: [Dennett, Intentional Systems (1971)](https://home.csulb.edu/~cwallis/382/readings/483/dennett.intentional.systems.1971.pdf), [Intentional stance (Wikipedia)](https://en.wikipedia.org/wiki/Intentional_stance), [LessWrong: Goal-directedness is behavioral, not structural](https://www.lesswrong.com/posts/9pxcekdNjE7oNwvcC/goal-directedness-is-behavioral-not-structural)

### 6.3 "Goal-Directedness Is in the Eye of the Beholder" (2025)

- **System studied**: Theoretical analysis of goal-directedness definitions applied to LLMs.
- **Key argument**: Goal-directedness is observer-relative -- whether a system appears goal-directed depends on the granularity and framing of the observer's analysis. This parallels Dennett's stance but with formal implications for AI safety.
- **Borrowable ideas**:
  - **Acknowledge observer-dependence explicitly**: Your choice of unit tests, prefill categories, and evaluation criteria determines what counts as "goal-directed." Being explicit about this strengthens the paper. Different evaluators might disagree on what constitutes "the same goal" across prefills.

**Sources**: [Goal-Directedness is in the Eye of the Beholder (arXiv 2025)](https://arxiv.org/abs/2508.13247)

---

## 7. Complex Systems / Cybernetics

### 7.1 Cybernetic Feedback and Goal Pursuit

- **System studied**: Abstract feedback control systems (thermostats, biological homeostasis, behavioral control loops).
- **Framework**: In the cybernetic tradition (Wiener 1948; Ashby 1956), a goal-directed system is one that: (1) has a reference state (goal/set-point), (2) has a sensor that measures the current state, (3) computes an error signal (difference between current and reference), (4) has an effector that acts to reduce the error. The system is goal-directed because it persistently acts to minimize the discrepancy between current state and goal state.
- **Key insight**: A cybernetic view forces the realization that *all* components -- sensing, error computation, and action -- are necessary for successful goal pursuit. A system that only has a reference state but cannot sense its current state is not goal-directed; a system that can sense but not act is not goal-directed.
- **Borrowable ideas**:
  - **Error minimization framework for LLM generation**: Model the code generation process as a control loop: the "goal" is encoded somewhere in the hidden states, the model "senses" what it has generated so far (via attention to previous tokens), computes an "error" (discrepancy between intended and actual code so far), and selects the next token to reduce this error. Your probes could test whether such error signals exist in the hidden states.
  - **Testing the control loop**: Cybernetics predicts that disrupting any component should impair goal pursuit. Test: (a) Ablate the "goal representation" (the reference state) -- does the model lose direction? (b) Corrupt the attention to previously generated tokens (the "sensor") -- does the model fail to adapt? (c) Ablate the MLP layers that compute the next token (the "effector") -- does goal-relevant generation degrade?
  - **Negative feedback signature**: Goal-directed systems show a characteristic signature: when perturbed away from the goal, they generate corrective actions that push back toward the goal. Measure whether adversarial prefills elicit "corrective" token sequences that undo the adversarial direction and return to goal-consistent code.

**Sources**: [Cybernetics: Circular Causal and Feedback Mechanisms (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC1525972/), [Control Theory, Goal Attainment, and Psychopathology (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6707771/)

---

## 8. Synthesis: Borrowable Methods for LLM Goal Analysis

### Summary Table of Methods and Their LLM Analogues

| Domain | Method | LLM Analogue | What It Tests |
|--------|--------|--------------|---------------|
| Neuroscience | Outcome devaluation | Change unit tests after generation begins | Sensitivity to goal value |
| Neuroscience | Lesion + behavior | Ablate layers/heads + test prefill adaptation | Necessity of components for goal-pursuit |
| Neuroscience | Optogenetics (temporally precise) | Steer at specific token positions | When goal commitment occurs |
| Dev. Psychology | Woodward habituation | Prefill probing with object/path swap | Whether model tracks goal vs. surface form |
| Animal Behavior | Tolman latent learning | Probe hidden states of untrained model | Latent goal knowledge before fine-tuning |
| Animal Behavior | Detour experiments | Adversarial prefills requiring "detours" | Inhibitory control / goal-directed flexibility |
| Biology | Levin bioelectric manipulation | Brief early-layer steering | Whether transient goal-state changes propagate |
| Philosophy | Persistence + plasticity | Multi-prefill + mid-generation perturbation | Operational goal-directedness score |
| Cybernetics | Error signal detection | Probe for discrepancy between generated and intended | Whether model tracks progress toward goal |
| ML (Meng/Bau) | Causal tracing | Corrupt + restore at each layer | Localize goal representations |
| ML (Geiger) | Interchange intervention | Swap "goal" activations between samples | Test causal role of goal representations |
| ML (Wu) | Myopic training comparison | Compare myopic vs. normal models on prefill adaptation | Whether goal-pursuit requires look-ahead |
| ML (Nainani) | Future-token probing + suppression | Probe for goal tokens at prefill position | Whether model pre-caches intended algorithm |
| ML (Turner/Rimsky) | Contrastive steering vectors | Compute and apply "goal direction" | Causally test goal encoding |
| ML (Chan) | Causal scrubbing | Resample non-goal activations | Sufficiency of proposed goal mechanism |
| ML (Everitt) | Goal-directedness evaluation | Capability-controlled task evaluation | Distinguish capability from goal-pursuit |

### Recommended Priority Methods for Your Setup

**Tier 1 -- Directly implementable with your existing infrastructure:**

1. **Causal tracing of goal representations** (Meng/Bau style): You already have `get_hs()` and `generate_with_patch()`. Run corrupt-and-restore experiments where the "corruption" is an adversarial prefill and the "restoration" is patching in activations from an aligned-prefill run. Sweep across layers and positions.

2. **Contrastive goal vectors** (CAA/RepE style): You already do contrastive directions. Formalize this as: collect activations from (goal-achieved, diverse-prefill) runs and (goal-failed, diverse-prefill) runs. The difference direction is the "goal vector." Steer with it to test causality.

3. **Persistence and plasticity metrics** (philosophy): Quantify your existing prefill probing results using the formal definitions of persistence (recovery from mid-generation perturbation) and plasticity (success across diverse starting states / prefills).

**Tier 2 -- Requires moderate extension:**

4. **Interchange interventions** (Geiger style): Take two tasks with different goals, swap candidate "goal representations" at a specific layer, and verify the model now pursues the swapped goal. This is the strongest causal test of goal encoding.

5. **Future-token probing** (Nainani/Wu style): Train probes to predict the final algorithm-critical tokens from hidden states at the prefill position. If a linear probe at layer L can predict what the model will eventually write (e.g., `return n % 2 != 0` vs. `return n & 1`), that is evidence for pre-planned goal states.

6. **Layer-specific necessity testing** (optogenetics analogue): For each layer, ablate it and measure the drop in prefill-adaptation performance. This creates a "lesion map" of which layers are necessary for goal-directed code generation.

**Tier 3 -- Ambitious extensions:**

7. **Causal scrubbing** (Redwood style): After identifying candidate goal circuits, resample all non-circuit activations and verify that goal-directed behavior is preserved. This is the strongest test of mechanistic sufficiency.

8. **Myopic training comparison** (Wu style): Train a code model with and without gradient propagation to past timesteps. Compare prefill-adaptation performance to test whether goal-pursuit requires forward-looking computation.

9. **Error signal detection** (cybernetics inspired): Probe hidden states during generation for a signal that correlates with "distance to correct solution." If the model maintains an internal error signal that decreases as it approaches the goal, this strongly supports the cybernetic model of LLM goal-pursuit.
