# Functional goal representations: research and introduction-framing brainstorm

September 20, 2026. Consolidates the literature research and discussion in this conversation, including the proposed connection to the Boundary Signal experiment in Appendix N. This is a research and planning report, not a replacement introduction. No Overleaf text has been changed, and no new model experiments have been run for this report.

## Recommended framing

Start with **“Do LLMs have goals?”** and turn it into an affirmative empirical investigation: what functional properties would justify attributing goals to a system, and what internal representations support those properties?

A working question for this paper is:

> Does the model contain a representation that specifies an upcoming sub-goal and organizes subsequent computation toward its realization?

The definition should apply across physical implementations. The experiments then identify how an LLM realizes these functions through particular neural components. This separates the functional question from the implementation question without treating the mechanism as irrelevant.

There are also two levels of evidence to distinguish. **Goal-directed behavior** concerns the system's actions across conditions. **A goal encoding** is a candidate internal representation whose causal role helps explain those actions. The paper's distinctive contribution is the connection between the two, through readout, transfer, and interventions during native generation.

Retain the proposed three-part framework—pre-commitment, plasticity, and persistence—with **causal guidance running through all three**. Distinguish temporal persistence from recovery after disruption. Introduce release after completion as the natural endpoint of persistence and the bridge to the boundary experiments.

## Literature and its role in the argument

### Dennett: goal attribution as a predictive strategy

Dennett motivates the opening question by treating beliefs and goals as part of a strategy for explaining and predicting a system's behavior. “Observer experience” is too weak a characterization: the predictive success of the attribution is empirically assessable. This gives us a route from attributing a goal to testing the mechanism that supports the predicted behavior. The intentional stance itself does not require discovering a localized goal vector.

Sources: Dennett, [*Intentional Systems* (1971)](https://home.csulb.edu/~cwallis/382/readings/483/dennett.intentional.systems.1971.pdf); [*The Intentional Stance* (1987), publisher page](https://mitpress.mit.edu/9780262040938/the-intentional-stance/). The accessible 1971 paper was consulted directly; it provides the basis for the characterization here.

### Bratman: a representation that organizes subsequent action

Bratman's planning theory motivates the functional role of prospective representations: plans guide further reasoning, constrain alternatives, and can specify ends while leaving implementation details open. The goal–implementation distinction is especially useful for interpreting cross-language transfer. Stability and revisability both matter; the theory does not equate commitment with permanently repeating the same behavior.

Sources: Bratman, *Intention, Plans, and Practical Reason* (1987), through the [project literature note](../claude/lit_review/bratman_1987_intention_plans_practical_reason.md); the directly consulted primary source is Bratman, Israel, and Pollack, [*Plans and Resource-Bounded Practical Reasoning* (1988), especially Sections 2–4](https://www.emse.fr/~boissier/enseignement/defiia/up5/pdf/Bratman-PlansPracticalResoning.pdf).

Use this connection to motivate the empirical questions, rather than present our proposed attributes as a complete test of Bratman's theory.

### Williams: intention-like representations in language models

Williams provides the closest conceptual connection to representations in LLMs. His account considers directive function, degrees of distality, degrees of abstraction, commitment, and planning. These dimensions help connect a representation's content to the actions it guides; they are not our three-part framework or a necessary-and-sufficient checklist. His empirical examples include function vectors and output/planning features. The application to our goal heads is our interpretation.

Source: Iwan Williams, [*Intention-like representations in language models?* (2026), Philosophical Studies](https://doi.org/10.1007/s11098-026-02605-y). See the [full paper review and terminology map](williams_intention_like_representations_review_2026_09_20.md) for section/page references and alternative interpretations.

The most useful connections for the introduction are:

- **Directive causal role:** the encoding helps bring about the operation it represents. Faithful verbalization identifies candidate content; transfer and native interventions test whether it guides generation (§§2.1, 4.1).
- **Proximal and abstract:** an imminent operation can admit multiple concrete implementations. Proximity concerns distance to action onset; abstraction concerns how much implementation detail is specified (§§2.2–2.3, 4.2–4.3). Cross-language transfer is particularly relevant to abstraction.
- **Commitment:** Williams distinguishes control of conduct, settling alternatives, and stability against reconsideration (§§2.4, 4.4). Our pre-implementation and causal evidence addresses the first most directly. It does not establish all three.

Two terminology choices matter. First, use **pre-commitment**, with an explicit operational definition below, rather than “prospective commitment”: Williams sometimes uses *prospective* to mean *distal*, whereas our target operation is imminent. Second, *mind-to-world* is a direction of fit, not a name for verbalization. Directive representations have **world-to-mind** fit: action brings about an outcome matching their content. The distinction motivates combining readout and intervention; it does not map one-to-one onto our two metrics.

### Lee and McShea: measurable persistence and plasticity

Lee and McShea distinguish returning toward a goal after perturbations from reaching it across different starting conditions. They treat these as graded dimensions, rather than a binary checklist that settles whether every system has goals. This is the closest direct source for the proposed persistence/plasticity vocabulary. Their particular spatial-trajectory metrics are not directly applicable to token sequences without additional modeling choices.

Source: Lee and McShea, [*Operationalizing Goal Directedness: An Empirical Route to Advancing a Philosophical Discussion* (2020)](https://sites.duke.edu/mcshearesearch/files/2023/11/Operationalizing-goal-directedness-Lee-and-McShea-2020.pdf).

An experimental distinction useful here is to vary the context before pursuit begins when testing plasticity, and introduce a recoverable disruption during pursuit when testing corrective persistence. These are complementary experimental axes, not a claim that the capacities are wholly independent.

### Cohen and Levesque: commitment has release conditions

Their formal account of persistent goals explicitly addresses when goals should be relinquished, including satisfaction and perceived impossibility, with a further formulation allowing dependence on background conditions. This motivates studying when a model stops pursuing a sub-goal as well as when it continues. It provides a conceptual link to completion and boundary signals; it does not establish that our boundary intervention implements their formal account.

Source directly consulted: Cohen and Levesque, [*Intention = Choice + Commitment* (1987)](https://cdn.aaai.org/AAAI/1987/AAAI87-073.pdf). This is the AAAI paper, distinct from their longer 1990 treatment.

### Contemporary LLM work: capability and goal pursuit

Everitt and colleagues evaluate how fully LLMs employ available capabilities toward a goal, distinguishing goal-directedness from task success alone. This matters when designing obstacle experiments: a model can fail because it lacks the required repair skill, so repair competence should be measured separately.

Source: Everitt et al., [*Evaluating the Goal-Directedness of Large Language Models* (2025)](https://arxiv.org/abs/2504.11844).

A further related-work candidate is [*A Behavioural and Representational Evaluation of Goal-Directedness in Language Model Agents* (2026)](https://arxiv.org/abs/2602.08964), which combines behavioral evaluation with analysis of internal representations in a spatial environment. Its abstract and article were inspected during the preliminary search; a detailed methodological comparison remains a report extension, not a completed novelty assessment.

## Citation support for the three-part framework

**Each leg has literature support; the exact three-part framework is our synthesis.** Present it as an operational characterization drawing on accounts of intention and goal-directedness. Do not attribute the entire definition to Bratman, Williams, or Lee and McShea individually. These accounts also operate at different levels: behavioral goal-directedness, planning agents, and candidate internal representations.

| Proposed attribute | Principal citation and relevant discussion | What it supports | Connection to our evidence |
| --- | --- | --- | --- |
| **Pre-commitment** | Bratman (1987); Bratman, Israel, and Pollack (1988), §§2–4; Williams (2026), §§2.2–2.4, 4.4 | An intention can precede action, organize subsequent reasoning, and guide execution. Commitment has stronger senses than merely occurring first. | Combine pre-implementation readout with evidence of causal guidance. Early decoding alone shows availability, not a settled commitment. |
| **Plasticity** | Lee and McShea (2020), §1; Williams (2026), §§2.3, 4.3 | Goal pursuit can accommodate varying conditions; abstract intentions leave details of realization open. Williams's abstraction discussion is the closer match to cross-language implementation transfer. | The same transplanted goal guides an implementation appropriate to a different programming-language context. This tests an adaptation of the behavioral concept, not Lee and McShea's spatial metric. |
| **Temporal persistence** | Bratman, Israel, and Pollack (1988), §§2, 4; Williams (2026), §§2.4, 4.4 | Intentions can continue organizing conduct and have stability across time, while remaining revisable. | Continued goal-specific information and causal influence across token positions. Stable content does not require an unchanged activation vector; affinity alone does not establish continued causal influence. |
| **Persistence through obstacles** | Lee and McShea (2020), §1 | Pursuit returns toward a goal following a perturbation. This is the most direct citation for corrective persistence. | Behavioral recovery is relevant, but connecting recovery to the extracted encoding requires goal-head interventions during recovery. Temporal availability does not substitute for this test. |
| **Release after completion** | Cohen and Levesque (1987), persistent achievement-goals and their release conditions | Goal commitment can end upon believed satisfaction or impossibility, and can depend on background conditions. | Motivate the Appendix N transition experiments as a companion question: when does the system stop pursuing the current sub-goal? The intervention does not by itself establish a semantic completion detector. |

Dennett belongs in the opening motivation for functional goal attribution. Bratman and Williams support the move to action-guiding representations. Lee and McShea provide the most direct plasticity/persistence vocabulary, and Cohen and Levesque supply the completion-and-release connection.

### Proposed framing language and citation placement

This is a paragraph candidate for deciding the framing, not a replacement introduction:

> We take a functional view of goal encodings as internal representations that guide the realization of an operation or outcome. Drawing on accounts of intention and goal-directedness, we characterize these representations along three dimensions: **pre-commitment**, representing and guiding an operation before its execution; **plasticity**, supporting different realizations of the same goal across contexts; and **persistence**, sustaining goal-directed computation over time and recovering from disruptions.

In the eventual introduction, split this into short sentences so each citation supports a specific claim:

1. Cite **Bratman (1987), Bratman et al. (1988), and Williams (2026)** when introducing action-guiding representations and pre-commitment.
2. Cite **Williams (2026) and Lee and McShea (2020)** when introducing abstraction and plasticity across conditions.
3. Cite **Bratman et al. (1988) and Williams (2026)** for stability over time, then **Lee and McShea (2020)** for recovery after disruption. Keep the two persistence questions explicit.
4. Cite **Cohen and Levesque (1987)** at the transition from persistence to release after completion and the Boundary Signal experiment.

The paragraph describes the functional profile we investigate. The subsequent contribution statements should identify which parts the experiments establish and which remain open. In particular, persistence through obstacles is currently a research question about the goal-head mechanism, not a demonstrated property of the extracted encoding.

Use **pre-commitment** in the following operational sense: *the system represents an upcoming operation before initiating it, and that representation guides its subsequent execution*. This preserves the intended framing while distinguishing it from stronger claims about settling competing intentions or resisting reconsideration. Likewise, say that implementations adapt across environments; the encoding itself need not change.

### Williams citation to add when editing the paper

The publisher lists the article online in 2026; omit volume, issue, and page range until assigned. This entry is provided here for later use; the paper bibliography has not been edited.

```bibtex
@article{williams2026intention,
  author  = {Williams, Iwan},
  title   = {Intention-like representations in language models?},
  journal = {Philosophical Studies},
  year    = {2026},
  doi     = {10.1007/s11098-026-02605-y},
  url     = {https://doi.org/10.1007/s11098-026-02605-y}
}
```

## Attributes and their experimental meaning

| Attribute | Functional question | Existing connection | Most useful next test |
| --- | --- | --- | --- |
| **Pre-commitment** | Is the upcoming operation represented before implementation, and does that representation guide subsequent computation? | Pre-implementation readout, transfer, and native causal interventions. | Bring temporal and causal evidence together; distinguish a readable prediction from an action-guiding state. |
| **Plasticity** | Can the same goal guide different concrete means of achieving it? | Transfer of an encoding between programming-language contexts. | Extend to different implementation constraints within a language, with goal identity held fixed. |
| **Temporal persistence** | Does goal-specific information remain available and behaviorally relevant during implementation? | Readout at pre-implementation positions; affinity across implementation spans. | Readout and transfer at intermediate implementation positions and across completion. |
| **Corrective persistence** | Does pursuit resume after a recoverable disruption? | Earlier behavioral prefill experiments. | Introduce disruptions after pursuit begins and test the goal heads' contribution to recovery. |
| **Release and transition** | Does the model stop pursuing a completed sub-goal and advance appropriately? | Boundary Signal experiments. | Align transition interventions with semantic completion and changes in goal readouts. |

For **pre-commitment**, the proposed operational emphasis is causal guidance by a pre-implementation representation. Readout establishes early availability; interventions establish a role in subsequent computation. Stronger claims about maintaining a settled choice under competing alternatives require additional evidence.

For **plasticity**, the central formulation is **goal identity is preserved while implementation adapts**. The extracted encoding need not itself change. A fixed encoding that guides appropriate implementations in different contexts is useful evidence of abstraction.

For **persistence**, distinguish persistence of goal-directed behavior, continued availability of goal information, and the mechanism maintaining that information. Repeatedly recovering the goal from context can support sustained pursuit even if no unchanged vector is carried forward through every token.

## What the affinity experiment currently contributes

Inspected sources: [goal-enc-affinity.png](../../Overleaf-draft/REF/goal-enc-affinity.png) and [14_goal_retention.ipynb](../../notebooks/clock/14_goal_retention.ipynb).

The plot compares a predicate's encoding at its assignment pivot with encodings sampled through implementation. It includes 640 conditions from 128 contexts, with comparisons separating predicate identity from shared context.

The useful observation is that **predicate identity remains distinguishable through much of the implementation**. At 60% progression, similarity for the same predicate in different contexts is approximately 0.652, compared with 0.292 for different predicates in different contexts. At the implementation endpoint these values are approximately 0.427 and 0.344. The cross-context comparison is more informative about goal identity than within-sequence similarity alone.

The notebook reports a median implementation length of **14 tokens**. Consequently, a claim about long implementations should be supported by a separate analysis of the longer, multi-step predicates rather than inferred from percentage-normalized trajectories alone.

Recommended extensions:

1. **Test functional availability along the trajectory.** At several positions, extract the encoding and evaluate both verbalization and transfer. Distinguish the enclosing sub-goal from local implementation steps and the next sub-goal.
2. **Align to completion.** Extend the trajectory beyond the implementation boundary and ask whether the current goal's readout gives way to the next goal. Declining similarity near completion may reflect an appropriate transition.
3. **Use context-level uncertainty estimates.** Pairwise cosine comparisons reuse the same encodings. Resample contexts, rather than treat every pair as an independent observation. Inspect per-predicate and implementation-length results.
4. **Investigate the maintenance mechanism separately.** Use temporally targeted interventions on access to task descriptions and earlier generated states to distinguish possible routes by which goal information remains available. Include clean-task controls because these interventions also change ordinary information access.

The affinity result alone cannot decide between retention and recomputation. Its strongest present interpretation is sustained goal-specific representational structure.

## Proposed obstacle-recovery experiments

The earlier [LM Performance on Prefill Probing report](../claude/qualitative_analysis/0_lm_performance.md) is a useful source of candidate disruptions. Its aggregate scores and broad interpretations were not revalidated for this brainstorm. In particular, neither ordinary task success nor a robustness difference by itself identifies an internal goal representation.

### Priority experiment: native recovery and goal-head mediation

1. Run a multi-step task normally and identify the current sub-goal before implementation.
2. After pursuit begins, introduce a syntactically valid, recoverable disruption. An incorrect intermediate assignment that later code can overwrite is a candidate; an irrevocably incorrect completed return is not a useful recovery opportunity.
3. Allow continuation without ongoing goal steering. Measure goal readout over time, the generated repair, and functional correctness using tests.
4. Compare recovery with and without temporally restricted goal-head interference. Include matched random-head interventions and the same interventions on undisrupted tasks.
5. Test a brief restoration of the original goal encoding, alongside a different-goal encoding and a matched control, then remove the intervention and observe continuation.

The central question is whether the identified mechanism contributes to restoring pursuit after disruption. A general reduction in coding performance under ablation would not establish a recovery-specific role; the clean-task comparison matters. Likewise, rescue shows that the injected encoding can support recovery, while spontaneous maintenance is a separate question.

Before interpreting failures, establish that the model can perform the relevant repair under a straightforward instruction. Keep goal-changing instructions separate from obstacles: following a legitimate new objective is different from abandoning an unchanged goal after a disruption.

### Complementary experiment: persistence of a transplanted goal

Transfer a goal encoding into a destination with an alternative implementation constraint or recoverable obstacle. Compare it with a destination lacking the obstacle. After a bounded intervention window, stop injecting the encoding and measure whether pursuit continues and succeeds.

This tests the extracted representation more directly. Continuous injection can test whether the representation guides an adaptive implementation under the intervention, but it cannot by itself show autonomous maintenance after the intervention ends.

## Boundary signals: persistence has an endpoint

The user specifically endorsed **release after completion** as a framing and segue to the Boundary Signal experiment in Appendix N. The corresponding source is [Boundary Signals: Moving On and Staying](../../Overleaf-draft/Goals-Overleaf/Text/Appendix/boundary.tex).

The functional idea is:

> The system should sustain a sub-goal while it remains unfinished, then move on after completion.

The citation anchor is Cohen and Levesque's (1987) account of release conditions, discussed above. Completion is one condition for release, rather than a demand to persist regardless of success, impossibility, or changed circumstances.

The current appendix reports that transferring signals from completed to incomplete steps can induce advancement, while reverse transfers can induce staying or repetition. The contrast is designed to reduce direct copying of a source variable as an explanation. This motivates distinguishing:

- **Goal identity:** what operation is being pursued?
- **Boundary control:** when should pursuit continue or give way to another sub-goal?

A possible bridge, for later introduction drafting:

> Pursuing a sub-goal requires both sustaining it during implementation and recognizing when to move on. We therefore also investigate signals that regulate transitions between sub-goals.

The present evidence concerns causal control over staying versus advancing. A next-variable change does not establish that the preceding task was semantically completed or that the next task is correctly implemented. Establishing a completion signal, or a full causal dissociation from goal identity, calls for continuation scoring and goal readouts under the boundary intervention.

A useful joint experiment would align three measurements around completion: affinity to the current goal, readout of current versus next goal, and the effect of the boundary intervention. Premature “move on” and prolonged “stay” interventions would then test whether changing the transition also changes which sub-goal is represented. This connects the temporal-persistence analysis to the boundary experiments without assuming their outcome.

## Recommended order of work

1. Agree on the functional attributes and their names. The present recommendation is pre-commitment, plasticity, and persistence, with temporal and corrective persistence distinguished and release after completion included as a companion question.
2. Strengthen the existing temporal evidence first: intermediate readout/transfer, length-stratified results, and completion-aligned trajectories.
3. Connect the boundary experiments to those trajectories, including what is represented before and after an induced transition.
4. Develop a small obstacle-recovery pilot with clean-task, capability, and matched-head controls before a large evaluation.
5. Use the resulting evidence map to structure the introduction. Motivate each property positively and state the corresponding experimental finding; keep detailed qualifications with the relevant results and methods.

The main-paper introduction has not been drafted under this new plan. The open decision is how prominently to feature corrective persistence before the experiments connect behavioral recovery to the identified goal-head mechanism.

## Internal notes used as research leads

- [Defining “Goal” for Code-LMs](../claude/1_defining_goal.md).
- [Bratman literature note](../claude/lit_review/bratman_1987_intention_plans_practical_reason.md).
- [OOD readings](../claude/lit_review/ood_readings.md).
- [Readings on goals of systems](../claude/lit_review/readings_goals_of_systems.md).
- [Causal and behavioral goal-analysis readings](../claude/lit_review/readings_causal_behavioral_goal_analysis.md).
- [Earlier Markdown framing draft](functional_subgoals_draft_2026_09_17.md).
- [Section 5 rewrite and evidence notes](section5_rewrite_2026_09_19.md).
- [Williams (2026): full review, terminology, and citation recommendations](williams_intention_like_representations_review_2026_09_20.md).

These notes were used as leads and conceptual proposals. Their stronger claims were not treated as independently verified results.
