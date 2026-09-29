# Presenting DIFF as an account of continued goal pursuit

September 23, 2026. Read the current LaTeX sources through Section 5, including Section 3's temporal-persistence subsection, and inspected the current teaser and both transfer figures. Also checked the OVReader appendix referenced in Section 5. This is a framing recommendation, not a revision of the manuscript or an independent validation of the saved experimental results.

## Main recommendation

Present DIFF as **a proposed computation that recovers the current sub-goal from the task and the work completed so far**. Its connection to persistence is that repeated evaluation can keep selecting the same unfinished operation as implementation proceeds. Once the operation is recognized as completed, the same computation can select another one.

This gives Section 5 a clear role in the existing narrative:

1. Section 3 identifies an early, readable, portable representation and finds goal-specific structure across implementation positions.
2. Section 4 establishes a causal role for the goal heads' information access during generation.
3. Section 5 asks what information allows the model to compute that representation again as the context changes.

The organizing question should be: **How does the model recover what it is still trying to do?** This connects persistence and transition more directly than the present title, “How does the LM decide the Next Step?”, which could also refer to choosing individual code tokens.

Suggested section title: **How Does the LM Recompute Its Current Sub-Goal?** A less mechanism-forward alternative is **From Task and Progress to the Current Sub-Goal**.

## The bridge from persistence to DIFF

The crucial distinction is between the abstract operation and its changing implementation state. While generating `max(len(w) for w in words)`, the local syntactic requirements change, but the operation “find the length of the longest word” can remain the relevant unfinished sub-goal. Continued goal pursuit requires both kinds of information: which operation is underway and how to continue its partially written implementation.

DIFF proposes an account of the first. It should not be presented as a complete algorithm for selecting each implementation token.

A useful conceptual decomposition is:

\[
P_t = \operatorname{InferProgress}(C_t),\qquad
U_t = G\setminus\operatorname{Completed}(P_t),\qquad
g_t = \operatorname{Select}(U_t,P_t,O),
\]

where \(C_t\) is the available context, \(G\) is the requested collection of sub-goals, and \(O\) represents ordering information. These are functional quantities, not claimed separately localized vectors or literal neural set operations.

- **During an implementation:** the current operation remains unfinished. A selector that continues work already underway can recover the same \(g_t\), while its neural encoding and the partial code change.
- **After completion:** the inferred completed work changes; the selector can recover a different sub-goal.

Importantly, set difference alone does not guarantee continued selection of the same operation when several remain unfinished. Persistence additionally requires a selection rule sensitive to order or ongoing work. The current experiments leave this rule partly unresolved. Making this explicit improves the theory: “difference” identifies eligible work; selection determines which eligible operation is pursued.

This also keeps the hierarchy consistent with Section 2: \(G\) denotes the full specification, \(g_k\) a sub-goal, and generated tokens implement \(g_k\). Do not silently change \(g_i\) into token-level implementation steps in the theory paragraph or Figure 1 caption.

## Suggested structure for Section 5

### Opening: from the observed encoding to the computation that supplies it

Suggested manuscript prose:

> The preceding sections identify a sub-goal encoding that can guide action and remains recognizable as its implementation unfolds. We now ask how the model computes this encoding from the changing context. We hypothesize that the LM repeatedly relates the task specification to the work completed so far, recovering an unfinished operation to pursue. During implementation, this computation can continue to identify the same sub-goal; once that sub-goal is completed, it can identify the next. We call this the DIFF account: the current sub-goal is selected from the difference between what the task requires and what has already been done.

Follow immediately with the scope of the experimental test:

> We test components of this account at the pre-implementation pivot, where task requirements and completed sub-goals can be varied separately. These interventions examine the information used to select a sub-goal. They provide a test of the proposed selection mechanism, while its repeated operation throughout an implementation remains an explanation of the temporal pattern observed in Section 3.

This replaces the current implication that the experiments directly validate an entire token-by-token persistence mechanism.

### 5.1: a possible implementation through attention

Bring the hypothesis before, or into the opening of, the current gaze subsection. Gaze then motivates where to intervene, rather than carrying the burden of deriving the theory.

The proposed mapping is:

| Component | Proposed role | Relevant evidence |
| --- | --- | --- |
| Task-span keys and values | Make task information available to an attention read | KV transfer changes the implemented operation and its readout. |
| Current-position queries | Condition the read on context, potentially including progress and order | Q transfer changes selection among destination tasks. |
| Head outputs | Supply operation-related information to later computation | Output transfer and access ablations from Sections 3–4. |

Describe this as **an attention read conditioned on progress**. The current assertion that task and progress encodings “should have opposite signs” is not a consequence of the functional hypothesis and is not tested by these interventions. Prefer compatibility with unfinished tasks over literal subtraction or a query equal to \(-E_{\mathrm{prog}}\).

Also, repeated computation does not require repeatedly rebuilding every input. Task information can remain accessible through cached K/V while the query and attention output are computed at each position. The claim concerns recovering the current sub-goal from available information, not erasing and reconstructing all memory at every step.

### 5.2: intervene on task information

Frame KV transfer as asking: **with completed work held fixed, can changing the task information read by the goal heads change the operation pursued?**

The answer is strong evidence for this information pathway. It does not, by itself, establish a progress comparison: the selected task content could also directly drive the next operation. This is why the Q experiment matters.

The current figure is a clear illustration of the task-content effect and cross-language realization. Its setup can remain; the prose should specify the predicted change before giving procedural details.

### 5.3: intervene on the context-dependent selection signal

Frame Q transfer as asking: **with the destination's task information fixed, can a query from a different progress state change which destination operation is selected?**

The strongest conceptual feature is that the intended patched operation \(d\) is absent from the source task. The source is about to do \(c\), but the patched destination does \(d\). This makes the result more informative than copying the source's next operation: source query information combines with destination task information to yield a different operation.

Then distinguish the two configurations:

- Order-matched transfer admits both an ordinal-position explanation and a completed-work explanation.
- Order-different transfer challenges the simple ordinal-position explanation. Successful shifts support a contribution beyond merely copying the source frontier's list index.

This is evidence consistent with progress-sensitive selection, not proof of a query containing only a progress vector. Other source–destination differences and the lower success rate constrain that interpretation. Keep the order comparison visible even if it is not separated into two main figures.

### Closing: connect the tested components back to persistence

Suggested manuscript prose:

> Together, the interventions support a computation in which task information accessed through keys and values is selected using context-dependent queries. We propose that repeated application of such a computation helps sustain goal pursuit: as the implementation grows, the model can recover the same unfinished sub-goal from the evolving context, and select another when progress indicates completion. Our experiments establish causal roles for these inputs at the pivot. They do not yet establish the complete selection rule or its operation at every implementation token.

This should be the main conclusion, rather than a claim of a fully validated neural subtraction algorithm followed by a broad retreat that the theory is underdeveloped.

## How to position OVReader

The current paragraph says the next sub-goal is “already decided” by the time the goal heads act. Successful decoding establishes availability of information, not an irreversible decision or where that information was first computed.

Present OVReader as suggesting that sub-goal information is available in intermediate residual states and that the observed head output may be one stage in a distributed computation. Goal heads need not be the unique place where an undecided model first selects its goal. This fits the readout, intervention, and ablation results without making OVReader proof of temporal reconstruction.

## The main remaining empirical gap

The evidence has two different time scales:

- Temporal affinity samples representations **within** a sub-goal's implementation.
- DIFF intervenes on selection **between** sub-goals, using a query captured at a pre-implementation pivot. The same query is imposed repeatedly during the patched rollout; this does not observe its native updating.

The proposed theory connects them, but the existing experiments do not directly close that gap. An eventual test could repeat matched task/progress interventions at several positions within an implementation, with attention to position alignment and the syntax already written. This is a possible extension, not a prerequisite for presenting the current account as a hypothesis with component-level causal evidence.

## Small issues to reconcile before polishing

- In the introduction, replace the claim that the LM “needs to” compare goal and progress with a proposal that it “may” or “we hypothesize that it” does so. Architecture does not uniquely entail DIFF.
- Avoid making “cannot hold such a state” the premise of Section 5. The useful question is how current goal content is computed from available context; it does not require excluding influence from cached representations.
- The Q figure currently plots causal-shift only, while the prose says it reports causal- and readout-shift. Its visible order-different bars also appear lower on average than the red placeholder value in the prose. Reconcile the actual aggregation before using a headline percentage.
- The KV figure's visible causal-shift bars appear around 0.96–0.97, whereas the prose says 99.2%. Check whether figure and text use the same run and aggregation. I have not re-evaluated the results here.
- The transfer captions still contain TODOs that the current images partly satisfy. The Q caption states progress transfer as established; “query transfer redirects selection among destination tasks” would describe the demonstrated effect more directly.

Sources: [introduction](../../Overleaf-draft/Goals-Overleaf/Text/01_intro.tex), [setup and notation](../../Overleaf-draft/Goals-Overleaf/Text/02_notations.tex), [readout and transfer](../../Overleaf-draft/Goals-Overleaf/Text/03_what.tex), [temporal persistence](../../Overleaf-draft/Goals-Overleaf/Text/persistance.tex), [head ablations](../../Overleaf-draft/Goals-Overleaf/Text/04_heads.tex), [Section 5](../../Overleaf-draft/Goals-Overleaf/Text/06_mechanism.tex), and [OVReader](../../Overleaf-draft/Goals-Overleaf/Text/Appendix/ovreader.tex).
