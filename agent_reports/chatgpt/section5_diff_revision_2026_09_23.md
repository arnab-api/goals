# Proposed revision of Section 5

September 23, 2026. A replacement draft based on the current [Section 5](../../Overleaf-draft/Goals-Overleaf/Text/06_mechanism.tex) and the [framing strategy](diff_framing_repeated_computation_2026_09_23.md). Existing LaTeX reference and citation keys are retained for eventual insertion. No `.tex` files have been edited. Numerical discrepancies are noted after the proposed text; the draft describes those results qualitatively pending reconciliation.

---

## 5. Selecting the Current Sub-Goal From Task and Progress

*Retain label: `sec:diff`.*

In the preceding sections, we examine an encoding of the sub-goal $g_k$: it precedes implementation, can guide different realizations, and remains recognizable as execution unfolds. We now ask how the LM determines which sub-goal to encode. To study this question, we place $g_k$ within the broader task $G$, where pursuing $g_k$ is itself the next action. This preserves the level of abstraction established by our earlier experiments. Within the implementation of $g_k$, individual steps can be closely tied to syntax and token selection. At the level of $G$, the next action is an operation that admits different implementations. We therefore investigate how the model selects this operation from the broader specification and its progress so far.

This perspective also connects sub-goal selection with persistence. As the model implements $g_k$, the code changes at every generation step, but $g_k$ can remain the relevant unfinished operation within $G$. Repeatedly recovering which operation to pursue could therefore sustain the same sub-goal content across changing neural states. We propose a *DIFF* account of this computation and test its components through interventions on the goal heads' attention reads.

### 5.1. A DIFF Account of Computing the Current Sub-Goal

At each token position, a transformer LM computes new residual states through a fixed number of layers. Earlier encodings remain accessible through cached representations, while the goal heads' outputs at the current position are computed anew. The available context tape (\Cref{fig:teaser}) contains both the specification of $G$ and the code written so far. We hypothesize that the LM uses this context to recover the task requirements and its progress, $E_{\mathrm{prog},t}$, then compares them to identify unfinished work. The current sub-goal $g_k$ is selected from this remaining work. We call this the *DIFF* account because selection depends on the difference between what is required and what has already been done.

During implementation, this computation can continue to recover $g_k$ while the model generates different steps toward its realization. Once the context indicates completion, it can select another sub-goal. This separates the question of *which operation to pursue* from *how to continue its implementation*: DIFF addresses the former, while the selected operation and partial code together inform subsequent generation. Identifying unfinished work does not by itself specify which operation to pursue when several remain; task order and the partially written implementation may also influence that selection.

**A possible implementation through attention.** In our setting, the sub-goals are explicitly described in the prompt. We hypothesize that the goal heads read these descriptions through attention conditioned on progress: task-span keys and values make the requested operations available, while queries computed from the current context influence which operation is read. The resulting head outputs supply sub-goal information to later layers for implementation. This is one possible realization of the DIFF account. It does not require literal subtraction of task and progress vectors; attention can favor unfinished operations through the compatibility of queries and task-span keys. Task information can remain available in the KV cache while queries and head outputs are computed at successive positions.

### 5.2. Intervening on Task Information and Sub-Goal Selection

We test two components of this proposed computation at the pre-implementation pivot, where we can specify which sub-goals have already been completed. First, holding completed work fixed, we change the task information available through the goal heads' keys and values. Second, holding the destination task fixed, we replace the queries with those from a context with different completed work. In both cases, we measure which abstract operation is encoded and implemented next.

We write a prompt as $p:\mathrm{\underline{a}b\underline{c}\rightarrow b}$. Letters identify sub-tasks in their specification order, underlines mark sub-tasks whose implementations have been prefilled, and the arrow indicates the next sub-goal in the unpatched run. We use the broader 250-head set from the causal-access experiments (\Cref{sec:mechanism:ablation}).

#### 5.2.1. Changing the Task Information Read by Goal Heads

*Retain label: `sec:diff:kv-transfer`; place the KV-transfer figure here.*

Can changing *what the goal heads read* change *what the LM does*? We construct source and destination prompts with the same completed sub-goals but different remaining operations:

$$
p_{\mathrm{src}}:\mathrm{\underline{a}\underline{b}c\rightarrow c},
\qquad
p_{\mathrm{dst}}:\mathrm{\underline{a}\underline{b}d\rightarrow d}.
$$

Both prompts have completed $\mathrm{a}$ and $\mathrm{b}$. The source next implements $\mathrm{c}$, whereas the destination next implements $\mathrm{d}$. If task information accessed through the goal heads guides this selection, replacing that information with the source's should cause the destination to implement $\mathrm{c}$.

In the patched destination run, we replace the goal heads' task-span key–value states with the corresponding source states. Other heads continue to read the destination states. The replacement is applied once and remains available through the KV cache during generation. For grouped-query attention, we maintain separate clean and patched caches so that goal heads read the patched states while other heads sharing their KV group retain the clean states. \Cref{fig:kv-transfer} illustrates the intervention.

We measure *causal-shift*: whether the patched continuation implements the source frontier $\mathrm{c}$ instead of the destination frontier $\mathrm{d}$. We also measure *readout-shift*: whether GoalScope's verbalization at the patched pivot describes $\mathrm{c}$. An oracle judges the generated implementation and readout against the target operation.

KV transfer redirects the upcoming operation with high consistency across four datatypes and all four Python/C++ transfer directions. The verbalization also shifts toward the source operation (\Cref{fig:kv-transfer}). These results establish that task information accessed through the goal heads can redirect both the model's implementation and the goal encoding exposed by the readout. We next test whether queries help determine which operation is selected from the available task information.

#### 5.2.2. Changing Which Destination Sub-Goal Is Selected

*Retain label: `sec:diff:q-transfer`; place the Q-transfer figure here.*

We now keep the destination task-span keys and values unchanged and intervene on what the goal heads *look for*. The source and destination differ in their completed work:

$$
p_{\mathrm{src}}:\mathrm{\underline{a}c\underline{b}\rightarrow c},
\qquad
p_{\mathrm{dst}}:\mathrm{\underline{a}bd\rightarrow b}.
$$

The source has completed $\mathrm{a}$ and $\mathrm{b}$ and is about to implement $\mathrm{c}$. The destination has completed only $\mathrm{a}$ and would normally implement $\mathrm{b}$. If the source queries carry information about completed work, transferring them should favor $\mathrm{d}$ in the destination: $\mathrm{b}$ is already complete according to the source progress, leaving $\mathrm{d}$ as the remaining destination operation. Notably, $\mathrm{d}$ is absent from the source specification. A successful transfer therefore combines information from the source query with the destination task, rather than copying the source's upcoming operation $\mathrm{c}$.

We cache the source goal-head queries at the pivot and substitute them for the destination queries. During the patched rollout, we apply the same cached queries at every generation step. We measure causal-shift toward $\mathrm{d}$ and evaluate whether the patched pivot readout identifies that operation. \Cref{fig:q-transfer} shows a concrete example.

To distinguish progress-sensitive selection from transfer of the frontier's list position, we compare two source configurations. In the *order-matched* configuration, $p_{\mathrm{src}}:\mathrm{\underline{a}\underline{b}c\rightarrow c}$, the source frontier $\mathrm{c}$ and the intended patched operation $\mathrm{d}$ are both third in their specifications. A shift to $\mathrm{d}$ could therefore reflect either completed-work information or a preference for the third task. In the *order-different* configuration shown above, the source frontier $\mathrm{c}$ is second, whereas $\mathrm{d}$ is third in the destination. Transferring only the source frontier's list position would favor $\mathrm{b}$; a shift to $\mathrm{d}$ provides evidence beyond this simple positional account.

Query transfer redirects the model toward $\mathrm{d}$ in both configurations, with a higher causal-shift rate when the orders match (\Cref{fig:q-transfer}). The order-different successes are consistent with the queries carrying progress-related information that influences selection among destination tasks. The performance difference also suggests that selection depends on specification order. These effects are weaker than KV transfer and do not identify the queries as a pure progress encoding. They support the DIFF account as a partial explanation of the information used to select the current sub-goal.

**Control.** We repeat both interventions with randomly selected heads from the later half of the LM. These controls show no measurable causal- or readout-shift, supporting the role of the identified goal heads in the transfer effects.

### 5.3. Implications for Continued Goal Pursuit

The interventions connect the sub-goal encoding studied in the preceding sections to its role within the broader task: changing task information or the queries used to access it can change which operation the model pursues. Repeated application of such a selection process offers an account of the temporal continuity observed in \Cref{sec:persistence}. The model could continue to recover $g_k$ as the current unfinished operation while generating its implementation, then select another sub-goal when its inferred progress indicates completion.

The computation may be distributed across layers. In \Cref{app:ovreader}, we introduce *OVReader*, a weights-only readout built from the goal heads' value and output projections. It exposes information about the upcoming operation in intermediate residual states, including before many goal heads act. The heads may therefore read and mediate sub-goal information whose computation has already begun elsewhere in the network.

Our interventions test selection at the pivot; they do not directly establish how native queries evolve throughout an implementation. In particular, query transfer holds the source pivot queries fixed during the rollout. The results also leave open how the model chooses among several unfinished operations. We thus view DIFF as a testable account of how repeated computation could sustain goal pursuit, supported by causal evidence about its inputs, with the full selection and updating mechanism still to be determined.

---

## Notes for review — not manuscript text

1. **Results needing reconciliation.** The existing KV prose reports 99.2% causal-shift and 96.1% readout-shift, whereas the current figure's causal bars appear closer to 96–97%. The Q prose contains red placeholders of 0.45 and 0.65; the visible order-different bars appear lower on average than 0.45. I have not selected one version or inferred corrected values from the pixels. The proposed text retains the qualitative findings and figure references until the run/aggregation is confirmed.
2. **Q readout results.** The current Q figure plots causal-shift only. The revised text still describes readout-shift as an evaluation, as in the existing section, but does not say the figure plots it or assert an unreported quantitative readout result.
3. **Control claim.** The control sentence preserves the current section's reported finding. This revision does not independently verify those runs; the final figure or appendix should expose their results.
4. **Interpretation of the query experiment.** A shift to a destination-only operation is stronger evidence than copying a source operation, but does not uniquely identify completed-work content. Queries could also carry order, selection preferences, and other contextual information. The text therefore says progress-related information is supported, rather than that the intervention isolates a pure progress representation.
5. **Recomputation and selection.** The draft distinguishes repeatedly recovering an operation from generating its next implementation token. It also states the selection requirement explicitly: identifying unfinished tasks is insufficient to guarantee continued selection of the operation underway. No literal subtraction, opposite-sign condition, unique completion detector, or continuously carried progress register is assumed.
6. **Structure and length.** The proposed organization separates the functional account, its attention implementation, its tests, and its implications. If space is tight, the final subsection can become two closing paragraphs of Section 5.2, with OVReader reduced to a sentence pointing to the appendix. Existing experiment labels can be retained even though their subsection numbers change.
7. **Figures.** The current KV caption names the destination environment as C++, but the displayed example has a Python destination. The Q caption presents progress transfer as established; a title such as “Query transfer redirects selection among destination tasks” better matches the claim in this revision. Neither caption has been edited.
8. **The depth argument.** The architectural motivation now follows the explanation of the task hierarchy. I omitted the argument that heads at layers 40–50 cannot influence an implementation longer than the remaining layer count. A representation at position $t$ can contribute to keys/values at a later layer that are read directly by positions $t+1$, $t+2$, and beyond; each read need not follow a chain through the intervening token positions. Depth limits a chain of successive latent updates across positions, not the number of future positions that can read an earlier state. Earlier information can also affect generated tokens, which then become context. Thus neither head depth nor a 30-token implementation establishes that a single early encoding cannot have sustained influence. This architectural distinction follows from the decoder and attention dependencies in [Vaswani et al. (2017), §§3.1–3.2](https://arxiv.org/html/1706.03762v7). The draft motivates the proposed repeated computation without presenting it as the only architecturally feasible mechanism. An implementation-length analysis would still help characterize the temporal-persistence result.
9. **Gaze removed from the main argument.** The attention hypothesis is now motivated directly by the task/progress decomposition and tested through KV/Q interventions. Neither a gaze subsection nor a passing gaze citation is necessary for that logic. The qualitative attention observations and `gandikota2026gaze` citation can move together to the appendix or related work. The proposed text no longer uses `sec:mechanism:gaze`; any remaining manuscript cross-references would need to be checked when this revision is adopted.
10. **Hierarchy and abstraction.** The opening now explicitly places the same $g_k$ encoding within the broader task $G$. The change is from characterizing the encoding to explaining its selection. Studying $g_k$ as the next action within $G$ keeps the selected action abstract enough to admit different implementations. The text says smaller implementation steps *can* be tied to tokens; it does not assume every sub-sub-goal is token-specific. The title now names the selected object directly: “Selecting the Current Sub-Goal From Task and Progress.” The distinction between selecting an operation and generating its implementation carries through the theory, experimental motivation, and conclusion.
