# Suggested rewrite of Section 5, with emphasis on Section 5.2

Reviewed September 19, 2026. This is a proposal only; the Overleaf draft and figures have not been edited. I rebuilt the current source into `/tmp/goals-section5-review-2026-09-19/` and inspected Figures 3 and 4 on pages 6 and 7. I also checked the intervention code and the saved records corresponding to the figures.

The main change in flow is to make each experiment follow the same sequence: **question → source/destination contrast → intervention → prediction → result → interpretation**. The crucial distinction is that KV-transfer aims to trigger the **source frontier**, whereas this Q-transfer experiment aims to trigger a **destination task that is absent from the source**. The latter is the prediction that makes the progress hypothesis testable.

The proposed prose below retains your DIFF hypothesis and its incomplete-account conclusion. Changes to numerical claims and interpretation are explained separately after the rewrite.

## Proposed manuscript text

### 5. How Is the Upcoming Sub-Goal Decided?

What information do the goal heads use to guide the next operation? We first examine where these heads read from, then use targeted interventions to test how task information and the model's progress contribute to selecting the next sub-goal.

### 5.1. Gaze to Relevant Task Descriptions

The goal heads often look back to the task they are about to implement. At a frontier position, their collective value-weighted attention concentrates on tokens in the prompt describing the upcoming predicate. This resembles the *gaze* behavior observed during generation in other settings (Gandikota and Bau, 2026). These qualitative patterns suggest that the heads read task information from the prompt. We next test this connection causally and ask what determines which part of the task description they read.

### 5.2. A DIFF Theory of Sub-Goal Selection

The observations in Section 5.1 raise a question: how do the goal heads identify what remains to be done? We hypothesize that, in pursuing a broader goal $G$, the LM maintains an encoding of its progress, $E_{\mathrm{prog}}$, which is updated as it completes each sub-goal. At a pivot, the model uses this progress together with $G$ to identify the remaining work and select the next operation $g_k$. We refer to this account as the *DIFF* theory.

In our coding setting, the task description explicitly specifies the sub-goals. One possible implementation of DIFF is therefore an attention read: the task-span keys and values provide information about $G$, while the queries use progress information to suppress completed sub-goals and select an unfinished one. A query component corresponding to $-E_{\mathrm{prog}}$ is one possible realization of this suppression. We test the roles of task information and progress through complementary interventions on keys and values, and on queries.

We write a task prompt as $p:\underline{a}b\underline{c}\rightarrow b$. The letters identify sub-tasks in their specification order; an underline indicates that the corresponding implementation has been prefilled. The arrow identifies the next sub-goal, or *frontier*, in the unpatched run. Thus, in this example, $a$ and $c$ have been implemented and $b$ remains to be done.

#### 5.2.1. Reading the Goal Encoding from the Task Span

We first ask whether changing the task information available to the goal heads changes the operation the LM implements. We construct source and destination prompts with the same completed sub-goals but different frontiers:

$$
p_{\mathrm{src}}:\underline{a}\underline{b}c\rightarrow c,
\qquad
p_{\mathrm{dst}}:\underline{a}\underline{b}d\rightarrow d.
$$

Both prompts contain implementations of $a$ and $b$. The source next requests $c$, while the destination requests $d$. In the destination run, we replace the task-span keys and values read by the selected goal heads with those cached from the source. Other heads retain access to the destination task-span states. The fixed source states remain available to the selected heads throughout generation; the destination prompt itself is unchanged.

Figure 3 illustrates this intervention with a shared initial sorting step. The C++ source next tests whether the shortest word starts with a vowel, while the Python destination tests whether the longest word is a palindrome. After KV-transfer, the destination instead generates a vowel test on the shortest word, expressed in Python. Thus, the intervention changes the operation while allowing it to be realized in the destination language.

If the goal heads read the goal information from the task description, replacing that information should cause the patched destination to implement $c$. We measure this effect as *causal-shift*, using an oracle to judge the generated continuation. We separately measure *readout-shift*: whether GoalScope's verbalization of the patched state identifies $c$. The former tests the behavioral effect; the latter tests whether the readout reflects the changed operation.

We use the broader 250-head set, which showed stronger causal effects in the ablation experiments in Section 4.2. Across the four datatypes and four language directions in Figure 3, causal-shift is 99.2% and readout-shift is 96.1%. These results show that the task information accessed through the goal heads can redirect the upcoming operation, including across programming languages. They support the task-information component of DIFF; the role of progress requires a separate test.

#### 5.2.2. Does the Query Carry Progress Information?

We next ask whether the queries determine what the goal heads look for by encoding the model's progress. We use a contrast in which source and destination share sub-tasks $a$ and $b$, but have different remaining tasks $c$ and $d$:

$$
p_{\mathrm{src}}:\underline{a}c\underline{b}\rightarrow c,
\qquad
p_{\mathrm{dst}}:\underline{a}bd\rightarrow b.
$$

The source has completed both $a$ and $b$ and is about to implement $c$. The destination has completed only $a$ and would normally implement $b$ next. We transplant the source pivot queries into the destination goal heads. Under the progress hypothesis, the transferred queries should treat $a$ and $b$ as completed and select $d$ from the destination specification. Importantly, $d$ is absent from the source, and the source's own frontier $c$ is absent from the destination. Selecting $d$ therefore tests whether the transferred state can identify unfinished work in a different task specification.

Figure 4 gives a concrete example. The shared sub-tasks are summing the input values ($a$) and counting distinct values ($b$). The source has completed both and next checks whether the list is sorted ($c$). The destination has completed only the sum and would next count distinct values. With the source queries patched in, it instead computes the difference between the maximum and minimum values ($d$), as predicted by transferring the source progress.

We compare two source configurations to distinguish this prediction from transfer of the frontier's position in the task list. In the *order-matched* configuration, $p_{\mathrm{src}}:\underline{a}\underline{b}c\rightarrow c$, the source frontier $c$ and predicted destination target $d$ are both third in their respective lists. Selecting $d$ is therefore consistent with either progress transfer or a positional instruction to perform the third listed task. In the *order-diff* configuration shown in Figure 4, $p_{\mathrm{src}}:\underline{a}c\underline{b}\rightarrow c$, the source frontier is second, while $d$ remains third in the destination. A transfer of the source frontier's list position would select $b$; the progress hypothesis instead predicts $d$.

For these experiments, causal-shift measures whether the patched continuation implements $d$, and readout-shift measures whether the patched verbalization identifies $d$. This differs from KV-transfer, where the target is the source frontier $c$. Figure 4 reports causal-shift for both source configurations across programming languages and head selections. With 250 heads and output amplification $\alpha=2$, causal-shift is 65.0% for order-matched prompts and 39.3% for order-diff prompts, pooled across the four datatypes and four language directions. The other configurations show the same qualitative pattern. In the Python-to-Python comparison, equally sized random-head controls produce substantially smaller causal shifts and no readout shifts.

The order-diff results are consistent with queries carrying progress information that can guide the selection of an unfinished destination task. The stronger order-matched results also suggest sensitivity to task order, although this contrast alone does not identify a separate encoding of order. Together, the experiments support aspects of DIFF without establishing it as a complete account of sub-goal selection. In particular, they do not show that queries literally encode $-E_{\mathrm{prog}}$, and other mechanisms may contribute to selecting the next operation. Our setup also leaves only one destination sub-task unfinished under the transferred progress. It therefore does not determine how the model chooses among several remaining sub-goals. DIFF captures a testable part of the proposed mechanism, but the full selection process remains unresolved.

### Optional transition to Section 5.3

These interventions test what information can redirect the next operation. We now ask where the upcoming sub-goal becomes readable within the model's computation, using the goal heads' value and output projections to examine intermediate residual states.

## Suggested figure captions

These are separate proposals; the figure files and their captions have not been changed.

**Figure 3. Task-span KV-transfer redirects the next operation.** The source and destination share a completed sorting step but request different subsequent operations. We make the selected goal heads read the source task-span keys and values while other heads retain destination task-span access. The patched Python continuation tests whether the shortest word starts with a vowel, matching the C++ source operation, and its readout shifts accordingly. Yellow marks the pivot; red, green, and purple mark source, destination, and patched continuations. The table reports causal-shift and readout-shift across datatypes and language directions, with 32 examples per cell.

**Figure 4. Query transfer tests whether progress guides the next operation.** The source has completed $a$ and $b$ and is about to implement $c$; the destination has completed only $a$ and would next implement $b$. After replacing the destination goal-head queries with source pivot queries, the illustrated continuation implements $d$, the destination task left unfinished under the source progress. The table compares order-matched and order-diff configurations. In order-matched prompts, both progress transfer and transfer of the source frontier's list position predict $d$. In order-diff prompts, progress transfer predicts $d$, while position transfer predicts $b$. Entries report causal-shift; configuration labels give the number of patched heads and their output amplification factor.

## Corrections and decisions to resolve before applying the rewrite

### Setup and intervention details

1. **The KV source notation contains a typo.** The current text writes $\underline{a}\underline{b}d\rightarrow c$. The source task list should contain $c$: $\underline{a}\underline{b}c\rightarrow c$. Figure 3 illustrates the same contrast with one shared completed task rather than two; the rewrite explicitly separates the general evaluated setup from this illustration.

2. **Figure 3's language direction is C++ → Python.** Its current caption says the destination environment is C++. The proposed prose and caption follow the image.

3. **The current KV footnote does not describe the implementation.** `headwise_kv_patch` computes attention with destination task-span states and with substituted source task-span states, then routes the appropriate per-head outputs. It retains the destination KV cache and reapplies this routing on every forward pass. It does not maintain two persistent clean/corrupted caches. The source span is cached once, but the intervention is maintained through generation and also affects prefill computations. Calling it a one-time intervention only at the pivot would be misleading. Suggested replacement footnote:

   > In grouped-query attention, several query heads share a KV head. We therefore route attention outputs per query head: selected goal heads use the substituted source task-span states, while other heads use destination states. The fixed source span is reused during prefill and decoding, without replacing the destination KV cache.

   “Other heads use destination states” refers to the direct intervention. Their subsequent activations can still change as the patched computation propagates through the network.

4. **Query transfer is also maintained during a steering window.** The current evaluation function caches the source pivot query, patches the destination pivot, and reuses that query during subsequent decoding passes. The CLI default for the new `diff_trigger` script is 16 forward passes including prefill; the exact invocation for a reported run should be documented before putting a duration into the paper. The rewrite avoids describing a single-token intervention followed by entirely unpatched generation.

5. **Figure 3's “no amp” label conflicts with the matching saved run.** Every displayed causal/readout cell matches the run `ph-goal_250__ro-goal_250__abl-off__a2__pad`, whose metadata specifies `alpha: 2.0`. No non-goal-head restriction is applied in this run, but the selected head outputs are amplified. The label should be reconciled with the intended data source. The draft rewrite reports the matched scores without asserting that they were obtained without amplification.

6. **Minor illustration inconsistency:** Figure 3 assigns the sorted words to `a` but uses `res` in the destination and patched continuations. This does not obscure the intended contrast, but the names should agree in a finished illustration.

### Results and their interpretation

The scores below are recomputed from the saved per-instance records, not averaged from the rounded numbers in the figures. The per-configuration summaries reproduce the same rates. Some top-level `summary.json` files contain only a subset of configurations, so I did not use those files as complete aggregates.

**Figure 4: causal-shift pooled across four datatypes and four language directions.** Each entry contains 512 instances.

| Head set and OV amplification | Order-matched | Order-diff |
| --- | ---: | ---: |
| 122 heads, $\alpha=2$ | 296/512 = 57.8% | 160/512 = 31.3% |
| 122 heads, $\alpha=4$ | 340/512 = 66.4% | 210/512 = 41.0% |
| 250 heads, $\alpha=2$ | 333/512 = 65.0% | 201/512 = 39.3% |

The current red **45%** placeholder does not match these aggregates. The **65%** placeholder matches the 250-head, $\alpha=2$ order-matched result; the corresponding order-diff rate is **39.3%**. I use that matched pair in the rewrite. I am not substituting the older approximately 94% source-goal query-transfer result: that is a different contrast and outcome.

**Figure 3:** the matching KV run contains 508/512 causal successes (99.2%) and 492/512 readout shifts (96.1%). These are oracle judgments of the target operation, not execution-based tests of complete program correctness. The causal criterion can credit a continuation containing both compared operations; it need not mean the destination operation has been eliminated entirely. The stricter joint B/C/D labels available for Q-transfer are a separate metric and should not be mixed with Figure 4's binary causal-shift scores.

**Random controls have nonzero causal effects.** The saved Q controls cover Python→Python across four datatypes, with 128 cases per variant:

| Control and OV amplification | Order-matched causal-shift | Order-diff causal-shift | Readout-shift |
| --- | ---: | ---: | ---: |
| 122 random heads, $\alpha=4$ | 12/128 = 9.4% | 8/128 = 6.3% | 0% in both variants |
| 250 random heads, $\alpha=2$ | 14/128 = 10.9% | 9/128 = 7.0% | 0% in both variants |

For comparison, the matching goal-head Python→Python causal rates are 75.8%/53.9% for 122 heads at $\alpha=4$, and 68.0%/50.0% for 250 heads at $\alpha=2$ (order-matched/order-diff). “Substantially smaller effects” is supported; “no sensitivity” is not. The proposed text limits this control statement to the checked Q-transfer comparison rather than asserting it for both experiments or all language directions.

**The order contrast is informative but not a unique identification of progress.** The order-diff setup distinguishes the progress prediction from copying either the source frontier or its list position. It does not identify the full query contents or a unique algorithm for computing them. Similarly, the order-matched advantage suggests an influence of ordering but does not by itself prove that the query contains explicit order IDs. The rewrite retains DIFF as your proposed account and makes these inferential limits explicit in the closing paragraph.

**Readout-shift belongs as a secondary result.** I would keep the behavioral causal-shift central and retain readout-shift as evidence that the readout reflects the intervention. Because the readout is based on the affected heads, it is not an independent localization of the causal mechanism. Figure 4 currently shows only causal-shift, so the rewrite does not say that it displays both measures. The full readout table could go in the appendix.

**The older query contrast and OVReader answer different questions.** The current ending combines an exploratory example where queries favor the source goal with the claim that the next goal is already “decided” and readable through OVReader. I suggest moving that contrast into a separate appendix discussion and using the short transition above to introduce Section 5.3. Reading an upcoming sub-goal at an intermediate layer locates its accessibility; it does not by itself identify the computation that selected it. This is a substantive change in interpretation, not merely a wording fix.

## Sources checked

- Current section: [Text/06_mechanism.tex](../../Overleaf-draft/Goals-Overleaf/Text/06_mechanism.tex).
- Figure 3: [image](../../Overleaf-draft/Goals-Overleaf/Figures/diff/kv-transfer.png), [caption](../../Overleaf-draft/Goals-Overleaf/Text/Figures/kv_transfer.tex).
- Figure 4: [image](../../Overleaf-draft/Goals-Overleaf/Figures/diff/q-transfer.png), [caption](../../Overleaf-draft/Goals-Overleaf/Text/Figures/q-transfer.tex).
- Supporting appendix: [Task and Query Transfer](../../Overleaf-draft/Goals-Overleaf/Text/Appendix/task_transfer.tex).
- Intervention implementation: [src/diff/functional.py](../../src/diff/functional.py), especially `run_q_transfer_intervention`, `greedy`, `headwise_kv_patch`, and `run_kv_transfer_intervention`.
- Evaluation setup and metrics: [scripts/diff/diff_trigger_eval.py](../../scripts/diff/diff_trigger_eval.py) and [scripts/diff/kv_transfer_eval.py](../../scripts/diff/kv_transfer_eval.py).
- KV records: `results/diff/Qwen2.5-Coder-32B-Instruct/kv_transfer/multi_pred/ph-goal_250__ro-goal_250__abl-off__a2__pad/*/results.json`.
- Q records: `results/diff/Qwen2.5-Coder-32B-Instruct/diff_trigger/{ph-goal_122__a2__piv-eq,ph-goal_122__a4__piv-eq,ph-goal_250__a2__piv-eq}/*/*/results.json`.
- Q control records: the same `diff_trigger` directory, under `ph-control122_122__a4__piv-eq` and `ph-control250_250__a2__piv-eq`.

The isolated PDF build completed, with existing unresolved references/citations and duplicate-label warnings. No LaTeX files or figure assets were changed.
