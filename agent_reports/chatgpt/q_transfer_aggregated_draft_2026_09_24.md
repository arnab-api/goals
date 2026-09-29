# Query-transfer subsection — aggregated-results draft

The proposed replacement below retains one concrete setup, pools the results across task-order variants, and delegates the ordering analysis to the appendix. The reported rates were recomputed from the saved per-example results, rather than averaged from the rounded numbers in the manuscript. No LaTeX source files were changed.

## Proposed replacement

```latex
\subsubsection{Query States Guide the Selection of the Next Sub-Goal}
\label{sec:diff:q-transfer}
\input{Text/Figures/q-transfer}

As a complementary experiment, we intervene on the query states to understand what the goal heads \emph{look for} in the task span. We construct source and destination prompts with different progress, for example,
\[
p_{\mathrm{src}}: \mathrm{\underline{a}c\underline{b}\rightarrow c},
\qquad
p_{\mathrm{dst}}: \mathrm{\underline{a}bd\rightarrow b}.
\]
The source has both $\mathrm{a}$ and $\mathrm{b}$ implementations prefilled in, while the destination has only $\mathrm{a}$. In the patched run, we replace the goal heads' queries with those cached at the source pivot, repeating the same replacement at every generation step. We use the 250-head set and amplify its output contributions by $\alpha=2$. If the queries carry the source progress, the \emph{DIFF} account predicts that the destination should skip $\mathrm{b}$, which is already completed in the source, and implement $\mathrm{d}$. The source frontier $\mathrm{c}$ is absent from the destination specification, so selecting $\mathrm{d}$ requires adapting to the available tasks.

\Cref{fig:q-transfer} illustrates the intervention and summarizes its effects. Aggregating across task-order variants, four datatypes, and four language directions, query transfer triggers $\mathrm{d}$ in $52.1\%$ of the 1,024 evaluations, with a readout-shift rate of $54.8\%$. We defer the individual configurations, their ordering controls, and the disaggregated results to \Cref{app:transfer:q}. Although weaker than KV transfer, these effects show that changing the queries can redirect the operation selected from the destination task span.

The results are consistent with progress-dependent selection, but do not isolate progress as the queries' only content. When the source frontier is available in the destination, an additional control finds that the transplanted queries strongly favor that operation (\Cref{app:transfer:expb}). Moreover, \textbf{OVReader}, built from the goal heads' value and output projections, exposes the upcoming operation in residual states before most goal heads act (\Cref{app:ovreader}). Together, these observations suggest that the queries can already reflect a selected sub-goal. DIFF may therefore operate earlier in the computation, with the goal heads helping retrieve and communicate its result.

The interventions support a causal role for task information and query-dependent selection, while leaving the precise computation unresolved. In particular, they do not establish the hypothesized opposite-sign relationship between queries and keys, or explain how the LM chooses among several pending sub-goals.

\paragraph{Control.} We repeat both KV- and query-transfer interventions with random heads drawn from the later half of the LM. We observe no measurable causal-shift or readout-shift with this control group.
```

## Suggested replacement caption

This caption assumes the main figure has been updated to show the pooled results. Its quantitative panel could mirror the KV figure, using one pair of causal-shift/readout-shift bars per language direction. The task-order breakdown would remain in the appendix.

```latex
\caption{\textbf{Query transfer redirects the next sub-goal.} In the illustrated destination, only $\mathrm{a}$ is complete and the native frontier is $\mathrm{b}$. We replace the goal heads' queries with states cached from a source in which both $\mathrm{a}$ and $\mathrm{b}$ are complete. Selecting $\mathrm{d}$ in the patched run is consistent with using the source progress to select a pending destination operation. Results show causal-shift and readout-shift toward $\mathrm{d}$, pooled across task-order variants and datatypes for each language direction (250 heads, $\alpha=2$). Individual configurations and ordering controls are in \Cref{app:transfer:q}.}
```

## Verified aggregation and editorial notes

The result cohort is:

`results/diff/Qwen2.5-Coder-32B-Instruct/diff_trigger/ph-goal_250__a2__piv-eq/{order_matched,order_diff}/`

This is the primary arm without attention restriction used by `notebooks/fig_n_table_generators/transfer.ipynb`. Each variant has 16 datatype/language cells with 32 examples each. I checked the `causal.triggered` and `readout.shifted` flags in all 32 `results.json` files against their `summary.json` files. No records were excluded; these are unconditional rates, not rates conditioned on clean-readout validity.

| Aggregation | Evaluations | Causal-shift | Readout-shift |
| --- | ---: | ---: | ---: |
| All configurations | 1,024 | 534/1,024 = **52.1%** | 561/1,024 = **54.8%** |
| Python → Python | 256 | 59.0% | 62.9% |
| Python → C++ | 256 | 47.3% | 53.5% |
| C++ → Python | 256 | 55.1% | 51.6% |
| C++ → C++ | 256 | 47.3% | 51.2% |

- **The existing aggregate prose needs reconciliation.** The saved order-matched arm is 65.0% causal / 70.9% readout, and the order-diff arm is 39.3% / 38.7%. These do not match the appendix prose's 69% / 68% and 41% / 40%. The draft uses the actual saved outcomes. All cells have equal sample sizes, so pooling records and equally averaging cells give the same result.
- **Heading:** I suggest replacing “The Progress Encoding Dictates What to Read” with the heading above because the interventions establish query-dependent selection more directly than a pure progress encoding.
- **Interpretation:** The main text keeps one brief pointer to the source-frontier-present control. Removing the ordering breakdown should not turn the pooled rate into evidence that the queries encode progress alone. The full control setup and numerical breakdown remain in the appendix.
- **Control sentence:** Retained from your supplied text. I verified the primary goal-head aggregate for this draft, but did not independently recompute the random-head controls. Their plotted values should be checked before retaining “no measurable.”
- **Figure:** The current `Text/Figures/q-transfer.tex` still contains a results-panel TODO. This draft supplies text and a caption only; no figure or manuscript source was modified.
