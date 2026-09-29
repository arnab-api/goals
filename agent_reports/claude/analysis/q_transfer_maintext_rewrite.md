# Rewrite: §q-transfer main text (aggregated only; matched/diff → appendix)

*2026-09-24. Draft rewrite of `\subsubsection{The Progress Encoding Dictates What to Read}`
(`sec:diff:q-transfer`) per the ask: **drop the order-matched vs order-diff breakdown from the main
text, report only the aggregated result, and show cross-language transfer in Figure 6 to match the
kv-transfer cross-lang figure (Figure 5).** A closing sentence defers the order-matched/order-diff
separation and the "source frontier also in the destination" variant to the appendix. Not applied to
`.tex`.*

*Numbers (250 heads, α=2, no-restrict; aggregate = mean of order-matched & order-diff, equal n=512
each): overall **causal 0.521 / readout 0.548**. By language direction below.*

---

## Rewritten subsection (LaTeX, drop-in)

```latex
\subsubsection{The Progress Encoding Dictates What to Read}
\label{sec:diff:q-transfer}
\input{Text/Figures/q-transfer}

As a counter-experiment, we intervene on the \emph{query} states to probe what the goal heads
\emph{look for} in the task span. We construct a source and destination that share a sub-goal but
differ in how far along they are:
\[
p_{\mathrm{src}}: \mathrm{\underline{a}c\underline{b}\rightarrow c},
\qquad
p_{\mathrm{dst}}: \mathrm{\underline{a}bd\rightarrow b.}
\]
The source has completed $\mathrm{a}$ and $\mathrm{b}$ and is implementing its frontier $\mathrm{c}$;
the destination has completed only $\mathrm{a}$ and would implement $\mathrm{b}$ next. In the patched
run we transplant the source's goal-head \emph{queries} at the pivot into the destination, repeating the
transfer at each generation step. If the query carries the source's \emph{progress} and drives frontier
selection --- the \emph{DIFF} prediction --- the destination should skip its own frontier $\mathrm{b}$
and instead implement $\mathrm{d}$, the sub-goal that remains once $\mathrm{a}$ and $\mathrm{b}$ are
treated as done.

\Cref{fig:q-transfer} reports the causal-shift and readout-shift by language direction, mirroring the
kv-transfer result in \Cref{fig:kv-transfer}. Aggregated across our query-transfer configurations, the
transfer shifts the destination to $\mathrm{d}$ with a causal-shift of $\sim\!0.52$ and a readout-shift
of $\sim\!0.55$, and the effect holds across all four language directions. This is weaker than the
near-ceiling kv-transfer, but a non-trivial, cross-lingually robust signal that the goal-head query
carries progress information that actively participates in selecting the next action. In
\Cref{app:ovreader} we introduce \textbf{OVReader}, a projection built from the goal heads' value and
output projections that reads goal encodings directly from the residual stream, and find that by the
time the goal heads act the next sub-goal has already been selected (\Cref{fig:ovreader}).

These interventions support a relationship between the goal ($G$) and progress ($E_\mathrm{prog}$)
encodings and the LM's next action, but they do not directly establish the opposite-sign query--key
relationship posited in \Cref{sec:diff}. The partial success rate indicates that the \emph{DIFF} theory
only partially captures how the LM decides what remains --- task order and other parallel signals likely
also contribute, and our single-frontier setup cannot say how the LM picks the \emph{first} among
several pending sub-goals. We read the result as encouraging but acknowledge the account is incomplete.

To separate transferred \emph{progress} from a simpler task-\emph{order} transfer, we run the query
transfer in two configurations (\emph{order-matched} and \emph{order-diff}) and additionally in a
variant where the source frontier is also present in the destination prompt; the full breakdowns and
analysis are in \Cref{app:task_transfer}.

\paragraph{Control.} Repeating the intervention with a random set of heads from the later half of the
LM produces no measurable causal- or readout-shift.
```

---

## Figure 6 (`fig:q-transfer`) — make it a cross-lang panel like Figure 5 (kv)

Currently `Text/Figures/q-transfer.tex` is a schematic + a right-panel TODO. To match Figure 5's format
(causal-shift solid + readout-shift striped, one group per language direction), regenerate a
**q cross-lang panel** with these **aggregated** values (mean of order-matched & order-diff, 250 heads,
α=2, no-restrict):

| direction | causal | readout |
|---|:-:|:-:|
| py→py   | 0.590 | 0.629 |
| py→cpp  | 0.473 | 0.535 |
| cpp→py  | 0.551 | 0.516 |
| cpp→cpp | 0.473 | 0.512 |
| **all** | **0.521** | **0.548** |

**Generated:** `notebooks/Figures/transfer/q_crosslang.pdf` (also copied to
`Overleaf-draft/.../Figures/generated/transfer/q_crosslang.pdf`). Produced by the new cell in
`notebooks/fig_n_table_generators/transfer.ipynb` (after the `q_barcharts` cell): it averages
`order_matched` and `order_diff` per language direction and draws the same `grouped_metric_bars` style as
the kv cross-lang panel (causal solid, readout white-striped, no 0.5 line, legend on top). Point
`\includegraphics` in `q-transfer.tex` at it (keep the left schematic if the figure is a composite, like
the kv figure).

*Note:* the earlier `q250_matched_vs_diff.png` and `q_right_panel.pdf` are **different** cuts (matched-vs-
diff / by-datatype) — not for the main text; they belong to the appendix breakdown.

---

## Appendix X (`app:task_transfer`) — what moves out of the main text

Fold the removed main-text detail here (the data already lives in
`agent_reports/claude/analysis/q_transfer_progress_vs_diff.md`):

1. **Order-matched vs order-diff** setup + purpose (separating progress transfer from ordinal/order
   transfer), with the split table: matched **0.650 / 0.709**, diff **0.393 / 0.387** (causal/readout),
   and the interpretation that matched − diff ≈ 0.26 is the order/position contribution while the
   non-zero order-diff level is the content/DIFF component.
2. **Source-frontier-in-destination variant** (`c_in_dst`): the case where the source frontier is also
   present in the destination prompt (progress available on both sides) — report its causal/readout and
   what it implies. *(Confirm the `c_in_dst` run exists / rerun if needed; the `--variants c_in_dst`
   path is in `scripts/diff/diff_trigger_eval.py`.)*
3. Per-language-direction split for each configuration (the §2a tables in the analysis doc).

Reconcile the `\Cref{app:task_transfer}` label with the actual appendix label for the q/kv transfer
section.
