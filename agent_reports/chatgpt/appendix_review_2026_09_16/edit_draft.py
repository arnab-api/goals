"""Apply the review to the saved author baseline; do not rerun over later edits."""
from pathlib import Path
import ast,json,re
ROOT=Path(__file__).resolve().parents[3];REPORT=Path(__file__).resolve().parent
PAPER=ROOT/'Overleaf-draft/Goals-Overleaf';APP=PAPER/'Text/Appendix';BEFORE=REPORT/'before'
def edit(name,replacements):
    p=APP/name;s=(BEFORE/'Text/Appendix'/name).read_text()
    for a,b in replacements:
        assert a in s,(name,a)
        s=s.replace(a,b,1)
    p.write_text(s)
edit('task.tex',[
('N subgoals',r'$N$ subgoals'),
("For a target {\\color{blue}frontier} the LM's generated code are prefilled in up to a pivot point.","For a target {\\color{blue}frontier}, the LM's generated code is prefilled up to a pivot point."),
('On the same figure we identify the pivot points probed using goalscope in {\\color{red} Sections ...}',r'On the same figure, we identify the pivot points probed using \goalscope in \Cref{app:goalscope,app:readout_sources}'),
('(\\colorbox{green!20}{\\texttt{``eq\'\'}}', '(\\colorbox{green!20}{\\texttt{``eq\'\'}}'),
('Generic \\texttt{<system>}', 'Generic \\texttt{<system>}'),
('tags are stand in for','tags stand in for'),
('\\texttt{``prev\\_end\'\'}}) \n','\\texttt{``prev\\_end\'\'}}).\n'),
('where we probe for the frontiner $g_k$ is most of our evaluations.','where we probe for the frontier $g_k$ in most of our evaluations.'),
(r'\arnab{add example python code-stubs in \Cref{tab:data-distribution}}'+'\n',''),
('copied verbatim from the task bank','with example Python stubs from the task bank'),
('We curate both python and C++ implementation stubs for each of the sub-tasks, which we use to prefill the code to the frontier, in the specified language in the prompt.','We curate both Python and C++ implementation stubs for each sub-task, which we use to prefill the code to the frontier in the language specified by the prompt. In the example stubs, \\texttt{a} denotes the result variable; \\texttt{n}, \\texttt{values}, \\texttt{s}, and \\texttt{words} denote the inputs.'),
])
edit('attn_head_localization.tex',[
('a transfer LM','a transformer LM'),
(r'\footnote{this assumption',r'\footnote{This assumption'),
('attention heads outputs in the source run $M(p_\\mathrm{src})$, when patched to the destination run $M(p_\\mathrm{dst})$, will trigger',"attention head outputs in the source run $M(p_\\mathrm{src})$ that, when patched into the destination run $M(p_\\mathrm{dst})$, will trigger"),
('surface level tokens','surface-level tokens'),
('In both source and destination runs the LM','In both source and destination runs, the LM'),
('To operationalize the idea in a single forward pass we use','To operationalize the idea in a single forward pass, we use'),
('in destination language','in the destination language'),
('In the destination run and the target run we prefill in','In the destination run and the target run, we prefill'),
('While paring up contrastive tasks we choose','While pairing up contrastive tasks, we choose'),
('To localize the heads we learn','To localize the heads, we learn'),
('over the heads, $\\mathrm{mask}^{\\ell j}$ dictates','over the heads. $\\mathrm{mask}^{\\ell j}$ dictates'),
("contribution in the source run are we bringing over in the patched run.","contribution in the source run we bring over to the patched run."),
('Concretely we perform','Concretely, we perform'),
('at the last token position and','at the last token position, and'),
('Fromally, the loss is:','Formally, the loss is:'),
])
edit('goalscope.tex',[
('used to measure the \\emph{faithfulness} and \\emph{causality}','used to measure \\emph{faithfulness} and \\emph{causality}'),
('last token position, and keep','last token position and keep'),
('Following the notation used throughout the paper we call the source context being probed as the source prompt $p_\\mathrm{src}$ and probe prompt as $p_\\mathrm{probe}$. For a selected goal heads $\\mathcal{G}$, we cache the output of each these heads at a specific token $\\mathrm{tok}_{\\mathrm{src}}$ being probed',
 'Following the notation used throughout the paper, we call the source context being probed the source prompt $p_\\mathrm{src}$ and the probe prompt $p_\\mathrm{probe}$. For a selected set of goal heads $\\mathcal{G}$, we cache the output of each of these heads at the position $t$ of the token $\\mathrm{tok}_{\\mathrm{src}}$ being probed'),
(r'$v_g=\sum_{[\ell,j]\in H}a_t^{\ell j}$',r'$v_g=\sum_{[\ell,j]\in\mathcal{G}}a_t^{\ell j}$'),
('We use  $v_g$','We use $v_g$'),
(r'\frac{g_t}{\|g_t\|}',r'\frac{v_g}{\|v_g\|}'),
(r'\arnab{does this eqn becomes recursive now? or is it okay?}','The right-hand side denotes the state before this injection; the arrow overwrites it once per layer and generation step.'),
("the goal heads contributions","the goal heads' contributions"),
(r'\text{, for each  } a^{\ell j} \in \mathcal{G}',r'\text{, for each } [\ell,j] \in \mathcal{G}'),
('cross model comparisons','cross-model comparisons'),
('with some minor adaptions','with some minor adaptations'),
("the judge response of", "the judge's response of"),
('make a subjective judgement whether','make a subjective judgement of whether'),
])

# Preserve all twelve author-selected examples; add their actual code templates.
tree=ast.parse((ROOT/'src/diff/data.py').read_text());q=next(n.value for n in tree.body if isinstance(n,ast.AnnAssign)and getattr(n.target,'id','')=='QTASKS')
bank={ast.literal_eval(k):{ast.literal_eval(p.args[0]):[ast.literal_eval(p.args[1]),ast.literal_eval(p.args[3])]for p in v.args[3].elts}for k,v in zip(q.keys,q.values)}
choices={'int':['is_even','digit_sum','is_perfect_square'],'list[int]':['value_range','mean_int','distinct_count'],'str':['is_palindrome','has_digit','word_count'],'list[str]':['count_long','count_starts_vowel','total_uppercase_letters']}
args={'int':'n','list[int]':'values','str':'s','list[str]':'words'};counts={'int':65,'list[int]':60,'str':64,'list[str]':64}
def esc(s):return ''.join({'_':r'\_','%':r'\%','&':r'\&','#':r'\#','{':r'\{','}':r'\}','`':r'\textasciigrave{}'}.get(c,c)for c in s)
# The caption no longer claims the author's descriptions are verbatim.
author_desc=re.findall(r'\\textit\{([^}]+)\}',(BEFORE/'Text/Appendix/tables/corpus.tex').read_text());assert len(author_desc)==12
author_desc[9]=author_desc[9].replace('longer than 5 words','longer than 5 characters')
lines=[r'\begin{tabular}{@{}lrp{.29\linewidth}p{.43\linewidth}@{}}\toprule',r'Datatype & \# Specs & Example sub-task description & Python stub\\\midrule'];i=0
for d,ids in choices.items():
    for j,k in enumerate(ids):
        code=bank[d][k][1].format(res='a',arg=args[d])
        lines.append((r'\texttt{'+d+'}'if j==0 else'')+' & '+(str(counts[d])if j==0 else'')+' & '+r'\textit{'+esc(author_desc[i])+'} & '+r'{\footnotesize\ttfamily '+esc(code)+'}'+r'\\[4pt]');i+=1
    lines.append(r'\addlinespace')
lines.append(r'\midrule Total & 253 & & \\\bottomrule\end{tabular}');(APP/'tables/corpus.tex').write_text('\n'.join(lines)+'\n')

# D/E are explicitly requested expansions; retain the author's section headings.
s=r'''\section{Are goal heads anything special? How many do we need for a decent readout?}
\label{app:readout_sources}
\label{sec:heads:residual}

\subsection{Comparing against full attention outputs and residual states}
Are the selected \goalheads important for the readout, or could we get a similar description by steering with a larger portion of the LM's state? To check this, we compare their aggregated contributions against the full attention module outputs and the residual stream. We use the same source prompts, readout prompts, and four pre-implementation pivots highlighted in \Cref{fig:task-prompt}.

All three kinds of source use norm-relative additive steering with $\alpha=0.1$, at probe layers 35--45. Concretely, we sum the 122 goal-head contributions, or average the full attention outputs or residual states over the specified source layers. We then normalize this vector before injecting it, following \Cref{eqn:additive-norm}. This keeps the intervention magnitude comparable despite the different numbers of heads and layers contributing to each source.

\begin{figure}[ht]\centering
\includegraphics[width=\linewidth]{Figures/appendix/source_comparison.pdf}
\caption{\textbf{Comparison of readout sources.} We compare the goal-head contributions with full attention outputs and residual states, using norm-relative additive steering at $\alpha=0.1$. Each bar averages the four pre-implementation pivots over the same 253 specifications ($n=1,012$ readouts). The source layer ranges are inclusive. The first implementation token is excluded.}
\label{tab:heads-vs-residual}
\label{fig:heads-vs-residual}
\end{figure}

We find that the goal-head contributions give substantially better readouts: 85.3\% faithfulness and 75.0\% causality, compared with 32.6\% and 29.2\% from the full attention outputs at layers 40--49. The residual baselines have low success under this intervention. So, selecting the goal heads helps expose the upcoming sub-goal to the readout prompt. The comparison holds the steering setup fixed; we did not separately tune its strength or layer range for each source. We return to extracting goal information from the residual with OVReader in \Cref{app:ovreader}.

\clearpage
\subsection{Readouts with different numbers of heads}
\label{app:sparsity_pivots}
How many of these heads do we need for a decent readout? We vary the selected set from 5 to 300 heads and repeat both readouts at each pivot. The main figure (\Cref{fig:sparsity}, left) averages over the four pre-implementation pivots. \Cref{fig:sparsity-other-pivots} shows the individual positions, so we can distinguish the effect of the head budget from when the goal becomes readable.

\begin{figure}[ht]\centering
\includegraphics[width=\linewidth]{Figures/appendix/head_budget_preimplementation.pdf}
\caption{\textbf{Head budget at each pre-implementation pivot.} Each point measures readout success on 253 specifications, with norm-relative additive steering at $\alpha=0.1$. Stars mark the separately selected 122-head set. The four panels contribute equally to the aggregate curve in \Cref{fig:sparsity}; none includes an implementation token.}
\label{fig:sparsity-other-pivots}
\end{figure}

At the assignment pivot, 25 heads give 93.7\% faithfulness and 90.5\% causality. Averaged over all four pivots, the same set gives 75.0\% and 74.2\%. The 122-head set gives 85.3\% and 75.0\% on this average. Thus, a small fraction of heads already supports effective readout, although performance depends on the pivot. Adding heads does not improve the readouts monotonically. These selections come from different mask thresholds and snapshots and are not strictly nested, so this sweep does not isolate head count from head membership.

\clearpage
\section{Across Model Comparisons}
\label{app:model_axes}
We next check whether the same setup works across LM sizes and training variants. We localize the \goalheads separately in each model and use that model for both the source and probe runs. All comparisons below use the same 253 specifications and four pre-implementation pivots. We use the \emph{replace} strategy with $\alpha=1.0$ throughout, so the intervention acts at each selected head's own layer. This differs from the additive setup used in \Cref{app:readout_sources}.

\begin{table}[ht]\centering\small
\input{Text/Appendix/tables/model_comparison_preimplementation}
\caption{Readout success across models, averaged over the four pre-implementation pivots ($n=1,012$ per model and mode). Each model uses its own selected heads. The five-position results, including the first implementation token, and exact selection names are in \Cref{app:model_comparison}.}
\label{tab:model-comparison-four-pivots}
\end{table}

\subsection{Model Size}
We compare Qwen2.5-Coder-Instruct at 14B and 32B parameters. The 32B model has better readouts on both measures: faithfulness increases from 51.9\% to 78.9\%, and causality from 30.5\% to 73.5\%, averaged over the four pivots. \Cref{fig:comparison-size} shows that the larger model's advantage is already visible before the assignment operator. We use 88 selected heads in the 14B model and 122 in the 32B model; this is a comparison of the localized readouts in each model, rather than a matched-head-budget test of size alone.

\begin{figure}[ht]\centering
\includegraphics[width=\linewidth]{Figures/appendix/model_size.pdf}
\caption{\textbf{Qwen2.5-Coder-Instruct: 14B vs.\ 32B.} Both models use head replacement at $\alpha=1.0$. Each point contains 253 readouts. The 32B model gives better readouts at all four pivots in both modes.}
\label{fig:comparison-size}
\end{figure}

\clearpage
\subsection{Code-finetuning helps}
We compare Qwen2.5-Coder-32B-Instruct with Qwen2.5-32B-Instruct to check how the readouts differ between code-specialized and general instruction models of the same size. The Coder model gives better readouts at every pivot in \Cref{fig:comparison-code}. Averaged over the four pivots, faithfulness increases from 23.8\% to 78.9\%, and causality from 44.1\% to 73.5\%. The comparison supports the use of a code-specialized model for these code tasks. It does not isolate code finetuning from other differences between the checkpoints or the independently selected head sets (108 vs.\ 122 heads).

\begin{figure}[ht]\centering
\includegraphics[width=\linewidth]{Figures/appendix/code_finetuning.pdf}
\caption{\textbf{Qwen2.5-Instruct: Coder vs.\ non-Coder at 32B.} The same prompts and replacement strategy are used in both models, with model-specific head sets. The advantage of the Coder model is visible in faithfulness and causality.}
\label{fig:comparison-code}
\end{figure}

\subsection{RL-finetuning also helps}
We compare OLMo-3.1-32B-Instruct with its SFT checkpoint, using 93 and 91 selected heads, respectively. The two readout measures move in different directions: causality increases from 45.2\% to 62.2\%, while faithfulness decreases from 52.9\% to 43.5\%. Thus, the improvement here is in triggering an implementation of the source sub-goal; the instruction model does not verbalize it more faithfully on average.

\begin{figure}[ht]\centering
\includegraphics[width=\linewidth]{Figures/appendix/rl_finetuning.pdf}
\caption{\textbf{OLMo-3.1-32B: instruction vs.\ SFT checkpoint.} Head replacement uses $\alpha=1.0$ in both models. Causality is higher for the instruction model at every pivot, whereas faithfulness is lower.}
\label{fig:comparison-rl}
\end{figure}

\clearpage
The instruction model often responds that it cannot answer without seeing the code. As a simple check on this wording, 402 of its 1,012 pre-implementation faithfulness readouts contain ``I cannot'' or ``I can't'', compared with 11 for SFT. The rate depends on the pivot: 192 of 253 readouts at the previous-line end contain one of these phrases, compared with 40 of 253 at the assignment operator. These are literal phrase counts, not an oracle classification of refusals.

\Cref{tab:olmo-readout-examples} shows four matched examples at the assignment pivot. Some responses decline to answer but still mention the source operation, which can receive a \texttt{YES} under the partial-content rubric in \Cref{app:goalscope}. Other responses provide no task information. This behavior can affect faithfulness without implying that the same state cannot trigger the source operation in the causality probe. The examples illustrate the difference in wording; they do not establish that it explains the entire faithfulness gap.

\begin{table}[ht]\centering\small
\input{Text/Appendix/tables/olmo_readout_examples}
\caption{\textbf{Matched OLMo faithfulness readouts at the assignment pivot.} Four selected examples with their saved oracle verdicts. The complete readouts are shown, including refusal-like wording and a case where the instruction model succeeds while SFT does not. These examples are illustrative; the rates in \Cref{tab:model-comparison-four-pivots} use all specifications.}
\label{tab:olmo-readout-examples}
\end{table}
'''
(APP/'readout_comparisons.tex').write_text(s)

candidates=json.loads((REPORT/'olmo_example_candidates.json').read_text())
ids=['int_python_003','int_python_001','list[int]_python_007','list[str]_python_000'];examples=[next(x for x in candidates if x['spec']==i)for i in ids]
lines=[r'\begin{tabular}{@{}lp{.70\linewidth}l@{}}\toprule',r'Model & Readout & Verdict\\\midrule']
for ex in examples:
    lines.append(r'\multicolumn{3}{@{}p{.97\linewidth}@{}}{\textit{'+esc(ex['condition'])+r'} (\texttt{'+esc(ex['spec'])+r'})}\\[2pt]')
    for k,label in [('sft','SFT'),('rl','Instruct')]:lines.append(label+' & '+esc(ex[k])+' & '+r'\texttt{'+esc(ex[k+'_verdict'])+'}'+r'\\[3pt]')
    lines.append(r'\addlinespace')
lines.append(r'\bottomrule\end{tabular}');(APP/'tables/olmo_readout_examples.tex').write_text('\n'.join(lines)+'\n');(REPORT/'olmo_selected_examples.json').write_text(json.dumps(examples,indent=2)+'\n')

# Explicitly authorized main-figure change; all main body text remains intact.
p=PAPER/'Text/Figures/sparsity_ablation.tex';s=(BEFORE/'Text/Figures/sparsity_ablation.tex').read_text();s=s.replace('versus head budget at the assignment pivot ($n=253$, $\\alpha=.1$)','versus head budget, averaged over the four pre-implementation pivots ($n=1,012$, $\\alpha=.1$)').replace('other capture positions appear in','individual pre-implementation pivots appear in');p.write_text(s)
print('Applied light A–C edits and populated D/E. Main prose untouched.')
