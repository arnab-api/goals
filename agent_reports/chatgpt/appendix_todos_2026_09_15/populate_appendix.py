"""Apply the September 15 appendix requests to the captured starting draft.

Uses this report's before/ snapshot. Do not rerun after further author edits.
"""
from pathlib import Path
import json, re, shutil
ROOT=Path(__file__).resolve().parents[3]
REPORT=Path(__file__).resolve().parent
PAPER=ROOT/'Overleaf-draft/Goals-Overleaf'
APP=PAPER/'Text/Appendix'
OLD=REPORT/'before/Text/Appendix'
def write(name,text): (APP/name).write_text(text.strip()+'\n')

# Retain the author's commented material and prompt caption, replacing only the
# active distribution table and the explicitly requested evaluation structure.
s=(OLD/'task.tex').read_text()
start=s.index('We use a fixed corpus')
end=s.index('\n% The main GoalScope evaluation',start)
s=s[:start]+r'''We use a fixed corpus of 253 Python specifications to compare GoalScope readouts across head budgets and capture positions (\Cref{app:readout_sources}), native ablations (\Cref{app:ablation,app:necessity}), and model families (\Cref{app:model_comparison}). To generate the corpus, we sample $N\in\{2,3,4\}$ sub-tasks from one of four datatypes: \texttt{int}, \texttt{str}, \texttt{list[int]}, and \texttt{list[str]}. \Cref{tab:data-distribution} gives the datatype counts and examples of the full descriptions used in the prompts. Each specification prefills at least one completed sub-task before the frontier $g_k$, with fixed function and intermediate-variable naming conventions. The four pre-implementation pivots end before the implementation of $g_k$; a fifth pivot includes its first token. \Cref{fig:task-prompt} shows a concrete example.

The corpus is stored in \texttt{data/goalscope\_eval.json}, generated with seed 123. Frontier identities are balanced within each datatype. There are 92 two-task, 82 three-task, and 79 four-task specifications. The complete archived OVReader sweep uses a separate 256-specification cohort (\Cref{app:ovreader}); query and KV transfer use source/destination pairs (\Cref{app:task_transfer}). These cohorts are reported separately.''' +s[end:]
start=s.index(r'\begin{table}');end=s.index('\n% \\begin{table}',start)
s=s[:start]+r'''\begin{table}[ht]\centering\small
\input{Text/Appendix/tables/corpus}
\caption{The locked evaluation corpus and illustrative sub-task descriptions, copied verbatim from the task bank. Each specification contains two to four sub-tasks; the examples are individual sub-tasks, not complete specifications. Five capture positions give 1,265 records per evaluation mode.}
\label{tab:data-distribution}
\end{table}
''' +s[end:]
s+=r'''

\subsection{Capture positions}
\label{app:pivots}
We cache the state at five positions around a prefilled frontier assignment: previous-line end, indentation, variable name, assignment operator, and first implementation token. Position names refer to tokenized boundaries, rather than whole lines. The first implementation token provides a comparison after implementation has begun. Each readout uses only the cached state at its selected position; subsequent tokens shown in an illustration are not available to that state.

\begin{figure}[ht]\centering
\includegraphics[width=.92\linewidth]{Figures/author/evaluation_setup.pdf}
\caption{\textbf{Capture positions and the two readouts.} Pink marks the previous-line end, purple indentation, yellow the variable, light green the assignment operator, and dark green the first implementation token. The arrows illustrate faithfulness and causality readouts at the variable and assignment positions. In the evaluation, both probes run separately at all five positions.}
\label{fig:pivot-probes-detail}
\end{figure}

For the evenness example \texttt{int\_python\_000}, the Qwen2.5-Coder-32B token indices are 102, 103, 104, 105, and 106, respectively. These indices depend on the model's tokenization. Five positions produce repeated measurements of the same specifications; rates pool positions only where explicitly stated.

\subsection{Scoring the readouts}
The runner extracts the generated frontier assignment as \texttt{impl\_span}. An oracle judges whether the probe's description (faithfulness) or C++ implementation (causality) captures a nontrivial part of that code. For \texttt{int\_python\_000}, the extracted assignment is \texttt{b = (n \% 2 == 0)}. The reference is the model's generated operation, not the natural-language task description. Agreement with that operation and correctness against the requested task are distinct measurements.

\Cref{app:oracle_prompts} provides the complete judging prompts, instantiated examples, and verdict rules. We count only \texttt{YES} verdicts as successes and retain all records in the denominator. Causality scores apply the deterministic trivial-readout sanitizer described there. The rubric allows partial agreement and does not certify execution correctness. Native ablations instead score the generated continuation against the requested frontier task.

In prompt illustrations, \texttt{\textbackslash n} marks an actual newline. Unmarked wrapping only fits the text to the box; the accompanying raw prompt files preserve the original newlines.
'''
write('task.tex',s)

write('goalscope.tex',r'''
\clearpage
\section{GoalScope}
\label{app:goalscope}
\subsection{Prompts for Faithfulness and Causality}
The following are the complete role-separated probe contents before applying the model family's chat template. Neither probe receives the source prompt as text. The faithfulness probe begins with an empty assistant message; the causality probe continues the fixed function prefix. The runner applies the template with \texttt{continue\_final\_message=True}.

\paragraph{Faithfulness.}
\PromptInput{faithfulness_probe.txt}

\paragraph{Causality.}
\PromptInput{causality_probe.txt}

\subsection{Steering setup}
At source position $t$, additive steering aggregates the selected heads' projected contributions, $g_t=\sum_{(\ell,j)\in H}a_t^{\ell j}$. At probe layer $\ell$ and position $s$, norm-relative injection uses
\begin{equation}
r_{\ell,s}\leftarrow r_{\ell,s}+\alpha\|r_{\ell,s}\|\frac{g_t}{\|g_t\|}.
\end{equation}
The main Qwen2.5-Coder-32B configuration uses $\alpha=.1$, layers 35--45 inclusive, and repeated injection during greedy decoding, with a 64-token generation budget. The source vector remains fixed. This coefficient specifies a fraction of the probe residual norm, rather than an absolute activation scale.

The replacement variant instead operates on each selected head's activation before its output projection, at that head's native layer. With a binary selection mask $m_h$, its update is $z_h\leftarrow(1-m_h)z_h+\alpha m_h z_h^{\mathrm{src}}$. We use $\alpha=1$ for the cross-model comparisons in \Cref{app:model_comparison}. Replacement does not use the additive layer band or normalize the source vector.

\begin{table}[ht]\centering\small
\begin{tabular}{lp{.31\linewidth}p{.31\linewidth}}\toprule
 & Norm-relative addition & Head replacement\\\midrule
Cached signal & Sum of projected head contributions & Each selected head's pre-projection activation\\
Injection sites & Probe layers 35--45 & Selected heads' own layers\\
Source coefficient $\alpha$ & .1 & 1\\
Duration & All decoding steps & All decoding steps\\
\bottomrule\end{tabular}
\caption{The two GoalScope intervention configurations. The additive settings are for the main Qwen2.5-Coder-32B evaluation; replacement is used for within-model replication across families and sizes.}
\label{tab:steering-config}
\end{table}

These settings were not selected by a systematic hyperparameter search. The localization sparsity coefficient $\lambda_{\mathrm{sparse}}=20$ (\Cref{app:head_loc}) is shared by the head sets used here; it is separate from the readout coefficient $\alpha$. The source and head-budget comparisons in \Cref{app:readout_sources} hold the additive configuration fixed. OVReader uses $\alpha=.15$ for transformed residual states and $.1$ for its head-output reference (\Cref{app:ovreader}).
''')
(APP/'prompts/faithfulness_probe.txt').write_text('System:\nYou are a helpful assistant proficient in programming.\n\nUser:\nWhat are you doing at this line of the code. Keep your answer as brief as possible.\n\nAssistant (empty):')
p=APP/'prompts/faithfulness_probe.txt'
(APP/'prompts/display'/p.name).write_bytes(p.read_bytes().replace(b'\n',b'\\n\n'))
s=(OLD/'oracle_prompts.tex').read_text()
a=s.index(r'\subsection{Generation prompt');b=s.index(r'\subsection{Verdict parsing',a)
s=s[:a]+r'The complete generation prompts for both probes are in \Cref{app:goalscope}.'+'\n\n'+s[b:]
write('oracle_prompts.tex',s)

s=(OLD/'readout_comparisons.tex').read_text()
a=s.index(r'\begin{table}');b=s.index(r'\end{table}',a)+len(r'\end{table}')
s=s[:a]+r'''\begin{figure}[ht]\centering
\includegraphics[width=\linewidth]{Figures/appendix/source_comparison.pdf}
\caption{\textbf{Comparison of injected sources.} Norm-relative injection at $\alpha=.1$ on the locked corpus, pooled over five positions ($n=1,265$ per bar). Layer endpoints are inclusive: the saved half-open source ranges \texttt{[40,50)} and \texttt{[25,56)} correspond to L40--49 and L25--55. The attention and residual sources are means over their stated layers. Poor raw-residual injection is a result about this probe configuration, not proof that the residual lacks subgoal information.}
\crefalias{figure}{figure}
\label{tab:heads-vs-residual}
\label{fig:heads-vs-residual}
\end{figure}''' +s[b:]
write('readout_comparisons.tex',s)

s=(OLD/'attn_head_localization.tex').read_text()
fig=s[s.index(r'\begin{figure}'):]
write('attn_head_localization.tex',r'''
\section{Localizing Goal Heads}
\label{app:head_loc}
We localize heads through a joint intervention on contrastive single-predicate prompts. The source and destination request different predicates in different programming languages. A third, target prompt requests the source predicate in the destination language. The destination and target implementations share a prefix but disagree at the next token. We select contrasts whose target token also differs from the source continuation, so copying the source token alone cannot solve the localization task.

For each attention head $h$, a learned mask $m_h\in[0,1]$ interpolates between destination and source activations before the output projection. We freeze the language model and minimize
\begin{equation}
\mathcal{L}(m)=\frac{1}{|\mathcal{B}|}\sum_{x\in\mathcal{B}}
D_{\mathrm{KL}}\!\left(P_{\mathrm{patched}}(\cdot\mid x;m)\,\|\,P_{\mathrm{target}}(\cdot\mid x)\right)
+\lambda_{\mathrm{sparse}}\frac{1}{|\mathcal{H}|}\sum_{h\in\mathcal{H}}m_h.
\end{equation}
Here $\mathcal{H}$ contains all candidate heads and $\mathcal{B}$ is a training batch. The target is the full next-token distribution of the target prompt. The sparsity term is the mean mask value, not an unnormalized sum. Optimizing the mask jointly allows several heads to contribute to the same counterfactual continuation; it does not require every selected head to have a large effect when patched alone.

Masks start at one and are clamped to $[0,1]$ after each AdamW update. All five saved runs use learning rate $.01$ and $\lambda_{\mathrm{sparse}}=20$. The cache is split in its stored order: the final $\lfloor .2N\rfloor$ instances form validation and the remainder form training. This is an instance split, not a held-out-predicate split. It is separate from the 253-specification readout corpus.

\begin{table}[ht]\centering\small
\input{Text/Appendix/tables/localization_training}
\caption{Localization settings from the saved mask checkpoints. Augmented runs resample surface forms for training and validation each epoch; unaugmented runs use precomputed source activations.}
\label{tab:localization-training}
\end{table}

Augmented runs use seed 123456 and choose between changing the source language (55\%), renaming variables (35\%), and toggling type hints (10\%). The training objective concerns one continuation position; subsequent GoalScope evaluation generates and judges a multi-token readout.

\begin{table}[ht]\centering\small
\input{Text/Appendix/tables/localization_selection}
\caption{Exported head selections used in the five-model evaluation. Layer endpoints are inclusive. Masks are thresholded within the export band; all listed memberships match their saved training checkpoints. SHA-256 prefixes identify the exact selection files. The OLMo instruction-model threshold includes values equal to one.}
\label{tab:localization-selection}
\end{table}

The core Qwen2.5-Coder-32B selection contains 122 heads, with selected layers spanning 28--55. The broader ablation set takes the top 250 mask values across all layers. Its depth coverage differs from the core selection. The small-head-budget sweep uses available snapshots and mask selections; these sets are not all nested prefixes of a single ranking.
'''+'\n'+fig)

# Keep old transfer prose/illustrations while replacing its requested table.
old=(OLD/'mechanistic_details.tex').read_text()
a=old.index(r'\section{Task and Query Transfer}');b=old.index(r'\section{OVReader}')
s=old[a:b]
a=s.index(r'\subsection{KV transfer by datatype}');b=s.index(r'The braced score',a)
s=s[:a]+r'''\subsection{KV transfer: aggregate and datatype results}
\begin{figure}[ht]\centering
\includegraphics[width=\linewidth]{Figures/appendix/kv_aggregate.pdf}
\caption{\textbf{KV transfer across languages.} Each language direction pools 128 pairs across four datatypes. Causal transfer counts source-or-both implementation verdicts; source-directed readout counts exclusively source-labeled patched descriptions. Both use all pairs in the denominator. Runs use 250 heads, assignment pivots, task-span padding, OV amplification 2, and no non-goal-head ablation.}
\label{fig:kv-aggregate}
\end{figure}

\begin{figure}[ht]\centering
\includegraphics[width=\linewidth]{Figures/appendix/kv_datatypes.pdf}
\caption{\textbf{KV transfer by datatype.} Breakdown of \Cref{fig:kv-aggregate}, with 32 pairs per language direction in each panel. Metrics, settings, and denominators are unchanged.}
\label{tab:kv-datatypes}
\label{fig:kv-datatypes}
\end{figure}
''' +s[b:]
s=s.replace('The braced score counts','The source-directed readout score counts').replace('match the braced scores here','match the source-directed readout scores here')
# Aggregate first, then datatype breakdown, then the qualitative illustration.
a=s.index(r'\begin{figure}[ht]\centering');b=s.index(r'\end{figure}',a)+len(r'\end{figure}')
kv_example=s[a:b];s=s[:a]+s[b:]
insert=s.index(r'\subsection{Query transfer')
s=s[:insert]+r'\subsection{A free-form KV-transfer example}'+'\n'+kv_example+'\n\n'+s[insert:]
write('task_transfer.tex',r'\clearpage'+'\n'+s)

write('ablation_sufficiency.tex',r'''
\section{Sufficiency of Goal-Head Access}
\label{app:ablation}
\label{app:sufficiency}
We test whether the model can implement the frontier when only the selected heads retain access to the frontier-description and progress span. At the final prompt token and subsequent decoding positions, heads outside the spared set in layers 20--63 cannot attend to this span. The blocked key interval starts one token before the frontier description and ends at the prompt's final assignment token. Earlier query positions and layers below 20 remain unchanged.

\begin{figure}[ht]\centering
\includegraphics[width=.70\linewidth]{Figures/author/ablation_sufficiency.pdf}
\caption{\textbf{Sufficiency intervention.} Only spared heads retain access to the green span in layers 20--63. The purple box marks the final \texttt{ =} query; the same restriction continues during generation. This illustrative prompt targets a palindrome test.}
\label{fig:ablation-setup}
\label{fig:sufficiency-setup}
\end{figure}

\begin{table}[ht]\centering\small
\input{Text/Appendix/tables/sufficiency}
\caption{Frontier implementation success on 253 specifications under span restriction. Goal and random rows differ in the set whose access is spared.}
\label{tab:goal-ablation-exact}
\end{table}

For the 122-head comparison, each condition spares 122 of the 1,760 heads in layers 20--63. For the broader selection, the goal condition spares 236 in-band heads and the random condition spares 170; selected heads below layer 20 lie outside the restricted band. Thus the broader control does not match the number of spared in-band heads. Random sets are sampled from the corresponding depth pool, without matching counts at each layer. The intervention tests access to this span, rather than isolation of goal heads from all other network computations.

\begin{table}[ht]\centering\small
\input{Text/Appendix/tables/sufficiency_example}
\caption{\textbf{An observed sufficiency example.} Exact continuations for \texttt{int\_python\_000}: digit count has been completed and the frontier is evenness. The 250-head goal condition preserves the clean implementation; the random condition implements a different computation. This example is separate from the illustrative palindrome prompt above.}
\label{tab:ablation-example}
\end{table}
\clearpage
''')
write('ablation_necessity.tex',r'''
\section{Necessity of Goal-Head Access}
\label{app:necessity}
We restrict selected heads' access at the final prompt token and all subsequent decoding positions. The sink intervention allows attention only to key position zero, denoted by generic \texttt{<bos>} in \Cref{fig:necessity-setup} (\texttt{<|im\_start|>} in Qwen). It preserves the sink value contribution and is not equivalent to zeroing a head. Two further interventions block only the frontier description, or the span from that description through the prompt's final assignment token. Earlier query positions remain unmodified.

\begin{figure}[ht]\centering
\includegraphics[width=.70\linewidth]{Figures/author/ablation_necessity.pdf}
\caption{\textbf{Necessity intervention with sink-only attention.} Selected heads cannot attend to the orange span and attend only to position zero. Purple marks the final assignment query. The same sink restriction continues throughout decoding.}
\label{fig:necessity-setup}
\end{figure}

\begin{table}[ht]\centering\small
\input{Text/Appendix/tables/necessity}
\caption{Frontier implementation success under three access restrictions ($n=253$). The clean baseline is .925. Each goal condition has a random-head control with the same total selected count, sampled from the corresponding depth pool.}
\label{tab:ablation-spans}
\end{table}

Blocking the frontier description alone has a smaller effect than blocking the description and progress span together. The sink intervention substantially reduces success for the broader 250-head set, whereas the 122-head core produces a smaller reduction. These outcomes depend on head membership, depth coverage, and the access restriction being tested.

\begin{table}[ht]\centering\small
\input{Text/Appendix/tables/necessity_example}
\caption{\textbf{An observed necessity example.} The same evenness frontier as \Cref{tab:ablation-example}, using the 250-head sink intervention. The goal-head restriction produces a malformed continuation; restricting random heads preserves the correct operation.}
\label{tab:necessity-example}
\end{table}
\clearpage
''')

write('ovreader.tex',r'''
\clearpage
\section{OVReader}
\label{app:ovreader}
\subsection{Constructing the readout direction}
Let $h_{L,t}\in\mathbb{R}^{d}$ be the residual state after source layer $L$ at position $t$. For selected head $j$, let $O_j\in\mathbb{R}^{d\times d_h}$ be its block of the attention output projection, and let $V_j\in\mathbb{R}^{d_h\times d}$ be its value projection. Query heads that share a KV head use the corresponding shared value map. Incorporating the head's pre-attention RMSNorm gain $\gamma_j$, define
\begin{equation}
B_j=V_j\operatorname{diag}(\gamma_j),\qquad A_j=O_j B_j.
\end{equation}
The normalization's scalar denominator is immaterial after normalizing each transformed vector. OVReader applies these weight maps directly to the residual at the chosen source layer, without computing a new attention distribution:
\begin{equation}
u_j(h)=\frac{A_jh}{\max(\|A_jh\|_2,\epsilon)},\qquad
g_{\mathrm{OV}}(h)=\frac{1}{|H|}\sum_{j\in H}u_j(h),\qquad \epsilon=10^{-8}.
\end{equation}
This is a weights-only readout, not the head's actual attention-weighted output. Per-head normalization also makes it different from a single linear map obtained by averaging the $A_j$. We inject $g_{\mathrm{OV}}$ with norm-relative coefficient $.15$ into probe layers 35--45 throughout decoding. The actual head-output reference uses $.1$.

\subsection{Faithfulness and causality across layers}
\Cref{fig:ov-faithfulness,fig:ov-causality} give separate sweeps for both modes at all five capture positions. Each faint curve corresponds to one datatype; the solid black curve averages the four datatype rates. The dashed reference uses actual goal-head outputs at the same pivot.

These plots use the complete archived evaluation: 64 specifications per datatype, 19 source layers, and a head-output reference. The same 256 specifications occur at every layer and pivot in both modes. This is the cohort used in the main OVReader figure, with rule-based implementation spans. The newer locked-corpus rerun is incomplete for causality, so we do not mix its records into these comparisons. The faint curves show differences across datatypes, not uncertainty over independent runs. Causality records include trivial-output sanitization.

\begin{figure}[p]\centering
\includegraphics[width=\linewidth]{Figures/appendix/ov_faithfulness.pdf}
\caption{\textbf{OVReader faithfulness at all five capture positions.} Each datatype contributes 64 specifications per point; the mean pools all 256. Source-layer indices are zero-based. The dashed line is mean faithfulness from the actual head contributions at the same position. The first implementation token is included as a comparison after implementation begins.}
\label{fig:ov-faithfulness}
\label{fig:ovreader-other-pivots}
\end{figure}

\begin{figure}[p]\centering
\includegraphics[width=\linewidth]{Figures/appendix/ov_causality.pdf}
\caption{\textbf{OVReader causality at all five capture positions.} The cohort, source layers, and aggregation match \Cref{fig:ov-faithfulness}. Scores measure oracle agreement of the C++ probe implementation with source code, rather than execution-based accuracy. The dashed line uses actual head contributions.}
\label{fig:ov-causality}
\end{figure}
\clearpage

\subsection{Alternative transformations of the residual}
The exploratory notebook compares the OV map with projections onto read and write subspaces and a leading-eigenvector subspace. With ${}^+$ denoting the Moore--Penrose pseudoinverse, the per-head transformations are:
\begin{table}[ht]\centering\small
\begin{tabular}{llp{.43\linewidth}}\toprule
Variant & Transformation of $h$ & Construction\\\midrule
OV & $O_jB_jh$ & Apply the value and output maps.\\
Write projection & $O_jO_j^+h$ & Project onto the output map's column space.\\
Read projection & $B_j^+B_jh$ & Project onto the value map's row space.\\
Eigen projection & $Q_jQ_j^\top h$ & Project onto an orthonormal basis built from leading eigenvectors.\\\bottomrule
\end{tabular}
\caption{Residual transformations compared in the saved exploratory example. Each variant normalizes per-head vectors and averages them before the same norm-relative injection.}
\label{tab:ov-variants}
\end{table}

The write and read spaces equal the column and row spaces of $A_j$ when $B_j$ has full row rank and $O_j$ has full column rank, respectively. These are projections onto the spaces, not onto their orthogonal complements. For the eigen variant, the notebook diagonalizes the smaller matrix $B_jO_j$, selects 64 eigenvectors by eigenvalue magnitude, and maps them into residual space using $O_j$. It constructs $Q_j$ by QR decomposition of their real and imaginary parts. This is the saved notebook's construction; the comparison is exploratory, not a tuned evaluation of these alternatives.

\begin{figure}[ht]\centering
\includegraphics[width=\linewidth]{Figures/slides/ov_example.pdf}
\caption{\textbf{Source prompt for the transformation comparison.} The source has counted empty and long words; at the highlighted assignment it is about to count vowel-initial words. The accompanying slide panel illustrates readouts across layers. \Cref{tab:ov-example} gives the full saved outputs for one layer.}
\label{fig:ovreader-example}
\end{figure}

\begin{table}[ht]\centering\small
\input{Text/Appendix/tables/ov_example}
\caption{\textbf{One saved comparison at layer 40 and the assignment pivot.} Full readouts from \texttt{07.1\_ov\_probe\_comparison.ipynb}; the frontier counts words starting with a vowel. OV, write, read, and eigen variants use $\alpha=.15$; the actual-head reference uses $.1$. Verdicts use the partial-content rubric, so \texttt{YES} need not recover the counting operation. This selected example is not an aggregate comparison of transformation quality.}
\label{tab:ov-example}
\end{table}
\clearpage
''')

# Extract untruncated layer-40 text from the saved notebook output.
nb=json.loads((ROOT/'notebooks/diff/07.1_ov_probe_comparison.ipynb').read_text())
outputs=[]
for o in nb['cells'][10]['outputs']:
    t=''.join(o.get('text',o.get('data',{}).get('text/plain',[])))
    outputs.append(re.sub(r'\x1b\[[0-9;]*m','',t))
(REPORT/'ov_variant_saved_outputs.txt').write_text('\n'.join(outputs))
def esc(s):
    return ''.join({'_':r'\_', '%':r'\%', '&':r'\&', '#':r'\#', '{':r'\{', '}':r'\}', '`':r'\textasciigrave{}'}.get(c,c)for c in s)
lines=[r'\begin{tabular}{lp{.64\linewidth}l}\toprule',r'Variant & Readout & Verdict\\\midrule']
for name,i in [('Actual heads',2),('OV',6),('Write projection',17),('Read projection',28),('Eigen projection',39)]:
    t=outputs[i].strip()
    if i==2:t=t.removeprefix('skyline (goalscope) ')
    else:t=re.sub(r'^L=40\s+','',t)
    verdict,t=t.split(None,1)
    # Rich truncates the NO_INFO verdict to NO_ in this display; retain it
    # verbatim rather than guessing a fuller label.
    lines.append(name+' & '+esc(' '.join(t.split()))+' & '+r'\texttt{'+esc(verdict)+'}'+r'\\')
lines.append(r'\bottomrule\end{tabular}')
(APP/'tables/ov_example.tex').write_text('\n'.join(lines))

write('mechanistic_details.tex',r'''
\input{Text/Appendix/ablation_sufficiency}
\input{Text/Appendix/ablation_necessity}
\input{Text/Appendix/task_transfer}
\input{Text/Appendix/ovreader}
''')

# Existing dissociation figures are available; do not replace them with blanks.
for name in ['goal_vs_decision_overlap','goal_vs_decision_moveon']:
    shutil.copy2(ROOT/'results/dissociation'/ 'Qwen2.5-Coder-32B-Instruct'/f'{name}.png',PAPER/'Figures/appendix'/f'{name}.png')
write('dissociation.tex',r'''
\clearpage
\section{Checking for Functional Dissociation}
\label{app:dissociation}
\subsection{Goal heads and impulse heads}
The saved exploratory comparison uses 122 goal heads and 181 decision/impulse heads in Qwen2.5-Coder-32B. The sets share 36 heads (Jaccard overlap .135). Membership overlap alone does not establish whether the sets perform the same function.

\begin{figure}[ht]\centering
\includegraphics[width=.9\linewidth]{Figures/appendix/goal_vs_decision_overlap.png}
\caption{\textbf{Goal-head and impulse-head membership.} The saved head sets have partial overlap, concentrated in the later layers. Colors distinguish goal-only, impulse-only, and shared heads.}
\label{fig:goal-impulse-overlap}
\end{figure}

The accompanying intervention transfers a mean ``move-on'' signal into either set at the notebook's fixed pivot (relative position $-2$), using source coefficient 2. The evaluation uses 256 multiline prompts with five conditions and letter-named variables. One metric asks whether the full-vocabulary next-token argmax names the next condition; the other asks whether that token has higher probability than the current condition's token. These are next-token measurements, not GoalScope oracle scores or complete-program correctness.

\begin{figure}[ht]\centering
\includegraphics[width=\linewidth]{Figures/appendix/goal_vs_decision_moveon.png}
\caption{\textbf{Transferring a move-on signal through different head sets.} Existing results from \texttt{notebooks/clock/11\_goal\_vs\_impulse.ipynb}. Impulse-head intervention increases the move-on rate more than goal-head intervention in this setup. The sets differ in size and depth coverage, so this comparison does not isolate head identity at a matched budget.}
\label{fig:goal-impulse-moveon}
\end{figure}
\clearpage
''')
write('cpp.tex',r'''
\section{Generalization across Programming Styles and Languages}
\label{app:cpp}
The cross-language KV-transfer evaluation is in \Cref{app:task_transfer}, with aggregate and datatype results in \Cref{fig:kv-aggregate,fig:kv-datatypes}. It covers Python and C++ in both source and destination roles. The free-form examples in \Cref{app:unspecified} provide a separate qualitative check beyond the fixed multi-predicate template.
''')
write('mlp.tex',r'''
\section{Role of MLPs}
\label{app:mlp}
\arnab{{\color{red} TODO:} path patching experiments: patch + freeze MLPs with dest values. Measure intervention effects.}
\begin{figure}[ht]\centering
\fbox{\begin{minipage}[c][1.2in][c]{.88\linewidth}\centering
\textbf{Placeholder: MLP path-patching results}\par\medskip
Compare the goal-head patch with the same patch while restoring MLP outputs to their clean destination values.\par\smallskip
Needed: frozen layers and positions, intervention effects, and matched controls.
\end{minipage}}
\caption{Reserved for the requested MLP path-patching experiment. A corresponding result figure has not yet been located; this panel reports no measurements.}
\label{fig:mlp-path-patching-placeholder}
\end{figure}
\clearpage
''')
print('Updated appendix files only. Main-paper sources remain untouched.')
