"""Populate figures while checking that main-paper prose is unchanged."""
from pathlib import Path
import re
import json

ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
PAPER = ROOT/'Overleaf-draft/Goals-Overleaf'
BEFORE = REPORT/'before'

def write(name, text):
    (PAPER/name).write_text(text)

def base(name):
    return (BEFORE/name).read_text()

def replace_float(text, kind, replacement):
    pattern = rf'\\begin\{{{kind}\}}.*?\\end\{{{kind}\}}'
    result, count = re.subn(pattern, lambda _: replacement, text, count=1, flags=re.S)
    assert count == 1
    return result

intro = base('Text/01_intro.tex').replace('\\label{sec:intro}', '\\label{sec:intro}\n\n\\input{Text/Figures/teaser}', 1)
write('Text/01_intro.tex', intro)
write('Text/03_what.tex', replace_float(base('Text/03_what.tex'), 'figure', r'\input{Text/Figures/goal_eval}'))
heads = replace_float(base('Text/04_heads.tex'), 'figure', r'\input{Text/Figures/sparsity_ablation}')
write('Text/04_heads.tex', replace_float(heads, 'table', ''))
mechanism = replace_float(base('Text/06_mechanism.tex'), 'figure', r'\input{Text/Figures/ovreader}')
mechanism = mechanism.replace('\\label{sec:steer:kv-transfer}', '\\label{sec:steer:kv-transfer}\n\n\\input{Text/Figures/kv_transfer}', 1)
write('Text/06_mechanism.tex', mechanism)

# Keep the original order within each topical group; put methodological material
# before exploratory generalization, and bound appendix float drift by section.
sections = ['task','attn_head_localization','goalscope','oracle_prompts',
            'readout_comparisons','mechanistic_details','gaze','olmo',
            'unspecified','non-code','boundary','cpp','dissociation','mlp']
breaks = {'task','attn_head_localization',
          'mechanistic_details','olmo','unspecified','non-code','boundary'}
write('Text/201_Appendix.tex', '\\appendix\n\\crefalias{section}{appendix}\n\n' +
      '\n'.join(('\\clearpage\n' if s in breaks else '')+f'\\input{{Text/Appendix/{s}}}' for s in sections)+'\n')

localization = base('Text/Appendix/attn_head_localization.tex')
localization += r'''
\begin{figure}[ht]
\centering
\includegraphics[width=\linewidth]{Figures/author/localization_heads.pdf}
\caption{\textbf{Locating the goal heads.} Left: source and destination prompts request different predicates in different programming languages. A separate target prompt requests the source predicate in the destination language. The two destination-language implementations share a prefix and differ in the next token. We patch source head contributions into the destination using the mask-weighted intervention in \Cref{eqn:opt-int}, and jointly train the masks to promote the target continuation. Here, Python's target token \texttt{[} differs from the source continuation \texttt{word}, discouraging direct token copying. Right: the learned mask and the selected 122 heads in Qwen2.5-Coder-32B. The target prompt determines the optimization target; it is not supplied to the patched destination.}
\label{fig:localization}
\end{figure}
'''
write('Text/Appendix/attn_head_localization.tex',localization)

scope = base('Text/Appendix/goalscope.tex')
scope = scope.replace('\\subsection{Aggregation and probe generation}', r'''
\begin{figure}[ht]\centering
\includegraphics[width=.92\linewidth]{Figures/author/evaluation_setup.pdf}
\caption{\textbf{Capture positions and the two probes, enlarged.} Pink marks the previous-line end, purple indentation, yellow the variable, light green the assignment operator, and dark green the first implementation token. The orange and green arrows illustrate faithfulness and causality readouts from the variable and assignment positions, respectively. In the evaluation, each probe is run separately at all five positions. Only source activations up to the selected pivot enter its cached vector.}
\label{fig:pivot-probes-detail}
\end{figure}

\subsection{Aggregation and probe generation}''')
write('Text/Appendix/goalscope.tex',scope)

task = base('Text/Appendix/task.tex')
task = task.replace('\\subsection{Exact source prompt and capture positions}',r'''
\begin{figure}[ht]\centering
\includegraphics[width=\linewidth]{Figures/author/task_prompt.pdf}
\caption{\textbf{Illustrative task prompt.} The assistant response is prefilled up to the target frontier (blue), here a palindrome check after character and uppercase counts. Generic \texttt{<system>}, \texttt{<user>}, and \texttt{<assistant>} tags stand in for the model family's chat tokens. The input ends at the final assignment operator. This illustrative prompt is distinct from the exact saved evaluation example below.}
\label{fig:task-prompt}
\end{figure}

\subsection{Exact source prompt and capture positions}''')
write('Text/Appendix/task.tex',task)

details = base('Text/Appendix/mechanistic_details.tex')
ablation_table = re.search(r'\\begin\{table\}.*?\\end\{table\}', base('Text/04_heads.tex'), re.S).group()
ablation_table = ablation_table.replace('{tab:goal-ablation}', '{tab:goal-ablation-exact}').replace('[t]', '[ht]')
ablation_add = r'''
\begin{figure}[p]
\centering
\textbf{(a) Sufficiency: spare goal-head access}\par\smallskip
\includegraphics[width=.70\linewidth]{Figures/author/ablation_sufficiency.pdf}\par\medskip
\textbf{(b) Necessity: restrict goal heads to position zero}\par\smallskip
\includegraphics[width=.70\linewidth]{Figures/author/ablation_necessity.pdf}
\caption{\textbf{Ablation setups.} (a) Only spared heads can attend to the green span; other heads in layers 20--63 are blocked from it. (b) Selected heads cannot attend to the orange span and instead attend to position zero, denoted by generic \texttt{<bos>} (\texttt{<|im\_start|>} in Qwen). Purple boxes mark the final \texttt{ =} query. The restriction continues at subsequent generation positions; earlier query rows are unchanged. Generic role tags stand in for model-specific delimiters.}
\label{fig:ablation-setup}
\end{figure}
'''
ablation_add += ablation_table + r'''

For the 122-head sufficiency run, both goal and random conditions block 1,638 heads in layers 20--63, leaving 122 with access in that band. For the broader run, the saved configurations block 1,524 heads in the goal condition and 1,590 in the random condition, leaving 236 and 170, respectively. Thus the broader sufficiency control does not exactly match the number of spared heads inside the intervention band. The sink comparisons use the stated selected-set sizes.

\begin{table}[ht]\centering\small
\begin{tabular}{lrrrr}\toprule
 & \multicolumn{2}{c}{122 heads} & \multicolumn{2}{c}{250 heads}\\
Blocked keys in selected heads & Goal & Random & Goal & Random\\\midrule
Task descriptions & .881 & .925 & .806 & .925\\
Task descriptions and progress & .372 & .925 & .249 & .925\\
All except sink & .854 & .925 & .158 & .921\\\bottomrule
\end{tabular}
\caption{Additional necessity interventions from the saved 253-specification evaluation. The clean baseline is .925 throughout. Restricting task descriptions alone has a smaller effect than restricting task and progress tokens together. These are access interventions at the current and subsequent query positions.}
\label{tab:ablation-spans}
\end{table}

\begin{table}[!ht]\centering\small
\begin{tabular}{lll}\toprule
Condition & Continuation after \texttt{b =} & Oracle verdict\\\midrule
Clean & \texttt{(n \% 2 == 0)} & YES\\
Spare goal-head access & \texttt{(n \% 2 == 0)} & YES\\
Spare random-head access & \texttt{(n * (n + 1)) // 2} & NO\\
Restrict goal heads to sink & \texttt{nn\textasciigrave} & MALFORMED\\
Restrict random heads to sink & \texttt{(n \% 2 == 0)} & YES\\\bottomrule
\end{tabular}
\caption{\textbf{An observed effect of ablation.} Exact saved continuations for \texttt{int\_python\_000}, whose prompt appears in \Cref{app:task}. The frontier is evenness after a completed digit-count operation. All intervention rows use the broader 250-head configuration. Goal-head access preserves the requested operation under span restriction; sink restriction yields malformed output. This selected example illustrates the aggregate results, rather than serving as a separate evaluation.}
\label{tab:ablation-example}
\end{table}

\clearpage
'''
# Keep the observed continuation alongside its setup on the same appendix page.
example_match = re.search(r'\\begin\{table\}\[!ht\].*?\\label\{tab:ablation-example\}.*?\\end\{table\}', ablation_add, re.S)
assert example_match
example = example_match.group()
ablation_add = ablation_add.replace(example,'')
example = example.replace(r'\begin{table}[!ht]',r'\begin{minipage}{\linewidth}').replace(r'\end{table}',r'\end{minipage}').replace(r'\caption{',r'\captionof{table}{')
example = re.sub(r'\\captionof\{table\}\{.*?\}\n\\label',lambda _: r'\captionof{table}{\textbf{An observed effect of ablation.} Exact saved continuations for \texttt{int\_python\_000}, whose prompt appears in \Cref{app:task}. The frontier is evenness after digit count. Intervention rows use the broader 250-head configuration. This selected example illustrates the aggregate results.}'+'\n'+r'\label',example,flags=re.S)
ablation_add = ablation_add.replace(r'\label{fig:ablation-setup}'+'\n'+r'\end{figure}',r'\label{fig:ablation-setup}'+'\n\\par\\medskip\n'+example+'\n'+r'\end{figure}')
setup_figure = re.search(r'\\begin\{figure\}\[p\].*?\\end\{figure\}',ablation_add,re.S).group()
ablation_add = ablation_add.replace(setup_figure,'').replace(r'\clearpage',setup_figure+'\n'+r'\clearpage',1)
details = details.replace('\\subsection{Task and query transfer}', ablation_add+'\n\\section{Task and Query Transfer}')
details = details.replace('The qualitative task-order example in the main text follows the saved figure with unequal source/destination progress.', 'The qualitative task-order example in \\Cref{fig:query-order} uses unequal source/destination progress.')
kv_tables = (REPORT/'kv_datatype_tables.tex').read_text()
details = details.replace('\\section{OVReader}', r'''
\begin{figure}[ht]\centering
\includegraphics[width=\linewidth]{Figures/author/kv_freeform.pdf}
\caption{\textbf{KV transfer after a shared initial sub-task.} The source and destination both sort the words by length (blue), then pursue different tasks: a vowel test on the shortest word versus a palindrome test on the longest word. Replacing the destination task-span keys and values available to the goal heads with those from the source changes the continuation to a vowel test on the shortest word. The readout shifts accordingly. Source, native destination, and patched continuations are red, green, and purple; yellow marks their pivots. This is an illustrative slide example, separate from the scored multi-predicate cohort. In the evaluated implementation, the fixed source span is routed to selected query heads at every forward pass; other heads retain destination KV access.}
\label{fig:kv-freeform}
\end{figure}

\subsection{KV transfer by datatype}
\begin{table}[ht]\centering\small
'''+kv_tables+r'''
\caption{Datatype breakdown of \Cref{tab:kv-transfer}: causal success \{source readout\}, in percent. Each cell has 32 examples, using 250 heads, assignment pivots, task-span padding, OV amplification 2, and no collateral ablation.}
\label{tab:kv-datatypes}
\end{table}
The braced score counts patched descriptions judged exclusively source-directed, with all examples in the denominator. The stored \texttt{valid} flag asks whether the original source and destination readouts match their respective frontiers, allowing a both-goal verdict; its rates are 99.2\%, 96.9\%, 96.1\%, and 94.5\% for Python--Python, Python--C++, C++--Python, and C++--C++, respectively. The stored \texttt{shifted} flag counts patched source-only or both-goal readouts, without requiring either clean readout to be correct. Its unconditional rates happen to match the braced scores here because none of these patched readouts is labeled both. The evaluation script also reports shift conditional on \texttt{valid}; that denominator differs from the unconditional scores shown here.

\subsection{Query transfer under a task-order contrast}
Let $A$ count characters, $B$ count uppercase letters, $C$ test whether any word is a palindrome, and $D$ count vowel-initial words. The source orders $(A,B,C,D)$ and has completed $(A,B)$; the destination orders $(A,B,D,C)$ and has completed only $A$. Its native next task is therefore $B$. A transferred state encoding only completed tasks $(A,B)$ would make $D$ next under destination order, whereas the source's next goal is $C$. The slide example shows a query-patched continuation implementing $C$, accompanied by a palindrome readout. This is a qualitative contrast, not a factorial evaluation of progress and goal identity.

\begin{figure}[ht]
\centering
\begin{minipage}[c]{.50\linewidth}
\centering\includegraphics[width=\linewidth]{Figures/slides/query_source_destination.pdf}
\end{minipage}\hfill
\begin{minipage}[c]{.47\linewidth}
\small\textbf{Source (upper prompt):} completed $A,B$; next $C$.\par\medskip
\textbf{Destination (lower prompt):} completed $A$; native next $B$.\par\medskip
\textbf{Query patch:} transfer the source pivot queries to destination goal heads.\par\medskip
Completed-task prediction: $D$.\par
Observed patched goal: $C$.
\end{minipage}\par\smallskip
\includegraphics[width=\linewidth]{Figures/slides/query_result.pdf}
\caption{\textbf{A query-transfer contrast from the exploratory slides.} The source and destination contain the same four tasks in different orders and have unequal prefills. The patched destination computes whether any word is a palindrome, matching the source's upcoming goal. Ellipses in the prompt panels mark the capture/patch boundary; the saved readout identifies position 141. This example does not establish that query states contain only goal information.}
\label{fig:query-order}
\end{figure}

\clearpage
\section{OVReader}''')
details += r'''

\begin{figure}[ht]\centering
\includegraphics[width=\linewidth]{Figures/ov_reader_other_pivots.pdf}
\caption{\textbf{OVReader at the remaining capture positions.} The cohort and aggregation match \Cref{fig:ovreader}: means and $\pm1$ sample standard deviation across four datatype rates, with 64 examples each. Dashed horizontal lines show the actual head-output reference for the same position. The first implementation token is included as a comparison after implementation begins.}
\label{fig:ovreader-other-pivots}
\end{figure}

The plots use the complete archived evaluation with rule-based frontier spans, rather than the incomplete frontier-span rerun. They contain 19 source-layer settings and a head-output reference, with 64 examples per datatype at every position and in both evaluation modes. The datatype bands describe variation across four task families; they are not confidence intervals over independent runs.

\begin{figure}[ht]\centering
\includegraphics[width=\linewidth]{Figures/slides/ov_example.pdf}
\caption{\textbf{Qualitative projection readouts from the exploratory slides.} The source has counted empty and long words; at the highlighted assignment it is about to count words beginning with a vowel. The slide compares an OV-transformed state with alternative write/read and eigenvector-based directions. These individual descriptions illustrate the experiment and are not additional aggregate evidence for the layer sweep.}
\label{fig:ovreader-example}
\end{figure}
'''
write('Text/Appendix/mechanistic_details.tex', details)

write('Text/Appendix/gaze.tex', r'''\section{Attention to Task Descriptions}
\label{app:gaze}
The query-transfer example in \Cref{fig:query-order} also provides a qualitative view of attention to the destination task list. Its native frontier is the uppercase-count task; after transferring source queries, the continuation and readout instead pursue the palindrome predicate. The slide highlights both descriptions in the task list. This visualization illustrates the association between the heads' access to task tokens and the operation expressed by the continuation.

\begin{figure}[ht]\centering
\includegraphics[width=\linewidth]{Figures/slides/query_gaze.pdf}
\caption{Task-description highlights accompanying the query-order experiment. The destination's native uppercase-count frontier is highlighted in red and the transferred palindrome goal in green. The slide supplies no calibrated color scale; this panel should be read as a qualitative visualization rather than a quantitative attention-mass comparison.}
\label{fig:query-gaze}
\end{figure}

A quantitative account requires a specified attention statistic, token-span definition, and aggregation across heads and examples. We leave comparison with independently localized gaze or concept-copying heads to a separate evaluation.
''')

write('Text/Appendix/unspecified.tex', r'''\section{Readouts without Enumerated Sub-Tasks}
\label{app:unspecified}
The controlled evaluation names the component predicates in the instruction. The examples below instead read intermediate operations from familiar programs whose component steps are not separately enumerated. They are qualitative illustrations from the exploratory slides, not part of the 253-specification evaluation.

\begin{figure}[ht]\centering
\includegraphics[width=\linewidth]{Figures/author/freeform_matching.pdf}
\caption{\textbf{Reading intent while writing code.} At the marked positions, GoalScope reads descriptions of the operations the model is about to implement: (a) checking for an unmatched man, (b) finding the next free man, and (c) proposing to the next woman on his preference list. These descriptions express an operation rather than a particular token sequence; for example, finding an unmatched man can be implemented with different loop forms. The figure illustrates this distinction and does not itself compare readouts across alternative implementations. Later code is shown for context.}
\label{fig:freeform-matching}
\end{figure}

\begin{figure}[ht]\centering
\includegraphics[width=.64\linewidth]{Figures/slides/freeform_atoi.pdf}
\caption{\textbf{A free-form string-to-integer example.} The highlighted source position is the indentation before \texttt{s = s.strip()}. A separate generic probe produces the displayed description about leading whitespace and signs. The later program is shown for context, but a cache at the highlighted token cannot depend on that future text. The description is a qualitative readout and need not match the exact granularity of the immediately following statement.}
\label{fig:freeform-atoi}
\end{figure}
''')

noncode = base('Text/Appendix/non-code.tex')
noncode += r'''
\begin{figure}[ht]\centering
\begin{minipage}[t]{.47\linewidth}
\centering\includegraphics[width=\linewidth]{Figures/slides/noncode_music.pdf}
\end{minipage}\hfill
\begin{minipage}[t]{.50\linewidth}
\centering\includegraphics[width=\linewidth]{Figures/slides/noncode_antonym.pdf}
\end{minipage}
\caption{\textbf{Reading a relation from few-shot examples.} The slide shows musician--instrument and antonym prompts, with capture positions highlighted on the answer prefix \texttt{A}. GoalScope describes the inferred relations. Green text denotes generated continuation shown for context. These selected examples are exploratory and are distinct from the token-indexed antonym run above.}
\label{fig:noncode-examples}
\end{figure}
'''
write('Text/Appendix/non-code.tex',noncode)

boundary = base('Text/Appendix/boundary.tex')
boundary += r'''
\begin{figure}[ht]\centering
\begin{minipage}[c]{.51\linewidth}
\centering\includegraphics[width=\linewidth]{Figures/slides/boundary_forward.pdf}
\end{minipage}\hfill
\begin{minipage}[c]{.46\linewidth}
\centering\includegraphics[width=\linewidth]{Figures/slides/boundary_reverse.pdf}
\end{minipage}
\caption{\textbf{A boundary intervention under different task orders.} Left: capture at the highlighted completed source expression and patch at the highlighted unfinished divisibility expression in the destination. The green continuation is the native completion of that expression; red text shows the patched run advancing to a new variable. Right: under reversed instruction order, the patched continuation advances to the next task in that order. These are selected examples; advancing to a variable does not certify that the preceding or following predicate is implemented correctly.}
\label{fig:boundary-examples}
\end{figure}
'''
write('Text/Appendix/boundary.tex',boundary)

def prose(text):
    for kind in ['figure','table']:
        text = re.sub(rf'\\begin\{{{kind}\}}.*?\\end\{{{kind}\}}','',text,flags=re.S)
    text = re.sub(r'\\input\{Text/Figures/[^}]+\}', '', text)
    return [line for line in text.splitlines() if line.strip()]

checks = {}
for before in (BEFORE/'Text').glob('*.tex'):
    if before.name=='201_Appendix.tex':
        continue
    after = PAPER/'Text'/before.name
    checks[before.name] = prose(before.read_text()) == prose(after.read_text())
assert all(checks.values()), checks
assert base('main.tex') == (PAPER/'main.tex').read_text()
assert base('packages.tex') == (PAPER/'packages.tex').read_text()
(REPORT/'main_prose_check.json').write_text(json.dumps(checks,indent=2))
print('Main-paper prose and non-float comments unchanged:', len(checks),'files')
