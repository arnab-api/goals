"""Populate appendix from locked data, saved prompts and within-model results."""
from pathlib import Path
import ast,json,hashlib,csv,os
from collections import Counter
os.environ.setdefault('MPLCONFIGDIR','/tmp/goalscope-paper-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
O=Path(__file__).resolve().parent;ROOT=O.parents[2]
R=ROOT/'Overleaf-draft/Goals-Overleaf';A=R/'Text/Appendix';P=A/'prompts'
DATA=ROOT/'results/evaluation/goalscope';MODEL='Qwen2.5-Coder-32B-Instruct'
def escape(s):
 return ''.join({'_':r'\_','%':r'\%','&':r'\&','#':r'\#','{':r'\{','}':r'\}','$':r'\$'}.get(c,c) for c in str(s))
def write(p,s):(p).write_text(s.strip()+'\n')
manifest=[]
def load(p):
 b=p.read_bytes();manifest.append({'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(b).hexdigest()});return json.loads(b)
def jl(p):
 b=p.read_bytes();manifest.append({'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(b).hexdigest()});return [json.loads(s) for s in b.splitlines() if s.strip()]
def inp(name):return '\\VerbatimInput[fontsize=\\footnotesize,breaklines=true,breakanywhere=true,frame=single]{Text/Appendix/prompts/'+name+'}\n'
tree=ast.parse((ROOT/'scripts/evaluation/eval_goalscope.py').read_text());constants={}
for node in tree.body:
 if isinstance(node,ast.Assign):
  for t in node.targets:
   if isinstance(t,ast.Name) and t.id in ['FAITH_HARD_SYS','CAUSAL_HARD_SYS','CAUSAL_INSTRUCTION','CAUSAL_PREFIX','ORACLE_MODEL']:
    constants[t.id]=ast.literal_eval(node.value)
anchors=['prev_end','indent','var','eq','first_impl']
baseline={}
for mode in ['faithfulness','causality']:
 p=next((DATA/MODEL/mode/'additive_norm/0.1').glob('heads=goal__*/config.json'))
 c=load(p);rr=jl(p.parent/'records.jsonl');ss=jl(p.parent/'samples.jsonl');baseline[mode]=(c,rr,ss)
 sys=constants['FAITH_HARD_SYS' if mode=='faithfulness' else 'CAUSAL_HARD_SYS']
 assert c['hard_rubric_tag']==hashlib.md5(sys.encode()).hexdigest()[:8]
 (P/f'{mode}_oracle_system.txt').write_text(sys)
 rec=next(r for r in rr if r['spec']=='int_python_000' and r['anchor']=='eq')
 assert rec['hard_prompt']
 (P/f'{mode}_oracle_example.txt').write_text(rec['hard_prompt'])
 (O/f'{mode}_example_record.json').write_text(json.dumps(rec,indent=2))
s=next(x for x in baseline['faithfulness'][2] if x['spec_id']=='int_python_000')
(P/'source_exact.txt').write_text(s['prompt'])
(P/'source_completion.txt').write_text(s['completion'])
(P/'faithfulness_oracle_user_template.txt').write_text('Code: {impl_span}\nDescription: {readout.strip()}\nAnswer:')
(P/'causality_oracle_user_template.txt').write_text('Python (gold): {impl_span}\nC++ (read-out): {readout.strip()}\nAnswer:')
(P/'causality_probe.txt').write_text('System:\nYou are a helpful assistant proficient in programming.\n\nUser:\n'+constants['CAUSAL_INSTRUCTION']+'\n\nAssistant (prefill):\n'+constants['CAUSAL_PREFIX'])
d=load(ROOT/'data/goalscope_eval.json');groups=d['datapoints'];allrows=[x for xs in groups.values() for x in xs]
assert len(allrows)==253
assert {x['spec_id'] for x in allrows}=={x['spec_id'] for x in baseline['faithfulness'][2]}
dist=[]
for typ,rs in groups.items():
 counts=Counter(x['frontier_id'] for x in rs);T=Counter(x['T'] for x in rs);j=Counter(x['j'] for x in rs)
 dist.append(dict(datatype=typ,specifications=len(rs),frontiers=len(counts),per_frontier=sorted(set(counts.values())),T=dict(T),j=dict(j),frontier_counts=dict(counts)))
with (O/'distribution.json').open('w') as f:json.dump(dist,f,indent=2)
task=r'''\section{Evaluation Corpus and Exact Prompt}
\label{app:task}
\subsection{Locked prefill-to-frontier corpus}
The main GoalScope evaluation uses \texttt{data/goalscope\_eval.json}, generated with seed 123 from the QTASKS predicate bank. It contains 253 Python specifications. Each specifies two, three, or four predicates and prefills one or more completed predicates before the frontier assignment. The function and intermediate-variable naming conventions are fixed; the exact example below uses \texttt{func} and successive letter variables. Sampling balances the frontier predicate within each datatype, rather than making the four datatype counts identical. The number of completed predicates is between one and three and always less than the total number of predicates.

\begin{table}[ht]\centering\small
\begin{tabular}{lrrrrrrr}\toprule
Datatype & Specs & Frontiers & Repeats & $T=2$ & $T=3$ & $T=4$ & Records/mode\\\midrule
'''
for x in dist:task+=f"{escape(x['datatype'])} & {x['specifications']} & {x['frontiers']} & {x['per_frontier'][0]} & {x['T'].get(2,0)} & {x['T'].get(3,0)} & {x['T'].get(4,0)} & {5*x['specifications']} \\\\\n"
task+=r'''\midrule Total & 253 & 57 & --- & 92 & 82 & 79 & 1265\\\bottomrule
\end{tabular}
\caption{Distribution of the locked evaluation corpus. $T$ is the number of predicates in the specification; repeats count distinct prefill contexts per frontier. Frontier identities are counted within datatype. Each specification contributes five capture positions per evaluation mode.}
\label{tab:data-distribution}
\end{table}

\begin{table}[ht]\centering\small
\begin{tabular}{lrrr}\toprule
Datatype & 1 completed & 2 completed & 3 completed\\\midrule
'''
for x in dist:task+=f"{escape(x['datatype'])} & {x['j'].get(1,0)} & {x['j'].get(2,0)} & {x['j'].get(3,0)} \\\\\n"
task+=r'''\midrule Total & 151 & 73 & 29\\\bottomrule\end{tabular}
\caption{Number of completed predicates in the prefilled context. These counts are specification-level, not token-position-level.}
\end{table}

The frontier bank excludes the integer predicates \texttt{is\_negative}, \texttt{is\_positive}, and \texttt{sign}, and the integer-list predicates \texttt{all\_positive}, \texttt{contains\_zero}, \texttt{count\_negative}, and \texttt{count\_positive}. This exclusion applies to measured frontier targets; it does not forbid such predicates in completed context. The following lists give every retained frontier ID; each occurs the number of times specified in \Cref{tab:data-distribution}.
'''
for x in dist:
 task+='\n\\paragraph{'+escape(x['datatype'])+' ('+str(x['per_frontier'][0])+' contexts per frontier).}\n'
 task+=', '.join('\\texttt{'+escape(k)+'}' for k in sorted(x['frontier_counts']))+'.\n'
task+=r'''
\subsection{Exact source prompt and capture positions}
The following is the complete saved prompt for \texttt{int\_python\_000} in the Qwen2.5-Coder-32B evaluation, including chat delimiters and the assistant prefill. The input ends immediately after the final \texttt{=} without an added newline. Visual line wrapping in the listing is typographic; the accompanying text file preserves the exact bytes.
'''+inp('source_exact.txt')+r'''
The first predicate, digit count, is already implemented; the frontier is evenness. The saved capture indices are 102 (previous-line end), 103 (indentation), 104 (variable), 105 (assignment operator), and 106 (first implementation token). They are indices in this model's tokenization, not universal character offsets. The saved clean continuation is:
'''+inp('source_completion.txt')+r'''
The gold implementation span used by the judge is \texttt{b = (n \% 2 == 0)}. More generally, the runner extracts the actual generated frontier assignment as \texttt{impl\_span}; it does not use the natural-language predicate description as the oracle's CODE field. This tests agreement with the generated operation and must be distinguished from clean correctness against the requested predicate. The controlled frontier identifies the assignment directly, without an oracle segmentation call.

Five capture positions yield 1,265 records per mode and model, with repeated observations of the same 253 specifications. Rates pool records unless a position or datatype is explicitly selected. This corpus is separate from the 64-per-datatype OVReader sampling and the source/destination-pair corpora used for Q/KV transfer; their counts should not be combined.
'''
write(A/'task.tex',task)

oracle=r'''\section{Exact Oracle Prompts and Scoring}
\label{app:oracle_prompts}
The evaluation calls \texttt{claude-sonnet-4-6} with a system rubric and a user message, a maximum response budget of 512 tokens, and caching enabled. The complete system prompts below are extracted verbatim from the runner. Their MD5-prefix rubric identifiers, \texttt{78bfd5f4} (faithfulness) and \texttt{9c106246} (causality), match the saved configurations used here. The code's mode name \emph{causality} denotes the implementation-transfer metric in the main text. Spelling, punctuation, and lenient scoring language are preserved as used, rather than edited for exposition.

\subsection{Faithfulness: system message}
'''+inp('faithfulness_oracle_system.txt')+r'''
\paragraph{User-message template.}
Braced fields below denote runtime substitutions, not literal strings sent to the oracle. \texttt{impl\_span} is the extracted Python frontier assignment; \texttt{readout.strip()} removes leading and trailing whitespace from the generated probe readout.
'''+inp('faithfulness_oracle_user_template.txt')+r'''
\paragraph{Exact instantiated example.}
This user message is copied from the saved assignment-pivot record for \texttt{int\_python\_000} in the 122-head norm-relative additive run.
'''+inp('faithfulness_oracle_example.txt')+r'''
\subsection{Causality / implementation transfer: system message}
'''+inp('causality_oracle_system.txt')+r'''
\paragraph{User-message template.}
The C++ \texttt{readout} includes the fixed assistant prefill \texttt{bool check(auto value)\{} concatenated with the generated continuation before grading.
'''+inp('causality_oracle_user_template.txt')+r'''
\paragraph{Exact instantiated example.}
The following is the saved user message for the same specification and assignment pivot in the implementation-transfer run.
'''+inp('causality_oracle_example.txt')+r'''
\subsection{Generation prompt for the implementation probe}
This is the role-separated probe instruction and assistant prefill, before the model-specific chat template is applied. It is distinct from the judging prompts above.
'''+inp('causality_probe.txt')+r'''
\subsection{Verdict parsing and deterministic exclusions}
The parser first looks for a verdict inside \texttt{<verdict>} tags and otherwise uses the last recognized bare verdict. Unrecognized responses become \texttt{UNPARSED}. The reported hard-YES rate counts only \texttt{YES} in the numerator and all records in the denominator; \texttt{NO}, \texttt{NO\_INFO}, \texttt{MALFORMED}, \texttt{HEAVY\_LOOP}, and \texttt{UNPARSED} are not successes. These rubrics explicitly permit partial content and wrong constants or thresholds. Consequently, their rates are not exact predicate accuracy or executable-program accuracy.

Before calling the oracle, empty readouts become \texttt{MALFORMED}. In the causality mode, a deterministic sanitizer assigns \texttt{NO} to continuations containing only the generic function scaffolding and sign/zero comparisons. The implementation removes the recognized scaffolding words, logical/comparison operators, standalone zero literals, and punctuation; a readout is trivial if nothing remains. Genuine arithmetic or bitwise operations and nonzero constants survive this test. Such short-circuited records have no oracle user message. This readout exclusion is distinct from the frontier exclusions in \Cref{app:task}.
'''
write(A/'oracle_prompts.tex',oracle)

# Recompute cross-model rates using the intended head-budget-matched OLMo selections.
chosen=[('Qwen2.5-Coder-32B-Instruct',122),('Qwen2.5-Coder-14B-Instruct',88),('Qwen2.5-32B-Instruct',108),('Olmo-3.1-32B-Instruct',93),('Olmo-3.1-32B-Instruct-SFT',91)]
stats=[];configs=[];keyset=None
for model,k in chosen:
 for mode in ['faithfulness','causality']:
  paths=[]
  for p in (DATA/model/mode/'replace/1').rglob('config.json'):
   c=json.loads(p.read_text())
   if c.get('n_specs')==253 and c.get('n_heads')==k and c.get('span_source')=='frontier':paths.append(p)
  assert len(paths)==1,(model,mode,paths)
  p=paths[0];c=load(p);rr=jl(p.parent/'records.jsonl')
  keys=[(x['spec'],x['anchor']) for x in rr];assert len(keys)==len(set(keys))==1265
  if keyset is None:keyset=set(keys)
  assert set(keys)==keyset
  sys=constants['FAITH_HARD_SYS' if mode=='faithfulness' else 'CAUSAL_HARD_SYS']
  assert c['hard_rubric_tag']==hashlib.md5(sys.encode()).hexdigest()[:8]
  configs.append(c)
  for anchor in anchors+['overall']:
   sub=[x for x in rr if anchor=='overall' or x['anchor']==anchor]
   stats.append(dict(model=model,heads=k,mode=mode,pivot=anchor,n=len(sub),yes=sum(x['hard']=='YES' for x in sub),rate=sum(x['hard']=='YES' for x in sub)/len(sub)))
with (O/'model_rates.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(stats[0]));w.writeheader();w.writerows(stats)
def rate(model,mode,p='overall'):return next(x['rate'] for x in stats if x['model']==model and x['mode']==mode and x['pivot']==p)
names=['Qwen Coder 32B','Qwen Coder 14B','Qwen Instruct 32B','OLMo Instruct 32B (+RL)','OLMo Instruct 32B (SFT)']
text=r'''\section{Generalization Across Model Families and Sizes}
\label{app:olmo}
\label{app:model_comparison}
\label{sec:steer:crossmodel}
We repeat localization and evaluation within five models: Qwen2.5-Coder at 14B and 32B parameters, Qwen2.5-32B-Instruct, and OLMo-3.1-32B instruction and SFT variants. Each model uses its own selected heads and its own probe generation. This tests replication of readable and transferable task information, not activation transplantation between models.

\subsection{Matched evaluation and head selection}
All ten runs use the same 253 specification identifiers and five capture positions, yielding 1,265 records per model and mode. Rates below are recomputed from saved records. The intervention is head replacement with source coefficient 1 and steering throughout decoding; the probe generation budget is 64 tokens. Replacement operates at the selected heads' own layers and ignores the additive steer-band setting. Thus the results are not directly interchangeable with the main norm-relative additive scores.

\begin{table}[ht]\centering\small
\begin{tabular}{lrrcc}\toprule
Model & Heads & Selected layer range & Faithfulness & Transfer\\\midrule
'''
for (model,k),name in zip(chosen,names):
 c=next(c for c in configs if c['model'].split('/')[-1]==model);layers=[x[0] for x in c['selected_heads']]
 text+=f"{name} & {k} & {min(layers)}--{max(layers)} & {rate(model,'faithfulness'):.3f} & {rate(model,'causality'):.3f} \\\\\n"
text+=r'''\bottomrule\end{tabular}
\caption{Within-model replication on the locked corpus, pooled over capture positions. Head sets are model-specific. A selected layer range gives its endpoints, not a claim that every layer is represented.}
\label{tab:model-comparison}
\end{table}

The Qwen sets contain 122, 88, and 108 heads, respectively. The OLMo comparison uses 93 instruction-model heads and 91 SFT-model heads from the candidate band 10--36, using the exported threshold selections rather than the older 130/79-head pair. Their counts are approximately matched, not identical. Head membership and its layer distribution remain model-specific. Full model identifiers and selection names are listed below for reproducibility.
'''
for c in configs[::2]:
 text+='\n\\noindent\\texttt{'+escape(c['model'])+'}\\\\\nSelection: \\texttt{'+escape(c['heads_name'])+'}.\\par\n'
text+=r'''
\subsection{Results by capture position}
\begin{figure}[ht]\centering
\includegraphics[width=\linewidth]{Figures/model_generalization_pivots.pdf}
\caption{Readout faithfulness and implementation transfer within each model across capture positions (253 records per point). The first implementation token is separated from the four earlier positions. The common scale exposes differences in timing and metric, which pooled rates can obscure.}
\label{fig:cross-model}
\end{figure}

\begin{table}[ht]\centering\scriptsize
\begin{tabular}{llrrrrr}\toprule
Model & Metric & Prev. end & Indent & Variable & Assignment & First impl.\\\midrule
'''
for (model,k),name in zip(chosen,names):
 for mode,label in [('faithfulness','Faith.'),('causality','Transfer')]:
  text+=name+' & '+label+' & '+' & '.join(f'{rate(model,mode,a):.3f}' for a in anchors)+r' \\'+'\n'
text+=r'''\bottomrule\end{tabular}
\caption{Exact per-position rates for the model comparison in \Cref{fig:cross-model}.}
\end{table}

\subsection{Interpretation}
The Coder-32B configuration scores above Coder-14B on both pooled metrics; it also outperforms the generic Qwen-32B instruction configuration on these code tasks. For OLMo, the instruction model has higher implementation-transfer scores, while the SFT model has higher readout faithfulness. Replication therefore does not imply identical performance or a single ordering of models across metrics.

These comparisons vary checkpoints, head sets, and potentially source-generation competence together. They support the presence of the phenomenon across families and sizes, but do not isolate causal effects of scale, code training, or RL. Capturing at equivalent named boundaries also need not mean identical tokenization across models. The approximate OLMo budget matching reduces one disparity without controlling every difference. All scores use the partial-content oracle rubric in \Cref{app:oracle_prompts}, rather than execution-based program correctness.
'''
write(A/'olmo.tex',text)
plt.rcParams.update({'font.size':8,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,axes=plt.subplots(1,2,figsize=(7,3.2),sharey=True)
colors=['#326C9B','#80A7BD','#AA6B2E','#6D5790','#A890B8']
for ax,mode,title in zip(axes,['faithfulness','causality'],['(a) Readout faithfulness','(b) Implementation transfer']):
 for (model,k),name,color in zip(chosen,names,colors):ax.plot(range(5),[rate(model,mode,a) for a in anchors],marker='o',ms=3,lw=1.2,label=name,color=color)
 ax.set_xticks(range(5));ax.set_xticklabels(['Previous\nline end','Indent','Variable','Assign.','First\nimpl.'],fontsize=7)
 ax.set_ylim(0,1.04);ax.set_title(title,loc='left',fontsize=9);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('Oracle agreement');handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,fontsize=7)
fig.tight_layout(rect=(0,.18,1,1))
for ext in ['pdf','svg','png']:fig.savefig(R/f'Figures/model_generalization_pivots.{ext}',dpi=220,bbox_inches='tight',pad_inches=.04)
# Remove superseded duplicate model subsection while retaining source and sparsity appendix material.
p=A/'readout_comparisons.tex';s=p.read_text()
if '\\subsection{Replication within multiple models}' in s:
 a=s.index('\\subsection{Replication within multiple models}');b=s.index('\\subsection{Head-budget results',a);s=s[:a]+s[b:]
s=s.replace('Readout Sources and Replication Across Models','Readout Sources and Head Budgets');p.write_text(s)
p=R/'Text/201_Appendix.tex';s=p.read_text();s=s.replace('\\input{Text/Appendix/oracle_prompts}\n','');s=s.replace('\\input{Text/Appendix/goalscope}','\\input{Text/Appendix/goalscope}\n\\input{Text/Appendix/oracle_prompts}');p.write_text(s)
p=R/'packages.tex';s=p.read_text();
if '\\usepackage{fvextra}' not in s:s+='\n\\usepackage{fvextra}\n'
p.write_text(s)
# Correct the main-text description of what the saved oracle actually judges.
p=R/'Text/03_what.tex';s=p.read_text().replace('An oracle judges readouts against the target description.','An oracle judges readouts against the extracted code for the generated frontier assignment (exact prompts in \\Cref{app:oracle_prompts}).');p.write_text(s)
p=A/'goalscope.tex';s=p.read_text().replace('An oracle judges whether the generated description or implementation matches a nontrivial part of the frontier description.','An oracle judges whether the generated description or implementation matches a nontrivial part of the extracted frontier code. The full system rubrics, user-message templates, and saved examples are in \\Cref{app:oracle_prompts}.');p.write_text(s)
(O/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('Populated exact prompts, distribution, and five-model comparison from saved records.')
