"""Rebuild appendix-only charts/tables from saved records; no inference calls."""
from pathlib import Path
from collections import Counter
import ast, hashlib, json, os, re
os.environ.setdefault('MPLCONFIGDIR', '/tmp/goals-appendix-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
PAPER = ROOT / 'Overleaf-draft/Goals-Overleaf'
FIG = PAPER / 'Figures/appendix'
TAB = PAPER / 'Text/Appendix/tables'
FIG.mkdir(exist_ok=True); TAB.mkdir(exist_ok=True)
MODEL = 'Qwen2.5-Coder-32B-Instruct'
DTYPES = ['int', 'list[int]', 'str', 'list[str]']
PIVOTS = ['prev_end', 'indent', 'var', 'eq', 'first_impl']
MODES = ['faithfulness', 'causality']
COLORS = ['#356ea5', '#c07625']
manifest = {}; results = {}

def track(p):
    p = Path(p); b = p.read_bytes()
    manifest[str(p.relative_to(ROOT))] = hashlib.sha256(b).hexdigest()
    return b

def read(p):
    b = track(p)
    return [json.loads(x) for x in b.splitlines() if x.strip()] if p.suffix == '.jsonl' else json.loads(b)

def tex(s):
    return ''.join({'_':r'\_', '%':r'\%', '&':r'\&', '#':r'\#', '{':r'\{', '}':r'\}', '`':r'\textasciigrave{}'}.get(c,c) for c in str(s))

def save(fig, name):
    for ext in ['pdf', 'png', 'svg']:
        dest = FIG / f'{name}.{ext}' if ext == 'pdf' else REPORT / f'{name}.{ext}'
        fig.savefig(dest, bbox_inches='tight', pad_inches=.04, dpi=170)
    plt.close(fig)

def style(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.set_ylim(0,1.08); ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis='y',alpha=.17); ax.set_axisbelow(True)

plt.rcParams.update({'font.size':9, 'axes.titlesize':10, 'pdf.fonttype':42, 'svg.fonttype':'none'})

# Complete locked corpus and exact task-bank descriptions.
corpus = read(ROOT/'data/goalscope_eval.json')
tree = ast.parse(track(ROOT/'src/diff/data.py'))
bank = next(x.value for x in tree.body if isinstance(x,ast.AnnAssign) and getattr(x.target,'id','')=='QTASKS')
desc = {ast.literal_eval(k): {ast.literal_eval(p.args[0]): ast.literal_eval(p.args[1]) for p in v.args[3].elts} for k,v in zip(bank.keys,bank.values)}
choices = {'int':['is_even','digit_sum','is_perfect_square'], 'list[int]':['has_even','total_sum','distinct_count'], 'str':['is_palindrome','count_uppercase','word_count'], 'list[str]':['has_palindrome','count_starts_vowel','total_uppercase_letters']}
lines = [r'\begin{tabular}{lrp{.67\linewidth}}\toprule',r'Datatype & \# Specs & Example sub-task descriptions\\\midrule']
for i,d in enumerate(DTYPES):
    for j,k in enumerate(choices[d]):
        lines.append((r'\texttt{'+d+'}' if j==0 else '')+' & '+(str(len(corpus['datapoints'][d])) if j==0 else '')+' & '+tex(desc[d][k])+r'\\')
    if i<3:lines.append(r'\addlinespace')
lines += [r'\midrule Total & 253 & \\\bottomrule\end{tabular}']
(TAB/'corpus.tex').write_text('\n'.join(lines)+'\n')

# Source comparison: exactly the same 253 x 5 specification/position keys.
sources={}; matched=None
for mode in MODES:
    for p in (ROOT/'results/evaluation/goalscope'/MODEL/mode).rglob('config.json'):
        c=json.loads(p.read_text())
        if c.get('n_specs')!=253 or c.get('span_source')!='frontier' or c.get('alpha')!=.1:continue
        if c['strategy']=='additive_norm' and c.get('n_heads')==122:key='heads'
        elif c['strategy']=='resid_norm':key=('attn' if c.get('source')=='attn' else 'resid')+str(c['source_layers'])
        else:continue
        read(p); rows=read(p.parent/'records.jsonl')
        keys={(r['spec'],r['anchor']) for r in rows}
        assert len(rows)==len(keys)==1265
        if matched is None:matched=keys
        assert keys==matched
        assert (mode,key) not in sources
        sources[mode,key]=sum(r['hard']=='YES' for r in rows)/len(rows)
order=['heads','attn[40, 50]','resid[25, 56]','resid[40, 50]']
assert len(sources)==8
fig,ax=plt.subplots(figsize=(6.5,3.1))
for i,mode in enumerate(MODES):
    v=[sources[mode,k] for k in order]
    bars=ax.bar(np.arange(4)+(i-.5)*.36,v,.34,color=COLORS[i],label=mode.capitalize())
    ax.bar_label(bars,labels=[f'{100*x:.1f}' for x in v],padding=3,fontsize=8)
style(ax);ax.set_xticks(range(4),['122 goal heads','Attention output\nL40–49','Residual\nL25–55','Residual\nL40–49']);ax.set_ylabel('Oracle agreement')
ax.legend(frameon=False,loc='upper right');fig.tight_layout();save(fig,'source_comparison')
results['source_comparison']=[{'mode':m,'source':s,'rate':v} for (m,s),v in sources.items()]

# Native ablations and exact saved continuations.
abl={}; examples={}; configs={}
for p in (ROOT/'results/evaluation/ablation'/MODEL).rglob('config.json'):
    c=json.loads(p.read_text())
    if c.get('n_specs')!=253:continue
    read(p); rows=read(p.parent/'records.jsonl');assert len(rows)==len({r['spec'] for r in rows})==253
    k=(c['axis'],c['strategy'],c['n_goal']); assert k not in abl
    abl[k]={a:sum(r[a]['verdict']=='YES' for r in rows)/253 for a in ['baseline','ablated','control']}
    examples[k]=next(r for r in rows if r['spec']=='int_python_000');configs[k]=c
results['ablations']=[{'axis':a,'strategy':s,'heads':h,**v,'config':configs[a,s,h],'example':examples[a,s,h]} for (a,s,h),v in abl.items()]
lines=[r'\begin{tabular}{lrr}\toprule',r'Condition & 122-head set & 250-head set\\\midrule']
for key,label in [('baseline','Clean'),('ablated','Spare goal-head access'),('control','Spare random-head access')]:
    lines.append(label+' & '+' & '.join(f'{abl["sufficiency","blind_span",h][key]:.3f}' for h in [122,250])+r'\\')
lines.append(r'\bottomrule\end{tabular}');(TAB/'sufficiency.tex').write_text('\n'.join(lines))
lines=[r'\begin{tabular}{lrrrr}\toprule',r' & \multicolumn{2}{c}{122-head set} & \multicolumn{2}{c}{250-head set}\\',r'Blocked keys & Goal & Random & Goal & Random\\\midrule']
for s,label in [('blind_task','Frontier description'),('blind_task_and_progress','Frontier description through pivot'),('sink','All except position zero')]:
    lines.append(label+' & '+' & '.join(f'{abl["necessity",s,h][k]:.3f}' for h in [122,250] for k in ['ablated','control'])+r'\\')
lines.append(r'\bottomrule\end{tabular}');(TAB/'necessity.tex').write_text('\n'.join(lines))
for axis,strategy in [('sufficiency','blind_span'),('necessity','sink')]:
    e=examples[axis,strategy,250]
    lines=[r'\begin{tabular}{lp{.48\linewidth}l}\toprule',r'Condition & Continuation after \texttt{b =} & Verdict\\\midrule']
    arms=[(e,'baseline','Clean'),(e,'ablated','Goal heads'),(e,'control','Random heads')]
    if axis=='necessity':
        arms=[(e,'baseline','Clean'),(examples[axis,'blind_task',250],'ablated','Goal heads, task only'),
              (examples[axis,'blind_task_and_progress',250],'ablated','Goal heads, task + progress'),
              (e,'ablated','Goal heads, sink'),(e,'control','Random heads, sink')]
    for row,k,label in arms:
        lines.append(label+' & '+r'\texttt{'+tex(row[k]['gen'])+'} & '+tex(row[k]['verdict'])+r'\\')
    lines.append(r'\bottomrule\end{tabular}');(TAB/f'{axis}_example.tex').write_text('\n'.join(lines))

# KV transfer, aggregate first then datatype facets.
kv=[]
for p in (ROOT/'results/diff'/MODEL/'kv_transfer/multi_pred/ph-goal_250__ro-goal_250__abl-off__a2__pad').glob('*/results.json'):
    if '__var_eq__' in str(p):kv.extend(read(p))
assert len(kv)==len({(r['key'],r['src_lang'],r['dst_lang']) for r in kv})==512
dirs=[('python','python'),('python','cpp'),('cpp','python'),('cpp','cpp')]; metrics={}
for d in ['all']+DTYPES:
    for s,t in dirs:
        rr=[r for r in kv if r['src_lang']==s and r['dst_lang']==t and (d=='all' or r['datatype']==d)]
        assert len(rr)==(128 if d=='all' else 32)
        metrics[d,s,t]=[sum(r['causal']['label'] in ['src','both'] for r in rr)/len(rr),sum(r['readout']['patched']=='src' for r in rr)/len(rr)]
def kvplot(ax,d):
    for i,label in enumerate(['Causal transfer','Source-directed readout']):
        vv=[metrics[d,s,t][i] for s,t in dirs];bs=ax.bar(np.arange(4)+(i-.5)*.38,vv,.36,label=label,color=COLORS[i])
        ax.bar_label(bs,labels=[f'{100*v:.1f}' for v in vv],padding=2,fontsize=6.5)
    style(ax);ax.set_xticks(range(4),['Py→Py','Py→C++','C++→Py','C++→C++']);ax.set_ylabel('Success');ax.set_xlabel('Source → destination language')
fig,ax=plt.subplots(figsize=(6.4,2.65));kvplot(ax,'all');ax.legend(loc='upper center',bbox_to_anchor=(.5,1.25),ncol=2,frameon=False);fig.tight_layout();save(fig,'kv_aggregate')
fig,axs=plt.subplots(2,2,figsize=(6.5,5.1))
for ax,d in zip(axs.flat,DTYPES):kvplot(ax,d);ax.set_title(d,loc='left')
fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False);fig.tight_layout(rect=(0,0,1,.94));save(fig,'kv_datatypes')
results['kv']=[{'datatype':d,'src':s,'dst':t,'causal':v[0],'readout':v[1]}for(d,s,t),v in metrics.items()]

# Complete archived OV cohort, retained for comparability with the main figure.
ov={}; ovkeys={}
for p in (ROOT/'results/evaluation/ov-reader__backup_span-rule_pre-locked_2026-09-14'/MODEL).rglob('config.json'):
    c=json.loads(p.read_text())
    if c.get('span_source')!='rule' or c.get('n_samples')!=64:continue
    read(p);rr=read(p.parent/'records.jsonl');assert len(rr)==len({(r['spec'],r['pivot'])for r in rr})==320
    k=(c['mode'],c['datatype'],str(c['layer_idx']));assert k not in ov
    ids={r['spec']for r in rr};assert len(ids)==64
    if c['datatype'] in ovkeys:assert ovkeys[c['datatype']]==ids
    ovkeys[c['datatype']]=ids
    ov[k]={piv:sum(r['hard']=='YES' for r in rr if r['pivot']==piv)/64 for piv in PIVOTS}
layers=sorted(int(l)for m,d,l in ov if m=='faithfulness' and d=='int' and l!='goalscope');assert len(ov)==160
for mode in MODES:
    fig,axs=plt.subplots(3,2,figsize=(6.65,6.35));handles=[]
    for ax,piv,title in zip(axs.flat,PIVOTS,['Previous-line end','Indentation','Variable name','Assignment operator','First implementation token']):
        for d,color in zip(DTYPES,['#3879b7','#db823a','#409b75','#a76ba5']):
            ax.plot(layers,[ov[mode,d,str(l)][piv]for l in layers],color=color,alpha=.36,lw=1.2,label=d)
        ax.plot(layers,[np.mean([ov[mode,d,str(l)][piv]for d in DTYPES])for l in layers],color='#222',lw=1.8,marker='o',ms=2.6,label='Mean across datatypes')
        ax.axhline(np.mean([ov[mode,d,'goalscope'][piv]for d in DTYPES]),color='#222',ls='--',lw=1.1,label='Head-output reference (mean)')
        style(ax);ax.set(xlim=(0,63),xticks=[0,20,40,63],xlabel='Source layer',ylabel='Oracle agreement');ax.set_title(title,loc='left')
    axs[-1,-1].axis('off');h,l=axs[0,0].get_legend_handles_labels();axs[-1,-1].legend(h,l,loc='center left',frameon=False,fontsize=8.5)
    fig.tight_layout();save(fig,f'ov_{mode}')
results['ov']=[{'mode':m,'datatype':d,'layer':l,'rates':v}for(m,d,l),v in ov.items()]

# Localization settings and exported-head provenance, read from checkpoints.
models=[(MODEL,'Qwen Coder 32B','.9',25,56),('Qwen2.5-Coder-14B-Instruct','Qwen Coder 14B','.9',0,48),('Qwen2.5-32B-Instruct','Qwen Instruct 32B','.9',25,56),('Olmo-3.1-32B-Instruct','OLMo Instruct 32B','1.0',10,36),('Olmo-3.1-32B-Instruct-SFT','OLMo SFT 32B','.6',10,36)]
meta=[]
for model,label,threshold,lo,hi in models:
    p=ROOT/'results/localization'/model/'lam20/head_mask.pt';track(p); ck=torch.load(p,map_location='cpu',weights_only=True); c=ck['config'];m=ck['mask_values']
    sel=ROOT/'head_sets'/model/f'goal__lam20_thr{float(threshold):.1f}_band{lo}-{hi}.json';hs=read(sel)
    assert set(map(tuple,hs))=={(i,j)for i in range(lo,hi)for j in range(m.shape[1])if m[i,j]>=float(threshold)}
    nv=max(1,int(c['n_instances']*.2))
    meta.append(dict(model=model,label=label,train=c['n_instances']-nv,val=nv,heads=len(hs),threshold=threshold,band=[lo,hi-1],selection=str(sel.relative_to(ROOT)),sha256=manifest[str(sel.relative_to(ROOT))],config=c))
lines=[r'\begin{tabular}{lrrrrc}\toprule',r'Model & Train & Validation & Epochs & Batch & Augment\\\midrule']
for r in meta:
    c=r['config'];lines.append(f"{r['label']} & {r['train']} & {r['val']} & {c['n_epochs']} & {c['batch_size']} & "+('Yes'if c['augment']else'No')+r'\\')
lines.append(r'\bottomrule\end{tabular}');(TAB/'localization_training.tex').write_text('\n'.join(lines))
lines=[r'\begin{tabular}{lrrrc}\toprule',r'Model & Threshold & Export layers & Heads & Selection SHA-256\\\midrule']
for r in meta:lines.append(f"{r['label']} & {r['threshold']} & {r['band'][0]}--{r['band'][1]} & {r['heads']} & "+r'\texttt{'+r['sha256'][:10]+'}'+r'\\')
lines.append(r'\bottomrule\end{tabular}');(TAB/'localization_selection.tex').write_text('\n'.join(lines));results['localization']=meta
(REPORT/'results.json').write_text(json.dumps(results,indent=2)+'\n')
(REPORT/'sources.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(f'Built 5 appendix figures and 7 table bodies from {len(manifest)} hashed inputs.')
