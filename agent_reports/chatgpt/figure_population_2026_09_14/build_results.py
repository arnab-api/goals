"""Build paper figures from saved verdicts; no model or oracle calls."""
from pathlib import Path
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict

os.environ.setdefault('MPLCONFIGDIR', '/tmp/goals-figure-population-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
FIG = ROOT / 'Overleaf-draft/Goals-Overleaf/Figures'
MODEL = 'Qwen2.5-Coder-32B-Instruct'
PIVOTS = ['prev_end', 'indent', 'var', 'eq', 'first_impl']
DTYPES = ['int', 'list[int]', 'str', 'list[str]']
MODES = ['faithfulness', 'causality']
COLORS = ['#356ea5', '#c07625']
MANIFEST = []
AGG = {}


def read(path):
    raw = path.read_bytes()
    MANIFEST.append({'path': str(path.relative_to(ROOT)),
                     'sha256': hashlib.sha256(raw).hexdigest()})
    if path.suffix == '.jsonl':
        return [json.loads(x) for x in raw.splitlines() if x.strip()]
    return json.loads(raw)


def save(fig, name):
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(FIG / f'{name}.{ext}', bbox_inches='tight', pad_inches=.035, dpi=220)
    plt.close(fig)


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylim(0, 1.045)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis='y', alpha=.18)
    ax.set_axisbelow(True)


plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                     'axes.titlesize': 9, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})

# Readout and sparsity share precisely the same specification/anchor pairs.
scope = {}
matched = None
for mode in MODES:
    for config in sorted((ROOT / 'results/evaluation/goalscope' / MODEL / mode / 'additive_norm').rglob('config.json')):
        c = json.loads(config.read_text())
        if c.get('alpha') != .1 or c.get('n_specs') != 253 or c.get('span_source') != 'frontier':
            continue
        nheads = c['n_heads']
        if nheads not in [5, 10, 25, 50, 100, 122, 150, 200, 250, 300]:
            continue
        read(config)
        rows = read(config.parent / 'records.jsonl')
        keys = {(r['spec'], r['anchor']) for r in rows}
        assert len(rows) == len(keys) == 1265
        if matched is None: matched = keys
        assert keys == matched
        assert (mode, nheads) not in scope
        scope[mode, nheads] = {p: sum(r['hard'] == 'YES' for r in rows if r['anchor'] == p) / 253 for p in PIVOTS}
assert len(scope) == 20
AGG['goalscope'] = [{'mode': m, 'heads': h, 'rates': v} for (m, h), v in scope.items()]

fig, ax = plt.subplots(figsize=(2.8, 2.65))
x = np.arange(5)
for i, mode in enumerate(MODES):
    vals = [scope[mode, 122][p] for p in PIVOTS]
    bars = ax.bar(x + (i-.5)*.37, vals, width=.35, color=COLORS[i], label=mode.capitalize())
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, v+.012, f'{100*v:.0f}', ha='center', fontsize=6.5)
style(ax)
ax.axvline(3.5, color='#888', linestyle=':', linewidth=.9)
ax.set_xticks(x, ['Prev.\nend', 'Indent', 'Variable', '=', 'First\nimpl.'])
ax.set_ylabel('Oracle agreement')
ax.set_xlabel('Source capture position')
ax.legend(loc='upper left', bbox_to_anchor=(0, 1.22), frameon=False, fontsize=7)
fig.tight_layout()
save(fig, 'goal_eval_matched')

# Native ablations: exact recorded YES verdicts, including newer alternatives.
abl = {}
abl_raw = {}
for config in sorted((ROOT / 'results/evaluation/ablation' / MODEL).rglob('config.json')):
    c = json.loads(config.read_text())
    if c.get('n_specs') != 253: continue
    read(config)
    rows = read(config.parent / 'records.jsonl')
    assert len(rows) == len({r['spec'] for r in rows}) == 253
    key = (c['axis'], c['strategy'], c['n_goal'])
    assert key not in abl
    abl[key] = {k: sum(r[k]['verdict'] == 'YES' for r in rows) / 253 for k in ['baseline', 'ablated', 'control']}
    abl_raw[key] = rows
AGG['ablations'] = [{'axis': a, 'strategy': s, 'heads': h, **v} for (a,s,h),v in abl.items()]
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.65), gridspec_kw={'width_ratios': [1, 1.05]})
for i, mode in enumerate(MODES):
    ks = sorted(h for m,h in scope if m == mode and h != 122)
    axes[0].plot(ks, [scope[mode,h]['eq'] for h in ks], color=COLORS[i],
                 marker=['o','s'][i], ms=3, ls=['-','--'][i], lw=1.2, label=mode.capitalize())
    axes[0].plot(122, scope[mode,122]['eq'], marker='*', ms=9, color=COLORS[i], mec='white', mew=.5)
style(axes[0]); axes[0].set(xlabel='Number of selected heads', ylabel='Oracle agreement', xlim=(0,310))
axes[0].set_xticks([0, 50, 100, 150, 200, 250, 300])
axes[0].set_title('(a) Sparse readout at =', loc='left')
axes[0].legend(loc='lower right', frameon=False, fontsize=7)
axes[0].text(.03,.09,'Stars: 122-head reference',transform=axes[0].transAxes,fontsize=6.7,color='#555')
keys = [('sufficiency','blind_span',122), ('sufficiency','blind_span',250),
        ('necessity','sink',122), ('necessity','sink',250)]
for i, (cond, color, label) in enumerate([('ablated', '#356ea5', 'Goal heads'), ('control', '#aaaaaa', 'Random heads')]):
    vals=[abl[k][cond] for k in keys]
    bars=axes[1].bar(np.arange(4)+(i-.5)*.36,vals,width=.34,color=color,label=label)
    for b,v in zip(bars,vals):
        axes[1].text(b.get_x()+b.get_width()/2,v+.012,f'{100*v:.1f}',ha='center',fontsize=6)
axes[1].axhline(abl[keys[0]]['baseline'], color='#333', ls=':', lw=1, label='Clean (92.5%)')
style(axes[1]); axes[1].set_xticks(range(4), ['Spare\n122','Spare\n250','Sink\n122','Sink\n250'])
axes[1].set_ylabel('Frontier implementation success')
axes[1].set_title('(b) Native implementation',loc='left')
axes[1].legend(loc='upper center',bbox_to_anchor=(.50,.68),frameon=True,facecolor='white',framealpha=.95,edgecolor='none',fontsize=6.6)
fig.tight_layout(w_pad=1.5); save(fig,'sparsity_ablation_paired')

# Complete archived OV cohort. The new frontier-corpus runs are still incomplete.
ovroot = ROOT / 'results/evaluation/ov-reader__backup_span-rule_pre-locked_2026-09-14' / MODEL
ov = {}
for config in sorted(ovroot.rglob('config.json')):
    c=json.loads(config.read_text())
    if c.get('span_source') != 'rule' or c.get('n_samples') != 64: continue
    read(config); rows=read(config.parent/'records.jsonl')
    assert len(rows) == len({(r['spec'],r['pivot']) for r in rows}) == 320
    key=(c['mode'],c['datatype'],str(c['layer_idx']))
    assert key not in ov
    ov[key]={p: sum(r['hard']=='YES' for r in rows if r['pivot']==p)/64 for p in PIVOTS}
layers=sorted(int(l) for m,d,l in ov if m=='faithfulness' and d=='int' and l!='goalscope')
assert len(ov)==8*(len(layers)+1)
AGG['ov_cohort']='archived span=rule, 64 per datatype; complete, independently sampled cohort'
AGG['ov']= [{'mode':m,'datatype':d,'layer':l,'rates':v} for (m,d,l),v in ov.items()]
heads=read(ROOT/'head_sets'/MODEL/'goal__lam20_thr0.9_band25-56.json')
assert len(heads)==122
hist=np.bincount([h[0] for h in heads],minlength=64)
AGG['head_counts_by_layer']=hist.tolist()

def ovplot(ax,pivot,legend=False):
    for i,mode in enumerate(MODES):
        values=np.array([[ov[mode,d,str(l)][pivot] for l in layers] for d in DTYPES])
        mean=values.mean(axis=0); sd=values.std(axis=0,ddof=1)
        ax.plot(layers,mean,color=COLORS[i],lw=1.5,marker=['o','s'][i],ms=2.7,label=mode.capitalize())
        ax.fill_between(layers,np.maximum(0,mean-sd),np.minimum(1,mean+sd),color=COLORS[i],alpha=.13,linewidth=0)
        ref=np.mean([ov[mode,d,'goalscope'][pivot] for d in DTYPES])
        ax.axhline(ref,color=COLORS[i],lw=.9,ls='--',alpha=.8)
    style(ax); ax.set_xlim(0,63);ax.set_xticks([0,10,20,30,40,50,63])
    ax.set_ylabel('Oracle agreement')
    if legend: ax.legend(loc='upper left',bbox_to_anchor=(0,.88),frameon=False,fontsize=8)

fig,(ax,lower)=plt.subplots(2,1,figsize=(6.4,3.05),sharex=True,gridspec_kw={'height_ratios':[3,1],'hspace':.08})
ovplot(ax,'eq',True)
ax.text(.01,.38,'Shading: ±1 SD across datatypes\nDashed: actual head-output readout',transform=ax.transAxes,fontsize=7,color='#555')
lower.bar(range(64),hist,color='#787586',width=.85)
lower.spines[['top','right']].set_visible(False)
lower.set(xlabel='Source layer',ylabel='Goal heads',ylim=(0,max(hist)+1))
lower.set_yticks([0,6,12]); fig.align_ylabels([ax,lower]);fig.subplots_adjust(left=.1,right=.99,bottom=.16,top=.99)
save(fig,'ov_reader_combined')
fig,axes=plt.subplots(2,2,figsize=(6.7,4.5))
for ax,p,title in zip(axes.flat,['prev_end','indent','var','first_impl'],['Previous-line end','Indentation','Variable name','First implementation token']):
    ovplot(ax,p);ax.set_title(title,loc='left');ax.set_xlabel('Source layer')
axes[0,0].legend(frameon=False,fontsize=7,loc='upper left')
fig.tight_layout();save(fig,'ov_reader_other_pivots')

# KV statistics. Preserve separate causal SRC-or-BOTH and exclusive patched-SRC readout metrics.
kvroot=ROOT/'results/diff'/MODEL/'kv_transfer/multi_pred/ph-goal_250__ro-goal_250__abl-off__a2__pad'
kv=[]
for p in sorted(kvroot.glob('*/results.json')):
    if '__var_eq__' not in str(p): continue
    rows=read(p)
    kv.extend(rows)
assert len(kv)==512
assert len({(r['key'],r['src_lang'],r['dst_lang']) for r in kv})==512
metrics=[]
for dtype in ['all']+DTYPES:
    for src in ['python','cpp']:
        for dst in ['python','cpp']:
            rr=[r for r in kv if r['src_lang']==src and r['dst_lang']==dst and (dtype=='all' or r['datatype']==dtype)]
            assert len(rr)==(128 if dtype=='all' else 32)
            metrics.append({'datatype':dtype,'src':src,'dst':dst,'n':len(rr),
                'causal':sum(r['causal']['label'] in ['src','both'] for r in rr)/len(rr),
                'readout':sum(r['readout']['patched']=='src' for r in rr)/len(rr),
                'shifted':sum(r['readout']['shifted'] for r in rr)/len(rr),
                'valid':sum(r['readout']['valid'] for r in rr)/len(rr)})
AGG['kv']=metrics
example=next(r for r in kv if r['src_lang']=='python' and r['dst_lang']=='cpp' and r['key']=='multi|int|shared=is_two_digit|src=is_even|dst=is_multiple_of_3')
(REPORT/'kv_example.json').write_text(json.dumps(example,indent=2))

def matrix_tex(dtype):
    lines=[r'\begin{tabular}{lcc}\toprule',r'Source $\backslash$ Destination & Python & C++\\\midrule']
    for src,label in [('python','Python'),('cpp','C++')]:
        cells=[]
        for dst in ['python','cpp']:
            r=next(r for r in metrics if r['datatype']==dtype and r['src']==src and r['dst']==dst)
            cells.append(f"{r['causal']*100:.1f}\\,\\{{{r['readout']*100:.1f}\\}}")
        lines.append(label+' & '+' & '.join(cells)+r'\\')
    lines.append(r'\bottomrule\end{tabular}')
    return '\n'.join(lines)
td=ROOT/'Overleaf-draft/Goals-Overleaf/Text/Figures'
(td/'kv_matrix_body.tex').write_text(matrix_tex('all')+'\n')
(REPORT/'kv_datatype_tables.tex').write_text('\n\n'.join('\n'+r'\paragraph{'+d.replace('[',r'[').replace(']',r']')+r'.}'+'\n'+matrix_tex(d) for d in DTYPES))
(REPORT/'results.json').write_text(json.dumps(AGG,indent=2))
(REPORT/'result_sources.json').write_text(json.dumps(MANIFEST,indent=2))
print('Matched GoalScope:', scope['faithfulness',122], scope['causality',122])
print('Ablations:',AGG['ablations'])
print('KV:',metrics[:4])
print('Complete OV layer count:',len(layers),'sources hashed:',len(MANIFEST))
