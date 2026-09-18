"""Recompute D/E comparisons from saved runs; exclude first_impl in pooled scores."""
from pathlib import Path
import ast, hashlib, json, os, re
os.environ.setdefault('MPLCONFIGDIR','/tmp/goals-review-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
ROOT=Path(__file__).resolve().parents[3]; REPORT=Path(__file__).resolve().parent
PAPER=ROOT/'Overleaf-draft/Goals-Overleaf'; FIG=PAPER/'Figures/appendix'; TAB=PAPER/'Text/Appendix/tables'
MODEL='Qwen2.5-Coder-32B-Instruct'; PIV=['prev_end','indent','var','eq']; ALL=PIV+['first_impl']; MODES=['faithfulness','causality']
COLORS=['#356ea5','#c07625']; manifest={}; data={}
def read(p):
    b=p.read_bytes();manifest[str(p.relative_to(ROOT))]=hashlib.sha256(b).hexdigest()
    return [json.loads(x)for x in b.splitlines()if x.strip()]if p.suffix=='.jsonl' else json.loads(b)
def tex(s):
    return ''.join({'_':r'\_','%':r'\%','&':r'\&','#':r'\#','{':r'\{','}':r'\}','\\':r'\textbackslash{}','~':r'\textasciitilde{}'}.get(c,c)for c in str(s))
def save(fig,name,main=False):
    fig.savefig((PAPER/'Figures'if main else FIG)/f'{name}.pdf',bbox_inches='tight',pad_inches=.04)
    fig.savefig(REPORT/f'{name}.png',bbox_inches='tight',pad_inches=.04,dpi=180);plt.close(fig)
def style(ax):
    ax.spines[['top','right']].set_visible(False);ax.set_ylim(0,1.06);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(axis='y',alpha=.17);ax.set_axisbelow(True)
plt.rcParams.update({'font.size':8,'axes.titlesize':9,'pdf.fonttype':42})

scope={};sources={};matched=None
for mode in MODES:
    for p in (ROOT/'results/evaluation/goalscope'/MODEL/mode).rglob('config.json'):
        c=json.loads(p.read_text())
        if c.get('n_specs')!=253 or c.get('span_source')!='frontier' or c.get('alpha')!=.1:continue
        if c['strategy']=='additive_norm':key=c['n_heads']
        elif c['strategy']=='resid_norm':key=('attn'if c.get('source')=='attn'else'resid')+str(c['source_layers'])
        else:continue
        read(p);rr=read(p.parent/'records.jsonl');keys={(r['spec'],r['anchor'])for r in rr};assert len(rr)==len(keys)==1265
        if matched is None:matched=keys
        assert keys==matched
        rates={piv:sum(r['hard']=='YES'for r in rr if r['anchor']==piv)/253 for piv in ALL}
        rates['pre_mean']=np.mean([rates[piv]for piv in PIV])
        if isinstance(key,int):assert(mode,key)not in scope;scope[mode,key]=rates
        else:assert(mode,key)not in sources;sources[mode,key]=rates
assert len(scope)==20 and len(sources)==6
data['scope']=[dict(mode=m,heads=h,**v)for(m,h),v in scope.items()]
data['sources']=[dict(mode=m,source=k,**v)for(m,k),v in sources.items()]
fig,ax=plt.subplots(figsize=(6.4,3))
for i,mode in enumerate(MODES):
    vv=[scope[mode,122]['pre_mean']]+[sources[mode,k]['pre_mean']for k in ['attn[40, 50]','resid[25, 56]','resid[40, 50]']]
    bs=ax.bar(np.arange(4)+(i-.5)*.36,vv,.34,color=COLORS[i],label=mode.capitalize());ax.bar_label(bs,labels=[f'{100*v:.1f}'for v in vv],padding=3,fontsize=8)
style(ax);ax.set_xticks(range(4),['122 goal heads','Attention outputs\nL40–49','Residual states\nL25–55','Residual states\nL40–49']);ax.set_ylabel('Readout success');ax.legend(frameon=False);fig.tight_layout();save(fig,'source_comparison')
def budget(ax,piv):
    for i,mode in enumerate(MODES):
        ks=sorted(h for m,h in scope if m==mode and h!=122)
        ax.plot(ks,[scope[mode,h][piv]for h in ks],color=COLORS[i],marker=['o','s'][i],ms=3,ls=['-','--'][i],lw=1.2,label=mode.capitalize())
        ax.plot(122,scope[mode,122][piv],marker='*',ms=9,color=COLORS[i],mec='white',mew=.5)
    style(ax);ax.set(xlim=(0,310),xticks=[0,50,100,150,200,250,300],xlabel='Number of selected heads',ylabel='Readout success')
fig,axs=plt.subplots(2,2,figsize=(6.7,4.7))
for ax,piv,title in zip(axs.flat,PIV,['Previous-line end','Indentation','Variable name','Assignment operator']):budget(ax,piv);ax.set_title(title,loc='left')
fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False);fig.tight_layout(rect=(0,0,1,.94));save(fig,'head_budget_preimplementation')
# Main figure: only the requested left panel changes. Reproduce right from the
# same native-ablation records/configuration used in the author baseline.
abl={}
for p in (ROOT/'results/evaluation/ablation'/MODEL).rglob('config.json'):
    c=json.loads(p.read_text())
    if c.get('n_specs')!=253 or c.get('strategy')not in ['blind_span','sink']:continue
    read(p);rr=read(p.parent/'records.jsonl');assert len(rr)==253
    abl[c['axis'],c['strategy'],c['n_goal']]={k:sum(r[k]['verdict']=='YES'for r in rr)/253 for k in ['baseline','ablated','control']}
fig,axs=plt.subplots(1,2,figsize=(6.8,2.65),gridspec_kw={'width_ratios':[1,1.05]})
budget(axs[0],'pre_mean');axs[0].set_title('(a) Mean over four pre-implementation pivots',loc='left',fontsize=8);axs[0].legend(loc='lower right',frameon=False,fontsize=7);axs[0].text(.03,.09,'Stars: 122-head reference',transform=axs[0].transAxes,fontsize=6.7,color='#555')
keys=[('sufficiency','blind_span',122),('sufficiency','blind_span',250),('necessity','sink',122),('necessity','sink',250)]
for i,(cond,color,label)in enumerate([('ablated',COLORS[0],'Goal heads'),('control','#aaa','Random heads')]):
    vv=[abl[k][cond]for k in keys];bs=axs[1].bar(np.arange(4)+(i-.5)*.36,vv,.34,color=color,label=label)
    axs[1].bar_label(bs,labels=[f'{100*v:.1f}'for v in vv],padding=2,fontsize=6)
axs[1].axhline(abl[keys[0]]['baseline'],color='#333',ls=':',lw=1,label='Clean (92.5%)');style(axs[1]);axs[1].set_xticks(range(4),['Spare\n122','Spare\n250','Sink\n122','Sink\n250']);axs[1].set_ylabel('Frontier implementation success');axs[1].set_title('(b) Native implementation',loc='left');axs[1].legend(loc='upper center',bbox_to_anchor=(.50,.68),frameon=True,facecolor='white',framealpha=.95,edgecolor='none',fontsize=6.6)
fig.tight_layout(w_pad=1.5);save(fig,'sparsity_ablation_paired',main=True)

# Use exactly the exported selections already documented in the later appendix.
models=[(MODEL,'Coder 32B','goal'),('Qwen2.5-Coder-14B-Instruct','Coder 14B','goal'),('Qwen2.5-32B-Instruct','Instruct 32B','goal'),('Olmo-3.1-32B-Instruct','OLMo +RL','json:goal__lam20_thr1.0_band10-36'),('Olmo-3.1-32B-Instruct-SFT','OLMo SFT','json:goal__lam20_thr0.6_band10-36')]
mr={};model_rows={};configs={}
for model,label,selection in models:
    for mode in MODES:
        found=[]
        for p in (ROOT/'results/evaluation/goalscope'/model/mode/'replace').rglob('config.json'):
            c=json.loads(p.read_text())
            if c.get('alpha')==1 and c.get('heads_name')==selection and c.get('n_specs')==253 and c.get('span_source')=='frontier':found.append((p,c))
        assert len(found)==1,(model,mode,len(found))
        p,c=found[0];read(p);rr=read(p.parent/'records.jsonl');assert len(rr)==1265 and {(r['spec'],r['anchor'])for r in rr}==matched
        mr[model,mode]={piv:sum(r['hard']=='YES'for r in rr if r['anchor']==piv)/253 for piv in ALL};mr[model,mode]['pre_mean']=np.mean([mr[model,mode][piv]for piv in PIV]);model_rows[model,mode]=rr;configs[model]=c
data['models']=[dict(model=m,mode=mode,**v)for(m,mode),v in mr.items()]
for name,pair in [('model_size',[models[1],models[0]]),('code_finetuning',[models[2],models[0]]),('rl_finetuning',[models[4],models[3]])]:
    fig,axs=plt.subplots(1,2,figsize=(6.5,2.65))
    for ax,mode in zip(axs,MODES):
        for i,(model,label,_)in enumerate(pair):
            ax.plot(range(4),[mr[model,mode][piv]for piv in PIV],color=COLORS[i],marker=['o','s'][i],ls=['-','--'][i],lw=1.4,ms=4,label=label)
        style(ax);ax.set_xticks(range(4),['Prev. end','Indent','Variable','=']);ax.set_title(mode.capitalize(),loc='left');ax.set_ylabel('Readout success');ax.legend(loc='lower right',frameon=False,fontsize=7)
    fig.tight_layout();save(fig,name)
lines=[r'\begin{tabular}{lrrr}\toprule',r'Model & Goal heads & Faithfulness & Causality\\\midrule']
for m,label,_ in models:lines.append(label+' & '+str(configs[m]['n_heads'])+' & '+' & '.join(f'{mr[m,mode]["pre_mean"]:.3f}'for mode in MODES)+r'\\')
lines.append(r'\bottomrule\end{tabular}');(TAB/'model_comparison_preimplementation.tex').write_text('\n'.join(lines)+'\n')

# Literal phrase counts are diagnostics, not a new semantic refusal classifier.
diagnostics={}
for m,_,_ in models[-2:]:
    rr=model_rows[m,'faithfulness'];diagnostics[m]={}
    for piv in PIV+['pre_mean']:
        selected=[r for r in rr if (r['anchor']in PIV if piv=='pre_mean'else r['anchor']==piv)]
        hits=[r for r in selected if re.search(r"\bI (?:cannot|can't)\b",r['readout'],re.I)]
        diagnostics[m][piv]={'n':len(selected),'literal_cannot':len(hits),'yes_among_literal_cannot':sum(r['hard']=='YES'for r in hits)}
data['refusal_phrase_diagnostic']=diagnostics
rl=model_rows[models[3][0],'faithfulness'];sft={(r['spec'],r['anchor']):r for r in model_rows[models[4][0],'faithfulness']}
candidates=[]
for r in rl:
    if r['anchor']!='eq':continue
    t=sft[r['spec'],'eq']
    if len(r['readout'])<230 and len(t['readout'])<180:
        candidates.append({'spec':r['spec'],'datatype':r['datatype'],'condition':r['condition'],'rl':r['readout'],'rl_verdict':r['hard'],'sft':t['readout'],'sft_verdict':t['hard']})
(REPORT/'olmo_example_candidates.json').write_text(json.dumps(candidates,indent=2)+'\n')
(REPORT/'results.json').write_text(json.dumps(data,indent=2)+'\n');(REPORT/'sources.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Source rates:',data['sources']);print('Head-budget 25/122:',[r for r in data['scope']if r['heads']in [25,122]])
print('Model averages:',[(m,[(mode,round(mr[m,mode]['pre_mean'],4))for mode in MODES])for m,_,_ in models]);print('Cannot diagnostics:',diagnostics)
