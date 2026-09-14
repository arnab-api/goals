"""Recompute per-pivot hard-YES rates from saved runs; no model calls."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/goalscope-paper-mpl')
from pathlib import Path
import json, csv, hashlib
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
O=Path(__file__).resolve().parent
ROOT=O.parents[2]
DATA=ROOT/'results/evaluation/goalscope/Qwen2.5-Coder-32B-Instruct'
FIG=ROOT/'Overleaf-draft/Goals-Overleaf/Figures'
ANCHORS=['prev_end','indent','var','eq','first_impl']
KS=[5,10,25,50,100,122,150,200,250,300]
rows=[]; manifest=[]; matched=None
for mode in ['faithfulness','causality']:
 seen=set()
 for p in sorted((DATA/mode/'additive_norm').rglob('config.json')):
  c=json.loads(p.read_text())
  if c.get('alpha')!=.1 or c.get('n_specs')!=253 or c.get('span_source')!='frontier':continue
  k=c['n_heads']
  if k not in KS:continue
  assert k not in seen,(mode,k)
  seen.add(k)
  path=p.parent/'records.jsonl';blob=path.read_bytes()
  rr=[json.loads(s) for s in blob.splitlines() if s.strip()]
  keys=[(r['spec'],r['anchor']) for r in rr]
  assert len(keys)==len(set(keys))==1265,(path,len(keys))
  if matched is None:matched=set(keys)
  assert set(keys)==matched, path
  assert {r['anchor'] for r in rr}==set(ANCHORS)
  for anchor in ANCHORS:
   records=[r for r in rr if r['anchor']==anchor]
   assert len(records)==253
   labels=Counter(r['hard'] for r in records)
   rows.append(dict(mode=mode,heads=k,pivot=anchor,n=len(records),yes=labels['YES'],rate=labels['YES']/len(records)))
  manifest.append(dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(blob).hexdigest(),config=c,labels=dict(Counter(r['hard'] for r in rr))))
 assert seen==set(KS),(mode,seen)
(O/'manifest.json').write_text(json.dumps(manifest,indent=2))
with (O/'per_pivot.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none'})
def draw(ax,pivot,title=None):
 for mode,color,style,marker,label in [('faithfulness','#326C9B','-','o','Readout faithfulness'),('causality','#AD6A27','--','s','Implementation transfer')]:
  rr=sorted([r for r in rows if r['mode']==mode and r['pivot']==pivot],key=lambda r:r['heads'])
  curve=[r for r in rr if r['heads']!=122]
  ax.plot([r['heads'] for r in curve],[r['rate'] for r in curve],color=color,ls=style,marker=marker,ms=3.8,lw=1.35,label=label)
  ref=next(r for r in rr if r['heads']==122)
  ax.plot(122,ref['rate'],marker='*',ms=10,color=color,markeredgecolor='white',markeredgewidth=.45,zorder=5)
 ax.set(xlim=(0,310),ylim=(0,1.04),xlabel='Number of selected heads',ylabel='Oracle agreement')
 ax.set_xticks([0,50,100,150,200,250,300]);ax.yaxis.set_major_formatter(PercentFormatter(1))
 ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
 if title:ax.set_title(title,fontsize=9,loc='left')
def save(fig,name):
 for ext in ['pdf','svg','png']:fig.savefig(FIG/f'{name}.{ext}',dpi=220,bbox_inches='tight',pad_inches=.04)
fig,ax=plt.subplots(figsize=(5.5,2.7));draw(ax,'eq');ax.legend(loc='lower right',frameon=False,fontsize=8)
ax.text(.03,.08,'★ 122-head reference',transform=ax.transAxes,ha='left',fontsize=7.5,color='#555555')
fig.tight_layout();save(fig,'sparsity_eq')
fig,axes=plt.subplots(2,2,figsize=(7,4.7))
for ax,piv,title in zip(axes.flat,['prev_end','indent','var','first_impl'],['(a) Previous-line end','(b) Indentation','(c) Variable name','(d) First implementation token']):draw(ax,piv,title)
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.5,1.005))
fig.tight_layout(rect=(0,0,1,.95));save(fig,'sparsity_other_pivots')
for k in [5,25,122]:print([r for r in rows if r['heads']==k and r['pivot']=='eq'])
print('Validated 20 runs, 253 matched specifications × 5 pivots each.')
