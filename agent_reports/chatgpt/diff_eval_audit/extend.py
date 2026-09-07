"""Derive frozen-snapshot tables, provenance and examples; optional publication plots."""
from pathlib import Path
import collections, csv, gzip, hashlib, json
from audit import metrics, csvwrite, ROOT, OUT

snapshot=json.load(gzip.open(OUT/'snapshot.json.gz','rt'))
cs=snapshot['configs']
source_paths=[ROOT/'agent_reports/claude/diff_eval_analysis.md']
for pat in ['scripts/diff/*.py','src/diff/*.py','notebooks/diff/*.ipynb','job_list/diff_eval/*.json','head_sets/Qwen2.5-Coder-32B-Instruct/**/*.json']:
    source_paths+=sorted(ROOT.glob(pat))
source_paths += [ROOT/'src/goals/goalscope.py',ROOT/'src/goals/head_loc_attn.py']
(OUT/'source_manifest.json').write_text(json.dumps([
    dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in source_paths],indent=2))

summaries=json.loads((OUT/'run_summary.json').read_text())
fmt=lambda x:'—' if x is None else f'{100*x:.2f}%'
lines=['# Frozen result tables',f"Snapshot: {snapshot['timestamp']}. Counts are records, not distinct independent experiments.",'',
       '| Mechanism/setup | Run | Pivot | N | Causal (SRC or BOTH) | Patched readout = SRC | Patched readout = SRC or BOTH | Valid | Readout shift given valid |',
       '|---|---|---|---:|---:|---:|---:|---:|---:|']
for r in summaries:
    lines.append(f"| {r['mechanism']}/{r['setup']} | `{r['run']}` | {r['variant']} | {r['n']} | {fmt(r['causal'])} | {fmt(r['ro_strict'])} | {fmt(r['ro'])} | {fmt(r['ro_valid'])} | {fmt(r['ro_given_valid'])} |")
(OUT/'tables.md').write_text('\n'.join(lines)+'\n')

examples=[]
for c in cs:
    if c['mechanism']=='kv_transfer' and c['setup']=='multi_pred' and '250' in c['run'] and 'abl-off' in c['run']:
        for r in c['rows']:
            if not r['causal']['triggered']:examples.append(dict(path=c['path'],record=r))
(OUT/'kv250_off_failures.json').write_text(json.dumps(examples,indent=2))

additional={}
kvrows=[r for c in cs if c['mechanism']=='kv_transfer' for r in c['rows']]
additional['kv_span_geometry']=dict(n=len(kvrows),different_start=sum(r['raw']['src_span'][0]!=r['raw']['dst_span'][0] for r in kvrows),
    different_length=sum(r['raw']['src_span'][1]-r['raw']['src_span'][0]!=r['raw']['dst_span'][1]-r['raw']['dst_span'][0] for r in kvrows))
additional['macro_vs_micro']=[]
groups=collections.defaultdict(list)
for c in cs:
    variant=('var_eq' if 'var_eq' in c['config'] else 'indent') if c['mechanism']=='kv_transfer' and c['setup']=='multi_pred' else '-'
    groups[(c['mechanism'],c['setup'],c['run'],variant)].append(c)
for key,cfgs in groups.items():
    stats=[metrics(c['rows']) for c in cfgs]
    additional['macro_vs_micro'].append(dict(group=list(key),macro_causal=sum(s['causal'] for s in stats)/len(stats),
        micro_causal=metrics([r for c in cfgs for r in c['rows']])['causal'],n_configs=len(cfgs)))
additional['pivot_comparison']=[]
for strategy in ['restrict','zero']:
    cfgs=[c for c in cs if c['mechanism']=='kv_transfer' and c['setup']=='multi_pred' and c['run']==f'ph-goal__abl-33-43-{strategy}__a2__pad']
    a={(c['config'].replace('__indent__','__PIVOT__'),r['key']):r for c in cfgs if '__indent__' in c['config'] for r in c['rows']}
    b={(c['config'].replace('__var_eq__','__PIVOT__'),r['key']):r for c in cfgs if '__var_eq__' in c['config'] for r in c['rows']}
    additional['pivot_comparison'].append(dict(strategy=strategy,n_indent=len(a),n_var_eq=len(b),shared_keys=len(a.keys()&b.keys())))
additional['filter_alpha_overlap']=[]
cfgs=[c for c in cs if c['mechanism']=='q_transfer' and c['setup']=='filter']
a={(c['config'],r['key']):r for c in cfgs if c['run']=='ph-goal__a1' for r in c['rows']}
b={(c['config'],r['key']):r for c in cfgs if c['run']=='ph-goal__a2' for r in c['rows']}
shared=a.keys()&b.keys()
additional['filter_alpha_overlap']=dict(shared=len(shared),same_after_rstrip=sum(all(a[k]['raw'][p].rstrip()==b[k]['raw'][p].rstrip() for p in ['src_prompt','dst_prompt']) for k in shared))
(OUT/'additional_checks.json').write_text(json.dumps(additional,indent=2))

# Meaningful arithmetic checks independent of saved booleans.
assert sum(s['n'] for s in summaries)==sum(len(c['rows']) for c in cs)
for r in summaries:
    assert sum(r['causal_labels'].values())==r['n_causal']
    assert r['c1_r1']+r['c1_r0']==r['causal_hits']
    assert r['c1_r1']+r['c0_r1']==r['ro_hits']
assert additional['kv_span_geometry']['different_length']==0

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
selected=[('q_transfer','multi_pred','ph-goal__a2','-','Q / multi · 122'),
          ('q_transfer','filter','ph-goal__a2','-','Q / filter · 122'),
          ('q_transfer','filter','ph-goal_250__ro-goal_250__a2','-','Q / filter · 250¹'),
          ('kv_transfer','dual_comp','ph-goal__abl-33-43-restrict__a2__pad','-','KV / dual · 122'),
          ('kv_transfer','dual_comp','ph-goal_250__ro-goal_250__abl-off__a2__pad','-','KV / dual · 250²'),
          ('kv_transfer','multi_pred','ph-goal__abl-33-43-restrict__a2__pad','var_eq','KV / multi · 122'),
          ('kv_transfer','multi_pred','ph-goal_250__ro-goal_250__abl-off__a2__pad','var_eq','KV / multi · 250')]
lookup={(r['mechanism'],r['setup'],r['run'],r['variant']):r for r in summaries}
stats=[lookup[k[:4]] for k in selected]
fig,axs=plt.subplots(1,2,figsize=(12,5.3),sharey=True)
y=np.arange(len(stats))
for ax,inc,strict,title in [(axs[0],'causal','causal_strict','Causal: how much success includes BOTH?'),(axs[1],'ro','ro_strict','Readout: unconditional versus selected cases')]:
    for j,r in enumerate(stats):ax.plot([r[strict],r[inc]],[j,j],color='#aab2bd',lw=3)
    ax.scatter([r[inc] for r in stats],y,label='SRC or BOTH',color='#267b9b',s=40,zorder=3)
    ax.scatter([r[strict] for r in stats],y,label='SRC only',color='#956327',marker='s',s=30,zorder=4)
    if inc=='ro':ax.scatter([r['ro_given_valid'] for r in stats],y,label='SRC or BOTH | valid',color='#7960ad',marker='x',s=40,zorder=5)
    ax.set_xlim(0,1.04);ax.set_xticks([0,.25,.5,.75,1]);ax.grid(axis='x',alpha=.2);ax.set_title(title,fontsize=11);ax.set_xlabel('Rate');ax.legend(loc='lower left',fontsize=8)
axs[0].set_yticks(y,[f"{k[4]} (n={r['n']})" for k,r in zip(selected,stats)]);axs[0].invert_yaxis()
fig.suptitle('DIFF audit: metric definitions materially change the picture',fontsize=14)
fig.text(.02,.015,'¹ list[int] only, n=32 per language.  ² Partial coverage at snapshot; not comparable to a complete grid. All configurations use α=2.',fontsize=9)
fig.tight_layout(rect=[0,.055,1,.95])
for ext in ['png','pdf']:fig.savefig(OUT/f'metric_definitions.{ext}',dpi=180,bbox_inches='tight')
plt.close(fig)
print('Wrote tables, source hashes, failure examples, additional checks, and metric_definitions plots. Arithmetic checks passed.')
