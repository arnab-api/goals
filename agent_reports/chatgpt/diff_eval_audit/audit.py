"""Read-only DIFF result audit. Standard library only; never imports GPU/oracle code.

Writes a frozen compressed snapshot, input hashes, per-config and per-run tables,
paired comparisons, and descriptive integrity checks alongside this script.
"""
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone
import csv, gzip, hashlib, json, math

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
BASE = ROOT / 'results/diff'
ACCEPT = {'src', 'both'}

def ratio(k,n): return k/n if n else None
def wilson(k,n):
    if not n:return [None,None]
    z=1.959963984540054;p=k/n;den=1+z*z/n
    mid=(p+z*z/(2*n))/den
    d=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [mid-d,mid+d]

def metrics(rows):
    ca=[r['causal'] for r in rows if r.get('causal')]
    ro=[r['readout'] for r in rows if r.get('readout')]
    ga=[r['gaze'] for r in rows if r.get('gaze')]
    valid=[r for r in ro if r['source'] in ACCEPT and r['dest'] in {'dst','both'}]
    parsed=[r for r in ca if r['label']!='parse_error']
    hit=sum(r['label'] in ACCEPT for r in ca)
    strict=sum(r['label']=='src' for r in ca)
    strict_valid=[r for r in ro if r['source']=='src' and r['dest']=='dst']
    both_pool=[r for r in rows if r.get('causal') and r.get('readout')]
    joint=Counter(f"c{int(r['causal']['label'] in ACCEPT)}_r{int(r['readout']['patched'] in ACCEPT)}" for r in both_pool)
    out=dict(n=len(rows),n_causal=len(ca),causal_hits=hit,causal=ratio(hit,len(ca)),
             causal_strict=ratio(strict,len(ca)),causal_parsed=ratio(hit,len(parsed)),
             causal_parse_errors=len(ca)-len(parsed),n_readout=len(ro),
             ro_hits=sum(r['patched'] in ACCEPT for r in ro),
             ro=ratio(sum(r['patched'] in ACCEPT for r in ro),len(ro)),
             ro_strict=ratio(sum(r['patched']=='src' for r in ro),len(ro)),
             n_valid=len(valid),ro_valid=ratio(len(valid),len(ro)),
             ro_given_valid=ratio(sum(r['patched'] in ACCEPT for r in valid),len(valid)),
             ro_strict_given_strict_valid=ratio(sum(r['patched']=='src' for r in strict_valid),len(strict_valid)),
             ro_src_baseline=ratio(sum(r['source'] in ACCEPT for r in ro),len(ro)),
             ro_dst_baseline=ratio(sum(r['dest'] in {'dst','both'} for r in ro),len(ro)),
             n_gaze=len(ga),gaze=ratio(sum(r['shifted'] for r in ga),len(ga)),
             gaze_clean=ratio(sum(r.get('clean_prefers_src',False) for r in ga),len(ga)),
             c1_r0=joint['c1_r0'],c0_r1=joint['c0_r1'],c1_r1=joint['c1_r1'],c0_r0=joint['c0_r0'])
    out['causal_ci_lo'],out['causal_ci_hi']=wilson(hit,len(ca))
    out['ro_ci_lo'],out['ro_ci_hi']=wilson(out['ro_hits'],len(ro))
    for name,col in [('causal_labels',[r['label'] for r in ca]),('patched_ro_labels',[r['patched'] for r in ro])]:
        out[name]=dict(Counter(col))
    return out

def csvwrite(name,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        w.writerows({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)

def main():
    configs=[]; manifest=[];issues=[]
    for p in sorted(BASE.rglob('results.json')):
        raw=p.read_bytes();rows=json.loads(raw)
        model,mechanism,setup,run,config=p.relative_to(BASE).parts[:-1]
        mp=p.with_name('meta.json');meta=json.loads(mp.read_text()) if mp.exists() else None
        info=dict(model=model,mechanism=mechanism,setup=setup,run=run,config=config,
                  path=str(p.relative_to(ROOT)),meta=meta,rows=rows)
        configs.append(info)
        manifest.append(dict(path=info['path'],sha256=hashlib.sha256(raw).hexdigest(),n=len(rows),
                             mtime_ns=p.stat().st_mtime_ns,has_meta=meta is not None))
        keys=[r['key'] for r in rows]
        if len(set(keys))!=len(keys):issues.append(dict(path=info['path'],kind='duplicate_keys'))
        for r in rows:
            ca,ro,ga=r.get('causal'),r.get('readout'),r.get('gaze')
            if ca and ca['triggered']!=(ca['label'] in ACCEPT):issues.append(dict(path=info['path'],key=r['key'],kind='causal_flag'))
            if ro and (ro['shifted']!=(ro['patched'] in ACCEPT) or ro['valid']!=(ro['source'] in ACCEPT and ro['dest'] in {'dst','both'})):
                issues.append(dict(path=info['path'],key=r['key'],kind='readout_flag'))
            if ga and all(k in ga for k in ['src_patched','dst_patched']):
                if ga['shifted']!=(ga['src_patched']>ga['dst_patched']):issues.append(dict(path=info['path'],key=r['key'],kind='gaze_flag'))
        stats=metrics(rows)
        if meta:
            for m,k in [('n_instances','n'),('causal_trigger_rate','causal'),('readout_shift_rate','ro'),('readout_valid_rate','ro_valid'),('readout_shift_rate_valid','ro_given_valid'),('gaze_shift_rate','gaze')]:
                if m in meta and meta[m] is not None and stats[k] is not None and abs(meta[m]-stats[k])>1e-10:
                    issues.append(dict(path=info['path'],kind='meta_mismatch',field=m,meta=meta[m],actual=stats[k]))
    stamp=datetime.now(timezone.utc).isoformat()
    snapshot=dict(timestamp=stamp,configs=configs)
    with gzip.open(OUT/'snapshot.json.gz','wt') as f:json.dump(snapshot,f)
    (OUT/'manifest.json').write_text(json.dumps(dict(timestamp=stamp,inputs=manifest,issues=issues),indent=2))
    groups=defaultdict(list);cfgrows=[];axes=defaultdict(list)
    for c in configs:
        variant=('var_eq' if 'var_eq' in c['config'] else 'indent') if c['mechanism']=='kv_transfer' and c['setup']=='multi_pred' else '-'
        key=(c['mechanism'],c['setup'],c['run'],variant)
        groups[key].extend(c['rows'])
        cfgrows.append({k:c[k] for k in ['mechanism','setup','run','config','path']}|dict(has_meta=c['meta'] is not None,variant=variant)|metrics(c['rows']))
        for r in c['rows']:
            for axis,value in [('datatype',r['datatype']),('language',r['src_lang']+'->'+r['dst_lang']),('naming',r.get('naming','?'))]:
                axes[key+(axis,value)].append(r)
    summaries=[dict(zip(['mechanism','setup','run','variant'],k))|metrics(v) for k,v in sorted(groups.items())]
    axrows=[dict(zip(['mechanism','setup','run','variant','axis','value'],k))|metrics(v) for k,v in sorted(axes.items())]
    csvwrite('run_summary.csv',summaries);csvwrite('config_summary.csv',cfgrows);csvwrite('axis_summary.csv',axrows)
    (OUT/'run_summary.json').write_text(json.dumps(summaries,indent=2))
    # Matched config+key AND exact source/destination text. Report overlap, not just config averages.
    byrun=defaultdict(dict)
    for c in configs:
        for r in c['rows']:byrun[(c['mechanism'],c['setup'],c['run'])][(c['config'],r['key'])]=r
    paired=[]
    def compare(mech,setup,a,b,filter_cfg=lambda cfg:True):
        aa={k:v for k,v in byrun[(mech,setup,a)].items() if filter_cfg(k[0])}
        bb={k:v for k,v in byrun[(mech,setup,b)].items() if filter_cfg(k[0])}
        shared=aa.keys()&bb.keys();same=[];mismatch=0
        for k in sorted(shared):
            x,y=aa[k],bb[k]
            if any(x['raw'][p]!=y['raw'][p] for p in ['src_prompt','dst_prompt']):mismatch+=1;continue
            same.append((x,y))
        if not shared:return
        result=dict(mechanism=mech,setup=setup,a=a,b=b,n_a=len(aa),n_b=len(bb),shared_keys=len(shared),prompt_mismatch=mismatch,n_exact=len(same))
        for met,label in [('causal','label'),('readout','patched')]:
            ps=[(x[met][label] in ACCEPT,y[met][label] in ACCEPT) for x,y in same if x.get(met) and y.get(met)]
            result.update({met+'_a':ratio(sum(x for x,y in ps),len(ps)),met+'_b':ratio(sum(y for x,y in ps),len(ps)),
                           met+'_gain':sum(not x and y for x,y in ps),met+'_loss':sum(x and not y for x,y in ps)})
        result['identical_patched_generation']=sum(x['raw']['gen_patched']==y['raw']['gen_patched'] for x,y in same)
        result['identical_patched_readout']=sum(x['raw']['ro_patched']==y['raw']['ro_patched'] for x,y in same)
        paired.append(result)
    r122='ph-goal__abl-33-43-restrict__a2__pad'
    r250='ph-goal_250__ro-goal_250__abl-33-43-restrict__a2__pad'
    off250='ph-goal_250__ro-goal_250__abl-off__a2__pad'
    compare('kv_transfer','multi_pred',r122,r250,lambda c:'var_eq' in c)
    compare('kv_transfer','multi_pred',r250,off250,lambda c:'var_eq' in c)
    compare('kv_transfer','dual_comp',r122,off250)
    for setup in ['multi_pred','filter']:
        compare('q_transfer',setup,'ph-goal__a1','ph-goal__a2')
        compare('q_transfer',setup,'ph-goal__a2','ph-goal_250__ro-goal_250__a2')
    csvwrite('paired_comparisons.csv',paired)
    (OUT/'paired_comparisons.json').write_text(json.dumps(paired,indent=2))
    # Head set composition and nesting.
    hroot=ROOT/'head_sets/Qwen2.5-Coder-32B-Instruct'
    a=set(map(tuple,json.loads((hroot/'goal__lam20_thr0.9_band25-56.json').read_text())))
    b=set(map(tuple,json.loads((hroot/'goal_topk/goal__lam20_top250.json').read_text())))
    heads=dict(n122=len(a),n250=len(b),intersection=len(a&b),removed=sorted(a-b),added=len(b-a),
               top250_layers=dict(sorted(Counter(l for l,h in b).items())),
               top250_bands=dict(Counter('0-27' if l<28 else '28-55' if l<=55 else '56-63' for l,h in b)))
    (OUT/'head_sets.json').write_text(json.dumps(heads,indent=2))
    print('SNAPSHOT',stamp,'configs',len(configs),'records',sum(len(c['rows']) for c in configs),'issues',len(issues))
    for s in summaries:
        fmt=lambda k:'-' if s[k] is None else f'{s[k]:.4f}'
        print(s['mechanism'],s['setup'],s['run'],s['variant'],'N',s['n'],'causal',fmt('causal'),'ro',fmt('ro'),'valid',fmt('ro_valid'),'ro|valid',fmt('ro_given_valid'),'gaze',fmt('gaze'),'base',fmt('gaze_clean'))
    print('PAIRS',json.dumps(paired,indent=2));print('HEADS',heads)

if __name__=='__main__':main()
