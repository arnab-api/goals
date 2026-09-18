from pathlib import Path
import hashlib,json,re
ROOT=Path(__file__).resolve().parents[3];REPORT=Path(__file__).resolve().parent
PAPER=ROOT/'Overleaf-draft/Goals-Overleaf';BEFORE=REPORT/'before'
allowed={'Text/Appendix/task.tex','Text/Appendix/attn_head_localization.tex','Text/Appendix/goalscope.tex','Text/Appendix/readout_comparisons.tex','Text/Appendix/tables/corpus.tex','Text/Figures/sparsity_ablation.tex'}
unchanged=[];changed=[]
for p in (BEFORE/'Text').rglob('*'):
    if not p.is_file():continue
    rel=str(p.relative_to(BEFORE));same=p.read_bytes()==(PAPER/rel).read_bytes()
    if not same:assert rel in allowed,rel;changed.append(rel)
    else:unchanged.append(rel)
for name in ['main.tex','packages.tex']:
    assert (BEFORE/name).read_bytes()==(PAPER/name).read_bytes();unchanged.append(name)
# Every already commented-out line in A/B/C remains intact, in the same order.
for name in ['task.tex','attn_head_localization.tex','goalscope.tex']:
    comments=lambda s:[l for l in s.splitlines()if l.lstrip().startswith('%')]
    assert comments((BEFORE/'Text/Appendix'/name).read_text())==comments((PAPER/'Text/Appendix'/name).read_text()),name
data=json.loads((REPORT/'results.json').read_text())
for family in ['scope','sources','models']:
    for r in data[family]:assert abs(r['pre_mean']-sum(r[k]for k in ['prev_end','indent','var','eq'])/4)<1e-12
log=Path('/tmp/goals-appendix-review-build/main.log').read_text()
assert not re.search(r'Overfull \\[hv]box|Float too large|^!',log,re.M)
undefined=sorted(set(re.findall(r"Reference `([^']+)' .*undefined",log)))
assert set(undefined)<={'app:pivots','app:oracle_prompts','fig:goal-eval'},undefined
duplicate=sorted(set(re.findall(r"Label `([^']+)' multiply defined",log)))
assert set(duplicate)<={'fig:freeform-matching','fig:freeform-matching@cref'},duplicate
out={'changed_existing_text_files':changed,'unchanged_existing_text_files':unchanged,'A_B_C_commented_material_preserved':True,'main_body_byte_identical':True,'four_pivot_means_checked':True,'build_overflows':False,'existing_undefined_references':undefined,'existing_duplicate_labels':duplicate}
(REPORT/'verification.json').write_text(json.dumps(out,indent=2)+'\n')
print('Verified: main body unchanged; A–C commented material preserved; four-pivot means correct; no overfull boxes.');print('Existing reference issues:',undefined,duplicate)
