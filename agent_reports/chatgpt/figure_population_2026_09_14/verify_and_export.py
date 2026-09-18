"""Check preservation and export the compiled review draft; no paper-source edits."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
PAPER = ROOT/'Overleaf-draft/Goals-Overleaf'
BEFORE = REPORT/'before'
BUILD = Path('/tmp/goals-figure-population-build')

def prose(text):
    for kind in ('figure','table'):
        text=re.sub(rf'\\begin\{{{kind}\}}.*?\\end\{{{kind}\}}','',text,flags=re.S)
    text=re.sub(r'\\input\{Text/Figures/[^}]+\}','',text)
    return [line for line in text.splitlines() if line.strip()]

checks={}
for before in (BEFORE/'Text').glob('*.tex'):
    if before.name=='201_Appendix.tex':continue
    checks[before.name]=prose(before.read_text())==prose((PAPER/'Text'/before.name).read_text())
assert all(checks.values()),checks
for name in ['main.tex','packages.tex']:
    assert (PAPER/name).read_bytes()==(BEFORE/name).read_bytes(),name
for before in (BEFORE/'Text/Appendix/prompts').rglob('*'):
    if before.is_file():
        assert before.read_bytes()==(PAPER/before.relative_to(BEFORE)).read_bytes()
for name in ['slide_crops.json','updated_figure_crops.json']:
    data=json.loads((REPORT/name).read_text())
    assert hashlib.sha256((ROOT/data['source']).read_bytes()).hexdigest()==data['sha256']

log=(BUILD/'main.log').read_text()
assert not re.search(r'(^!|Reference .* undefined|Citation .* undefined|Float too large|Overfull \\[hv]box)',log,re.M)
aux=(BUILD/'main.aux').read_text()
labels={m[0]:{'number':m[1],'page':int(m[2])} for m in re.findall(r'\\newlabel\{((?:fig|tab):[^}]+)\}\{\{([^}]+)\}\{(\d+)\}',aux)}
expected=['fig:teaser','fig:goal-eval','fig:sparsity','fig:kv-transfer','tab:kv-transfer','fig:ovreader',
          'fig:task-prompt','fig:localization','fig:ablation-setup','tab:ablation-example','fig:kv-freeform','fig:query-order']
assert all(k in labels for k in expected)
for name in ['goal_eval_matched','sparsity_ablation_paired','ov_reader_combined','ov_reader_other_pivots','kv_transfer_example']:
    assert (PAPER/'Figures'/f'{name}.pdf').is_file()

shutil.copy2(BUILD/'main.pdf',PAPER/'main.pdf')
shutil.copy2(BUILD/'main.pdf',REPORT/'revised_draft.pdf')
shutil.copy2(BUILD/'main.log',REPORT/'build.log')
(REPORT/'main_prose_check.json').write_text(json.dumps(checks,indent=2))
(REPORT/'figure_locations.json').write_text(json.dumps(labels,indent=2))
pdfinfo=subprocess.check_output(['pdfinfo',str(BUILD/'main.pdf')],text=True)
validation={
    'main_prose_unchanged':checks,
    'main_tex_and_packages_unchanged':True,
    'exact_prompt_and_oracle_files_unchanged':True,
    'reference_pdfs_unchanged':True,
    'unresolved_references_or_citations':False,
    'overfull_boxes_or_oversized_floats':False,
    'pdf_pages':int(re.search(r'Pages:\s+(\d+)',pdfinfo).group(1)),
    'pdf_sha256':hashlib.sha256((BUILD/'main.pdf').read_bytes()).hexdigest(),
    'remaining_warnings':[s.strip() for s in log.splitlines() if 'Warning:' in s],
}
(REPORT/'validation.json').write_text(json.dumps(validation,indent=2))
print(json.dumps({'pages':validation['pdf_pages'],'main_prose_files_checked':len(checks),'main_figures':{k:labels[k] for k in expected[:6]}},indent=2))
