"""Verify appendix-only scope, exact prompts, asset references, and build warnings."""
from pathlib import Path
import hashlib,json,re,subprocess

ROOT=Path(__file__).resolve().parents[3]
REPORT=Path(__file__).resolve().parent
PAPER=ROOT/'Overleaf-draft/Goals-Overleaf'
BEFORE=REPORT/'before'
checks={}
protected=[p for p in (BEFORE/'Text').rglob('*') if p.is_file() and 'Appendix' not in p.relative_to(BEFORE).parts]
protected += [BEFORE/'main.tex', BEFORE/'packages.tex']
for p in protected:
    assert p.read_bytes()==(PAPER/p.relative_to(BEFORE)).read_bytes(),p
checks['main_sources_byte_identical']=[str(p.relative_to(BEFORE))for p in protected]
changed=subprocess.check_output(['git','diff','--name-only'],cwd=PAPER,text=True).splitlines()
assert all(p.startswith('Text/Appendix/') or p=='main.pdf' for p in changed),changed
checks['tracked_changes_within_scope']=True
for p in (BEFORE/'Text/Appendix/prompts').rglob('*.txt'):
    assert p.read_bytes()==(PAPER/p.relative_to(BEFORE)).read_bytes(),p
for p in (PAPER/'Text/Appendix/prompts').glob('*.txt'):
    assert (p.parent/'display'/p.name).read_bytes()==p.read_bytes().replace(b'\n',b'\\n\n'),p
checks['existing_raw_and_display_prompts_unchanged']=True
checks['display_newline_markers_match_raw_prompts']=True
for p in (PAPER/'Text/Appendix').glob('*.tex'):
    for asset in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',p.read_text()):
        assert (PAPER/asset).exists(),(p,asset)
checks['appendix_figure_paths_exist']=True
log=Path('/tmp/goals-appendix-todos-build/main.log').read_text()
assert not re.search(r'Overfull \\[hv]box|Float too large|^!',log,re.M)
undefined=set(re.findall(r"Reference `([^']+)' .*undefined",log))
assert undefined <= {'fig:goal-eval'},undefined
checks['undefined_references']=sorted(undefined)
checks['undefined_reference_note']='Existing main-paper fig:goal-eval references; its input is commented out in the author baseline. Left unchanged.'
checks['overfull_boxes_or_oversize_floats']=False
(REPORT/'verification.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
