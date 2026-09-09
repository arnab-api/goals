"""Paper-sized GoalScope design alternatives inspired by goal-poster.pdf."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/goalscope-paper-mpl')
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.backends.backend_pdf import PdfPages
O=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42})
INK='#25252B'; GRAY='#66666E'; LINE='#BCC1C6'; GOLD='#B77A20'; GREEN='#28765D'; PURPLE='#76559A'
figs=[]
def new(w,h):
 f,a=plt.subplots(figsize=(w,h)); f.subplots_adjust(0,0,1,1);a.set(xlim=(0,w),ylim=(h,0));a.axis('off');return f,a
def t(a,x,y,s,size=8,c=INK,b=False,mono=False,ha='left'):
 return a.text(x,y,s,fontsize=size,color=c,weight='bold' if b else 'normal',fontfamily='DejaVu Sans Mono' if mono else 'DejaVu Sans',va='center',ha=ha,linespacing=1.35)
def box(a,x,y,w,h,c='white',edge=LINE):
 a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.015,rounding_size=0.04',facecolor=c,edgecolor=edge,linewidth=.65))
def arr(a,x,y,xx,yy,c=GRAY,rad=0):
 a.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle='-|>',mutation_scale=7,linewidth=.85,color=c,connectionstyle=f'arc3,rad={rad}'))
def mark(a,x,y,token='=',c=GOLD):
 q=t(a,x,y,token,8,c,True,True);q.set_bbox(dict(facecolor='#FFF0CB' if c==GOLD else '#E3F1EA',edgecolor='none',pad=1.4))
def source(a,x,y,w=2.6):
 t(a,x,y,'(a) Source computation',8.5,b=True)
 box(a,x,y+.17,w,1.12)
 t(a,x+.1,y+.35,'Check whether n has more than',8)
 t(a,x+.1,y+.52,'two ones in binary.',8)
 t(a,x+.1,y+.78,'def check(n):',8,mono=True)
 t(a,x+.1,y+1.00,'    a ',8,mono=True);mark(a,x+.5,y+1.00)
 t(a,x+.7,y+1.00,'← capture at t',7,c=GOLD)
 return x+.55,y+1.12
def target(a,x,y,w,kind):
 color=GOLD if kind=='describe' else GREEN
 box(a,x,y,w,.92)
 if kind=='describe':
  t(a,x+.1,y+.17,'What are you doing at this line?',7.7)
  t(a,x+.1,y+.43,'⊕',11,c=color)
  t(a,x+.33,y+.43,'Assistant response',7.5,c=GRAY)
  t(a,x+.1,y+.72,'“Counting the ones in binary.”',8,c=color)
 else:
  t(a,x+.1,y+.17,'Check a condition in a C++ function.',7.7)
  t(a,x+.1,y+.43,'bool check(int value) { return',7.4,mono=True)
  t(a,x+.1,y+.72,'⊕',11,c=color)
  t(a,x+.32,y+.72,'__builtin_popcount(value)>2;',7.4,c=color,mono=True)
def save(f,n):
 for ext in ['pdf','svg','png']:f.savefig(O/f'{n}.{ext}',dpi=220,facecolor='white')
 figs.append((f,n))

# 1: poster's fork, compressed for a two-column paper.
f,a=new(7,2.95)
px,py=source(a,.12,.2,2.45)
arr(a,px,py,1.34,1.85,GOLD)
box(a,.48,1.86,2.04,.65,'#FAF8F1')
t(a,1.5,2.04,'Selected attention heads',8,ha='center')
t(a,1.5,2.30,r'$g_t=\sum_{(\ell,h)\in H}W_O^{\ell,h}a_t^{\ell,h}$',10,ha='center')
t(a,3.78,.20,'(b) Describe the sub-step',8.5,b=True)
target(a,3.78,.38,3.07,'describe')
t(a,3.78,1.59,'(c) Implement the sub-step',8.5,b=True)
target(a,3.78,1.77,3.07,'implement')
arr(a,2.54,2.14,3.76,.82,GOLD,rad=-.1)
arr(a,2.54,2.25,3.76,2.43,GREEN)
t(a,3.23,1.56,'inject',7,c=GRAY,ha='center')
t(a,.14,2.78,'Shaded token: source capture     ⊕: activation injection into a separate model run',7,c=GRAY)
save(f,'P1_describe_and_implement')

# 2: early encoding, horizontal methods figure.
f,a=new(7,2.72)
t(a,.12,.17,'(a) Capture before the sub-step is written',8.5,b=True)
box(a,.12,.37,3.1,1.33)
t(a,.23,.57,'Task: more than two ones in binary',8)
t(a,.23,.88,'def check(n):',8,mono=True)
t(a,.23,1.12,'    a ',8,mono=True);mark(a,.63,1.12)
t(a,.77,1.12," bin(n).count('1') > 2",8,c=GRAY,mono=True)
t(a,.23,1.49,'Capture t',7,c=GOLD)
arr(a,.55,1.4,.66,1.23,GOLD)
t(a,1.29,1.49,'Later continuation (not captured)',7,c=GRAY)
arr(a,.7,1.24,.7,2.10,GOLD)
t(a,1.5,2.18,r'$g_t=\sum_{(\ell,h)\in H}W_O^{\ell,h}a_t^{\ell,h}$',10,ha='center')
arr(a,2.83,2.17,3.62,2.17,GOLD)
t(a,3.38,1.93,'inject',7,c=GRAY,ha='center')
t(a,3.68,.17,'(b) Read out in a new context',8.5,b=True)
box(a,3.68,.37,3.16,1.33)
t(a,3.8,.60,'What are you doing at this line?',8)
t(a,3.8,.96,'Probe residual stream   ⊕',8)
t(a,3.8,1.38,'“Counting the ones in binary.”',8,c=GOLD)
box(a,3.68,1.91,3.16,.52,'#FAF8F1')
t(a,5.25,2.17,r'$r_{\ell,s}\leftarrow r_{\ell,s}+\alpha\|r_{\ell,s}\|\widehat{g}_t$',11,ha='center')
arr(a,5.8,1.89,5.8,1.72,GOLD)
t(a,.12,2.55,'Source code is absent from the probe prompt. The captured direction steers probe generation.',7,c=GRAY)
save(f,'P2_early_goal_readout')

# 3: single-column poster concept with vertical compact flow.
f,a=new(3.35,4.75)
source(a,.12,.15,3.10)
arr(a,1.68,1.45,1.68,1.72,GOLD)
t(a,1.68,1.90,r'$g_t=\sum_{(\ell,h)\in H}W_O^{\ell,h}a_t^{\ell,h}$',11,ha='center')
t(a,1.68,2.16,'Selected head outputs → new context',7.5,c=GRAY,ha='center')
t(a,.12,2.46,'(b) Describe',8.5,b=True)
target(a,.12,2.62,3.1,'describe')
t(a,.12,3.81,'(c) Implement',8.5,b=True)
target(a,.12,3.97,3.1,'implement')
# Increase canvas slightly to leave clean bottom margin.
f.set_size_inches(3.35,5.04);a.set_ylim(5.04,0)
a.plot([2.95,3.31,3.31],[1.90,1.90,4.69],color=GRAY,lw=.7)
arr(a,3.31,3.05,3.21,3.05,GOLD)
arr(a,3.31,4.69,3.21,4.69,GREEN)
save(f,'P3_single_column')

with PdfPages(O/'all_options.pdf') as pdf:
 for f,_ in figs:pdf.savefig(f,facecolor='white')
captions='''# GoalScope paper figure drafts

These revisions follow the central GoalScope panel in [the poster](../../../Overleaf-draft/REF/goal-poster.pdf): capture selected-head outputs at a marked code position, then steer a new context to describe or implement the sub-step. Figures are sized for a paper (7-inch full-width or 3.35-inch single-column), with 7–8.5 pt body text, no presentation title, and editable SVG/PDF output.

The code and outputs are schematic, not newly measured examples. The source task follows the poster's binary-count example, but the C++ illustration uses the matching threshold >2. The poster displays >5, so its exact output should not be presented as complete task transfer without discussing that mismatch. The faithfulness illustration concerns the current operation; the implementation illustration conveys the intended full predicate. Prompt text is abbreviated. Use actual saved continuations before publication.

## P1 — Describe and implement (recommended)

**Suggested caption.** GoalScope captures a sub-step signal and transfers it to new contexts. (a) At the marked source token t, we sum the selected attention heads' contributions in residual-stream coordinates to obtain g_t. The extracted signal steers separate runs of the same model to (b) describe the sub-step under a verbalization prompt or (c) implement it under a code-generation prompt. Source code is not included as text in either target prompt. Colored target text denotes schematic generated continuations; ⊕ denotes activation injection, not an input token. Generation is steered at the target's current position across a chosen layer band. Examples are illustrative.

The fork is the key message; source capture and target injections must remain visually distinct. This is the closest to the poster's high-level framing.

## P2 — Goal readout before implementation

**Suggested caption.** GoalScope verbalizes the sub-step encoded before its implementation appears in generated text. (a) We capture selected-head outputs at the highlighted assignment token. The gray continuation is shown only to identify the upcoming computation; it is causally downstream of the captured state. (b) Their summed direction steers a separate model run with a fixed verbalization prompt. In the norm-relative additive variant, r_{ℓ,s} ← r_{ℓ,s} + α‖r_{ℓ,s}‖ĝ_t at selected probe layers and decoding positions. Examples are schematic; the figure explains the extraction/injection procedure, not a measured success rate.

This is narrower than P1 and is suitable if the paper figure should introduce only the GoalScope readout. Do not infer early-encoding performance from the illustrative example. The diagram depicts the norm-relative variant used in DIFF evaluation; the poster also reports an unnormalized additive setting. Keep their alpha values and settings separate.

## P3 — Single-column describe/implement

Use the P1 caption. This version stacks the branches for a 3.35-inch column and retains readable text without shrinking a wide figure. Right-side colored arrows indicate the two target contexts; they are not a sequential dependency between target runs.

## Provenance and implementation

Reference inspected: Overleaf-draft/REF/goal-poster.pdf, page 1, central GoalScope panel. Code consulted in the preceding drafts: src/goals/goalscope.py and src/goals/head_loc_attn.py. The current drafts intentionally omit head counts, alpha values, layer ranges, and performance bars; these depend on the specific experiment and belong in the caption/methods once the result to illustrate is selected. The original poster has multiple source capture positions; the drafts focus on one early position to keep the procedure readable.

Run `python agent_reports/chatgpt/goalscope_paper_figures/generate.py` to regenerate. PDFs preserve physical dimensions and embed fonts; SVG text remains editable.
'''
(O/'captions.md').write_text(captions)
html='''<!doctype html><meta charset="utf-8"><title>GoalScope paper figures</title><style>body{font:15px system-ui;color:#25252b;margin:30px auto;max-width:1000px;background:#f3f3f3}section,header{padding:24px;margin:20px 0;background:white}img{display:block;max-width:100%;margin:20px auto}a{color:#28765d}p{line-height:1.5}</style><header><h1>GoalScope · paper figure alternatives</h1><p>Poster-inspired revisions at paper dimensions. All outputs are schematic. <a href="all_options.pdf">Combined PDF</a> · <a href="captions.md">Captions and design notes</a></p></header>'''
for _,n in figs:
 html+=f'<section><h2>{n.replace("_"," ")}</h2><a href="{n}.pdf">PDF</a> · <a href="{n}.svg">SVG</a><img src="{n}.svg" style="width:{672 if n[:2]!="P3" else 322}px"></section>'
(O/'index.html').write_text(html)
print('Created three paper-sized options and captions.')
