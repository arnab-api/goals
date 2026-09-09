"""Poster-style GoalScope teaser: code pivot → head contributions → verbalization."""
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
INK='#28282D'; GRAY='#77777E'; GOLD='#B87922'; PALE='#FFF0CB'; BORDER='#B9BABF'
figs=[]
def new(h):
 f,a=plt.subplots(figsize=(7,h));f.subplots_adjust(0,0,1,1);a.set(xlim=(0,7),ylim=(h,0));a.axis('off');return f,a
def t(a,x,y,s,z=8,c=INK,b=False,mono=False,ha='left'):
 return a.text(x,y,s,fontsize=z,color=c,weight='bold' if b else 'normal',family='DejaVu Sans Mono' if mono else 'DejaVu Sans',ha=ha,va='center',linespacing=1.3)
def box(a,x,y,w,h,fill='white',edge=BORDER):
 a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.015,rounding_size=0.06',facecolor=fill,edgecolor=edge,lw=.7))
def arr(a,x,y,xx,yy,rad=0,dash=False):
 a.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle='-|>',mutation_scale=8,color=GOLD,lw=1,linestyle='--' if dash else '-',connectionstyle=f'arc3,rad={rad}'))
def source(a,x,y):
 t(a,x,y,'Source computation',8.5,b=True)
 t(a,x,y+.32,'…',9)
 t(a,x,y+.57,'n has more than two ones',8)
 t(a,x,y+.77,'in its binary representation.',8)
 t(a,x,y+1.03,'…',9)
 t(a,x,y+1.28,'def check(n):',8,mono=True)
 t(a,x,y+1.54,'    a ',8,mono=True)
 q=t(a,x+.40,y+1.54,'=',8,GOLD,True,True)
 q.set_bbox(dict(facecolor=PALE,edgecolor='none',pad=2))
 t(a,x+.49,y+1.54," bin(n).count('1') > 2",8,GRAY,mono=True)
 t(a,x+.46,y+1.80,'pivot',7,GOLD,ha='center')
 return x+.44,y+1.61
def heads(a,x,y,compact=False):
 # Each tile is a projected head contribution at the same source position.
 for i in range(4):
  xx=x+i*.24
  a.add_patch(Rectangle((xx,y),.17,.30,facecolor=PALE,edgecolor=GOLD,lw=.65))
  for j in range(3):
   a.plot([xx+.035,xx+.13],[y+.07+j*.075]*2,color=GOLD,lw=.8,alpha=.4+.15*((i+j)%3))
def readout(a,x,y,w=2.72):
 box(a,x,y,w,1.40)
 t(a,x+.10,y+.16,'GoalScope readout',8.5,b=True)
 box(a,x+.33,y+.34,w-.45,.44,'#F3F3F4')
 t(a,x+.43,y+.56,'What are you doing at this line?',7.4)
 box(a,x+.10,y+.91,w-.25,.34,'#FFFAEC',GOLD)
 t(a,x+.18,y+1.08,'“Counting the ones in binary.”',7.8)
 return x+.10,y+1.08
def save(f,n):
 for e in ['svg','pdf','png']:f.savefig(O/f'{n}.{e}',dpi=220,facecolor='white')
 figs.append((f,n))

# T1: closest to poster, annotate a single diagonal transfer arrow.
f,a=new(2.65)
px,py=source(a,.15,.25)
rx,ry=readout(a,4.1,.62)
arr(a,px,py,2.37,1.46,rad=-.07,dash=True)
heads(a,2.40,1.04)
t(a,2.87,.65,'Cache selected',7.8,b=True,ha='center')
t(a,2.87,.83,'head contributions',7.8,b=True,ha='center')
t(a,2.87,1.53,'sum & inject',7.4,c=GOLD,ha='center')
arr(a,3.40,1.40,rx-.03,ry,rad=-.04,dash=True)
t(a,.15,2.40,'Gray code: later continuation',7,c=GRAY)
save(f,'T1_poster_with_cache')

# T2: small magnified inset beside pivot; continuous arrow to readout.
f,a=new(2.9)
px,py=source(a,.15,.17)
rx,ry=readout(a,4.10,.60)
box(a,1.57,2.05,2.14,.64,'#FFFCF4',GOLD)
heads(a,1.70,2.25)
t(a,2.76,2.25,'Selected head',7.2)
t(a,2.76,2.44,'contributions',7.2)
arr(a,px,py,1.59,2.25,rad=.05)
t(a,.76,2.10,'cache',7.3,c=GOLD)
arr(a,3.72,2.35,rx-.03,ry,rad=.12,dash=True)
t(a,3.85,2.57,'sum & inject',7.3,c=GOLD,ha='center')
save(f,'T2_pivot_inset')

# T3: clean three-stage strip; source example shorter, no future continuation.
f,a=new(2.35)
t(a,.13,.22,'Source prefix',8.5,b=True)
t(a,.13,.62,'Task: more than two',8)
t(a,.13,.82,'ones in binary.',8)
t(a,.13,1.14,'def check(n):',8,mono=True)
t(a,.13,1.40,'    a ',8,mono=True)
q=t(a,.53,1.40,'=',8,GOLD,True,True);q.set_bbox(dict(facecolor=PALE,edgecolor='none',pad=2))
t(a,.53,1.71,'pivot',7,GOLD,ha='center')
arr(a,.68,1.41,2.23,1.41)
box(a,2.28,.74,1.50,1.11,'#FFFCF4',GOLD)
t(a,3.03,.96,'Cache head',8,b=True,ha='center')
t(a,3.03,1.14,'contributions',8,b=True,ha='center')
heads(a,2.58,1.36)
arr(a,3.81,1.41,4.15,1.41,dash=True)
t(a,4.00,1.91,'sum & inject',7.2,c=GOLD,ha='center')
readout(a,4.20,.61,2.65)
save(f,'T3_compact_strip')

with PdfPages(O/'all_options.pdf') as p:
 for f,_ in figs:p.savefig(f,facecolor='white')
(O/'captions.md').write_text('''# Poster-style GoalScope teaser revisions

Only the faithfulness/readout context is shown. No equations or causal implementation branch. Each gold tile represents a selected attention head's contribution projected into residual-stream coordinates at the highlighted source pivot. The contributions are summed and injected into a separate run of the same model. Tiles are schematic, not an actual count or layer arrangement.

**Recommended: T1**, closest to the poster's direct source-to-readout composition. T2 moves the cache to a pivot-linked inset. T3 is a more explicit compact sequence, showing only the source prefix.

**Proposed caption.** GoalScope reads out the sub-step encoded at a code position. At the highlighted pivot, we cache the selected attention heads' contributions and sum them into a steering vector. Injecting this vector into a separate run of the same model, prompted to describe its current activity, yields a natural-language readout. The target prompt does not include the source instructions or code. In T1/T2, gray text shows the later continuation, which is not part of the captured prefix. Gold tiles schematically denote projected head contributions. Prompt wording is abbreviated and the example readout is illustrative; replace it with a recorded example before publication.

7-inch paper figures with 7–8.5 pt text. Editable SVG, embedded-font PDF, PNG, combined PDF, and HTML comparison included. Reference: Overleaf-draft/REF/goal-poster.pdf, central GoalScope panel. Previous options remain unchanged. Regenerate with `python agent_reports/chatgpt/goalscope_teaser_figures/generate.py`.
''')
html='''<!doctype html><meta charset="utf-8"><title>GoalScope teaser revisions</title><style>body{max-width:950px;margin:30px auto;font:15px system-ui;background:#f4f4f4;color:#28282d}header,section{padding:22px;background:white;margin:20px 0}img{width:100%}a{color:#946018}</style><header><h1>GoalScope teaser · poster-style revisions</h1><p>Faithfulness only; selected head contributions made visible; no equations.</p><a href="all_options.pdf">All three PDFs</a> · <a href="captions.md">Caption and notes</a></header>'''
for _,n in figs:html+=f'<section><h2>{n.replace("_"," ")}</h2><a href="{n}.pdf">PDF</a> · <a href="{n}.svg">SVG</a><img src="{n}.svg"></section>'
(O/'index.html').write_text(html)
print('Saved three poster-style teaser variants.')
