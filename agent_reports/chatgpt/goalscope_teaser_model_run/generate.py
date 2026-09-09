"""Figure-1-inspired pivot magnification with explicit source model run."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/goalscope-paper-mpl')
# Reuse drawing primitives without executing earlier figure generation.
base=Path(__file__).resolve().parents[1]/'goalscope_teaser_figures/generate.py'
exec(base.read_text().split('# T1:')[0])
O=Path(__file__).resolve().parent
figs=[]
def run(a,x,y,w=2.62):
 box(a,x,y,w,1.26,'#FAFAFB')
 t(a,x+.10,y+.14,'Source model run',8.2,b=True)
 # Token-aligned columns through a schematic layer stack.
 a.add_patch(Rectangle((x+1.52,y+.30),.46,.75,facecolor=PALE,edgecolor=GOLD,lw=.65))
 for j in range(3):
  yy=y+.38+j*.24
  a.plot([x+.35,x+2.11],[yy+.06]*2,color='#CFD1D5',lw=2,zorder=1)
  for k in range(3):
   a.add_patch(Rectangle((x+.45+k*.36,yy),.13,.12,facecolor='#DDDEE2',edgecolor='none',zorder=2))
  for k in range(2):
   a.add_patch(Rectangle((x+1.57+k*.18,yy),.12,.12,facecolor='#F4CF88',edgecolor=GOLD,lw=.6,zorder=3))
 t(a,x+.20,y+.66,'layers',6.6,c=GRAY,ha='center')
 t(a,x+.88,y+1.12,'…    a',7.4,mono=True,ha='center')
 t(a,x+1.76,y+1.12,'=',8,c=GOLD,b=True,mono=True,ha='center')
 t(a,x+1.68,y+.27,'selected head outputs',6.3,c=GOLD,ha='center')
 return x+1.76,y+.69
def dotted(a,x,y,xx,yy):a.plot([x,xx],[y,yy],color=GOLD,lw=.7,linestyle=(0,(2,2)))

# U1: source prompt with a pivot-linked magnification beneath, as in filter Fig. 1.
f,a=new(3.65)
px,py=source(a,.15,.12)
rx,ry=readout(a,4.13,.52)
# Dotted magnification boundaries originate around the highlighted token.
dotted(a,px-.08,py+.02,.29,2.21)
dotted(a,px+.08,py+.02,2.86,2.21)
cx,cy=run(a,.27,2.22)
arr(a,cx+.10,cy,3.19,2.88)
t(a,3.29,2.18,'Cache contributions',7.7,b=True)
heads(a,3.36,2.37)
t(a,3.81,2.86,'sum',7.2,c=GOLD,ha='center')
arr(a,4.38,2.49,5.39,1.94,rad=.12,dash=True)
t(a,4.91,2.46,'inject',7.2,c=GOLD)
save(f,'U1_pivot_magnification')

# U2: explicit source-run stage across the middle, a compact linear read.
f,a=new(3.1)
t(a,.15,.18,'Source prefix',8.5,b=True)
t(a,.15,.51,'Task: more than two ones in binary.',8)
t(a,.15,.86,'def check(n):',8,mono=True)
t(a,.15,1.10,'    a ',8,mono=True)
q=t(a,.55,1.10,'=',8,GOLD,True,True);q.set_bbox(dict(facecolor=PALE,edgecolor='none',pad=2))
t(a,.55,1.35,'pivot',7,GOLD,ha='center')
dotted(a,.48,1.18,.18,1.63);dotted(a,.65,1.18,2.79,1.63)
cx,cy=run(a,.17,1.65)
rx,ry=readout(a,4.16,.56)
box(a,3.03,1.80,.82,.87,'#FFFCF4',GOLD)
t(a,3.44,1.98,'Cache',7.8,b=True,ha='center')
for j in range(3):
 a.add_patch(Rectangle((3.14+j*.21,2.16),.14,.28,facecolor=PALE,edgecolor=GOLD,lw=.65))
arr(a,cx+.17,cy,3.00,2.32)
arr(a,3.88,2.25,4.26,1.65,rad=.05,dash=True)
t(a,4.10,2.49,'sum & inject',7.3,c=GOLD,ha='center')
save(f,'U2_model_run_to_cache')

with PdfPages(O/'all_options.pdf') as p:
 for f,_ in figs:p.savefig(f,facecolor='white')
(O/'captions.md').write_text('''# GoalScope teaser: explicit model-run capture

Reference inspected: Figure 1, page 1, Overleaf-draft/REF/papers/filter.pdf. Its dotted pivot-to-internals magnification is adapted here to show output capture rather than query transplantation.

**U1** keeps the poster's source-code/faithfulness-readout composition, adding a pivot-linked view of the source model run underneath. **U2** makes the model-run → cache → readout sequence more explicit.

**Proposed caption.** GoalScope reads out a sub-step from selected attention-head outputs. The highlighted source token is expanded into a schematic view of the source model run: rows indicate layers and the gold column identifies the same pivot position across those layers. We cache selected heads' output contributions at that position, sum them in residual-stream coordinates, and inject the resulting signal into a separate run of the same model with a fixed verbalization prompt. The probe does not receive the source code as text. In U1, gray code shows the later continuation, which is not part of the captured prefix. Head counts, layers, and token columns are schematic; tiles represent projected contributions, not raw queries or keys. The readout is illustrative and should be replaced with a saved example for publication.

Both are 7-inch paper figures, with editable SVG and embedded-font PDF. No equations or causality branch. The small repeated gold tiles show selected contributions across layers at one token, not contributions across multiple source positions.
''')
html='''<!doctype html><meta charset="utf-8"><title>GoalScope model-run capture</title><style>body{max-width:1000px;margin:30px auto;font:15px system-ui;background:#f3f3f3}header,section{padding:24px;background:white;margin:20px 0}img{width:100%}a{color:#95651d}</style><header><h1>GoalScope · capture at the pivot inside the model run</h1><a href="all_options.pdf">Combined PDF</a> · <a href="captions.md">Caption</a></header>'''
for _,n in figs:html+=f'<section><h2>{n.replace("_"," ")}</h2><a href="{n}.pdf">PDF</a> · <a href="{n}.svg">SVG</a><img src="{n}.svg"></section>'
(O/'index.html').write_text(html)
print('Created two model-run capture variants.')
