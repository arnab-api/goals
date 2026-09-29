"""Vector redraw based on Overleaf-draft/REF/fig1_draft.png.

All neural values and generation snippets are illustrative.
"""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/goals-refinement-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, PathPatch
from matplotlib.path import Path as MPath

OUT=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42})
W,H=13.6,7.65
fig,ax=plt.subplots(figsize=(W,H));fig.subplots_adjust(0,0,1,1)
ax.set(xlim=(0,W),ylim=(H,0));ax.axis('off')
INK='#242C35'; GRAY='#737C85'; LINE='#D5D9DE'
BLUE='#376DA0'; READ='#5686AE'; ORANGE='#B46622'; PALE='#FFF1DB'
WRITE='#AD5139'; PURPLE='#725992'
def t(x,y,s,size=11,color=INK,bold=False,ha='left',mono=False):
    return ax.text(x,y,s,fontsize=size,color=color,fontweight='bold' if bold else 'normal',
        ha=ha,va='center',fontfamily='DejaVu Sans Mono' if mono else 'DejaVu Sans',linespacing=1.35)
def box(x,y,w,h,edge=LINE,fill='white',dash=False,lw=.9):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=.09',
        fc=fill,ec=edge,lw=lw,linestyle=(0,(4,3)) if dash else '-'))
def line(ps,color=GRAY,dash=False,lw=1):
    ax.plot(*zip(*ps),color=color,lw=lw,ls=(0,(2,3)) if dash else '-',solid_capstyle='round')
def arrow(x,y,xx,yy,color=GRAY,lw=1.1,dash=False):
    ax.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle='-|>',mutation_scale=10,
        color=color,lw=lw,shrinkA=0,shrinkB=0,linestyle=(0,(4,3)) if dash else '-'))
def bez(points,color=READ,lw=.85,head=False):
    p=MPath(points,[MPath.MOVETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4])
    if head:
        ax.add_patch(FancyArrowPatch(path=p,arrowstyle='-|>',mutation_scale=10,color=color,lw=lw))
    else: ax.add_patch(PathPatch(p,fc='none',ec=color,lw=lw))

xs=[1.80,5.18,8.22,11.04]; ws=[2.17,2.16,1.58,2.20]
cs=[x+w/2 for x,w in zip(xs,ws)]
start,finish=4.53,10.43
t(cs[0],.30,'Intent formation',13,bold=True,ha='center')
t(7.42,.30,'Action phase',13,bold=True,ha='center')
t(cs[3],.30,'Transition',13,bold=True,ha='center')
t(start,.66,'Action onset',10.5,GRAY,ha='center')
t(finish,.66,'Exit',10.5,GRAY,ha='center')
for x in [start,finish]: line([(x,.84),(x,5.76)],'#B3BAC1',True,.85)

t(.28,1.29,'(i) Functional',11.3,BLUE,True)
t(.28,1.52,'account',11.3,BLUE,True)
labs=[r'About to do $g_k$',r'Doing $g_k$','Done?', 'Move to next']
for x,w,lab in zip(xs,ws,labs):
    box(x,1.08,w,.65,BLUE,fill='#F7FAFD')
    t(x+w/2,1.40,lab,13,ha='center')
for a,b in zip(range(3),range(1,4)):
    arrow(xs[a]+ws[a]+.025,1.40,xs[b]-.045,1.40,GRAY)
bez([(cs[2],1.05),(cs[2],.64),(cs[1],.64),(cs[1],1.05)],GRAY,1.05,True)
t(7.7,.94,'not yet',10.3,GRAY,ha='center')
t(10.03,1.58,'yes',10.3,GRAY,ha='center')
line([(.25,1.98),(13.3,1.98)],LINE,.0,.8)

t(.28,2.30,'Context tape',11,bold=True)
t(7.0,2.20,r'$g_k$: get the length of the longest word',11,ha='center')
pieces=[('<task>',1.00,'past'),('…',.50,'past'),('b',.40,'past'),('=',.40,'pivot'),
        ('max',.64,'code'),('(',.25,'code'),('len',.50,'code'),('(',.25,'code'),
        ('w',.33,'code'),(')',.25,'code'),('for',.51,'code'),('w',.33,'code'),
        ('in',.40,'write'),('words',.70,'future'),(')',.25,'future')]
centers=[]; x=1.80
for token,w,kind in pieces:
    if kind=='code' and token=='max': x=4.72
    fill={'past':'#EEF2F6','pivot':PALE,'code':'#F1F3F5','write':'#F9E6DF','future':'#FAFAFA'}[kind]
    edge={'pivot':ORANGE,'write':WRITE}.get(kind,LINE)
    ax.add_patch(Rectangle((x,2.48),w,.39,fc=fill,ec=edge,lw=.8,
        linestyle=(0,(2,2)) if kind=='future' else '-'))
    t(x+w/2,2.675,token,11.5,GRAY if kind=='future' else INK,ha='center',mono=True)
    centers.append((token,x+w/2,kind));x+=w+.07

# Reading arcs terminate at the current neural computation, not at a persistent register.
join=(5.74,4.03)
for token,c,kind in centers:
    if kind in ('past','pivot'):
        bez([(c,2.90),(c,3.35),(5.30,3.40),join],READ,.85)
    elif kind=='code' and token in ('max','len','for','w'):
        bez([(c,2.90),(c,3.38),(5.94,3.59),join],READ,.85)
    elif kind=='write': write_x=c
# Single reverse-direction arrow identifies token production.
bez([(6.68,4.03),(6.68,3.40),(write_x,3.46),(write_x,2.91)],WRITE,1.45,True)
t(11.07,3.30,'Read available context',10.2,READ)
line([(10.70,3.30),(10.99,3.30)],READ,lw=1)
t(11.07,3.59,'Write next token',10.2,WRITE)
arrow(10.70,3.59,10.99,3.59,WRITE)

t(.28,4.24,'(ii) LLM',11.3,ORANGE,True)
t(.28,4.47,'counterpart',11.3,ORANGE,True)
for x,w,lab in zip(xs,ws,labs):
    box(x,4.06,w,.70,ORANGE,'#FFFCF7',dash=True)
    t(x+w/2,4.38,lab,13,ha='center')
for a,b in zip(range(3),range(1,4)):
    arrow(xs[a]+ws[a]+.025,4.39,xs[b]-.045,4.39,GRAY,1,True)
t(cs[1],4.98,'recomputed from context*',9.7,ORANGE,ha='center')
t(cs[2],4.98,'schematic test',9.5,GRAY,ha='center')

t(.28,5.48,'Sampled',10.4,GRAY)
t(.28,5.70,'head outputs',10.4,GRAY)
sample_x=[4.11,5.20,6.22,7.24,8.26,9.28]
levels=[.32,.55,.75,.95,.44,.68]
for i,c in enumerate(sample_x):
    box(c-.34,5.33,.68,.29,ORANGE,PALE,lw=1.1 if i==0 else .7)
    # Constant mean intensity, different patterns: no implied decay curve.
    order=[levels[(j+i)%6] for j in range(6)]
    for j,a in enumerate(order):
        ax.add_patch(Rectangle((c-.28+j*.094,5.385),.065,.18,fc=ORANGE,alpha=a,ec='none'))
t(3.63,5.47,'capture at “=”',9.1,ORANGE,ha='right')
t(11.67,5.47,'Same goal content;',10.3,ORANGE,ha='center')
t(11.67,5.69,'changing activations',10.3,ORANGE,ha='center')

t(5.05,6.03,'Plasticity: transfer the pre-implementation encoding to C++',11.5,PURPLE,True)
box(3.15,6.26,10.04,.90,edge='#D3C7DF',fill='#FBF9FD')
t(3.32,6.45,'std::size_t longest(const auto& words) {',10.5,GRAY,mono=True)
t(3.32,6.73,'  std::size_t v = 0; for (const auto& w : words) v = std::max(v, w.size());',10.3,mono=True)
t(3.32,6.98,'  return v; }',10.3,mono=True)
bez([(sample_x[0]-.06,5.66),(3.76,6.01),(2.77,5.92),(3.12,6.73)],PURPLE,1.25,True)
t(.28,7.43,'* Proposed reconstruction account. Code, sampled activations, and completion test are schematic.',9.5,GRAY)

for ext in ['svg','pdf','png']:
    fig.savefig(OUT/f'goal_pursuit_reference_refined.{ext}',dpi=200,facecolor='white')
plt.close(fig)
print(OUT/'goal_pursuit_reference_refined.svg')
