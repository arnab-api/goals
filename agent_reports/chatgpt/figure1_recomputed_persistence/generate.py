"""Figure 1 revision: evolving context closes the goal-pursuit loop.

Run with .venv/bin/python. All examples and activation patterns are schematic.
"""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/goals-figure-recomputation')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans', 'svg.fonttype':'none',
                     'pdf.fonttype':42, 'font.size':11})
INK='#202833'; GRAY='#64707C'; BORDER='#D4DAE0'
BLUE='#24647E'; BLUE_PALE='#EDF5F8'; GOLD='#A56818'; GOLD_PALE='#FFF2DB'
W,H=12,8.0
fig,ax=plt.subplots(figsize=(W,H))
fig.subplots_adjust(0,0,1,1)
ax.set(xlim=(0,W),ylim=(H,0));ax.axis('off')
def t(x,y,s,size=11,color=INK,bold=False,ha='left',mono=False,rotation=0):
    return ax.text(x,y,s,fontsize=size,color=color,weight='bold' if bold else 'normal',
        ha=ha,va='center',family='DejaVu Sans Mono' if mono else 'DejaVu Sans',
        linespacing=1.4,rotation=rotation)
def box(x,y,w,h,fill='white',edge=BORDER,r=.07,lw=.9):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',
        fc=fill,ec=edge,lw=lw))
def path(points,color=GRAY,lw=1.2,dashed=False):
    ax.plot(*zip(*points),color=color,lw=lw,ls=(0,(3,3)) if dashed else '-',
        solid_capstyle='round',solid_joinstyle='round')
def arrow(x,y,xx,yy,color=GRAY,lw=1.2):
    ax.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle='-|>',mutation_scale=11,
        lw=lw,color=color,shrinkA=0,shrinkB=0))
def vec(x,y,values):
    for j,a in enumerate(values):
        ax.add_patch(Rectangle((x+j*.17,y),.135,.20,fc=GOLD,alpha=a,ec='none'))

cs=[1.9,6.0,10.1]
t(.35,.28,'(a) Functional persistence through repeated goal reconstruction',13,bold=True)
t(11.65,.53,'PROPOSED ACCOUNT',9,GRAY,ha='right')
for x in [.35,4.45,8.55]:
    box(x,.76,3.10,.78,GOLD_PALE if x==4.45 else 'white',
        '#DDC293' if x==4.45 else BORDER)
t(1.9,1.04,'Task + current progress',12,bold=True,ha='center')
t(1.9,1.32,'What has been done so far?',10.4,GRAY,ha='center')
t(6,1.04,'Reconstruct the current sub-goal',11.5,GOLD,True,ha='center')
t(6,1.32,'The same goal g, while unfinished',10.4,GOLD,ha='center')
t(10.1,1.04,'Take the next action',12,bold=True,ha='center')
t(10.1,1.32,'A changing step toward g',10.4,GRAY,ha='center')
arrow(3.48,1.13,4.40,1.13)
arrow(7.58,1.13,8.50,1.13)
path([(10.1,1.57),(10.1,2.12),(1.9,2.12),(1.9,1.81)],BLUE,1.5)
arrow(1.9,1.81,1.9,1.57,BLUE,1.5)
t(6,1.88,'Update progress; compute again',11.5,BLUE,True,ha='center')
path([(.35,2.42),(11.65,2.42)],BORDER,.9)

t(.35,2.70,'(b) LLM counterpart: generated code changes the next computation',13,bold=True)
box(.35,3.04,11.3,.42,BLUE_PALE,edge='#C5DCE5')
t(.53,3.25,'Task: check whether any word is a palindrome.',11.5,BLUE,bold=True)
t(11.45,3.25,'Instruction remains in context',10.3,BLUE,ha='right')

prefixes=['b =','b = any(','b = any(w == w[::-1]']
outputs=['any','w','for']
profiles=[[.29,.84,.43,.99,.57,.37],[.48,.71,.60,.85,.33,.66],[.69,.46,.83,.65,.42,.30]]
for i,c in enumerate(cs):
    x=c-1.55; sub='₀₁₂'[i]
    arrow(c,3.49,c,3.85,BLUE,1.05)
    box(x,3.88,3.10,.71,fill='#FAFBFC')
    t(c,4.07,f'Available prefix at t{sub}',10.5,GRAY,ha='center')
    t(c,4.37,prefixes[i],11,ha='center',mono=True)
    if i==0:
        # Emphasize the actual assignment pivot without marking future code as input.
        box(c+.035,4.245,.16,.25,GOLD_PALE,edge='#D4B482',r=.02)
        t(c,4.37,prefixes[i],11,ha='center',mono=True)
        t(c+.48,4.37,'← capture pivot',9.4,GOLD)
    arrow(c,4.64,c,4.91)
    box(x+.19,4.94,2.72,.91,fill='#F0F2F5',edge='#BEC7D1')
    t(c,5.15,'Current forward computation',10.8,bold=True,ha='center')
    vec(c-.75,5.36,profiles[i])
    t(c+.40,5.47,f'h{sub}',13,GOLD,True)
    t(c,5.72,'selected head outputs',9.6,GOLD,ha='center')
    arrow(c,5.89,c,6.19)
    box(x+.28,6.22,2.54,.46,fill=BLUE_PALE,edge='#B9D3DF')
    t(c,6.45,f'Next token:  {outputs[i]}',11,BLUE,ha='center',mono=True)

# Connect generated tokens to the following sampled prefix, not h_i to h_j.
# Ellipses make omitted intermediate generation steps explicit.
for i in range(2):
    c=cs[i]; nc=cs[i+1]; gx=c+2.02
    path([(c+1.32,6.45),(gx,6.45),(gx,4.35)],BLUE,1.5)
    arrow(gx,4.35,nc-1.61,4.35,BLUE,1.5)
    t(gx-.14,5.30,'Append to context',10.3,BLUE,ha='center',rotation=90)
    t(gx,6.16,'⋮',16,BLUE,ha='center')

t(6,7.04,'Functional continuity: the goal content can recur as the prefix and activations change.',11.5,GOLD,True,ha='center')
t(6,7.35,'Measurement: capture each h separately and compare goal-related information across positions.',10.8,GRAY,ha='center')
t(.35,7.75,'Schematic snapshots; intermediate tokens omitted. The reconstruction mechanism is a hypothesis.',10,GRAY)

for ext in ['svg','pdf','png']:
    fig.savefig(OUT/f'recomputed_persistence.{ext}',dpi=200,facecolor='white')
plt.close(fig)
print(OUT/'recomputed_persistence.svg')
