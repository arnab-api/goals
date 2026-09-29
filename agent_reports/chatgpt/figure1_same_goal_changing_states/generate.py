"""Simple Figure 1 proposal: same sub-goal, changing actions and activations."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/goals-figure-same-goal')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42})
INK='#25303B'; GRAY='#697480'; BORDER='#D7DCE1'
GOLD='#A46616'; PALE='#FFF3DF'; BLUE='#39738A'; PURPLE='#725393'
W,H=12.3,6.25
fig,ax=plt.subplots(figsize=(W,H));fig.subplots_adjust(0,0,1,1)
ax.set(xlim=(0,W),ylim=(H,0));ax.axis('off')
def t(x,y,s,size=11,color=INK,bold=False,ha='left',mono=False):
    return ax.text(x,y,s,fontsize=size,color=color,fontweight='bold' if bold else 'normal',
        ha=ha,va='center',fontfamily='DejaVu Sans Mono' if mono else 'DejaVu Sans',linespacing=1.35)
def box(x,y,w,h,fill='white',edge=BORDER,r=.06):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',
        fc=fill,ec=edge,lw=.85))
def path(ps,color=GRAY,dash=False,lw=1.2):
    ax.plot(*zip(*ps),color=color,lw=lw,ls=(0,(3.5,3)) if dash else '-',solid_capstyle='round')
def arrow(x,y,xx,yy,color=GRAY,dash=False,lw=1.2):
    ax.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle='-|>',mutation_scale=10,
        color=color,lw=lw,linestyle=(0,(3.5,3)) if dash else '-',shrinkA=0,shrinkB=0))
def vector(cx,y,vals):
    for j,a in enumerate(vals):
        ax.add_patch(Rectangle((cx-.48+j*.17,y),.135,.18,fc=GOLD,alpha=a,ec='none'))

cs=[3.1,6.7,10.3]
t(.30,.28,'(a) Functional view',12.5,bold=True)
box(1.52,.61,10.38,.47,PALE,edge='#DFBF89')
t(6.71,.845,'Sub-goal: check whether n is prime',13,GOLD,True,ha='center')
for c,label in zip(cs,['Adopt the goal','Pursue the goal','Complete']):
    t(c,1.33,label,11.5,ha='center')
arrow(4.0,1.33,5.70,1.33)
arrow(7.62,1.33,9.60,1.33)
path([(.30,1.67),(11.95,1.67)],BORDER,lw=.8)

t(.30,1.97,'(b) LLM',12.5,bold=True)
t(cs[0],1.98,'Before action',11,BLUE,True,ha='center')
t(8.5,1.98,'Continued pursuit',11,BLUE,True,ha='center')
# Short bracket joins the later snapshots semantically, without linking latent vectors.
path([(5.45,2.17),(5.45,2.11),(11.55,2.11),(11.55,2.17)],BLUE,lw=.9)
profiles=[[.30,.92,.52,.77,.38,.67],[.52,.71,.34,.95,.62,.43],[.74,.43,.66,.82,.31,.57]]
for i,c in enumerate(cs):
    box(c-1.25,2.30,2.50,.66,PALE,edge='#DFBF89')
    t(c,2.49,'“check primality”',11,GOLD,ha='center')
    vector(c-.17,2.68,profiles[i])
    t(c+.64,2.77,'h'+'₀₁₂'[i],12,GOLD,True)
    arrow(c,3.02,c,3.40)

t(.30,2.48,'Head',10,GRAY)
t(.30,2.68,'outputs',10,GRAY)
t(.30,3.70,'Generated',10,GRAY)
t(.30,3.90,'code',10,GRAY)

for c,title in zip(cs,['Initialize','Test divisors','Return the result']):
    box(c-1.60,3.45,3.20,1.12,fill='#F8F9FA')
    t(c,3.65,title,11,bold=True,ha='center')
# Code after the capture point is a future continuation, not source text at that pivot.
t(1.68,3.99,'is_prime = n >= 2',10.6,mono=True)
box(2.47,3.875,.135,.225,PALE,edge='#DAB980',r=.015)
t(1.68,3.99,'is_prime = n >= 2',10.6,mono=True)
t(cs[0],4.33,'capture at “=”',9.7,GOLD,ha='center')
t(5.27,3.94,'for d in range(2, n):',10.3,mono=True)
t(5.27,4.17,'    if n % d == 0:',10.3,mono=True)
t(5.27,4.40,'        is_prime = False',10.3,mono=True)
t(cs[2],4.09,'return is_prime',10.6,ha='center',mono=True)

# Small feedback paths show newly generated code entering the next sampled computation.
# These are explanatory hypotheses about goal recurrence, not measured mediation paths.
for i in range(2):
    c,nc=cs[i],cs[i+1]
    gx=c+1.80
    path([(c+1.63,4.04),(gx,4.04),(gx,2.64)],BLUE,True,1.15)
    arrow(gx,2.64,nc-1.30,2.64,BLUE,True,1.15)
t(8.50,4.82,'New code becomes context for the next computation',10.3,BLUE,ha='center')

# One optional realization of the first captured encoding, in another language.
path([(1.80,2.64),(1.29,2.64),(1.29,5.51)],PURPLE,lw=1.2)
arrow(1.29,5.51,1.47,5.51,PURPLE)
t(1.55,4.88,'Different means',11,PURPLE,True)
box(1.50,5.10,5.68,.85,fill='#FAF8FC',edge='#D2C8DD')
t(1.67,5.31,'C++',10.5,PURPLE,True)
t(2.32,5.31,'bool prime = n >= 2;',10.3,mono=True)
t(2.32,5.65,'for (int d = 2; d < n; ++d) { ... }',10.3,mono=True)
t(7.55,5.35,'Same sub-goal.',12,GOLD,True)
t(7.55,5.66,'Different actions and activations.',11.5,GOLD)
t(.30,6.10,'Schematic example. Dashed arrows illustrate a proposed reconstruction account; code steps are selected snapshots.',9.5,GRAY)

for ext in ['svg','pdf','png']:
    fig.savefig(OUT/f'same_goal_changing_states.{ext}',dpi=200,facecolor='white')
plt.close(fig)
print(OUT/'same_goal_changing_states.svg')
