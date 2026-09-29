"""Editable, schematic Figure 1. Run with the repository's .venv/bin/python."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/goal-figure-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = Path(__file__).resolve().parent
plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none',
                     'pdf.fonttype': 42, 'font.size': 11})
INK = '#202833'
GRAY = '#626C78'
LINE = '#D2D8DF'
GOLD = '#A76816'
PALE = '#FFF2D9'
PURPLE = '#66528F'
W, H = 10.8, 7.5
fig, ax = plt.subplots(figsize=(W, H))
fig.subplots_adjust(0, 0, 1, 1)
ax.set(xlim=(0, W), ylim=(H, 0))
ax.axis('off')

def text(x, y, s, size=11, color=INK, bold=False, mono=False, ha='left'):
    return ax.text(x, y, s, fontsize=size, color=color, ha=ha, va='center',
                   fontweight='bold' if bold else 'normal',
                   fontfamily='DejaVu Sans Mono' if mono else 'DejaVu Sans',
                   linespacing=1.4)

def box(x, y, w, h, fill='white', edge=LINE, radius=.07):
    ax.add_patch(FancyBboxPatch((x,y), w,h,
        boxstyle=f'round,pad=0,rounding_size={radius}', facecolor=fill,
        edgecolor=edge, linewidth=.9))

def arrow(x,y,xx,yy,color=GRAY,lw=1.15):
    ax.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle='-|>',
        mutation_scale=10,linewidth=lw,color=color,shrinkA=0,shrinkB=0))

def path(points, color=GRAY, lw=1.15, dashed=False):
    ax.plot(*zip(*points),color=color,lw=lw,
            linestyle=(0,(3,3)) if dashed else '-',solid_capstyle='round')

def vector(x,y,values):
    for i,alpha in enumerate(values):
        ax.add_patch(Rectangle((x+i*.15,y),.125,.19,
            facecolor=GOLD,alpha=alpha,edgecolor='none'))

text(.35,.28,'(a) Functional description',13,bold=True)
text(10.42,.28,'SCHEMATIC',9,GRAY,ha='right')
box(.40,.60,10.0,.45,PALE,edge='#E1C28D')
text(5.40,.825,'One sub-goal: check whether any word is a palindrome',12,GOLD,True,ha='center')
centers=[1.80,5.40,9.0]
for c,lab in zip(centers,['Select a goal before acting','Act using suitable means','Continue pursuing the goal']):
    text(c,1.28,lab,10.8,ha='center')
arrow(3.03,1.28,4.04,1.28)
arrow(6.68,1.28,7.70,1.28)
path([(.35,1.61),(10.45,1.61)],LINE,.9)

text(.35,1.89,'(b) LLM implementation',13,bold=True)
text(.40,2.20,'Task: write a Python function that checks whether any word is a palindrome.',11)

prefixes=['b =','b = any(','b = any(w == w[::-1]']
next_tokens=['any','w','for']
labels=['Before implementation','During implementation','Later in implementation']
profiles=[[.28,.85,.45,.98,.57,.36],[.40,.70,.62,.88,.30,.58],[.67,.48,.80,.69,.44,.27]]
for i,c in enumerate(centers):
    x=c-1.40
    box(x,2.48,2.80,2.00,fill='#FAFBFC')
    text(c,2.68,labels[i],10.5,GRAY,ha='center')
    # Literal prefix shown immediately before the token produced by this step.
    text(c,3.00,prefixes[i],11,mono=True,ha='center')
    if i==0:
        # 'b =' is 3 monospaced characters, centered on c.
        box(c+.035,2.875,.16,.25,PALE,edge='#DAB877',radius=.02)
        text(c,3.00,prefixes[i],11,mono=True,ha='center')
        text(c+.62,3.00,'← pivot',9.3,GOLD)
    arrow(c,3.17,c,3.36)
    box(x+.35,3.38,2.10,.55,fill='#F0F2F5',edge='#BEC7D1')
    subscript = '₀₁₂'[i]
    text(c,3.54,f'LM at position t{subscript}',11,ha='center')
    # Highlighted tiles represent selected heads within the current model run.
    for j in range(6):
        ax.add_patch(Rectangle((c-.47+j*.165,3.72),.12,.10,
            facecolor=GOLD,alpha=profiles[i][j],edgecolor='none'))
    text(c,4.20,f'next token:  {next_tokens[i]}',10.5,mono=True,ha='center')
    arrow(c,3.96,c,4.07)
    # Head contributions have a separate output path from ordinary generation.
    path([(c+1.05,3.73),(x+2.65,3.73),(x+2.65,4.42),(c-.10,4.42)],GOLD,1)
    arrow(c-.10,4.42,c-.10,4.56,GOLD,1)
    vector(c-.80,4.585,profiles[i])
    text(c+.21,4.69,f'h{subscript}',12,GOLD,True)

text(.42,4.98,'Selected head outputs',10.5,GOLD,bold=True)
text(7.22,4.98,'Temporal persistence: goal-related content across positions',10,GRAY,ha='center')
# Extract only the pre-implementation vector for the two separate probe runs.
# Route along the left margin so the capture arrow cannot be confused with native time.
path([(1.00,4.68),(.22,4.68),(.22,5.33),(8.04,5.33)],PURPLE,1.2)
arrow(2.77,5.33,2.77,5.60,PURPLE)
arrow(8.04,5.33,8.04,5.60,PURPLE)
text(.42,5.17,'Cache h₀ once; inject it into separate runs',10.5,PURPLE,bold=True)

box(.40,5.62,4.75,1.32,fill='#FBF9FD',edge='#C9BFDA')
text(.58,5.84,'Readout before action',11.5,bold=True)
text(.58,6.16,'Probe: “What operation are you performing?”',10,GRAY)
text(.58,6.57,'“Check whether any word is a palindrome.”',11,GOLD)

box(5.66,5.62,4.75,1.32,fill='#FBF9FD',edge='#C9BFDA')
text(5.84,5.84,'Plasticity: the same operation in C++',11.5,bold=True)
text(5.84,6.13,'return std::any_of(words.begin(), words.end(),',9.6,mono=True)
text(5.84,6.37,'  [](const auto& w) { return std::equal(',9.6,mono=True)
text(5.84,6.61,'    w.begin(), w.end(), w.rbegin()); });',9.6,mono=True)

text(.40,7.20,'Illustrative code and readouts. Shared color denotes goal content, not an unchanged stored vector.',10,GRAY)

for ext in ['svg','pdf','png']:
    fig.savefig(OUT/f'functional_vs_llm.{ext}',facecolor='white',dpi=200)
plt.close(fig)
print(OUT/'functional_vs_llm.svg')
