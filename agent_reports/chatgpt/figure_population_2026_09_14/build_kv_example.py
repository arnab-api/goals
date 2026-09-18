"""Typeset an actual saved KV-transfer example, without chat wrappers."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
OUT = ROOT / 'Overleaf-draft/Goals-Overleaf/Figures'
record = json.loads((REPORT / 'kv_example.json').read_text())
assert record['raw']['gen_patched'].startswith(' (n % 2 == 0);')
plt.rcParams.update({'font.family':'DejaVu Sans', 'pdf.fonttype':42, 'svg.fonttype':'none'})
fig, ax = plt.subplots(figsize=(6.8, 3.2))
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.set(xlim=(0,1), ylim=(0,1)); ax.axis('off')
blue, orange = '#326ca6', '#ba6418'
for x, title, color, task in [(0.005,'(a) Source: Python',blue,'(2) whether the number is even'),(.515,'(b) Destination: C++',orange,'(2) whether the number is a multiple of three')]:
    ax.add_patch(FancyBboxPatch((x,.39),.48,.60,boxstyle='round,pad=0.003,rounding_size=0.012',linewidth=.7,edgecolor='#b7bec5',facecolor='white'))
    ax.text(x+.012,.954,title,color=color,weight='bold',fontsize=10,va='top')
    ax.text(x+.012,.859,'Task tokens: cache / replace K,V',fontsize=8.2,color='#555')
    ax.add_patch(FancyBboxPatch((x+.009,.669),.46,.154,boxstyle='round,pad=0.003,rounding_size=0.008',linewidth=0,facecolor=color,alpha=.11))
    ax.text(x+.017,.795,'(1) whether the number has exactly two digits',fontsize=7.5,va='top')
    ax.text(x+.017,.732,task,fontsize=7.5,va='top',color=color)
    # Exact excerpt. Long C++ line is split for display only, explicitly marked.
    code = ('def func(n: int):\n    a = (10 <= abs(n) <= 99)\n    b =' if x<.5 else
            'auto func(int n){\n    auto a = (std::abs(n) >= 10\n              && std::abs(n) <= 99);\n    auto b =')
    ax.text(x+.014,.633,code,fontfamily='DejaVu Sans Mono',fontsize=7.4,va='top',linespacing=1.35)
ax.annotate('',xy=(.54,.845),xytext=(.465,.845),arrowprops={'arrowstyle':'-|>','color':blue,'lw':1.6})
ax.text(.5,.347,r'$(K,V)_{\mathrm{dst}}\leftarrow(K,V)_{\mathrm{src}}$   in 250 goal heads',ha='center',fontsize=10)
ax.text(.015,.264,'(c) Destination continuation',fontsize=9.5,weight='bold')
ax.text(.02,.19,'Original',fontsize=8.5,color=orange)
ax.text(.15,.19,'auto b = (n % 3 == 0);',fontsize=9,fontfamily='DejaVu Sans Mono')
ax.text(.02,.107,'KV patched',fontsize=8.5,color=blue)
ax.text(.15,.107,'auto b = (n % 2 == 0);',fontsize=9,fontfamily='DejaVu Sans Mono',color=blue)
ax.text(.02,.026,'GoalScope: “I’m checking if the number is even (i.e., divisible by 2).”',fontsize=8.1)
for ext in ['pdf','svg','png']:
    fig.savefig(OUT/f'kv_transfer_example.{ext}',dpi=220)
