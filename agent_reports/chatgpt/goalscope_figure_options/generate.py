"""Editable vector figure drafts. All verbalizations are illustrative, not measurements."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/goalscope-figure-mpl')
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.backends.backend_pdf import PdfPages

OUT = Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans', 'svg.fonttype':'none', 'pdf.fonttype':42})
INK='#172A3A'; MUTED='#617080'; BLUE='#2875B9'; TEAL='#16877B'; GOLD='#BC791D'
PALE='#EFF5FA'; GREEN='#EDF8F4'; BORDER='#D5DFE6'; PURPLE='#7755A7'
figures=[]
def canvas(title, subtitle, height=7.6):
    f,a=plt.subplots(figsize=(15,height)); f.subplots_adjust(0,0,1,1)
    a.set(xlim=(0,1500),ylim=(height*100,0)); a.axis('off')
    txt(a,45,44,title,24,weight='bold'); txt(a,45,76,subtitle,12,color=MUTED)
    return f,a
def txt(a,x,y,s,size=13,color=INK,weight='normal',ha='left',mono=False):
    a.text(x,y,s,fontsize=size,color=color,weight=weight,ha=ha,va='center',
           family='DejaVu Sans Mono' if mono else 'DejaVu Sans',linespacing=1.5)
def box(a,x,y,w,h,fill='white',edge=BORDER):
    a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=12',
                             linewidth=1.1,facecolor=fill,edgecolor=edge))
def arrow(a,x,y,xx,yy,color=BLUE,lw=2):
    a.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle='-|>',mutation_scale=14,
                              linewidth=lw,color=color,connectionstyle='arc3,rad=0'))
def dot(a,x,y,label='',color=BLUE,r=15):
    a.add_patch(Circle((x,y),r,facecolor=color,edgecolor='white',linewidth=1))
    if label: txt(a,x,y,label,11,'white','bold',ha='center')
def footer(a,y):
    txt(a,45,y,'SCHEMATIC EXAMPLES · Verbalizations are illustrative, not recorded outputs.  |  Additive norm-relative GoalScope.',10,color=MUTED)
def equation(a,x,y):
    txt(a,x,y,r'$r_{\ell,s}\ \leftarrow\ r_{\ell,s}+\alpha\,\|r_{\ell,s}\|\,\widehat{g}_t$',18)
def save(f,name):
    for ext in ['svg','pdf','png']:
        f.savefig(OUT/f'{name}.{ext}',dpi=150,facecolor='white')
    figures.append((f,name))

# A: full example, left-to-right.
f,a=canvas('A  /  From a code position to a goal readout',
           'Extract selected attention-head contributions, then use the same model in a separate verbalization run.')
for x,label in [(45,'(a) SOURCE COMPUTATION'),(560,'(b) EXTRACT & AGGREGATE'),(1005,'(c) PROBE GENERATION')]:
    txt(a,x,128,label,13,weight='bold')
box(a,45,158,440,408,PALE)
txt(a,67,190,'TASK INSTRUCTIONS',11,BLUE,'bold')
txt(a,67,225,'Keep palindromes that start with a vowel.',12)
txt(a,67,262,'CODE PREFIX',11,BLUE,'bold')
for y,s in [(302,'ys = [s for s in xs'),(335,'      if s == s[::-1]]'),(380,'zs')]: txt(a,67,y,s,15,mono=True)
box(a,107,363,30,33,'#D3E9FB',BLUE); txt(a,122,380,'=',15,BLUE,'bold',ha='center',mono=True)
txt(a,67,440,'t: activation at the final “=” token',12,BLUE)
txt(a,67,485,'Completed: palindrome filter',12)
txt(a,67,516,'Remaining: vowel-initial filter',12,TEAL,'bold')
arrow(a,485,380,565,380)
box(a,560,158,365,408)
txt(a,742,199,'Selected heads H, at token t',13,ha='center')
for i,(x,y) in enumerate([(615,270),(690,295),(770,260),(858,285)]):
    dot(a,x,y); arrow(a,x,y+18,741,369)
txt(a,742,232,'across source layers',11,MUTED,ha='center')
dot(a,742,390,'Σ',r=24); arrow(a,742,416,742,453)
txt(a,742,482,r'$g_t=\sum_{(\ell,h)\in H}W_O^{\ell,h}\,a_t^{\ell,h}$',18,ha='center')
txt(a,742,532,'Project → sum → normalize',12,BLUE,ha='center')
arrow(a,925,380,1004,380)
box(a,1005,158,450,408,GREEN)
txt(a,1027,190,'FIXED TEXT PROMPT',11,TEAL,'bold')
txt(a,1027,232,'“What are you doing at this line …?”',12)
txt(a,1027,263,'Source code is not in the probe prompt.',11,MUTED)
for y in [306,348,390]:
    box(a,1090,y,300,29,'white'); txt(a,1240,y+15,'probe residual stream',10,ha='center')
    txt(a,1060,y+15,'⊕',20,TEAL,ha='center'); arrow(a,1075,y+15,1090,y+15,TEAL)
txt(a,1025,352,r'$\widehat{g}_t$',17,TEAL,ha='center')
arrow(a,1240,424,1240,449,TEAL)
box(a,1027,454,404,79,'white',TEAL)
txt(a,1229,493,'“Keep strings starting with a vowel.”',12,TEAL,'bold',ha='center')
box(a,45,594,1410,97,'#FAFBFC')
equation(a,75,640)
txt(a,695,626,'Same source vector at each selected probe layer and generation step.',12)
txt(a,695,660,'DIFF readout settings: α = 0.1; layers 35–45; steering throughout decoding.',11,MUTED)
footer(a,729); save(f,'A_code_to_readout')

# B: two explicit depth diagrams.
f,a=canvas('B  /  Two model runs, connected by one activation vector',
           'Source head outputs are pooled across layers; the pooled direction is added at multiple probe layers.',8.4)
box(a,45,119,465,615,PALE); box(a,990,119,465,615,GREEN)
txt(a, 70,150,'(a) SOURCE RUN',13,BLUE,'bold')
txt(a,1015,150,'(b) PROBE RUN · SAME MODEL',13,TEAL,'bold')
txt(a,70,201,'Keep palindromes starting with a vowel.',12)
txt(a,70,243,'ys = [s for s in xs',14,mono=True)
txt(a,70,272,'      if s == s[::-1]]',14,mono=True)
txt(a,70,309,'zs =',14,mono=True); dot(a,153,309,'t',r=13)
txt(a,1015,205,'“What are you doing at this line …?”',12)
txt(a,1015,242,'Fixed prompt; no source code text',11,MUTED)
arrow(a,280,338,280,368); arrow(a,1240,270,1240,368,TEAL)
for j,y in enumerate([378,441,504,567]):
    box(a,90,y,340,43,'white')
    txt(a,110,y+22,f'Layer ℓ{j+1}',11,MUTED)
    for k in range(5): dot(a,245+k*34,y+22,color=BLUE if (j+k)%3==0 else '#CDD8E0',r=9)
    arrow(a,430,y+22,630,473)
    box(a,1090,y,300,43,'white')
    txt(a,1110,y+22,'Residual stream',11)
    txt(a,1358,y+22,'⊕',21,TEAL,ha='center')
    arrow(a,881,473,1088,y+22,TEAL)
    if j<3:
        arrow(a,280,y+44,280,y+61,MUTED,1.5)
        arrow(a,1240,y+44,1240,y+61,TEAL,1.5)
box(a,628,399,253,148,'white',BLUE)
txt(a,754,430,'PROJECT & SUM',12,BLUE,'bold',ha='center')
txt(a,754,478,r'$g_t=\sum_{(\ell,h)\in H}W_O^{\ell,h}a_t^{\ell,h}$',15,ha='center')
txt(a,754,521,r'Normalize to $\widehat{g}_t$',12,ha='center')
txt(a,280,647,'Read selected heads at one token t',12,BLUE,ha='center')
txt(a,280,678,'Extraction layers determined by H',11,MUTED,ha='center')
arrow(a,1240,612,1240,645,TEAL)
txt(a,1223,676,'“Keep vowel-initial strings.”',13,TEAL,'bold',ha='center')
txt(a,754,604,'Arrows carry projected head outputs,',10,MUTED,ha='center')
txt(a,754,628,'not text or source residual states.',10,MUTED,ha='center')
equation(a,555,767)
txt(a,70,767,'Depth ↓  ·  Layers shown schematically',11,MUTED)
footer(a,813); save(f,'B_two_model_runs')

# C: aligned cursor snapshots.
f,a=canvas('C  /  Follow the goal as the computation progresses',
           'Each marked token supplies a separate activation snapshot to the same fixed GoalScope probe.',8.1)
box(a,45,111,1410,68,PALE)
txt(a,68,145,'SHARED INSTRUCTIONS',11,BLUE,'bold')
txt(a,325,145,'Keep strings that are palindromes and start with a vowel; return the filtered list.',13)
txt(a,45,219,'(a) SOURCE PREFIX AT EACH SNAPSHOT',12,weight='bold')
txt(a,721,219,'(b) GOALSCOPE',12,weight='bold')
txt(a,1080,219,'(c) ILLUSTRATIVE READOUT',12,weight='bold')
rows=[(250,'1','ys =','Filter palindromes.',BLUE),
      (393,'2','ys = [s for s in xs if s == s[::-1]]\nzs =','Keep vowel-initial strings.',TEAL),
      (536,'3','ys = [s for s in xs if s == s[::-1]]\nzs = [s for s in ys if s[:1].lower() in "aeiou"]\nreturn','Return the filtered list.',PURPLE)]
for y,n,code,out,col in rows:
    box(a,45,y,625,125,PALE); dot(a,73,y+27,n,col)
    lines=code.split('\n')
    for line_i,line in enumerate(lines):
        line_y=y+65+(line_i-(len(lines)-1)/2)*23
        txt(a,105,line_y,line,11,mono=True)
    end=lines[-1]
    if end.endswith('='):
        token_x=105+(len(end)-1)*9.2
        box(a,token_x-3,line_y-13,17,26,'#D3E9FB',col)
        txt(a,token_x,line_y,'=',11,col,'bold',mono=True)
    else:
        a.plot([105,160],[line_y+13,line_y+13],color=col,lw=2)

    # Precise final extraction token is separately identified, avoiding a cursor-as-token fiction.
    txt(a,643,y+104,'t'+n,11,col,'bold',ha='right')
    arrow(a,671,y+64,721,y+64,col)
    box(a,721,y+23,263,82,'white',col)
    txt(a,852,y+49,'Head outputs → gₜ',12,col,ha='center')
    txt(a,852,y+80,'Inject into fixed probe',11,ha='center')
    arrow(a,985,y+64,1065,y+64,col)
    box(a,1065,y+23,390,82,'white',col)
    txt(a,1260,y+64,'“'+out+'”',13,col,'bold',ha='center')
box(a,45,687,1410,64,'#FAFBFC')
txt(a,68,710,'Extraction tokens: t₁ = final “=” · t₂ = final “=” · t₃ = final token of “return”.',11)
txt(a,68,735,'Each row is an independent probe generation; source snapshots are not accumulated inside the probe.',11,MUTED)
footer(a,785); save(f,'C_goal_over_progress')

# D: DIFF alignment with explicitly defined donors.
f,a=canvas('D  /  A shared readout for clean and DIFF-patched computations',
           'Hold the destination prompt fixed. Change the source computation, then extract its selected-head signal.',9.1)
box(a,45,113,1410,169,PALE)
txt(a,68,144,'SHARED DESTINATION',12,BLUE,'bold')
txt(a,68,181,'Tasks: T₁ = palindrome; T₂ = vowel-initial',12)
txt(a,68,217,'ys = [s for s in xs if s == s[::-1]]',13,mono=True)
txt(a,68,251,'zs =',13,mono=True); txt(a,155,251,'← extract at the final “=” token t',11,BLUE)
txt(a,865,153,'DONOR SETUPS',11,weight='bold')
txt(a,865,189,'Q donor: same {T₁, T₂}; completed {T₂}.',11)
txt(a,865,219,'KV donor: {T₁, T₃}; completed {T₁}.',11)
txt(a,865,249,'T₃ = more than one distinct character.',11)
for x,s in [(45,'(a) COMPUTATION INSPECTED'),(527,'(b) EXTRACT'),(816,'(c) FIXED PROBE'),(1134,'(d) READOUT')]:txt(a,x,321,s,11,weight='bold')
rows=[(352,'Clean destination',r'$g_t^{\mathrm{clean}}$','Keep vowel-initial\nstrings.',TEAL),
      (490,'Query-patched destination',r'$g_t^{Q\mathrm{\ patch}}$','Filter\npalindromes.',BLUE),
      (628,'KV-patched destination',r'$g_t^{KV\mathrm{\ patch}}$','Keep strings with >1\ndistinct character.',PURPLE)]
for y,title,g,out,col in rows:
    box(a,45,y,415,111,'white',col); txt(a,65,y+31,title,13,col,'bold')
    sub={'Clean destination':'Original destination forward',
         'Query-patched destination':'Transplant donor Q at selected heads',
         'KV-patched destination':'Transplant donor task-span K,V'}[title]
    txt(a,65,y+75,sub,11)
    arrow(a,461,y+55,525,y+55,col)
    box(a,527,y,214,111,PALE); txt(a,634,y+37,'Selected head sum',11,ha='center');txt(a,634,y+76,g,21,col,ha='center')
    arrow(a,742,y+55,807,y+55,col)
    box(a,810,y,251,111,GREEN);txt(a,935,y+31,'Same prompt + model',11,ha='center')
    txt(a,935,y+73,r'$r\ \leftarrow\ r+\alpha\|r\|\widehat{g}_t$',15,TEAL,ha='center')
    arrow(a,1062,y+55,1127,y+55,col)
    box(a,1129,y,326,111,'white',col);txt(a,1292,y+55,out,12,col,'bold',ha='center')
box(a,45,773,1410,68,'#FAFBFC')
txt(a,68,797,'The DIFF transplant occurs in the computation being inspected. GoalScope is a separate activation-to-text intervention.',11)
txt(a,68,824,'Rows show predicted readouts for the illustrative setup, not measured transfer results or behavioral continuations.',11,MUTED)
footer(a,881);save(f,'D_diff_readout_alignment')

with PdfPages(OUT/'all_options.pdf') as pdf:
    for f,_ in figures: pdf.savefig(f,facecolor='white')
titles=['A · Code → signal → readout','B · Two model runs','C · Goal over progress','D · DIFF + GoalScope']
cards='\n'.join(f'<section id="{name[0]}"><h2>{title}</h2><p><a href="{name}.pdf">PDF</a> · <a href="{name}.svg">Editable SVG</a> · <a href="{name}.png">PNG</a></p><img src="{name}.svg" alt="{title}"></section>' for (_,name),title in zip(figures,titles))
(OUT/'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><title>GoalScope figure options</title><style>body{font:16px system-ui;background:#edf2f6;color:#172a3a;margin:32px auto;max-width:1500px;padding:0 24px}header,section{background:white;padding:24px;border-radius:14px;margin-bottom:24px}h1{margin:0 0 12px}h2{font-size:20px}a{color:#2875b9}img{width:100%;height:auto}nav{display:flex;gap:24px;position:sticky;top:0;padding:16px;background:#edf2f6}p{line-height:1.6}</style><header><h1>GoalScope · four figure drafts</h1><p>Compare layout and explanatory emphasis. Every readout is illustrative, not an experimental observation. The diagrams depict additive norm-relative GoalScope.</p><a href="all_options.pdf">Download all four as one PDF</a></header><nav><a href="#A">A · Pipeline</a><a href="#B">B · Architecture</a><a href="#C">C · Progress</a><a href="#D">D · DIFF</a></nav>'''+cards+'</html>')
(OUT/'README.md').write_text('''# GoalScope figure drafts

Open [the comparison page](index.html) or [the four-page PDF](all_options.pdf).

- A: concrete code-to-readout pipeline; recommended general methods figure.
- B: source/probe model stacks and extraction/injection distinction.
- C: independent snapshots as source computation progresses.
- D: shared destination, Q/KV interventions, aligned GoalScope readouts.

All verbalizations are illustrative. These are design drafts, not empirical result figures. Each final source token supplies one summed vector across selected heads/layers. The norm-relative additive strategy injects its direction repeatedly into a separate probe run. Replacement GoalScope is not depicted. Probe question is abbreviated; actual default wording: “What are you doing at this line of the code. Keep your answer as brief as possible.” The system prompt is “You are a helpful assistant proficient in programming.” DIFF defaults: alpha 0.1, layers 35–45, all readout steps. Source extraction layers need not match probe injection layers.

SVG text remains editable. PDF embeds fonts. Regenerate with `python agent_reports/chatgpt/goalscope_figure_options/generate.py` from the repository root. Requires matplotlib.
''')
print(f'Wrote {len(figures)} options, comparison HTML, and combined PDF to {OUT}')
