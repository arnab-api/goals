"""Generate editable, dependency-free SVG concepts for the DIFF paper's Figure 1."""
from pathlib import Path
from html import escape

ROOT = Path(__file__).parent
C = dict(ink='#202b3b', muted='#647184', line='#dce2e9', q='#267b9b', qbg='#edf7fa',
         kv='#956327', kvbg='#fcf5e9', goal='#7960ad', goalbg='#f3effa', panel='#f8fafc')

class SVG:
    def __init__(self, w, h, title):
        self.w, self.h = w, h
        self.s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-labelledby="title"><title id="title">{escape(title)}</title>',
          '<defs>' + ''.join(f'<marker id="arrow-{k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="{C[k]}"/></marker>' for k in ['ink','q','kv','goal']) + '</defs>',
          '<rect width="100%" height="100%" fill="white"/>']
    def box(self,x,y,w,h,fill='white',stroke='line',r=10):
        self.s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{C.get(fill,fill)}" stroke="{C.get(stroke,stroke)}"/>')
    def text(self,x,y,t,size=20,color='ink',weight=400,mono=False):
        font='DejaVu Sans Mono, monospace' if mono else 'DejaVu Sans, Arial, sans-serif'
        self.s.append(f'<text x="{x}" y="{y}" font-family="{font}" font-size="{size}" xml:space="preserve" font-weight="{weight}" fill="{C.get(color,color)}">{escape(t)}</text>')
    def line(self,x1,y1,x2,y2,color='line',arrow=False,dash=False):
        self.s.append(f'<path d="M{x1},{y1} L{x2},{y2}" fill="none" stroke="{C.get(color,color)}" stroke-width="2"'+(f' marker-end="url(#arrow-{color})"' if arrow else '')+(' stroke-dasharray="6 5"' if dash else '')+'/>')
    def path(self,d,color='ink',dash=False):
        self.s.append(f'<path d="{d}" fill="none" stroke="{C[color]}" stroke-width="2.2" marker-end="url(#arrow-{color})"'+(' stroke-dasharray="6 5"' if dash else '')+'/>')
    def save(self,name):
        (ROOT/name).write_text('\n'.join(self.s+['</svg>']))

def label(s,x,y,letter,title,sub=None):
    s.text(x,y,f'({letter})  {title}',24,weight=700)
    if sub:s.text(x,y+29,sub,17,'muted')

def footer(s,y):
    s.text(28,y,'SCHEMATIC EXAMPLES · Text summarizes interventions; arrows do not specify patch duration or layer count.',15,'muted')

def req(s,x,y,w,second='start with a vowel',header='TASK REQUIREMENTS · K,V'):
    s.box(x,y,w,102,'kvbg','kv')
    s.text(x+16,y+27,header,16,'kv',700)
    s.text(x+16,y+57,'Keep words that are palindromes',19)
    s.text(x+16,y+84,f'and {second}.',19)

def progress(s,x,y,w,kind='palindrome',header='COMPLETED WORK · Q',code=True):
    s.box(x,y,w,100,'qbg','q')
    s.text(x+16,y+27,header,16,'q',700)
    s.text(x+16,y+56,('Palindrome filter implemented' if kind=='palindrome' else 'Vowel filter implemented'),18)
    if code:
        s.text(x+16,y+82,'res = [w for w in words',16,'muted',mono=True)
        s.text(x+16,y+103,('       if w == w[::-1]]' if kind=='palindrome' else '       if w[0] in "aeiou"]'),16,'muted',mono=True)

def outcome(s,x,y,w,target='vowel',code=True):
    title,desc,expr={
       'vowel':('Keep vowel-initial words','“Keep words starting with a vowel.”','if w[0] in "aeiou"]'),
       'palindrome':('Keep palindromes','“Keep words that are palindromes.”','if w == w[::-1]]'),
       'length':('Keep words longer than 3','“Keep words longer than three letters.”','if len(w) > 3]'),
    }[target]
    s.box(x,y,w,153 if code else 76,'goalbg','goal')
    s.text(x+16,y+25,'SELECTED NEXT SUBGOAL',15,'goal',700)
    s.text(x+16,y+54,title,22,'goal',700)
    if code:
        s.text(x+16,y+85,'return [w for w in res',17,mono=True)
        s.text(x+16,y+110,'        '+expr,17,mono=True)
        s.text(x+16,y+137,'Illustrative continuation',14,'muted')

def columns():
    s=SVG(1500,790,'DIFF Figure 1 concept A: three parallel panels')
    s.text(28,40,'DIFF: select the next subgoal from task requirements and completed work',27,weight=700)
    for x in [509,1003]:s.line(x,73,x,738)
    panels=[(28,'a','What remains?','Natural selection at the pivot','palindrome','vowel'),
            (522,'b','Change progress','Same task · transplant the query','vowel','palindrome'),
            (1016,'c','Change requirements','Same progress · transplant task K,V','palindrome','length')]
    for x,l,t,sub,done,target in panels:
        label(s,x,91,l,t,sub)
        req(s,x,145,456,'are longer than 3 characters' if target=='length' else 'start with a vowel',
            'TRANSPLANTED TASK · K,V' if l=='c' else 'TASK REQUIREMENTS · K,V')
        progress(s,x,269,456,done,'TRANSPLANTED PROGRESS · Q' if l=='b' else 'COMPLETED WORK · Q',False)
        s.path(f'M{x+440},247 L{x+440},409 L{x+293},409','kv',l=='c')
        s.path(f'M{x+110},369 L{x+110},409 L{x+157},409','q',l=='b')
        s.box(x+158,389,134,48,'white','ink')
        s.text(x+195,420,'DIFF',22,weight=700)
        s.text(x+138,461,'at the next-code pivot',16,'muted')
        s.line(x+225,472,x+225,494,'goal',True)
        outcome(s,x,510,456,target)
        s.text(x+8,699,{'a':'Select what remains to be done',
                         'b':'Next: vowel → palindrome',
                         'c':'Next: vowel → length > 3'}[l],20,weight=700)
    footer(s,770);s.save('A_parallel_panels.svg')

def hero():
    s=SVG(1500,925,'DIFF Figure 1 concept B: worked example and two causal interventions')
    label(s,28,40,'a','The next subgoal is what remains to be done')
    req(s,28,65,465)
    s.box(28,185,465,155,'qbg','q')
    s.text(45,213,'CODE SO FAR · PROGRESS',16,'q',700)
    s.text(45,244,'res = [w for w in words',19,mono=True)
    s.text(45,271,'       if w == w[::-1]]',19,mono=True)
    s.text(45,311,'▸',23,'goal',700)
    s.text(72,309,'Pivot: before the next computation',17,'muted')
    s.path('M493,115 L625,115 L625,191 L679,191','kv')
    s.text(525,100,'K,V',20,'kv',700)
    s.path('M493,266 L625,266 L625,226 L679,226','q')
    s.text(533,290,'Q',20,'q',700)
    s.box(682,166,185,84,'white','ink')
    s.text(735,201,'DIFF',27,weight=700)
    s.text(702,229,'attention read',17,'muted')
    s.path('M867,208 L947,208 L947,118 L979,118','goal')
    s.text(878,233,'head',14,'goal'); s.text(878,251,'output',14,'goal')
    outcome(s,981,76,490,'vowel',False)
    s.path('M1226,153 L1226,178 L1060,178 L1060,196','goal')
    s.path('M1226,178 L963,178 L963,303 L978,303','goal')
    s.box(980,203,491,48,'white','line')
    s.text(997,234,'GoalScope: “Keep vowel-initial words.”',20,'goal')
    s.box(980,272,491,68,'panel','line')
    s.text(997,300,'return [w for w in res',18,mono=True)
    s.text(997,325,'        if w[0] in "aeiou"]',18,mono=True)
    s.line(28,373,1471,373)
    s.text(28,410,'Two interventions test the inputs to this selection',24,weight=700)
    for x,l,title,change,target in [(28,'b','Change progress','Q','palindrome'),(775,'c','Change requirements','K,V','length')]:
        label(s,x,462,l,title)
        s.text(x,491,'Hold task requirements fixed' if l=='b' else 'Hold completed work fixed',18,'muted')
        s.box(x,513,695,101,'qbg' if l=='b' else 'kvbg','q' if l=='b' else 'kv')
        s.text(x+18,542,'DONOR CONTEXT',15,'q' if l=='b' else 'kv',700)
        s.text(x+18,573,'Vowel filter implemented' if l=='b' else 'Task: palindromes AND words longer than 3',21)
        s.text(x+18,600,'Same task: palindromes AND vowel-initial words' if l=='b' else 'Same progress: palindrome filter already implemented',16,'muted')
        s.line(x+347,614,x+347,672,'q' if l=='b' else 'kv',True,True)
        s.text(x+365,651,'transplant '+change,18,'q' if l=='b' else 'kv',700)
        outcome(s,x,688,695,target,False)
        s.text(x+18,800,'Clean next goal: vowel  →  Patched: '+('palindrome' if l=='b' else 'length > 3'),20,weight=700)
        s.text(x+18,835,'Destination keeps its own task K,V.' if l=='b' else 'Destination retains its progress context.',18,'muted')
    footer(s,903);s.save('B_worked_example.svg')

def rows():
    s=SVG(1500,695,'DIFF Figure 1 concept C: compact causal comparison')
    s.text(28,41,'DIFF: two inputs determine the next subgoal',29,weight=700)
    for x,t,c in [(310,'Task requirements · K,V','kv'),(712,'Completed work · Q','q'),(1205,'Next subgoal','goal')]:s.text(x,92,t,20,c,700)
    scenarios=[(119,'a','Natural behavior','Read at the pivot','start with a vowel','Palindrome filter','vowel'),
               (295,'b','Transplant Q','Same task','start with a vowel','Vowel filter','palindrome'),
               (471,'c','Transplant K,V','Same progress','are longer than 3','Palindrome filter','length')]
    for y,l,title,sub,second,done,target in scenarios:
        s.box(28,y,1444,150,'panel','line')
        s.text(45,y+42,f'({l}) {title}',21,weight=700)
        s.text(45,y+73,sub,17,'muted')
        s.box(310,y+22,373,104,'kvbg','kv')
        s.text(326,y+52,'Keep words that are',18)
        s.text(326,y+80,'palindromes AND',20,weight=700)
        s.text(326,y+109,second,20,'kv',700 if l=='c' else 400)
        s.box(712,y+22,310,104,'qbg','q')
        s.text(728,y+58,done,21,'q',700 if l=='b' else 400)
        s.text(728,y+88,'already implemented',18)
        s.path(f'M683,{y+42} L698,{y+42} L698,{y+9} L1083,{y+9} L1083,{y+48}','kv',l=='c')
        s.line(1022,y+88,1050,y+88,'q',True,l=='b')
        s.box(1051,y+49,88,53,'white','ink')
        s.text(1064,y+83,'DIFF',20,weight=700)
        s.line(1139,y+75,1176,y+75,'goal',True)
        s.box(1190,y+22,261,104,'goalbg','goal')
        lines={'vowel':['Keep vowel-','initial words'], 'palindrome':['Keep','palindromes'], 'length':['Keep words','longer than 3']}[target]
        for i,t in enumerate(lines):s.text(1207,y+60+29*i,t,22,'goal',700)
    s.text(28,653,'Dashed arrows: transplanted activations. All rows refer to the same destination pivot.',16,'muted')
    footer(s,680);s.save('C_compact_comparison.svg')

columns();hero();rows()

DESIGNS=[('B_worked_example','B · Worked example + two causal edits','Recommended starting point. The top panel grounds the mechanism in code and separates the GoalScope readout from continuation; two lower panels explain the interventions.'),
         ('A_parallel_panels','A · Three parallel panels','The most symmetric overview. Readers compare the two inputs and the selected subgoal across all three cases. More compact vertically; repeats some task text.'),
         ('C_compact_comparison','C · Compact causal comparison','The most economical first-page teaser. Reads as a controlled comparison and preserves room for the caption, but leaves code and GoalScope to a later figure.')]
html='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>DIFF · Figure 1 concepts</title><style>
*{box-sizing:border-box}body{margin:0;background:#eef1f5;color:#202b3b;font:17px/1.6 system-ui,sans-serif}main{max-width:1548px;margin:auto;padding:36px 24px}h1{font-size:34px;line-height:1.2;margin:0 0 12px}h2{font-size:23px;margin:0}p{max-width:1000px}a{color:#267b9b}nav{display:flex;gap:22px;margin:24px 0}section{margin:32px 0 48px}header{display:flex;justify-content:space-between;align-items:center;gap:20px}.figure{background:white;border:1px solid #dce2e9;border-radius:12px;overflow:auto;box-shadow:0 5px 24px #202b3b08}.figure img{display:block;width:100%;min-width:950px}.note{color:#647184}button{border:1px solid #bfcad6;background:white;padding:9px 14px;border-radius:7px;cursor:pointer}.narrow .figure{max-width:1000px}.narrow .figure img{min-width:0} @media print{body{background:white}nav,button{display:none}section{break-before:page}main{padding:0}.figure{border:0;box-shadow:none}.figure img{min-width:0}p{font-size:13px}}
</style><main><h1>DIFF · Figure 1 concepts</h1><p>Three editable layouts of the same story: select what remains, change progress by transferring Q, and change requirements by transferring K,V.</p><p class="note">Design mockups, not measured results. Code assumes nonempty lowercase words. Readout wording is illustrative. The diagrams summarize the intervention and do not imply a single head, layer, or forward pass.</p><nav>'''
html+=''.join(f'<a href="#{name}">{title.split(" · ")[0]}</a>' for name,title,_ in DESIGNS)
html+='''<button onclick="document.body.classList.toggle('narrow')">Toggle smaller figure preview</button></nav>'''
for name,title,desc in DESIGNS:
    html+=f'<section id="{name}"><header><h2>{title}</h2><a href="{name}.svg" download>SVG</a> · <a href="{name}.pdf" download>PDF</a> · <a href="{name}.png" download>PNG</a></header><p>{desc}</p><div class="figure"><img src="{name}.svg" alt="{title}"></div></section>'
html+='''<p class="note">Suggested caption: <b>DIFF selects the next subgoal by relating task requirements to completed work.</b> (a) After implementing a palindrome filter, the next subgoal is to keep vowel-initial words. (b) Transplanting a query from a context where the vowel filter is already implemented changes the selected subgoal to palindromes while preserving the destination task. (c) Transplanting task keys and values changes the remaining requirement to word length while preserving the completed palindrome computation. Schematic examples; final outputs should be replaced with recorded model continuations and GoalScope readouts.</p></main></html>'''
(ROOT/'index.html').write_text(html)
