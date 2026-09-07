from pathlib import Path
from drawing import SVG,C

ROOT=Path(__file__).parent
C.update(clean='#38745d',cleanbg='#eff7f2')

def t(s,x,y,text,size=20,color='ink',bold=False,mono=False):
    s.text(x,y,text,size,color,700 if bold else 400,mono)

def math(s,x,y,parts,size=22,color='ink'):
    """parts: strings or (base, subscript). Keep math as editable text."""
    from html import escape
    a=[f'<text x="{x}" y="{y}" font-family="DejaVu Serif,serif" font-size="{size}" fill="{C[color]}" xml:space="preserve">']
    for p in parts:
        if isinstance(p,tuple):
            a.append(escape(p[0])+f'<tspan baseline-shift="sub" font-size="70%">{escape(p[1])}</tspan>')
        else:a.append(escape(p))
    s.s.append(''.join(a)+'</text>')

def prompt(s,x,y,w,kind):
    """One visible prompt. Return actual marked span and pivot anchors."""
    dest=kind=='dst'; q=kind=='q'
    color='ink' if dest else ('q' if q else 'kv')
    s.box(x,y,w,341,'white',color,8)
    math(s,x+18,y+31,[('p','dst' if dest else ('src_q' if q else 'src_kv'))],25,color)
    t(s,x+160,y+30,'SHARED DESTINATION' if dest else ('QUERY DONOR' if q else 'TASK DONOR'),17,color,True)
    math(s,x+18,y+68,['G = {',('T','1'),', ',('T','2' if (dest or q) else '3'),'}    P = {',('T','2' if q else '1'),'}'],21)
    s.line(x+16,y+84,x+w-16,y+84)
    t(s,x+18,y+111,'Keep words satisfying both conditions:',18)
    # K,V extraction/replacement span: full two-condition instruction.
    if not q:s.box(x+12,y+120,w-24,60,'kvbg','kv',3)
    t(s,x+19,y+144,'T₁: the word is a palindrome',19)
    t(s,x+19,y+170,'T₂: the word starts with a vowel' if (dest or q) else 'T₃: the word has >1 unique characters',19)
    t(s,x+18,y+207,'def func(words: list[str]):',18,mono=True)
    t(s,x+18,y+234,'    res = [w for w in words',18,mono=True)
    t(s,x+18,y+261,'           if '+('w[0] in "aeiou"]' if q else 'w == w[::-1]]'),18,mono=True)
    prefix='    return [w for w in res '
    t(s,x+18,y+302,prefix,18,mono=True)
    px=x+18+len(prefix)*18*.602
    s.box(px-3,y+283,30,26,'qbg','q',3)
    t(s,px,y+302,'if',18,'q',mono=True)
    t(s,x+18,y+329,'π: final token of the prompt',15,'muted')
    return dict(q=(px+12,y+312),qleft=(px-4,y+296),qright=(px+29,y+296),
                kvleft=(x+12,y+150),kvright=(x+w-12,y+150),bottom=(x+w/2,y+341))

def result(s,x,y,w,kind,panel=None):
    col={'clean':'clean','q':'q','kv':'kv'}[kind]
    num={'clean':'2','q':'1','kv':'3'}[kind]
    title={'clean':'Clean run','q':'Query-patched run','kv':'K/V-patched run'}[kind]
    if panel: title=f'({panel}) '+title
    s.box(x,y,w,183,col+'bg',col,7)
    t(s,x+16,y+28,title,19,col,True)
    parts=['M(',('p','dst'),')']
    if kind=='q':parts+=[' [',('q','dst'),' ← ',('q','src_q'),']']
    if kind=='kv':parts+=[' [',('(K,V)','dst'),' ← ',('(K,V)','src_kv'),']']
    math(s,x+16,y+65,parts,20)
    s.line(x+14,y+82,x+w-14,y+82)
    math(s,x+16,y+113,[('T',num),'  ',{'clean':'vowel-initial','q':'palindrome','kv':'>1 unique characters'}[kind]],23,col)
    t(s,x+16,y+151,{'clean':'w[0] in "aeiou"]','q':'w == w[::-1]]','kv':'len(set(w)) > 1]'}[kind],20,mono=True)

def top(s,title,sub):
    t(s,28,40,title,30,bold=True)
    t(s,28,73,sub,19,'muted')

def foot(s,y):
    t(s,28,y,'Schematic continuations · Each patch is applied separately to the same p_dst; the destination text is never replaced.',17,'muted')
    t(s,28,y+26,'G: task set; P: completed work; π: pivot. Nonempty lowercase words assumed. Arrows identify activation sites; patch settings belong in the caption.',15,'muted')

def flanks():
    s=SVG(1800,955,'DIFF: two donors, one destination, three continuations')
    top(s,'Two interventions, one destination prompt','Change the progress query or the task keys/values to change which condition the model implements next.')
    q=prompt(s,28,153,550,'q')
    d=prompt(s,625,153,550,'dst')
    k=prompt(s,1222,153,550,'kv')
    # Direct KV arrow connects actual instruction spans; the text is not replaced.
    s.path(f'M{k["kvleft"][0]},{k["kvleft"][1]} L{d["kvright"][0]},{d["kvright"][1]}','kv',True)
    math(s,1310,125,[('G','src_kv'),' ≠ ',('G','dst'),'; ',('P','src_kv'),' = ',('P','dst')],18,'kv')
    math(s,55,125,[('G','src_q'),' = ',('G','dst'),'; ',('P','src_q'),' ≠ ',('P','dst')],18,'q')
    # Source Q travels below cards, ends on the destination's boxed if.
    s.path(f'M{q["q"][0]},{q["q"][1]} L{q["q"][0]},540 L{d["q"][0]},540 L{d["q"][0]},{d["q"][1]}','q',True)
    math(s,515,531,[('q','dst'),' ← ',('q','src_q')],23,'q')
    # KV label uses a leader to the short right-to-center patch arrow.
    math(s,1244,533,[('(K,V)','dst'),' ← ',('(K,V)','src_kv')],23,'kv')
    t(s,28,603,'Read the same destination under three conditions',23,bold=True)
    # One neutral branch stems from the destination prompt, independent of donor paths.
    s.line(1175,470,1205,470,'ink')
    s.line(1205,470,1205,628,'ink')
    s.line(310,628,1490,628,'ink')
    for x,kind in [(28,'clean'),(625,'q'),(1222,'kv')]:
        s.line(x+275,628,x+275,655,'ink',True)
        result(s,x,666,550,kind)
    foot(s,900)
    s.save('F_shared_destination.svg')

def left_donors():
    s=SVG(1900,1160,'DIFF: labeled interventions and their aligned effects on one destination')
    s.s.append('<defs><marker id="arrow-clean" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="#38745d"/></marker></defs>')
    top(s,'DIFF: two interventions on the same destination','(a) → (b): replace task K,V.     (c) → (d): replace progress Q.     Dashed: transplant. Solid: resulting continuation.')
    # Pair each donor with its corresponding effect across the page.
    t(s,28,122,'(a) Task / goal transfer',24,'kv',True)
    math(s,28,158,[('(K,V)','dst'),' ← ',('(K,V)','src_kv')],25,'kv')
    k=prompt(s,28,181,550,'kv')
    t(s,47,554,'Different task set; shared progress {T₁}',20,'kv',True)
    t(s,28,632,'(c) Progress transfer',24,'q',True)
    math(s,28,668,[('q','dst'),' ← ',('q','src_q')],25,'q')
    q=prompt(s,28,691,550,'q')
    t(s,47,1062,'Same task set; different progress {T₂}',20,'q',True)
    d=prompt(s,738,440,550,'dst')
    # Both input arrows end at the exact activation site in the single destination.
    s.path(f'M{k["kvright"][0]},{k["kvright"][1]} L649,{k["kvright"][1]} L649,{d["kvleft"][1]} L{d["kvleft"][0]},{d["kvleft"][1]}','kv',True)
    s.path(f'M{q["qright"][0]},{q["qright"][1]} L605,{q["qright"][1]} L605,1083 L{d["q"][0]},1083 L{d["q"][0]},{d["q"][1]}','q',True)
    # Local tags bind each injection site to its donor panel and effect.
    s.box(699,571,39,29,'kvbg','kv',4)
    t(s,704,592,'(a)',18,'kv',True)
    s.box(1106,721,39,29,'qbg','q',4)
    t(s,1111,742,'(c)',18,'q',True)
    # Clean baseline is separate from the two paired intervention rows.
    result(s,738,142,550,'clean')
    s.line(1013,440,1013,334,'clean',True)
    t(s,1032,382,'No patch',18,'clean',True)
    # The solid colored links originate at the affected state, not the prompt header.
    s.path(f'M{d["kvright"][0]},{d["kvright"][1]} L1328,{d["kvright"][1]} L1328,341 L1380,341','kv')
    t(s,1409,225,'Effect of (a)',18,'kv',True)
    result(s,1394,250,478,'kv','b')
    # Route from the right edge of the boxed query; avoid its local (c) label.
    s.path(f'M{d["qright"][0]},{d["qright"][1]} L1093,{d["qright"][1]} L1093,767 L1348,767 L1348,852 L1380,852','q')
    t(s,1409,736,'Effect of (c)',18,'q',True)
    result(s,1394,761,478,'q','d')
    t(s,754,846,'One destination prompt.',20,'muted')
    t(s,754,877,'Two separate patched runs.',20,'muted')
    foot(s,1115)
    s.save('G_converging_donors.svg')

flanks();left_donors()
items=[('G_converging_donors','G · Labeled interventions and aligned effects','Revised: (a) task transfer → (b) K/V-patched run on top; (c) progress transfer → (d) query-patched run below. Separate colored links run from each affected destination site to its effect. The clean baseline sits above the shared destination.'),('F_shared_destination','F · Donors flank one destination','Earlier alternative: the destination appears once, between the query donor and the task donor. Three aligned result cards compare the clean and separately patched runs.')]
page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>DIFF · one destination</title><style>*{box-sizing:border-box}body{margin:0;background:#eef1f5;color:#202b3b;font:18px/1.55 system-ui,sans-serif}main{max-width:1950px;margin:auto;padding:28px}h1{font-size:32px}h2{font-size:24px}p{max-width:1100px}a{color:#267b9b}.figure{overflow:auto;background:white;border:1px solid #dce2e9}.figure img{display:block;width:100%;min-width:1200px}section{margin:40px 0 60px}.note{color:#647184}.small .figure{max-width:1100px}.small img{min-width:0}button{padding:10px;cursor:pointer}</style><main><h1>DIFF: two interventions on one destination</h1><p>Following your exact T₁/T₂/T₃ construction. One visible destination; independent Q and K,V replacements; three continuations. These are explanatory mockups, not new experimental results.</p><button onclick="document.body.classList.toggle('small')">Toggle smaller preview</button>'''
for n,title,desc in items:
    page+=f'<section><h2>{title}</h2><p>{desc}</p><p><a href="{n}.svg">SVG</a> · <a href="{n}.pdf">PDF</a> · <a href="{n}.png">PNG</a></p><div class="figure"><img src="{n}.svg" alt="{title}"></div></section>'
page+='''<p class="note">The Q-patched destination repeats palindrome filtering even though its visible code already performed it. This is the intended diagnostic effect of importing the donor’s progress query. T₃ means len(set(w)) &gt; 1; it is absent from the destination instruction. The two patches are separate runs, not a combined intervention.</p></main></html>'''
(ROOT/'index.html').write_text(page)
