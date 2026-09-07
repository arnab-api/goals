from pathlib import Path
import html

ROOT=Path(__file__).parent
from drawing import SVG, C
C.update(green='#34765d',greenbg='#edf6f0')

def txt(s,x,y,t,size=19,c='ink',bold=False,mono=False):
    s.text(x,y,t,size,c,700 if bold else 400,mono)

def code(s,x,y,line,size=17,c='ink'):
    txt(s,x,y,line,size,c,mono=True)

def pivot(s,x,y,prefix,token='if',size=17):
    cw=size*.602
    code(s,x,y,prefix,size)
    px=x+len(prefix)*cw
    s.box(px-3,y-size, len(token)*cw+8,size+7,'qbg','q',3)
    code(s,px,y,token,size,'q')
    return px+len(token)*cw/2,y+8

def qp(s,x,y,w,role,source=False,size=17):
    s.box(x,y,w,220,'white','line',7)
    txt(s,x+15,y+27,role,16,'muted',True)
    txt(s,x+15,y+58,'Keep words that are palindromes',18)
    
    if role=='CLEAN PROMPT': s.box(x+12,y+65,250,25,'goalbg','goal',3)
    txt(s,x+15,y+84,'and start with a vowel.',18)
    s.line(x+15,y+99,x+w-15,y+99)
    code(s,x+15,y+126,'def func(words):',size)
    code(s,x+15,y+152,'    res = [w for w in words',size)
    code(s,x+15,y+178,'           if '+('w[0] in "aeiou"]' if source else 'w == w[::-1]]'),size)
    return pivot(s,x+15,y+205,'    return [w for w in res ',size=size)

def kp(s,x,y,w,role,source=False,size=17):
    s.box(x,y,w,220,'white','line',7)
    txt(s,x+15,y+27,role,16,'muted',True)
    s.box(x+10,y+38,w-20,57,'kvbg','kv',3)
    txt(s,x+15,y+60,'Return whether the '+('shortest' if source else 'longest')+' word',18)
    txt(s,x+15,y+85,'starts with a vowel.' if source else 'is a palindrome.',18)
    s.line(x+15,y+106,x+w-15,y+106)
    code(s,x+15,y+138,'def func(words):',size)
    code(s,x+15,y+166,'    res = sorted(words, key=len)',size)
    p=pivot(s,x+15,y+202,'    ','return',size)
    return (x+w-7,y+68),p

def qarrow(s,p1,p2,edge,label_y):
    s.path(f'M{p1[0]},{p1[1]} L{p1[0]},{label_y-18} L{edge},{label_y-18} L{edge},{p2[1]+20} L{p2[0]},{p2[1]+20} L{p2[0]},{p2[1]}','q',True)

def result(s,x,y,w,label,lines,color='goal',size=17):
    s.box(x,y,w,44+len(lines)*25,'goalbg' if color=='goal' else 'greenbg',color,6)
    txt(s,x+13,y+26,label,16,color,True)
    for i,line in enumerate(lines):code(s,x+13,y+53+25*i,line,size)

def compact():
    s=SVG(1710,1120,'DIFF Figure 1 revision: code and exact intervention sites')
    txt(s,28,39,'DIFF: task requirements and completed code determine the next subgoal',29,bold=True)
    txt(s,28,69,'Boxed token = query site at the pivot. Shaded instruction = task K,V span. Dashed arrow = activation transplant.',18,'muted')
    xs=[28,596,1164]
    for x in [581,1149]:s.line(x,95,x,1050)
    for x,letter,title,sub in [(xs[0],'a','Read what remains','A goal before the next condition is written'),(xs[1],'b','Transplant progress','Same task; different completed code'),(xs[2],'c','Transplant requirements','Different tasks; same completed code')]:
        txt(s,x,119,f'({letter}) {title}',25,bold=True)
        txt(s,x,149,sub,18,'muted')
    # Natural behavior: same concrete prompt as the Q destination.
    p=qp(s,28,181,518,'CLEAN PROMPT',False)
    # Qualitative attention read, explicitly labeled, no invented intensities.
    s.path(f'M{p[0]},{p[1]} L565,{p[1]} L565,265 L290,265','goal')
    txt(s,42,437,'At the boxed if token:',19,bold=True)
    txt(s,42,468,'Palindrome filtering is already in the code.',18)
    txt(s,42,497,'The remaining condition is vowel-initial.',18,'goal',True)
    txt(s,42,539,'Goal-head attention reads the task span.',18,'muted')
    result(s,28,575,518,'CLEAN CONTINUATION',['w[0] in "aeiou"]'],'green')
    s.box(28,685,518,98,'goalbg','goal')
    txt(s,44,714,'GOALSCOPE · SUMMARY',16,'goal',True)
    txt(s,44,748,'Check whether the word starts with a vowel.',19,'goal')
    s.box(28,831,518,166,'panel','line')
    txt(s,44,862,'The DIFF account',21,bold=True)
    txt(s,44,897,'Q at the pivot: completed work',19,'q')
    txt(s,44,928,'K,V in the instruction: task requirements',19,'kv')
    txt(s,44,959,'Head output: the selected next subgoal',19,'goal')
    # Q source and destination, arrow attached to exact if token.
    p1=qp(s,596,181,518,'SOURCE · vowel condition completed',True)
    p2=qp(s,596,477,518,'DESTINATION · palindrome condition completed',False)
    qarrow(s,p1,p2,1133,445)
    txt(s,614,445,'Cache Q(if) → overwrite destination Q(if)',18,'q',True)
    result(s,596,761,518,'CLEAN NEXT CONDITION',['w[0] in "aeiou"]'],'green')
    result(s,596,857,518,'PATCHED NEXT CONDITION',['w == w[::-1]]'])
    txt(s,610,983,'Readout: vowel check → palindrome check',18,'goal')
    txt(s,610,1016,'The destination code prefix is unchanged.',17,'muted')
    # KV source and destination, arrow attached to instruction spans.
    k1,_=kp(s,1164,181,518,'SOURCE · shortest word / vowel',True)
    k2,_=kp(s,1164,477,518,'DESTINATION · longest word / palindrome',False)
    s.path(f'M{k1[0]},{k1[1]} L1695,{k1[1]} L1695,{k2[1]} L{k2[0]},{k2[1]}','kv',True)
    txt(s,1179,432,'Cache K,V over the shaded source instruction;',17,'kv',True)
    txt(s,1179,458,'overwrite K,V at the destination instruction.',17,'kv',True)
    result(s,1164,761,518,'CLEAN CONTINUATION',['res[-1] == res[-1][::-1]'],'green')
    result(s,1164,857,518,'PATCHED CONTINUATION',['res[0][0] in "aeiou"'])
    txt(s,1178,983,'Readout: palindrome check → vowel check',18,'goal')
    txt(s,1178,1016,'The sorting prefix and pivot stay in place.',17,'muted')
    txt(s,28,1075,'Adapted from goal-diff slides 1, 3 and 5. Prompts, code and readouts are shortened schematic examples, not verbatim model outputs.',17,'muted')
    txt(s,28,1101,'Nonempty lowercase words assumed. The arrow in (a) is schematic; patch duration, head sets and span alignment belong in the caption.',16,'muted')
    s.save('D_code_first.svg')

def wide_prompt(s,x,y,w,role,kind,source):
    h=252 if kind=='q' else 285
    s.box(x,y,w,h,'white','line',7)
    txt(s,x+17,y+29,role,17,'muted',True)
    if kind=='q':
        txt(s,x+17,y+63,'Keep words that are palindromes and start with a vowel.',18)
        s.line(x+17,y+80,x+w-17,y+80)
        code(s,x+17,y+110,'def func(words: list[str]):',18)
        code(s,x+17,y+141,'    res = [w for w in words',18)
        code(s,x+17,y+170,'           if '+('w[0].lower() in "aeiou"]' if source else 'w == w[::-1]]'),18)
        p=pivot(s,x+17,y+210,'    return [w for w in res ',size=18)
        txt(s,x+17,y+240,'↑ Prompt ends at the boxed token.',15,'muted')
        return p
    s.box(x+12,y+42,w-24,66,'kvbg','kv',4)
    txt(s,x+20,y+68,'Return True if the '+('shortest word starts' if source else 'longest word is'),19)
    txt(s,x+20,y+95,'with a vowel.' if source else 'a palindrome.',19)
    if source:
        code(s,x+17,y+144,'def func(words: list[str]):',18)
        code(s,x+17,y+175,'    res = sorted(words, key=len)',18)
        pivot(s,x+17,y+238,'    ','return',18)
    else:
        code(s,x+17,y+144,'auto func(std::vector<std::string> words){',17)
        code(s,x+17,y+175,'    std::stable_sort(words.begin(), words.end(),',17)
        code(s,x+17,y+204,'        /* ascending word-length comparator */);',17)
        pivot(s,x+17,y+238,'    ','return',18)
    txt(s,x+17,y+269,'Shared progress: words sorted by length.',16,'muted')
    return x+w/2,y+42

def slides():
    s=SVG(1700,1280,'DIFF Figure 1 revision: prompt-to-prompt transplants with slide evidence')
    txt(s,28,39,'DIFF at the pivot: transplant progress or transplant the task',30,bold=True)
    txt(s,28,73,'Source and destination prompts show exactly where the cached activation is taken from and inserted.',19,'muted')
    txt(s,28,119,'(a) Query transfer',25,bold=True)
    txt(s,295,119,'Same instruction · different completed conditions',20,'muted')
    p1=wide_prompt(s,28,160,665,'SOURCE · vowel filtering completed','q',True)
    p2=wide_prompt(s,1006,160,665,'DESTINATION · palindrome filtering completed','q',False)
    s.path(f'M{p1[0]},{p1[1]} L{p1[0]},442 L{p2[0]},442 L{p2[0]},{p2[1]}','q',True)
    txt(s,725,235,'Cache Q at if',20,'q',True)
    txt(s,725,271,'↓',26,'q')
    txt(s,725,304,'Patch Q at if',20,'q',True)
    txt(s,725,342,'Selected goal heads',17,'muted')
    s.box(28,468,665,107,'goalbg','goal')
    txt(s,46,499,'GOALSCOPE · SUMMARIZED FROM SLIDE 3',16,'goal',True)
    txt(s,46,531,'Source: palindrome     Destination: vowel',20)
    txt(s,46,558,'Patched destination: palindrome',20,'goal',True)
    result(s,719,468,952,'PATCHED CONTINUATION · SLIDE 3 EXCERPT',[
      'word == word[::-1] and word.lower() == word.lower()[::-1]]'],size=18)
    txt(s,737,566,'Reported continuation retained, including its repeated palindrome test.',17,'muted')
    s.line(28,606,1671,606)
    txt(s,28,649,'(b) Key/value transfer',25,bold=True)
    txt(s,360,649,'Different instructions · shared sorting progress',20,'muted')
    k1=wide_prompt(s,28,700,665,'SOURCE · Python','kv',True)
    k2=wide_prompt(s,1006,700,665,'DESTINATION · C++','kv',False)
    # Route source-span to destination-span through the central gutter, no arrows over code.
    s.path('M681,775 L724,775 L724,687 L985,687 L985,775 L1018,775','kv',True)
    txt(s,747,807,'Cache task K,V',20,'kv',True)
    txt(s,747,845,'↓',26,'kv')
    txt(s,734,881,'Patch task K,V',20,'kv',True)
    txt(s,747,920,'Shaded spans only',17,'muted')
    s.box(28,1017,665,129,'goalbg','goal')
    txt(s,46,1048,'GOALSCOPE · SUMMARIZED FROM SLIDE 5',16,'goal',True)
    txt(s,46,1080,'Source: vowel     Destination: palindrome',20)
    txt(s,46,1111,'Patched destination: vowel',20,'goal',True)
    result(s,719,1017,952,'PATCHED CONTINUATION · SLIDE 5 EXCERPT',[
      '!words.empty() &&',
      'std::string("aeiouAEIOU").find(words.front()[0])',
      '    != std::string::npos;'],size=18)
    txt(s,28,1190,'Q carries which work is done; task K,V supplies what is required. The resulting read selects the next subgoal.',23,bold=True)
    txt(s,28,1230,'Source: goal-diff.pdf, slides 3 and 5. Prompt excerpts are shortened; readouts are summarized; output excerpts preserve the reported expressions.',16,'muted')
    txt(s,28,1256,'Query arrow marks the prefill sites; the slide also reports repeated decode patching. K,V replaces the destination task span, not its code prefix.',16,'muted')
    s.save('E_slide_grounded.svg')

compact();slides()

items=[('D_code_first','D · Three panels, code first','Natural DIFF behavior, Q transfer, and K/V transfer. Shortened Python examples keep the code legible; outputs are schematic.'),('E_slide_grounded','E · Prompt-to-prompt transplants','Closer to the supplied slides: explicit source/destination prompts, exact marked intervention sites, and reported output excerpts. Retains Python → C++ for KV transfer.')]
page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>DIFF Figure 1 · revision 2</title><style>*{box-sizing:border-box}body{margin:0;background:#eef1f5;color:#202b3b;font:18px/1.55 system-ui,sans-serif}main{max-width:1750px;margin:auto;padding:30px 24px}h1{font-size:34px}h2{font-size:24px}p{max-width:1100px}a{color:#267b9b}section{margin:42px 0 64px}.figure{background:white;border:1px solid #dce2e9;overflow:auto}.figure img{width:100%;min-width:1150px;display:block}.note{color:#647184}button{padding:9px 14px;cursor:pointer}.small .figure{max-width:1050px}.small .figure img{min-width:0}</style><main><h1>DIFF Figure 1 · revision 2</h1><p>Prompts and code now carry the explanation: Q transfers connect boxed pivot tokens; K,V transfers connect highlighted instruction spans. Destination code prefixes remain visible.</p><p><button onclick="document.body.classList.toggle('small')">Toggle smaller preview</button></p>'''
for n,t,d in items:
    page+=f'<section><h2>{t}</h2><p>{d}</p><p><a href="{n}.svg">SVG</a> · <a href="{n}.pdf">PDF</a> · <a href="{n}.png">PNG</a></p><div class="figure"><img src="{n}.svg" alt="{html.escape(t)}"></div></section>'
page+='''<p class="note">D normalizes the examples for explanation and must not be treated as measured output. E retains the reported continuation expressions, with surrounding tokens omitted and line breaks adjusted. No model experiments were run for these figures. See README.md for provenance and caption details.</p></main></html>'''
(ROOT/'index.html').write_text(page)
