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
