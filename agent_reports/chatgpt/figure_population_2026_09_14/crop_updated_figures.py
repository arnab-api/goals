"""Vector crops of the author's 2026-09-15 figures, plus logged illustration fixes.

Coordinates are top-left points on a normalized 720 x 408 page. The original
2160 x 1224 PDF remains unchanged. Captions are typeset in the paper, not cropped.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import shutil

ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
SOURCE = ROOT/'Overleaf-draft/REF/goal-figures.pdf'
OUT = ROOT/'Overleaf-draft/Goals-Overleaf/Figures/author'
TMP = Path('/tmp/goal-figures-paper-crops')
CROPS = {
    'goalscope_words': (2,(48,0,688,294)),
    'evaluation_setup': (3,(108,0,632,320)),
    'task_prompt': (4,(67,11,667,314)),
    'localization_heads': (5,(47,52,658,271)),
    'freeform_matching': (6,(25,37,697,294)),
    'ablation_setup': (7,(8,56,705,275)),
    'ablation_sufficiency': (7,(8,56,352,275)),
    'ablation_necessity': (7,(360,56,705,275)),
    'kv_freeform': (9,(15,29,698,308)),
}

def run(args):
    subprocess.run(list(map(str,args)),check=True,stdout=subprocess.DEVNULL)

def main():
    TMP.mkdir(exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
    for page in {c[0] for c in CROPS.values()}:
        run(['pdfseparate','-f',page,'-l',page,SOURCE,TMP/'page-%d.pdf'])
    patches = {
      'task_prompt': r'''
\fill[white] (107,276.5) rectangle (480,292);
\node[anchor=north west,inner sep=0,font=\ttfamily\fontsize{12}{14}\selectfont] at (107.88,277.7) {b = sum(1 for w in words for c in w if c.isupper())};
''',
      'ablation_setup': r'''
\definecolor{spanGreen}{RGB}{217,234,211}
\definecolor{spanOrange}{RGB}{252,229,205}
\fill[spanGreen] (39,245) rectangle (289,258);
\node[anchor=north west,inner sep=0,font=\ttfamily\fontsize{8.1}{10}\selectfont] at (39.43,247.7) {b = sum(1 for w in words for c in w if c.isupper())};
\draw[rounded corners=1.5bp,draw=violet,line width=.7bp] (48,257.5) rectangle (56,268.5);
\fill[spanOrange] (390,247) rectangle (640,257.5);
\node[anchor=north west,inner sep=0,font=\ttfamily\fontsize{8.1}{10}\selectfont] at (390.94,247.7) {b = sum(1 for w in words for c in w if c.isupper())};
''',
      'kv_freeform': r'\fill[white] (45,137) rectangle (272,149);',
    }
    patches['ablation_sufficiency']=patches['ablation_setup']
    patches['ablation_necessity']=patches['ablation_setup']
    for name,(page,(x0,y0,x1,y1)) in CROPS.items():
        box=f'{3*x0} {3*(408-y1)} {3*x1} {3*(408-y0)}'
        raw=TMP/f'{name}_raw.pdf'
        run(['pdfcrop','--bbox',box,TMP/f'page-{page}.pdf',raw])
        width,height=x1-x0,y1-y0
        tex=r'''\documentclass{article}
\usepackage[paperwidth=WIDTHbp,paperheight=HEIGHTbp,margin=0bp]{geometry}
\usepackage{graphicx,tikz,DejaVuSansMono}
\pagestyle{empty}
\begin{document}\null
\begin{tikzpicture}[remember picture,overlay,x=1bp,y=-1bp]
\node[anchor=north west,inner sep=0] at (current page.north west) {\includegraphics[width=WIDTHbp]{CROP}};
\begin{scope}[shift={(current page.north west)}]
\begin{scope}[shift={(-X0,-Y0)}]
PATCH
\end{scope}\end{scope}
\end{tikzpicture}\end{document}'''
        for k,v in {'WIDTH':width,'HEIGHT':height,'CROP':raw,'X0':x0,'Y0':y0,'PATCH':patches.get(name,'')}.items():
            tex=tex.replace(k,str(v))
        source=TMP/f'{name}.tex';source.write_text(tex)
        for _ in range(2):
            run(['pdflatex','-interaction=nonstopmode','-halt-on-error','-output-directory',TMP,source])
        shutil.copy2(source.with_suffix('.pdf'),OUT/f'{name}.pdf')
        run(['pdftoppm','-r',140,'-png','-singlefile',OUT/f'{name}.pdf',TMP/name])
    (REPORT/'updated_figure_crops.json').write_text(json.dumps({
        'source':str(SOURCE.relative_to(ROOT)),
        'sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'coordinates':'720 x 408 normalized points; original dimensions 3x larger',
        'crops':CROPS,
        'corrections':{
            'task_prompt':'Reorder uppercase-count comprehension loops (w before c).',
            'ablation_setup':'Same loop correction in both panels; move left query marker from b= to c=.',
            'kv_freeform':'Remove accidentally repeated C++ function declaration; no other source or readout edited.',
        },
    },indent=2))

if __name__=='__main__':main()
