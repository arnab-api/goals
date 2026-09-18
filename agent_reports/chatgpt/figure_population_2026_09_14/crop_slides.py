"""Extract vector-preserving crops from the author's slide deck (PDF points).

Boxes use top-left coordinates on 720 x 405 pt slides. The evaluation crop
removes only the unrelated chart corner below its source prompt.
"""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
SOURCE = ROOT / 'Overleaf-draft/REF/goals-iclr27.pdf'
OUT = ROOT / 'Overleaf-draft/Goals-Overleaf/Figures/slides'
TMP = Path('/tmp/goals-paper-slide-crops')
CROPS = {
    'goalscope_teaser': (3, (50, 45, 648, 308)),
    'localization': (2, (5, 78, 329, 380)),
    'evaluation_setup': (6, (313, 15, 715, 253)),
    'freeform_atoi': (4, (4, 36, 301, 402)),
    'freeform_matching': (5, (24, 35, 697, 285)),
    'readout_sources': (7, (9, 60, 710, 325)),
    'model_size': (9, (8, 67, 710, 346)),
    'model_code_tuning': (10, (7, 74, 712, 348)),
    'model_rl': (11, (7, 108, 712, 368)),
    'ablation_necessity_setup': (13, (15, 127, 314, 351)),
    'ablation_sufficiency_setup': (14, (3, 126, 310, 363)),
    'query_source_destination': (18, (5, 34, 287, 324)),
    'query_gaze': (18, (325, 211, 692, 311)),
    'query_result': (18, (8, 327, 701, 399)),
    'ov_example': (19, (5, 62, 710, 327)),
    'noncode_relations': (22, (3, 103, 333, 332)),
    'noncode_music': (22, (3, 114, 182, 329)),
    'noncode_antonym': (22, (160, 103, 348, 254)),
    'noncode_head_overlap': (22, (372, 49, 696, 229)),
    'boundary_forward': (23, (7, 109, 273, 398)),
    'boundary_reverse': (23, (280, 216, 515, 398)),
    'boundary_naming': (23, (460, 35, 707, 205)),
}


def run(args):
    subprocess.run(list(map(str, args)), check=True, stdout=subprocess.DEVNULL)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(exist_ok=True)
    for page in sorted({c[0] for c in CROPS.values()}):
        run(['pdfseparate', '-f', page, '-l', page, SOURCE, TMP / 'page-%d.pdf'])
    for name, (page, (x0, y0, x1, y1)) in CROPS.items():
        bbox = f'{x0} {405-y1} {x1} {405-y0}'
        run(['pdfcrop', '--bbox', bbox, TMP / f'page-{page}.pdf', OUT / f'{name}.pdf'])
    # Preserve vector content while masking unrelated overlapping slide panels.
    def overlay(name, width, height, drawing):
        tex = r'''\documentclass{article}
\usepackage[paperwidth=WIDTHbp,paperheight=HEIGHTbp,margin=0bp]{geometry}
\usepackage{graphicx,tikz}
\pagestyle{empty}
\begin{document}\null
\begin{tikzpicture}[remember picture,overlay,x=1bp,y=1bp]
\node[anchor=south west,inner sep=0] at (current page.south west) {\includegraphics{CROP}};
\begin{scope}[shift={(current page.south west)}]
DRAWING
\end{scope}
\end{tikzpicture}
\end{document}'''.replace('CROP', str(OUT/f'{name}.pdf')).replace('WIDTH',str(width)).replace('HEIGHT',str(height)).replace('DRAWING',drawing)
        source = TMP/f'{name}_clean.tex'
        source.write_text(tex)
        for _ in range(2):
            run(['pdflatex','-interaction=nonstopmode','-halt-on-error','-output-directory',TMP,source])
        import shutil
        shutil.copy2(source.with_suffix('.pdf'),OUT/f'{name}.pdf')
    overlay('evaluation_setup',402,238,r'\fill[white] (0,0) rectangle (217,43);')
    overlay('noncode_music',179,215,r'\fill[white] (150,66) rectangle (181,217);')
    overlay('ov_example',705,265,r'\fill[white] (240,242) rectangle (707,267);')
    # The teaser's unused lower-right area can show the same code-steering
    # example from slide 6, without shrinking the head-output cache diagram.
    overlay('goalscope_teaser',598,263,r'''
\draw[rounded corners=6bp,draw=black!65,fill=white] (342,5) rectangle (595,70);
\node[anchor=north,font=\sffamily\fontsize{11}{12}\selectfont] at (468,65) {(d) Causality: steer a C++ run};
\node[anchor=north west,align=left,font=\ttfamily\fontsize{8.5}{11}\selectfont] at (351,45) {bool check(auto value)\{\\\phantom{xx}return \_\_builtin\_popcount(value)>2;};
\draw[->,line width=1.4bp,draw=orange!80!yellow] (290,154) .. controls (332,142) and (324,35) .. (343,35);
\node[anchor=south west,fill=white,inner sep=1bp,font=\sffamily\fontsize{9}{10}\selectfont] at (6,2) {Cache gold head outputs at the highlighted pivot};
''')
    for name in CROPS:
        run(['pdftoppm', '-r', '140', '-png', '-singlefile',
             OUT / f'{name}.pdf', TMP / name])
    (REPORT / 'slide_crops.json').write_text(json.dumps({
        'source': str(SOURCE.relative_to(ROOT)),
        'sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'coordinates': 'PDF points, top-left origin',
        'crops': CROPS,
        'evaluation_mask': [0, 0, 217, 43],
    }, indent=2))


if __name__ == '__main__':
    main()
