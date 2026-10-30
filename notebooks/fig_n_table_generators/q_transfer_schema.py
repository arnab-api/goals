"""gsr schematic WITH the clean destination run -> notebooks/Figures/transfer/q_transfer_schema.{svg,pdf,png}.

Two rows:
  2×2 grid · row 1: (i) source run · (ii) clean destination run (attention + readout at the pivot)
  row 2: (iii) select (q_src drops in from (i)) · (iv) restrict and generate
(ii) and (iii) share the colour / bar scale.
Render: cairosvg + IBM Plex Sans/Mono.
Data: one real instance (scripts/nrgs/q_transfer_example_attention.py)."""
import glob
import json
import os
from xml.sax.saxutils import escape as esc

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

EXAMPLE = dict(
    json="results/diff/Qwen2.5-Coder-32B-Instruct/gsr_example/order_diff__int__python-python__*.json",
    dst_order=["is_prime", "is_positive", "last_digit"],          # a, b, d
    signature="def func(n: int):",
    src_tasks=[("a", "(1) whether the number is prime"), ("c", "(2) the number of set bits"),
               ("b", "(3) whether the number is positive")],
    src_prefill=[("a", "x = n > 1 and all(...)"), ("b", "y = (n > 0)")],
    src_gen="bin(n).count('1')",
    dst_prefill="x = n > 1 and all(n % i != 0 ...)",
    clean_gen="n > 0",
    patched_gen="n % 10",
)

INK, MUTED, FAINT, MASKTXT = "#1f1e1c", "#8f8b82", "#c9c5bc", "#a9a59c"
GREEN, GREEN_BG = "#3f8f5a", "#e7f3eb"
RED, RED_BG, SALMON = "#d23c35", "#fbe3e1", "#e0706b"
PIVOT_BG, PIVOT_BD = "#fde9a9", "#e3a72f"
VIOLET, PANEL_BG, MASK_BG, ARROW = "#6a4bc4", "#f7f6f3", "#efede8", "#4a4a4a"
DSTGEN = "#3a7d2c"                                   # clean destination generation (Figure 5/6 green)
SANS, MONO, SERIF, SYM = ("'IBM Plex Sans', sans-serif", "'IBM Plex Mono', monospace",
                          "'Times New Roman', serif", "'DejaVu Serif', serif")
FS, LH, FL = 14.5, 23, 15
CW = FS * 0.6

fp = [f for f in glob.glob(os.path.join(ROOT, EXAMPLE["json"]))
      if list(json.load(open(f))["task_desc"]) == EXAMPLE["dst_order"]]
assert len(fp) == 1, fp
EX = json.load(open(fp[0]))

out = []
add = out.append
def text(x, y, s, size=FL, fam=SANS, fill=INK, weight=None, style=None, anchor=None):
    a = (f' font-weight="{weight}"' if weight else '') + (f' font-style="{style}"' if style else '') \
        + (f' text-anchor="{anchor}"' if anchor else '')
    add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{fam}" font-size="{size}" fill="{fill}"{a}>'
        f'{esc(s)}</text>')
def rect(x, y, w, h, fill, stroke="none", rx=0, sw=1, dash=None, op=None):
    add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}"'
        + (f' fill-opacity="{op:.3f}"' if op is not None else '')
        + f' stroke="{stroke}" stroke-width="{sw}"' + (f' stroke-dasharray="{dash}"' if dash else '') + '/>')
def path(d, stroke, sw=1.5, marker=None):
    add(f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round"'
        + (f' marker-end="url(#{marker})"' if marker else '') + '/>')
def code(x, y, s, col=INK):
    xs = ' '.join(f'{x + i * CW:.2f}' for i in range(len(s)))   # per-char x (cairo PNG rounds advances)
    add(f'<text x="{xs}" y="{y:.1f}" xml:space="preserve" font-family="{MONO}" font-size="{FS}" fill="{col}">{esc(s)}</text>')
    return x + CW * len(s)
def pivot(x, y):
    """the pivot token ' =': box spans the leading space + '='; returns the box centre."""
    rect(x - CW + 3.0, y - 15, 2 * CW - 1.5, 20, PIVOT_BG, PIVOT_BD, rx=3, sw=1.3); code(x, y, "=")
    return (x, y)
def letter(x, y, l, under=False, col=RED):
    text(x, y, l, size=17, fam=SERIF, fill=col, anchor="middle")
    if under: add(f'<line x1="{x-4.5}" y1="{y+3.5}" x2="{x+4.5}" y2="{y+3.5}" stroke="{col}" stroke-width="1.2"/>')
IT = lambda s: f'<tspan font-style="italic">{s}</tspan>'
SUB = lambda s, after='': f'<tspan font-size="14" dy="4.5">{s}</tspan><tspan dy="-4.5">{after}</tspan>'
DARROW = f'<tspan font-family="{SYM}" font-size="17">⇒</tspan>'
def title(x, y, raw, size=21, anchor=None):
    a = f' text-anchor="{anchor}"' if anchor else ''
    add(f'<text x="{x}" y="{y}" font-family="{SERIF}" font-size="{size}" fill="{INK}"{a}>{raw}</text>')

POS = EX["positions"]
TOK = dict(zip(POS, EX["tokens"]))
ATT = {"clean": dict(zip(POS, EX["attn_clean"])), "patched": dict(zip(POS, EX["attn_patched"]))}
SCORE = {"clean": EX["scores_clean"], "patched": EX["scores_patched"]}
ORDER = EXAMPLE["dst_order"]
LET = dict(zip(ORDER, ["a", "b", "d"]))
SEL = EX["cl_selected"]
assert SEL == ORDER[-1] == max(SCORE["patched"], key=SCORE["patched"].get)
# one colour scale and one bar scale for (ii) and (iii)
AMAX = max(max(ATT[m][q] for p in ORDER for q in EX["task_positions"][p]) for m in ATT)
GMAX = max(max(SCORE[m].values()) for m in SCORE)
BW = 92 / GMAX

ROW2 = 412
C1, C2, PW = 8, 576, 560          # 2×2 grid: two equal columns
W, H = C2 + PW + 8, ROW2 + 466
add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
add('<defs>')
for mid, col, sz in [("mk", ARROW, 9), ("mg", GREEN, 9), ("ms", SALMON, 9)]:
    add(f'<marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="{sz}" markerHeight="{sz}" '
        f'markerUnits="userSpaceOnUse" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{col}"/></marker>')
add(f'<pattern id="hatch" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
    f'<rect width="7" height="7" fill="{MASK_BG}"/><line x1="0" y1="0" x2="0" y2="7" stroke="#dedad1" stroke-width="3"/></pattern>')
add('</defs>')
rect(0, 0, W, H, "#ffffff")


def dst_prompt(x0, y0, mode):
    """mode: 'clean' / 'patched' (token shading ∝ attention, shared scale) or 'mask'."""
    y, info = y0, {}
    dim = MASKTXT if mode == "mask" else INK
    code(x0, y, "...", dim); y += LH
    for k, pid in enumerate(ORDER, 1):
        l = LET[pid]
        line = f"({k})" + "".join(TOK[q] for q in EX["task_positions"][pid])
        w = CW * len(line) + 8
        if mode in ATT:
            cx = x0 + CW * 3
            for q in EX["task_positions"][pid]:
                t = TOK[q]
                rect(cx + CW - 1, y - 15, CW * (len(t) - 1) + 2, 20, GREEN, rx=2,
                     op=0.04 + 0.66 * min(1.0, ATT[mode][q] / AMAX))
                cx += CW * len(t)
            letter(x0 - 17, y, l); code(x0, y, line)
        elif pid == SEL:
            rect(x0 - 4, y - 15.5, w, 21, GREEN_BG, GREEN, rx=4, sw=1.5)
            letter(x0 - 17, y, l, col=GREEN); code(x0, y, line)
        else:
            rect(x0 - 4, y - 15.5, w, 21, "url(#hatch)", rx=4)
            letter(x0 - 17, y, l, col=FAINT); code(x0, y, line, MASKTXT)
        info[l] = (y, x0 - 4 + w)
        y += LH
    code(x0, y, "...", dim); y += LH
    for s in ["<code>", EXAMPLE["signature"]]:
        code(x0, y, s, dim); y += LH
    s = EXAMPLE["dst_prefill"]
    if mode == "mask": rect(x0 + 33, y - 15.5, CW * len(s) + 8, 21, "url(#hatch)", rx=4)
    letter(x0 + 24, y, "a", under=True, col=FAINT if mode == "mask" else RED); code(x0 + 38, y, s, dim)
    y += LH
    info["piv_y"] = y
    return info


def gaze_column(px, pw, info, piv, mode, below=False):
    """per-task mean attention (gaze) to the right of each task span, fed from the pivot."""
    y = info["piv_y"]
    BX = px + pw - 150
    text(BX, info["a"][0] - 26, "gaze", size=14, fill=MUTED)
    best = max(SCORE[mode], key=SCORE[mode].get)
    for pid in ORDER:
        ty = info[LET[pid]][0]; g = SCORE[mode][pid]
        rect(BX, ty - 12, BW * g, 14, GREEN, rx=2, op=0.35 + 0.65 * g / GMAX)
        text(BX + BW * g + 6, ty, f"{g:.2f}", size=13.5, fill=INK if pid == best else MUTED,
             weight="600" if pid == best else None)
    ax = BX - 16
    if below:   # out of the bottom of the pivot box, under the generated text
        x0, yb = piv[0] + 0.5 * CW, y + 12
        path(f"M{x0:.1f},{y+5:.1f} L{x0:.1f},{yb-6:.1f} Q{x0:.1f},{yb:.1f} {x0+6:.1f},{yb:.1f} L{ax-12:.1f},{yb:.1f} "
             f"Q{ax:.1f},{yb:.1f} {ax:.1f},{yb-12:.1f} L{ax:.1f},{info['a'][0]-5:.1f}", GREEN, sw=1.8)
    else:       # from the right edge of the pivot box
        path(f"M{piv[0]+CW+1.5:.1f},{y-6:.1f} L{ax-14:.1f},{y-6:.1f} Q{ax:.1f},{y-6:.1f} {ax:.1f},{y-20:.1f} "
             f"L{ax:.1f},{info['a'][0]-5:.1f}", GREEN, sw=1.8)
    for pid in ORDER:
        add(f'<circle cx="{ax}" cy="{info[LET[pid]][0]-5}" r="2.8" fill="{GREEN}"/>')
    return BX


def readout(x, y_piv, piv, ro, note="GoalScope readout at the pivot", side="left"):
    by = y_piv + 50
    words, lines, cur = ro.strip().split(), [], ""
    for wd in words:
        if len(cur) + len(wd) + 1 > 64: lines.append(cur); cur = wd
        else: cur = (cur + " " + wd).strip()
    lines.append(cur)
    bh = 16 + 19 * len(lines)
    sx = piv[0] - 4 if side == "left" else piv[0] + 0.55 * CW     # leave from the bottom-left / bottom-right of the pivot
    path(f"M{sx:.1f},{y_piv+5.5:.1f} C{sx:.1f},{by-14:.1f} {sx+28:.1f},{by-8:.1f} {sx+34:.1f},{by:.1f}", MUTED, sw=1.1)
    rect(x - 10, by, 500, bh, "#ffffff", "#9a968d", rx=10, sw=1.1)
    for i, ln in enumerate(lines):
        s = ("“" if i == 0 else "") + ln + ("”" if i == len(lines) - 1 else "")
        text(x + 4, by + 20 + 19 * i, s, size=14.5, style="italic")
    text(x - 10, by + bh + 18, note, size=13, fill=MUTED)


# ═══════════ (i) source run · row 1, left ═══════════
X0 = C1 + 6
title(X0, 36, f'(i) Source run, {IT("M")}({IT("p")}{SUB("src", ")  ")}{DARROW}  {IT("c")}')
LX, CX = X0 + 8, X0 + 26
y = 92
code(CX, y, "..."); y += LH
for l, s in EXAMPLE["src_tasks"]:
    letter(LX, y, l); code(CX, y, s); y += LH
code(CX, y, "..."); y += LH
code(CX, y, "<code>"); y += LH
code(CX, y, EXAMPLE["signature"]); y += LH
for l, s in EXAMPLE["src_prefill"]:
    letter(CX + 24, y, l, under=True); code(CX + 38, y, s); y += LH
text(CX - 6, y, "→", size=17, fill=RED); letter(CX + 24, y, "c")
xe = code(CX + 38, y, "z ")
SP = pivot(xe, y)
code(xe + 2 * CW, y, EXAMPLE["src_gen"], RED)
SRC_PIV_Y = y
# q_src pill under the source pivot; label to its right
QX, QY = SP[0] - 34, SRC_PIV_Y + 62
path(f"M{SP[0]:.1f},{SRC_PIV_Y+6:.1f} L{SP[0]:.1f},{QY-16:.1f}", SALMON, sw=1.8, marker="ms")
rect(QX, QY - 15, 68, 32, RED_BG, SALMON, rx=9, sw=1.7)
title(QX + 34, QY + 8, f'{IT("q")}{SUB("src")}', anchor="middle")
text(QX + 80, QY - 2, "goal-head queries at the source pivot", size=14, fill=MUTED)
text(QX + 80, QY + 16, "(carry the source progress)", size=14, fill=MUTED)

# ═══════════ (ii) clean destination run · row 1, right ═══════════
PX = C2
rect(PX, 8, PW, ROW2 - 24, PANEL_BG, rx=12)
title(PX + 18, 36, f'(ii) Clean destination run, {IT("M")}({IT("p")}{SUB("dst", ")  ")}{DARROW}  {IT("b")}')
text(PX + 18, 62, "no intervention: the goal heads read b, the next listed task", size=14.5, fill=MUTED)
TXc = PX + 44
info_c = dst_prompt(TXc, 92, "clean")
y = info_c["piv_y"]
xe = code(TXc + 38, y, "y ")
PVc = pivot(xe, y)
cend = code(xe + 2 * CW, y, EXAMPLE["clean_gen"], DSTGEN)
gaze_column(PX, PW, info_c, PVc, "clean", below=True)
text(PVc[0] + 22, y + 30, "attention × |v| from the pivot, mean over goal heads", size=13, fill=GREEN)
readout(TXc, y, PVc, EX["ro_dst"])

# ═══════════ (iii) select · row 2, left ═══════════
P1X, P1Y = C1, ROW2
rect(P1X, P1Y + 8, PW, H - P1Y - 16, PANEL_BG, rx=12)
title(P1X + 18, P1Y + 36, f'(iii) Select, {IT("M")}({IT("p")}{SUB("dst", " | ← ")}{IT("q")}{SUB("src", ")")}')
text(P1X + 18, P1Y + 62, "transfer the source query; read which task the goal heads attend to", size=14.5, fill=MUTED)
TX = P1X + 44
info1 = dst_prompt(TX, P1Y + 92, "patched")
y = info1["piv_y"]
xe = code(TX + 38, y, "y ")
PV1 = pivot(xe, y)
# q_src -> (iii) pivot: down the panel's left margin, then along the band under the pivot line
BAND, MX = y + 30, P1X + 12
QIN = PV1[0] - 0.45 * CW                     # left half of the " =" box
path(f"M{QX:.1f},{QY:.1f} C{MX:.1f},{QY:.1f} {MX:.1f},{QY+20:.1f} {MX:.1f},{QY+44:.1f} L{MX:.1f},{BAND-12:.1f} "
     f"Q{MX:.1f},{BAND:.1f} {MX+12:.1f},{BAND:.1f} L{QIN-12:.1f},{BAND:.1f} "
     f"Q{QIN:.1f},{BAND:.1f} {QIN:.1f},{y+14:.1f} L{QIN:.1f},{y+5.5:.1f}", SALMON, sw=1.9, marker="ms")
BX1 = gaze_column(P1X, PW, info1, PV1, "patched")
readout(TX, y, PV1, EX["ro_patched"], "GoalScope readout at the pivot (readout-shift)", side="right")

# ═══════════ (iv) restrict and generate · row 2, right ═══════════
P2X = C2
rect(P2X, P1Y + 8, PW, H - P1Y - 16, PANEL_BG, rx=12)
title(P2X + 18, P1Y + 36, f'(iv) Restrict and Generate, {IT("M")}({IT("p")}{SUB("dst", " | mask")}'
      f'<tspan font-size="14" dy="4.5">d</tspan><tspan dy="-4.5">)  </tspan>{DARROW}  {IT("d")}')
text(P2X + 18, P1Y + 62, "fresh run, no query transfer; goal heads may read only d", size=14.5, fill=MUTED)
TX2 = P2X + 44
info2 = dst_prompt(TX2, P1Y + 92, "mask")
y = info2["piv_y"]
text(TX2 - 6, y, "→", size=17, fill=VIOLET); letter(TX2 + 24, y, "d", col=VIOLET)
xe = code(TX2 + 38, y, "y ", MASKTXT)
PV2 = pivot(xe, y)
gen = EXAMPLE["patched_gen"]
gx = xe + 2 * CW
rect(gx - 4, y - 15.5, CW * len(gen) + 8, 21, "#ffffff", VIOLET, rx=4, sw=1.2, dash="3.5 3")
gend = code(gx, y, gen, VIOLET)
dy, dxe = info2["d"]
rx_ = max(gend, dxe, TX2 + 38 + CW * len(EXAMPLE["dst_prefill"]) + 4) + 16
path(f"M{gend+4:.1f},{y-5:.1f} L{rx_-12:.1f},{y-5:.1f} Q{rx_:.1f},{y-5:.1f} {rx_:.1f},{y-17:.1f} "
     f"L{rx_:.1f},{dy-5+12:.1f} Q{rx_:.1f},{dy-5:.1f} {rx_-12:.1f},{dy-5:.1f} L{dxe+4:.1f},{dy-5:.1f}",
     GREEN, sw=2.0, marker="mg")
ly = y + 44
text(TX2 - 10, ly, "From the pivot on, goal heads may attend only to", size=15, weight="600")
items = [(GREEN_BG, GREEN, None, "the selected task's span (d)"), (PIVOT_BG, PIVOT_BD, None, 'the pivot token " ="'),
         ("#ffffff", VIOLET, "3.5 3", "the tokens generated so far")]
for i, (f, s_, dash, lbl) in enumerate(items):
    yy = ly + 25 + i * 23
    rect(TX2 - 6, yy - 13, 24, 17, f, s_, rx=3, sw=1.4, dash=dash)
    text(TX2 + 28, yy, lbl, size=15)
yy = ly + 25 + 3 * 23
rect(TX2 - 6, yy - 13, 24, 17, "url(#hatch)", rx=3)
text(TX2 + 28, yy, "all other keys are masked (pre-softmax)", size=15, fill=MUTED)
text(TX2 - 10, yy + 28, "Non-goal heads read the full context.", size=15, fill=MUTED)
hy = info1["d"][0] - 5
path(f"M{BX1 + BW * SCORE['patched'][SEL] + 44:.1f},{hy:.1f} L{P2X+12:.1f},{hy:.1f}", ARROW, sw=1.6, marker="mk")

add('</svg>')
FIG = os.path.join(HERE, "..", "Figures", "transfer")
svg_path = os.path.join(FIG, "q_transfer_schema.svg")
open(svg_path, "w").write("\n".join(out))
try:
    import cairosvg
    cairosvg.svg2pdf(url=svg_path, write_to=svg_path[:-4] + ".pdf")
    cairosvg.svg2png(url=svg_path, write_to=svg_path[:-4] + ".png", scale=1.3)
except ImportError:
    print("cairosvg not installed: wrote the svg only")
print("ok")
