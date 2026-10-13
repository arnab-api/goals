"""gsr schematic (draft 5) -> notebooks/Figures/transfer/gsr_schema.{svg,pdf,png}.

Render: cairosvg + IBM Plex Sans/Mono (not in the goals env; see agent_reports/claude/q-transfer-update-todos.md).
Palette/type follow Figure 1 (Levels2b.dc.html) and Figures 5/6.
Data: one real instance (EXAMPLE below), re-computed with scripts/diff/gsr_example_attention.py; its per-task
scores match the stored cl_select_scores. Panel (ii) shades every task-span token by its real pivot attention
(value-weighted, mean over the 250 goal heads, with q_src) and shows the per-task mean (the gaze score) on the right."""
import json
import os
from xml.sax.saxutils import escape as esc

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

# ── the example (dst prompt text + token attention come from the JSON; the rest is display text) ─────────────
EXAMPLE = dict(
    json="results/diff/Qwen2.5-Coder-32B-Instruct/gsr_example/order_diff__int__python-python__{hash}.json",
    dst_order=["is_prime", "is_positive", "last_digit"],          # a, b, d
    signature="def func(n: int):",
    src_tasks=[("a", "(1) whether the number is prime"), ("c", "(2) the number of set bits"),
               ("b", "(3) whether the number is positive")],
    src_prefill=[("a", "x = n > 1 and all(...)"), ("b", "y = (n > 0)")],
    src_gen="bin(n).count('1')",
    dst_prefill="x = n > 1 and all(n % i != 0 ...)",
    patched_gen="n % 10",
)

INK, MUTED, FAINT, MASKTXT = "#1f1e1c", "#8f8b82", "#c9c5bc", "#a9a59c"
GREEN, GREEN_BG = "#3f8f5a", "#e7f3eb"
RED, RED_BG, SALMON = "#d23c35", "#fbe3e1", "#e0706b"
PIVOT_BG, PIVOT_BD = "#fde9a9", "#e3a72f"
VIOLET, PANEL_BG, MASK_BG, ARROW = "#6a4bc4", "#f7f6f3", "#efede8", "#4a4a4a"
SANS, MONO, SERIF, SYM = ("'IBM Plex Sans', sans-serif", "'IBM Plex Mono', monospace",
                          "'Times New Roman', serif", "'DejaVu Serif', serif")
FS = 14.5             # code size
CW = FS * 0.6         # Plex Mono advance
LH = 23
FL = 15               # label size


def _load():
    import glob
    fp = glob.glob(os.path.join(ROOT, EXAMPLE["json"].replace("{hash}", "*")))
    ex = [json.load(open(f)) for f in fp]
    ex = [e for e in ex if [p for p in e["task_desc"]] == EXAMPLE["dst_order"]] or ex
    assert len(ex) == 1, fp
    return ex[0]
EX = _load()

out = []
add = out.append
def text(x, y, s, size=FL, fam=SANS, fill=INK, weight=None, style=None, anchor=None, raw=False):
    a = (f' font-weight="{weight}"' if weight else '') + (f' font-style="{style}"' if style else '') \
        + (f' text-anchor="{anchor}"' if anchor else '')
    add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{fam}" font-size="{size}" fill="{fill}"{a}>'
        f'{s if raw else esc(s)}</text>')
def rect(x, y, w, h, fill, stroke="none", rx=0, sw=1, dash=None, op=None):
    add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}"'
        + (f' fill-opacity="{op:.3f}"' if op is not None else '')
        + f' stroke="{stroke}" stroke-width="{sw}"' + (f' stroke-dasharray="{dash}"' if dash else '') + '/>')
def path(d, stroke, sw=1.5, marker=None, dash=None, op=None):
    add(f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round"'
        + (f' marker-end="url(#{marker})"' if marker else '') + (f' stroke-dasharray="{dash}"' if dash else '')
        + (f' stroke-opacity="{op}"' if op else '') + '/>')
def code(x, y, s, col=INK):
    # per-character x: cairo's PNG backend rounds glyph advances, which drifts along long lines
    xs = ' '.join(f'{x + i * CW:.2f}' for i in range(len(s)))
    add(f'<text x="{xs}" y="{y:.1f}" xml:space="preserve" font-family="{MONO}" font-size="{FS}" fill="{col}">{esc(s)}</text>')
    return x + CW * len(s)
def pivot(x, y):
    rect(x - 1.5, y - 15, CW + 3, 20, PIVOT_BG, PIVOT_BD, rx=3, sw=1.3); code(x, y, "=")
    return (x + CW / 2, y)
def letter(x, y, l, under=False, col=RED):
    text(x, y, l, size=17, fam=SERIF, fill=col, anchor="middle")          # upright, as in Figure 6
    if under: add(f'<line x1="{x-4.5}" y1="{y+3.5}" x2="{x+4.5}" y2="{y+3.5}" stroke="{col}" stroke-width="1.2"/>')
IT = lambda s: f'<tspan font-style="italic">{s}</tspan>'
SUB = lambda s, after='': f'<tspan font-size="14" dy="4.5">{s}</tspan><tspan dy="-4.5">{after}</tspan>'
DARROW = f'<tspan font-family="{SYM}" font-size="17">⇒</tspan>'
def title(x, y, raw, size=21, anchor=None):
    a = f' text-anchor="{anchor}"' if anchor else ''
    add(f'<text x="{x}" y="{y}" font-family="{SERIF}" font-size="{size}" fill="{INK}"{a}>{raw}</text>')

# ── data ───────────────────────────────────────────────────────────────────────
POS = EX["positions"]
TOK = dict(zip(POS, EX["tokens"]))
A_P = dict(zip(POS, EX["attn_patched"]))
ORDER = EXAMPLE["dst_order"]
LET = dict(zip(ORDER, ["a", "b", "d"]))
SP_, SC_ = EX["scores_patched"], EX["scores_clean"]
AMAX_P = max(A_P[q] for p in ORDER for q in EX["task_positions"][p])
GMAX = max(SP_.values())
SEL = max(SP_, key=SP_.get)
assert SEL == ORDER[-1] == EX["cl_selected"], (SEL, EX["cl_selected"])

W, H = 1452, 478
add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
add('<defs>')
for mid, col, sz in [("mk", ARROW, 9), ("mg", GREEN, 9), ("ms", SALMON, 9)]:
    add(f'<marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="{sz}" markerHeight="{sz}" '
        f'markerUnits="userSpaceOnUse" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{col}"/></marker>')
add(f'<pattern id="hatch" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
    f'<rect width="7" height="7" fill="{MASK_BG}"/><line x1="0" y1="0" x2="0" y2="7" stroke="#dedad1" stroke-width="3"/></pattern>')
add('</defs>')
rect(0, 0, W, H, "#ffffff")

# ═══════════ (i) source run ═══════════
X0 = 14
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
QX, QY = SP[0] - 34, y + 70
path(f"M{SP[0]:.1f},{y+6:.1f} L{SP[0]:.1f},{QY-16:.1f}", SALMON, sw=1.8, marker="ms")
rect(QX, QY - 15, 68, 32, RED_BG, SALMON, rx=9, sw=1.7)
title(QX + 34, QY + 8, f'{IT("q")}{SUB("src")}', anchor="middle")
text(QX - 6, QY + 42, "goal-head queries at the", size=14, fill=MUTED)
text(QX - 6, QY + 60, "source pivot (src progress)", size=14, fill=MUTED)

# ── p_dst renderer (shared by (ii) heat and (iii) mask) ──────────────────────────
def dst_prompt(x0, y0, mode):
    """mode: 'heat' (token shading ∝ patched attention) or 'mask'. Returns line geometry."""
    y, info = y0, {}
    dim = MASKTXT if mode == "mask" else INK
    code(x0, y, "...", dim); y += LH
    for k, pid in enumerate(ORDER, 1):
        l = LET[pid]
        line = f"({k})" + "".join(TOK[q] for q in EX["task_positions"][pid])
        w = CW * len(line) + 8
        if mode == "heat":
            cx = x0 + CW * 3
            for q in EX["task_positions"][pid]:
                t = TOK[q]
                rect(cx + CW - 1, y - 15, CW * (len(t) - 1) + 2, 20, GREEN, rx=2, op=0.04 + 0.66 * A_P[q] / AMAX_P)
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

# ═══════════ (ii) select ═══════════
P1X, P1W = 392, 560
rect(P1X, 8, P1W, H - 16, PANEL_BG, rx=12)
title(P1X + 18, 36, f'(ii) Select, {IT("M")}({IT("p")}{SUB("dst", " | ← ")}{IT("q")}{SUB("src", ")")}')
text(P1X + 18, 62, "transfer the source query; read which task the goal heads attend to", size=14.5, fill=MUTED)
TX = P1X + 44
info1 = dst_prompt(TX, 92, "heat")
y = info1["piv_y"]
text(TX - 6, y, "→", size=17, fill=INK); letter(TX + 24, y, "b", col=INK)
xe = code(TX + 38, y, "y ")
PV1 = pivot(xe, y)
BAND = y + 30
path(f"M{QX+68:.1f},{QY:.1f} C{QX+170:.1f},{QY:.1f} {TX-60:.1f},{BAND:.1f} {TX-10:.1f},{BAND:.1f} "
     f"L{PV1[0]-22:.1f},{BAND:.1f} Q{PV1[0]-4:.1f},{BAND:.1f} {PV1[0]-4:.1f},{y+8:.1f}", SALMON, sw=1.9, marker="ms")

# gaze scores (mean token attention per task) to the right of each task span, fed from the pivot
BX = P1X + P1W - 150
BW = 92 / GMAX                              # px per unit gaze (largest bar = 92 px)
text(BX, info1["a"][0] - 26, "gaze", size=14, fill=MUTED)
for pid in ORDER:
    ty = info1[LET[pid]][0]
    g = SP_[pid]
    rect(BX, ty - 12, BW * g, 14, GREEN, rx=2, op=0.35 + 0.65 * g / GMAX)
    text(BX + BW * g + 6, ty, f"{g:.2f}", size=13.5, fill=INK if pid == SEL else MUTED,
         weight="600" if pid == SEL else None)
ax = BX - 16
path(f"M{PV1[0]+CW:.1f},{y-6:.1f} L{ax-14:.1f},{y-6:.1f} Q{ax:.1f},{y-6:.1f} {ax:.1f},{y-20:.1f} "
     f"L{ax:.1f},{info1['a'][0]-5:.1f}", GREEN, sw=1.8)
for pid in ORDER:
    add(f'<circle cx="{ax}" cy="{info1[LET[pid]][0]-5}" r="2.8" fill="{GREEN}"/>')
text(PV1[0] + 18, y + 18 + 0, "attention × |v| from the pivot, mean over goal heads", size=13, fill=GREEN)
cb, cd = SC_[ORDER[1]], SC_[SEL]
text(BX, info1["d"][0] + 26, "clean run:", size=12.5, fill=MUTED)
text(BX, info1["d"][0] + 42, f"b {cb:.2f}, d {cd:.2f}", size=12.5, fill=MUTED)

# readout bubble (selection pass, readout-shift)
ro = EX["ro_patched"].strip()
by = y + 50
words, lines, cur = ro.split(), [], ""
for wd in words:
    if len(cur) + len(wd) + 1 > 64: lines.append(cur); cur = wd
    else: cur = (cur + " " + wd).strip()
lines.append(cur)
bh = 16 + 19 * len(lines)
path(f"M{PV1[0]+2:.1f},{y+6:.1f} C{PV1[0]+6:.1f},{by-14:.1f} {PV1[0]+30:.1f},{by-8:.1f} {PV1[0]+36:.1f},{by:.1f}", MUTED, sw=1.1)
rect(TX - 10, by, 500, bh, "#ffffff", "#9a968d", rx=10, sw=1.1)
for i, ln in enumerate(lines):
    s = ("“" if i == 0 else "") + ln + ("”" if i == len(lines) - 1 else "")
    text(TX + 4, by + 20 + 19 * i, s, size=14.5, style="italic")
text(TX - 10, by + bh + 18, "GoalScope readout at the pivot (readout-shift)", size=13, fill=MUTED)

# ═══════════ (iii) restrict and generate ═══════════
P2X, P2W = 968, 470
rect(P2X, 8, P2W, H - 16, PANEL_BG, rx=12)
title(P2X + 18, 36, f'(iii) Restrict and Generate, {IT("M")}({IT("p")}{SUB("dst", " | mask")}'
      f'<tspan font-size="14" dy="4.5">d</tspan><tspan dy="-4.5">)  </tspan>{DARROW}  {IT("d")}')
text(P2X + 18, 62, "fresh run, no query transfer; goal heads may read only d", size=14.5, fill=MUTED)
TX2 = P2X + 44
info2 = dst_prompt(TX2, 92, "mask")
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
ly = y + 46
text(TX2 - 10, ly, "From the pivot on, goal heads may attend only to", size=15, weight="600")
items = [(GREEN_BG, GREEN, None, "the selected task's span (d)"), (PIVOT_BG, PIVOT_BD, None, "the pivot token  ="),
         ("#ffffff", VIOLET, "3.5 3", "the tokens generated so far")]
for i, (f, s_, dash, lbl) in enumerate(items):
    yy = ly + 26 + i * 24
    rect(TX2 - 6, yy - 13, 24, 17, f, s_, rx=3, sw=1.4, dash=dash)
    text(TX2 + 28, yy, lbl, size=15)
yy = ly + 26 + 3 * 24
rect(TX2 - 6, yy - 13, 24, 17, "url(#hatch)", rx=3)
text(TX2 + 28, yy, "all other keys are masked (pre-softmax)", size=15, fill=MUTED)
text(TX2 - 10, yy + 30, "Non-goal heads read the full context.", size=15, fill=MUTED)

# hand-off (ii) -> (iii): the highest-gaze task is kept
hy = info1["d"][0] - 5
path(f"M{BX + BW * GMAX + 44:.1f},{hy:.1f} L{P2X+12:.1f},{hy:.1f}", ARROW, sw=1.6, marker="mk")

add('</svg>')
FIG = os.path.join(HERE, "..", "Figures", "transfer")
svg_path = os.path.join(FIG, "gsr_schema.svg")
open(svg_path, "w").write("\n".join(out))
try:
    import cairosvg
    cairosvg.svg2pdf(url=svg_path, write_to=svg_path[:-4] + ".pdf")
    cairosvg.svg2png(url=svg_path, write_to=svg_path[:-4] + ".png", scale=1.3)
except ImportError:
    print("cairosvg not installed: wrote the svg only")
print("ok")
