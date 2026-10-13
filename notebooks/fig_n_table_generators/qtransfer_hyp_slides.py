"""Slide figures for the query-transfer alternative-hypothesis study
(agent_reports/claude/analysis/query-transfer-report.md) -> notebooks/Figures/slides/qtransfer_hypotheses/.

One 16:9 figure per slide (svg + png + pdf), meant to be pasted into a slide deck:
  01_setup · 02_configs · 03_set_diff · 04_order_id · 05_complete_count · 06_induction · 07_results · 08_margins
Predictions are computed from the four rules below and asserted against the report's triplets.
Render: cairosvg + IBM Plex Sans/Mono (see agent_reports/claude/q-transfer-update-todos.md)."""
import os
from xml.sax.saxutils import escape as esc

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "Figures", "slides", "qtransfer_hypotheses"))

# ── the design (report: Experimental setup) ───────────────────────────────────────
SRC_LIST = list("abcd")
CONFIGS = {   # source completed set + writing order, donor list + donor's own frontier
    "ALT1": dict(src_done=["a", "b", "d"], src_written=["a", "b", "d"], src_frontier="c",
                 donor=list("abde"), donor_frontier="b"),
    "ALT2": dict(src_done=["a", "b", "d"], src_written=["a", "b", "d"], src_frontier="c",
                 donor=list("adef"), donor_frontier="d"),
    "ALT3": dict(src_done=["a", "b", "c"], src_written=["a", "b", "c"], src_frontier="d",
                 donor=list("acbe"), donor_frontier="c"),
}
ALTS = list(CONFIGS)

def set_diff(c):       return next(t for t in c["donor"] if t not in c["src_done"])
def order_id(c):       return c["donor"][SRC_LIST.index(c["src_frontier"])]
def complete_count(c): return c["donor"][len(c["src_done"])]
def induction(c):
    last = c["src_written"][-1]
    return c["donor"][c["donor"].index(last) + 1]

HYPS = [  # name, rule (as written for the deck), function, colour
    ("set_diff", "first_of(dst_task_list \\ source_completed)", set_diff, "#0072B2"),
    ("order_id", "dst[order of the source frontier]", order_id, "#D55E00"),
    ("complete_count", "dst[|source_completed| + 1]", complete_count, "#009E73"),
    ("induction", "I follow <source last prefilled>", induction, "#CC79A7"),
]
PRED = {h: "".join(f(CONFIGS[a]) for a in ALTS) for h, _, f, _ in HYPS}
assert PRED == {"set_diff": "eee", "order_id": "dee", "complete_count": "efe", "induction": "eeb"}, PRED

# ── results (report: KV transfer + source-query replacement, 128 matched groups) ──────────
WIN = {"ALT1": {"e": 65, "d": 57}, "ALT2": {"e": 56, "f": 47}, "ALT3": {"e": 85, "b": 35}}
COMPLETE = {"set_diff": 28, "order_id": 11, "complete_count": 16, "induction": 15}
NONE_OF = 58
MARGINS = [  # setup, contrast, singled-out hypothesis, mean, lo, hi (pooled, 95% bootstrap)
    ("ALT1", "e − d", "order_id", 0.006402, -0.006476, 0.01938),
    ("ALT2", "e − f", "complete_count", -0.0009371, -0.01373, 0.01182),
    ("ALT3", "e − b", "induction", 0.04119, 0.02765, 0.05498),
]
assert sum(COMPLETE.values()) + NONE_OF == 128

# ── style ───────────────────────────────────────────────────────────────────────
INK, MUTED, FAINT, RULE = "#1f1e1c", "#77736b", "#b9b5ad", "#e3e0d8"
SALMON, SALMON_BG = "#e0706b", "#fbe3e1"      # source progress (as in the paper's q_src)
GREEN, GREEN_BG = "#3f8f5a", "#e7f3eb"        # task list / kv donor (paper's "reads task span")
PANEL = "#f7f6f3"
SANS, MONO = "'IBM Plex Sans', sans-serif", "'IBM Plex Mono', monospace"
HC = {h: c for h, _, _, c in HYPS}
W, H = 1600, 900

class Slide:
    def __init__(self):
        self.o = []
    def add(self, s): self.o.append(s)
    def text(self, x, y, s, size=22, fill=INK, weight=None, style=None, anchor=None, fam=SANS, raw=False):
        a = (f' font-weight="{weight}"' if weight else '') + (f' font-style="{style}"' if style else '') \
            + (f' text-anchor="{anchor}"' if anchor else '')
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{fam}" font-size="{size}" fill="{fill}"{a}>'
                 f'{s if raw else esc(s)}</text>')
    def rect(self, x, y, w, h, fill, stroke="none", rx=0, sw=1, dash=None, op=None):
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" '
                 + (f'fill-opacity="{op}" ' if op is not None else '')
                 + f'stroke="{stroke}" stroke-width="{sw}"' + (f' stroke-dasharray="{dash}"' if dash else '') + '/>')
    def line(self, x1, y1, x2, y2, col, sw=2, marker=None, dash=None):
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{col}" stroke-width="{sw}"'
                 + (f' marker-end="url(#{marker})"' if marker else '') + (f' stroke-dasharray="{dash}"' if dash else '') + '/>')
    def path(self, d, col, sw=2, marker=None, dash=None):
        self.add(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="{sw}" stroke-linecap="round"'
                 + (f' marker-end="url(#{marker})"' if marker else '') + (f' stroke-dasharray="{dash}"' if dash else '') + '/>')

    def header(self, title, sub=None):
        self.text(64, 84, title, size=40, weight="600")
        if sub: self.text(64, 124, sub, size=22, fill=MUTED)

    # task chip: states -- plain · done (source progress) · frontier · target(colour) · dim
    def chip(self, x, y, t, state="plain", col=None, s=54, note=None):
        r = 10
        if state == "done":
            self.rect(x, y, s, s, SALMON_BG, SALMON, rx=r, sw=2)
            self.text(x + s / 2, y + s / 2 + 9, t + "′", size=26, fill="#9b3b36", weight="600", anchor="middle")
        elif state == "target":
            self.rect(x - 4, y - 4, s + 8, s + 8, "none", col, rx=r + 3, sw=3.5)
            self.rect(x, y, s, s, col, rx=r)
            self.text(x + s / 2, y + s / 2 + 9, t, size=26, fill="#ffffff", weight="600", anchor="middle")
        elif state == "frontier":
            self.rect(x, y, s, s, "#ffffff", INK, rx=r, sw=2.5)
            self.text(x + s / 2, y + s / 2 + 9, t, size=26, fill=INK, weight="600", anchor="middle")
        elif state == "dim":
            self.rect(x, y, s, s, "#ffffff", RULE, rx=r, sw=2)
            self.text(x + s / 2, y + s / 2 + 9, t, size=26, fill=FAINT, weight="600", anchor="middle")
        else:
            self.rect(x, y, s, s, "#ffffff", "#9a968d", rx=r, sw=2)
            self.text(x + s / 2, y + s / 2 + 9, t, size=26, fill=INK, weight="600", anchor="middle")
        if note:
            self.text(x + s / 2, y + s + 26, note, size=16, fill=MUTED, anchor="middle")

    def chips(self, x, y, seq, states, cols=None, gap=12, s=54, notes=None):
        xs = []
        for i, t in enumerate(seq):
            xx = x + i * (s + gap)
            self.chip(xx, y, t, states[i], (cols or {}).get(i), s=s, note=(notes or {}).get(i))
            xs.append(xx)
        return xs

    def save(self, name):
        os.makedirs(OUT, exist_ok=True)
        defs = ('<defs>' + ''.join(
            f'<marker id="m{k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="12" markerHeight="12" '
            f'markerUnits="userSpaceOnUse" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
            for k, c in [("ink", INK), ("muted", MUTED), ("salmon", SALMON), ("green", GREEN)]
            + [(h, c) for h, c in HC.items()]) + '</defs>')
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">{defs}'
               f'<rect width="{W}" height="{H}" fill="#ffffff"/>' + "\n".join(self.o) + '</svg>')
        p = os.path.join(OUT, name + ".svg")
        open(p, "w").write(svg)
        try:
            import cairosvg
            cairosvg.svg2png(url=p, write_to=p[:-4] + ".png", scale=1.5)
            cairosvg.svg2pdf(url=p, write_to=p[:-4] + ".pdf")
        except ImportError:
            pass
        return p


def config_row(S, x, y, alt, src_states=None, donor_states=None, donor_cols=None, donor_notes=None, label=True):
    """ALT row: source list (done = source progress) | kv-donor list."""
    c = CONFIGS[alt]
    if label:
        S.text(x, y + 36, alt, size=26, weight="600")
    sx = x + 110
    st = src_states or ["done" if t in c["src_done"] else "frontier" for t in SRC_LIST]
    S.chips(sx, y, SRC_LIST, st)
    dx = sx + 4 * 66 + 150
    dst = donor_states or [("done" if t == "a" else ("frontier" if t == c["donor_frontier"] else "plain")) for t in c["donor"]]
    xs = S.chips(dx, y, c["donor"], dst, donor_cols, notes=donor_notes)
    return sx, dx, xs


# ═════════════════════════ 01 · setup ═════════════════════════
def slide_setup():
    S = Slide()
    S.header("Which donor task does a transferred query select?",
             "three prompts per run · Qwen2.5-Coder-32B · 250 goal heads · read gaze, no generation")
    y0 = 236
    boxes = [
        (64, "Source", "gives the query", SALMON, SALMON_BG,
         ["lists tasks a b c d", "some are already written", "query at its final ' =' → q_src"]),
        (584, "KV donor", "gives the task list", GREEN, GREEN_BG,
         ["lists 4 tasks, only a written", "its task-list keys/values", "are copied into the recipient"]),
        (1104, "Recipient", "is the run we read", INK, PANEL,
         ["lists a x y z, only a written", "goal heads read the donor's list", "and use q_src at its ' ='"]),
    ]
    for x, name, role, col, bg, lines in boxes:
        S.rect(x, y0, 432, 310, bg, col, rx=16, sw=2)
        S.text(x + 28, y0 + 54, name, size=32, weight="600", fill=col if col != INK else INK)
        S.text(x + 28, y0 + 90, role, size=22, fill=MUTED)
        for i, l in enumerate(lines):
            S.text(x + 28, y0 + 150 + 44 * i, l, size=23)
    S.path(f"M{280},{y0 - 2} C{280},{y0 - 46} {320},{y0 - 46} {400},{y0 - 46} L{1240},{y0 - 46} "
           f"C{1300},{y0 - 46} {1320},{y0 - 40} {1320},{y0 - 6}", SALMON, sw=3, marker="msalmon")
    S.text(800, y0 - 58, "q_src  (replaces the recipient's query at ' =')", size=21, fill=SALMON, weight="600", anchor="middle", fam=MONO)
    S.line(1016 + 4, y0 + 165, 1104 - 8, y0 + 165, GREEN, sw=3, marker="mgreen")
    S.text(1060, y0 + 152, "K, V", size=20, fill=GREEN, weight="600", anchor="middle", fam=MONO)
    # readout
    y1 = 600
    S.rect(64, y1, 1472, 210, PANEL, rx=16)
    S.text(96, y1 + 56, "Readout: gaze at the recipient's final ' ='", size=28, weight="600")
    S.text(96, y1 + 100, "attention × |value| from the pivot to each of the donor's four task descriptions,", size=23)
    S.text(96, y1 + 134, "averaged over the goal heads · the highest-gaze task is the winner", size=23)
    S.text(96, y1 + 180, "Question: which donor task wins, given what the source has already done?", size=23,
           fill=SALMON, weight="600")
    return S.save("01_setup")


# ═════════════════════════ 02 · configs ═════════════════════════
def legend_chips(S, x, y):
    S.chip(x, y, "a", "done", s=40); S.text(x + 52, y + 28, "written (in the source: progress to transfer)", size=19, fill=MUTED)
    S.chip(x + 560, y, "c", "frontier", s=40); S.text(x + 612, y + 28, "the prompt's own next task", size=19, fill=MUTED)

def slide_configs():
    S = Slide()
    S.header("Three matched configurations", "each separates one hypothesis from the other three · 128 groups each")
    x0, y0 = 64, 210
    S.text(x0 + 110, y0 - 22, "Source  [a b c d]", size=22, fill=SALMON, weight="600")
    S.text(x0 + 110 + 4 * 66 + 150, y0 - 22, "KV donor", size=22, fill=GREEN, weight="600")
    S.text(x0 + 110 + 4 * 66 + 150 + 300, y0 - 22, "notation", size=22, fill=MUTED)
    for i, alt in enumerate(ALTS):
        y = y0 + i * 150
        c = CONFIGS[alt]
        sx, dx, _ = config_row(S, x0, y, alt)
        S.text(sx, y + 88, "written in order " + ", ".join(c["src_written"]), size=18, fill=MUTED)
        S.text(dx + 4 * 66 + 34, y + 36,
               "src " + "".join(t + ("′" if t in c["src_done"] else "") for t in SRC_LIST) + f" → {c['src_frontier']}"
               + "   ·   donor " + "".join(t + ("′" if t == "a" else "") for t in c["donor"]) + f" → {c['donor_frontier']}",
               size=22, fam=MONO)
    S.line(64, 690, 1536, 690, RULE, sw=1.5)
    legend_chips(S, 64, 720)
    S.text(64, 830, "Recipient in every config: a′xyz → x.  Without q_src, the donor's own frontier (b, d, c) is the expected winner.",
           size=20, fill=MUTED)
    return S.save("02_configs")


# ═════════════════════════ 03–06 · one hypothesis per slide ═════════════════════════
def pred_table(S, x, y, upto, results=False):
    """cumulative prediction table; rows = hypotheses introduced so far."""
    cw = [236, 92, 92, 92] + ([150] if results else [])
    hdr = ["hypothesis"] + ALTS + (["all three"] if results else [])
    S.text(x, y, "predicted winner", size=20, fill=MUTED)
    yy = y + 40
    cx = x
    for j, h in enumerate(hdr):
        S.text(cx + (0 if j == 0 else cw[j] / 2), yy, h, size=20, fill=MUTED, weight="600",
               anchor=None if j == 0 else "middle")
        cx += cw[j]
    S.line(x, yy + 14, x + sum(cw), yy + 14, RULE, sw=1.5)
    for i, (h, _, _, col) in enumerate(HYPS[:upto]):
        ry = yy + 56 + i * 58
        cur = (i == upto - 1) and not results
        if cur:
            S.rect(x - 14, ry - 36, sum(cw) + 28, 52, col, rx=8, op=0.12)
        S.rect(x, ry - 20, 14, 14, col, rx=3)
        S.text(x + 24, ry - 7, h, size=22, weight="600" if cur else None, fam=MONO)
        cx = x + cw[0]
        others = [PRED[o] for o, *_ in HYPS if o != h]
        for j, alt in enumerate(ALTS):
            t = PRED[h][j]
            unique = all(o[j] != t for o in others)
            if results:
                n = WIN[alt].get(t)
                S.text(cx + cw[j + 1] / 2 - 18, ry - 7, t, size=24, weight="600", fill=col if unique else INK, anchor="middle")
                S.text(cx + cw[j + 1] / 2 + 18, ry - 7, f"{n}", size=20, fill=MUTED, anchor="middle")
            else:
                S.text(cx + cw[j + 1] / 2, ry - 7, t, size=26, weight="600", fill=col if unique else INK, anchor="middle")
            cx += cw[j + 1]
        if results:
            S.text(cx + cw[-1] / 2, ry - 7, f"{COMPLETE[h]}", size=24, weight="600", anchor="middle")
    return yy + 56 + upto * 58

def slide_hyp(k):
    h, rule, f, col = HYPS[k]
    S = Slide()
    S.header(h, None)
    S.text(64, 130, rule, size=28, fam=MONO, fill=col)
    x0, y0 = 64, 210
    S.text(x0 + 110, y0 - 46, "Source", size=22, fill=SALMON, weight="600")
    S.text(x0 + 110 + 4 * 66 + 150, y0 - 46, "KV donor (dst)", size=22, fill=GREEN, weight="600")
    for i, alt in enumerate(ALTS):
        c = CONFIGS[alt]
        y = y0 + i * 168
        tgt = f(c)
        ti = c["donor"].index(tgt)
        src_st = ["done" if t in c["src_done"] else "frontier" for t in SRC_LIST]
        if h == "set_diff":        # source progress copied onto the donor list; first task left is the target
            dst_st = ["done" if t in c["src_done"] else "dim" for t in c["donor"]]
            note = "first not done: " + tgt
        elif h == "order_id":      # position of the source frontier, read off the donor list
            pos = SRC_LIST.index(c["src_frontier"]) + 1
            dst_st = ["dim"] * 4
            note = f"source frontier {c['src_frontier']} is #{pos} → dst[{pos}] = {tgt}"
        elif h == "complete_count":
            n = len(c["src_done"])
            dst_st = ["dim"] * 4
            note = f"{n} done → dst[{n + 1}] = {tgt}"
        else:                      # induction: successor of the last task the source wrote
            last = c["src_written"][-1]
            dst_st = ["dim"] * 4
            dst_st[c["donor"].index(last)] = "plain"
            note = f"source wrote {last} last → after {last} comes {tgt}"
        dst_st[ti] = "target"
        sx, dx, xs = config_row(S, x0, y, alt, src_states=src_st, donor_states=dst_st, donor_cols={ti: col})
        # per-hypothesis cue on the source side
        if h == "order_id":
            fi = SRC_LIST.index(c["src_frontier"])
            S.text(sx + fi * 66 + 27, y - 8, f"#{fi + 1}", size=18, fill=col, weight="600", anchor="middle")
            S.text(xs[ti] + 27, y - 12, f"#{ti + 1}", size=18, fill=col, weight="600", anchor="middle")
        elif h == "complete_count":
            S.text(sx + 2 * 66, y + 88, f"|done| = {len(c['src_done'])}", size=19, fill=col, weight="600", anchor="middle")
            S.text(xs[ti] + 27, y - 12, f"#{ti + 1}", size=18, fill=col, weight="600", anchor="middle")
        elif h == "induction":
            S.text(sx, y + 88, "wrote " + " → ".join(c["src_written"]), size=18, fill=MUTED)
            li = c["donor"].index(c["src_written"][-1])
            S.path(f"M{xs[li] + 27:.1f},{y + 60:.1f} Q{(xs[li] + xs[ti]) / 2 + 27:.1f},{y + 92:.1f} {xs[ti] + 27:.1f},{y + 62:.1f}",
                   col, sw=2.5, marker="m" + h)
        elif h == "set_diff":
            S.text(sx + 2 * 66, y + 88, "done = {" + ", ".join(c["src_done"]) + "}", size=19, fill=SALMON,
                   weight="600", anchor="middle")
            S.path(f"M{sx + 4 * 66 + 8:.1f},{y + 27:.1f} L{dx - 18:.1f},{y + 27:.1f}", SALMON, sw=2.5, marker="msalmon", dash="6 5")
        S.text(dx, y + 92 if h != "induction" else y + 118, note, size=20, fill=col, weight="600")
    if h == "set_diff":
        S.text(64, 735, "Source progress is copied onto the donor list; the first task not done is the target.", size=20, fill=MUTED)
    pred_table(S, 1036, 214, k + 1)
    return S.save(f"{k + 3:02d}_{h}")


# ═════════════════════════ 07 · results table ═════════════════════════
def slide_results():
    S = Slide()
    S.header("Results: no rule wins in a majority of groups",
             "KV transfer + source query · winner = highest gaze among the donor's four tasks · counts out of 128")
    x, y = 64, 230
    cw = [290, 560, 150, 150, 150, 160]
    hdr = ["hypothesis", "rule", "ALT1", "ALT2", "ALT3", "all three"]
    cx = x
    for j, hd in enumerate(hdr):
        S.text(cx + (0 if j < 2 else cw[j] / 2), y, hd, size=21, fill=MUTED, weight="600",
               anchor=None if j < 2 else "middle")
        cx += cw[j]
    S.line(x, y + 16, x + sum(cw), y + 16, RULE, sw=1.5)
    for i, (h, rule, _, col) in enumerate(HYPS):
        ry = y + 76 + i * 78
        S.rect(x, ry - 22, 16, 16, col, rx=3)
        S.text(x + 28, ry - 7, h, size=24, weight="600", fam=MONO)
        S.text(x + cw[0], ry - 7, rule, size=19, fam=MONO, fill=MUTED)
        cx = x + cw[0] + cw[1]
        others = [PRED[o] for o, *_ in HYPS if o != h]
        for j, alt in enumerate(ALTS):
            t = PRED[h][j]
            unique = all(o[j] != t for o in others)
            mid = cx + cw[j + 2] / 2
            if unique:
                S.rect(mid - 62, ry - 38, 124, 50, col, col, rx=8, sw=1.5, op=0.13)
            S.text(mid - 24, ry - 4, t, size=28, weight="600", fill=col if unique else INK, anchor="middle")
            S.text(mid + 22, ry - 4, f"{WIN[alt][t]}", size=22, fill=INK if unique else MUTED, anchor="middle")
            cx += cw[j + 2]
        mid = cx + cw[-1] / 2
        S.text(mid, ry - 4, f"{COMPLETE[h]}", size=28, weight="600", anchor="middle")
    ry = y + 76 + 4 * 78
    S.line(x, ry - 44, x + sum(cw), ry - 44, RULE, sw=1.5)
    S.text(x + 28, ry - 7, "none of the four", size=22, fill=MUTED)
    S.text(x + sum(cw[:-1]) + cw[-1] / 2, ry - 4, f"{NONE_OF}", size=28, weight="600", fill=MUTED, anchor="middle")
    S.text(64, 770, "Boxed cells: the one setup where a hypothesis disagrees with the other three (its test).", size=20, fill=MUTED)
    S.text(64, 804, "Unboxed cells share a prediction, so they share a count: e in ALT1 (65) is not evidence for any one rule.",
           size=20, fill=MUTED)
    S.text(64, 838, "'all three' = groups where the rule's predicted task wins in ALT1, ALT2 and ALT3.", size=20, fill=MUTED)
    return S.save("07_results")


# ═════════════════════════ 08 · distinguishing comparisons ═════════════════════════
def slide_margins():
    S = Slide()
    S.header("The three tests, pooled over 128 groups",
             "gaze margin between the shared prediction (e) and the singled-out one · mean and 95% bootstrap interval")
    x0, x1, y0 = 500, 1180, 250
    lo, hi = -0.02, 0.06
    X = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)
    S.line(X(0), y0 - 30, X(0), y0 + 3 * 150 - 60, FAINT, sw=2, dash="6 6")
    for v in [-0.02, 0.0, 0.02, 0.04, 0.06]:
        S.text(X(v), y0 + 3 * 150 - 24, f"{v:+.2f}" if v else "0", size=18, fill=MUTED, anchor="middle")
    S.text(X(-0.02), y0 + 3 * 150 + 12, "← favours the singled-out rule", size=18, fill=MUTED)
    S.text(X(0.06), y0 + 3 * 150 + 12, "favours e (the other three) →", size=18, fill=MUTED, anchor="end")
    verdict = {"order_id": ("unresolved", "interval spans 0"), "complete_count": ("unresolved", "interval spans 0"),
               "induction": ("against induction", "interval excludes 0")}
    for i, (alt, con, h, m, l, u) in enumerate(MARGINS):
        y = y0 + i * 150
        col = HC[h]
        S.text(64, y + 8, f"{alt}:  {con}", size=26, weight="600", fam=MONO)
        S.text(64, y + 42, f"tests {h}", size=20, fill=col, weight="600", fam=MONO)
        S.line(X(l), y, X(u), y, col, sw=5)
        S.line(X(l), y - 12, X(l), y + 12, col, sw=3); S.line(X(u), y - 12, X(u), y + 12, col, sw=3)
        S.add(f'<circle cx="{X(m):.1f}" cy="{y}" r="10" fill="{col}"/>')
        S.text(X(m), y - 22, f"{m:+.4f}", size=18, fill=INK, anchor="middle")
        S.text(x1 + 40, y - 2, verdict[h][0], size=23, fill=INK if h == "induction" else MUTED, weight="600")
        S.text(x1 + 40, y + 26, verdict[h][1], size=19, fill=MUTED)
    S.text(64, 820, "A positive margin favours e but not set_diff alone: two other rules also predict e in each setup.",
           size=20, fill=MUTED)
    return S.save("08_margins")


if __name__ == "__main__":
    for fn in [slide_setup, slide_configs, *[(lambda k=k: slide_hyp(k)) for k in range(4)], slide_results, slide_margins]:
        print("wrote", fn())
