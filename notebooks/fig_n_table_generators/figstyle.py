"""Shared figure style for the paper's figure/table generators.

Every generator notebook in ``notebooks/fig_n_table_generators/`` imports this so the paper's charts share
one visual language (font, palette, mark conventions) and save to a consistent place.

Conventions (see the dicts below):
- **Font:** Times New Roman (msttcorefonts) for all text + math.
- **Metric encoding:** faithfulness = SOLID fill, causality = HATCHED fill ('///'). Same colour for a
  given series across both metrics; the fill pattern carries the metric. (transfer analogue:
  causal-shift = solid, readout-shift = hatched.)
- **Series colour:** a fixed, CVD-safe Okabe–Ito categorical order — assigned per entity (model / head-set
  / condition) and never cycled by rank.
- **Pivots** (prev_end, indent, var, eq, first_impl) get a fixed colour each (a cool→warm ramp), so the
  pivot always reads the same colour across figures.

Save PDFs with :func:`save_fig` → ``notebooks/Figures/<group>/<name>.pdf``.
"""
from __future__ import annotations

import os

import matplotlib as mpl
import matplotlib.pyplot as plt

# ── paths ────────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
FIG_ROOT = os.path.abspath(os.path.join(_HERE, "..", "Figures"))  # notebooks/Figures

FONT = "Times New Roman"

# ── palettes ──────────────────────────────────────────────────────────────────────
# Okabe–Ito: colourblind-safe categorical order (assign in order, never cycle).
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]

# fixed colour per model (extend as needed; order is the paper's usual order)
MODEL_COLOR = {
    "Qwen2.5-Coder-32B-Instruct": "#0072B2",
    "Qwen2.5-Coder-14B-Instruct": "#56B4E9",
    "Qwen2.5-32B-Instruct":       "#009E73",
    "Qwen2.5-72B-Instruct":       "#E69F00",
    "Olmo-3.1-32B-Instruct":      "#D55E00",
    "Olmo-3.1-32B-Instruct-SFT":  "#CC79A7",
    "GLM-4-32B-0414":             "#8C564B",
    # the comparison campaign's models (repro/PLAN.md). Chosen to stay apart from the four paper colours above they
    # share figures with (Coder-32B, Coder-14B, Qwen2.5-32B, Olmo-3.1): over those seven, the worst pair is OKLab
    # dE 9.6 under protan/deutan simulation and 15.2 under normal vision. #DDAA33 is close to Qwen2.5-72B's
    # #E69F00 (dE 2.8): do not plot GLM-4.5-Air and Qwen2.5-72B in one figure.
    "Llama-3.3-70B-Instruct":     "#762A83",
    "GLM-4.5-Air":                "#DDAA33",
    "Devstral-Small-2-24B-Instruct-2512": "#B2182B",
}

DEFAULT_MODEL = "Qwen2.5-Coder-32B-Instruct"  # the model the per-group generator notebooks plot by default


def model_slug(default: str = DEFAULT_MODEL) -> str:
    """The results slug a per-group generator notebook reads: ``$GOALS_MODEL`` (an HF id or its slug), else
    ``default`` (Coder-32B, the paper's). E.g. ``GOALS_MODEL=zai-org/GLM-4.5-Air jupyter nbconvert --execute ...``."""
    return os.environ.get("GOALS_MODEL", default).split("/")[-1]

# Canonical pivots shown in figures/tables. `first_impl` is DELIBERATELY EXCLUDED from all paper
# figures & tables (it is the first implementation token, not a pre-implementation pivot). Use `PIVOTS`.
PIVOTS = ["prev_end", "indent", "var", "eq"]
# full token order (incl. first_impl) — for colour lookup / internal use only, NOT for plotting.
PIVOT_ORDER = ["prev_end", "indent", "var", "eq", "first_impl"]
# per-pivot colours — kept consistent with notebooks/fig_n_table_generators/poster.ipynb (ANCHOR_COLOR)
PIVOT_COLOR = {
    "prev_end":   "#b5675f",
    "indent":     "#6b6aa1",
    "var":        "#d1a054",
    "eq":         "#a7c48a",
    "first_impl": "#3f8079",  # excluded from figures; kept only for lookup consistency
}

# metric fill conventions
HATCH = "////"
def metric_kw(metric: str, color: str) -> dict:
    """Bar kwargs for a metric: faithfulness/causal-shift = solid, causality/readout-shift = hatched."""
    m = metric.lower()
    # SOLID: faithfulness, causal-shift (transfer primary). HATCHED: causality (goalscope), readout-shift.
    hatched = m in ("causality", "readout", "readout_shift", "readout-shift", "readout shift")
    kw = dict(color=color, edgecolor="white", linewidth=0.6)
    if hatched:
        kw.update(hatch=HATCH, edgecolor="white")  # coloured FILL + white diagonal stripes
    return kw


# temporal-persistence family scheme (validated separately: teal=same sub-goal, orange=different)
TP_TEAL, TP_ORANGE = "#0a8f7e", "#c2410c"


# ── rcParams ────────────────────────────────────────────────────────────────────
def set_style(base_fontsize: int = 10):
    """Apply the shared rcParams. Call once at the top of a generator notebook."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": [FONT, "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "custom",
        "mathtext.rm": FONT,
        "mathtext.it": f"{FONT}:italic",
        "mathtext.bf": f"{FONT}:bold",
        "font.size": base_fontsize,
        "axes.titlesize": base_fontsize + 1,
        "axes.labelsize": base_fontsize,
        "xtick.labelsize": base_fontsize - 1,
        "ytick.labelsize": base_fontsize - 1,
        "legend.fontsize": base_fontsize - 1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,   # embed TrueType (editable text, not paths)
        "ps.fonttype": 42,
    })


# ── legend proxies + save ─────────────────────────────────────────────────────────
def metric_legend_handles(labels=("faithfulness", "causality")):
    """Two proxy handles explaining the solid/hatched metric convention (colour fill + white stripes)."""
    from matplotlib.patches import Patch
    return [
        Patch(facecolor="grey", edgecolor="white", label=labels[0]),
        Patch(facecolor="grey", edgecolor="white", hatch=HATCH, label=labels[1]),
    ]


def legend_top(ax, handles=None, labels=None, title_pad=26, **kw):
    """Place the legend in a single row at the TOP of the axes, just below the title.
    Call after set_title. Pass handles/labels or let it read them from the axes."""
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    if ax.get_title():
        ax.set_title(ax.get_title(), pad=title_pad)
    ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=len(labels), frameon=False, handlelength=1.6, columnspacing=1.4,
              borderaxespad=0.2, **kw)


def save_fig(fig, group: str, name: str, also_png: bool = True):
    """Save ``fig`` to notebooks/Figures/<group>/<name>.pdf (+ .png). Returns the pdf path."""
    out = os.path.join(FIG_ROOT, group)
    os.makedirs(out, exist_ok=True)
    pdf = os.path.join(out, f"{name}.pdf")
    fig.savefig(pdf)
    if also_png:
        fig.savefig(os.path.join(out, f"{name}.png"))
    return pdf


# ── tables: emit .tex for the paper + a true rendered preview in the notebook ─────
def save_table(tex_body: str, group: str, name: str) -> str:
    """Write a LaTeX table snippet to notebooks/Figures/<group>/<name>.tex (for \\input). Returns path."""
    out = os.path.join(FIG_ROOT, group)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"{name}.tex")
    with open(path, "w") as f:
        f.write(tex_body if tex_body.endswith("\n") else tex_body + "\n")
    return path


def render_latex_table(tex_body: str, preamble: str = r"\usepackage{booktabs,array,multirow,xcolor}"):
    """Compile a LaTeX table snippet to a PNG and return an IPython Image for INLINE display — a faithful
    preview of the actual .tex (Jupyter's MathJax cannot render `tabular`). Falls back to None on failure
    (caller should then rich_print the numbers)."""
    import subprocess, tempfile
    doc = (
        "\\documentclass[border=6pt]{standalone}\n" + preamble + "\n"
        "\\begin{document}\n" + tex_body + "\n\\end{document}\n"
    )
    try:
        with tempfile.TemporaryDirectory() as d:
            tex = os.path.join(d, "t.tex")
            with open(tex, "w") as f:
                f.write(doc)
            r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "t.tex"],
                               cwd=d, capture_output=True, timeout=60)
            pdf = os.path.join(d, "t.pdf")
            if not os.path.exists(pdf):
                return None
            subprocess.run(["pdftoppm", "-png", "-r", "200", "-singlefile", "t.pdf", "t"],
                           cwd=d, capture_output=True, timeout=60)
            png = os.path.join(d, "t.png")
            if not os.path.exists(png):
                return None
            from IPython.display import Image
            return Image(data=open(png, "rb").read())
    except Exception:
        return None
