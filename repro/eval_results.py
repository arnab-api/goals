"""Headline numbers of the evaluation-wave results of one model, as CLAUDE.md section 4 reports them: GoalScope
mean4 and per pivot (4.1 / 4.4), ablation baseline / ablated / control (4.6), obstacle ADAPTED rate by level (4.8),
KV transfer pooled causal / read-out rates per arm (4.9), Q transfer pooled rates per arm and variant and the
c_in_dst joint rates (4.12), and the temporal-persistence record counts (4.7). With --curves, also the retention
curves of temporal_persistence.ipynb cells 2-3 (cos of the pivot encoding with the encoding x% into the
implementation, impl spans 10-40 tokens; goal heads, and goal vs control on the matched conditions). With
--holdout N, also each GoalScope leaf on the specs outside the N-spec round-robin subset (eval_goalscope --limit N)
that a fit check tuned its constants on.

    python repro/eval_results.py <HF id or slug> [<results dir>] [--curves] [--holdout N]       from the clone root
"""
import glob
import json
import os
import sys

argv = sys.argv[1:]
HOLDOUT = None
if "--holdout" in argv:
    i = argv.index("--holdout")
    HOLDOUT = int(argv[i + 1])
    del argv[i:i + 2]
args = [a for a in argv if a != "--curves"]
slug = args[0].split("/")[-1]
R = args[1] if len(args) > 1 else "results"
PIVOTS = ("prev_end", "indent", "var", "eq")
TUNED = set()
if HOLDOUT:
    sys.path.insert(0, ".")
    from scripts.evaluation.eval_goalscope import (  # noqa: E402
        ALL_DATATYPES, DEFAULT_DATASET, limit_round_robin, load_locked_datapoints,
    )
    TUNED = {sid for sid, _ in limit_round_robin(load_locked_datapoints(DEFAULT_DATASET, ALL_DATATYPES)[0], HOLDOUT)}


def load(p):
    with open(p) as f:
        return json.load(f)


def short(path, root):
    return os.path.relpath(os.path.dirname(path), root)


print(f"== {slug}")
root = f"{R}/evaluation/goalscope/{slug}"
for p in sorted(glob.glob(f"{root}/*/*/*/*/summary.json")):
    s, c = load(p), load(os.path.join(os.path.dirname(p), "config.json"))
    ys = [s["by_anchor"][a]["hard_yes"] for a in PIVOTS if a in s["by_anchor"]]
    print(f"GoalScope {short(p, root)[:90]}\n    n={s.get('n_samples')} mean4={sum(ys) / len(ys):.3f} "
          + " ".join(f"{a}={s['by_anchor'][a]['hard_yes']:.3f}" for a in PIVOTS if a in s["by_anchor"])
          + f" rubric={c.get('hard_rubric_tag')}")
    if TUNED:
        recs = [json.loads(line) for line in open(os.path.join(os.path.dirname(p), "records.jsonl")) if line.strip()]
        held = [r for r in recs if r["spec"] not in TUNED]
        by = {a: [r["hard"] == "YES" for r in held if r["anchor"] == a] for a in PIVOTS}
        hy = [sum(v) / len(v) for v in by.values() if v]
        print(f"    held-out (not in the {HOLDOUT} tuning specs): n={len({r['spec'] for r in held})} "
              f"mean4={sum(hy) / len(hy):.3f} " + " ".join(f"{a}={sum(v) / len(v):.3f}" for a, v in by.items() if v))

root = f"{R}/evaluation/ablation/{slug}"
for p in sorted(glob.glob(f"{root}/*/*/summary.json")):
    s = load(p)
    o = s["overall"]
    print(f"ablation {short(p, root)[:100]}\n    n={s['n']} " + " ".join(f"{k}={v:.3f}" for k, v in o.items()))

p = f"{R}/obstacle_robustness/{slug}/summary.json"
if os.path.exists(p):
    s = load(p)
    print(f"obstacle n={s['n']} " + " ".join(f"{lvl}={s['by_level'][lvl]['adapted']:.2f}" for lvl in s["by_level"])
          + f" verdicts={s['verdict_totals']}")


def pooled(cells, rate, n_key):
    n = sum(c[n_key] for c in cells)
    vals = [(c[rate], c[n_key]) for c in cells if c.get(rate) is not None]
    return (sum(v * k for v, k in vals) / sum(k for _, k in vals) if vals else None), n


def fmt(x):
    return "n/a" if x is None else f"{x:.3f}"


root = f"{R}/diff/{slug}/path_patch"
for arm in sorted(glob.glob(f"{root}/*/")):
    cells = [load(p) for p in sorted(glob.glob(f"{arm}*/meta.json"))]
    if not cells:
        continue
    c, n = pooled(cells, "causal_trigger_rate", "n_instances")
    r, _ = pooled(cells, "readout_shift_rate", "n_instances")
    print(f"KV {os.path.basename(arm.rstrip('/'))[:100]}\n    cells={len(cells)} n={n} causal={fmt(c)} readout={fmt(r)}")

root = f"{R}/diff/{slug}/diff_trigger"
for arm in sorted(glob.glob(f"{root}/*/")):
    for var in sorted(glob.glob(f"{arm}*/")):
        cells = [load(p) for p in sorted(glob.glob(f"{var}*/summary.json"))]
        if not cells:
            continue
        name = f"{os.path.basename(arm.rstrip('/'))[:80]} / {os.path.basename(var.rstrip('/'))}"
        if os.path.basename(var.rstrip("/")) == "c_in_dst":
            js = {k: pooled(cells, k, "n")[0] for k in ("joint_native_rate", "joint_c_rate", "joint_d_rate",
                                                         "joint_neither_rate")}
            print(f"Q {name}\n    cells={len(cells)} " + " ".join(f"{k[6:-5]}={fmt(v)}" for k, v in js.items()))
        else:
            c, n = pooled(cells, "causal_trigger_rate", "n")
            r, _ = pooled(cells, "readout_shift_rate", "n")
            print(f"Q {name}\n    cells={len(cells)} n={n} causal={fmt(c)} readout={fmt(r)}")

for sub in ("", "control/"):
    d = f"{R}/temporal_persistence/{slug}/{sub}"
    if os.path.isdir(d):
        print(f"temporal {sub or 'goal/'}: {len(glob.glob(d + 'cond_*.pt'))} cond records")

if "--curves" in sys.argv and os.path.isdir(f"{R}/temporal_persistence/{slug}"):
    import torch

    sys.path.insert(0, ".")
    from src.goals import temporal_persistence as TP

    grid = [0.0, 0.05, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 1.0]
    d = f"{R}/temporal_persistence/{slug}/"
    goal = [torch.load(p)["record"] for p in glob.glob(d + "cond_*.pt")]
    ctrl = [torch.load(p)["record"] for p in glob.glob(d + "control/cond_*.pt")]
    in_span = lambda rs: [r for r in rs if 10 <= r["n_impl"] <= 40]  # noqa: E731
    key = lambda r: (r["datatype"], r["lang"], r["pred_id"], r["ctx_key"], r["pos"])  # noqa: E731
    ck = {key(r) for r in ctrl}
    sets = [("goal heads, all conditions", in_span(goal))]
    if ctrl:
        sets += [("goal heads, matched to control", in_span([r for r in goal if key(r) in ck])),
                 ("control heads", in_span(ctrl))]
    print(f"temporal heads_key {sorted({r.get('heads_key', str(TP.HEAD_BUDGET)) for r in goal})}")
    for title, rs in sets:
        print(f"  {title}: n={len(rs)}")
        for name, cur in TP.compute_curves(rs, grid).items():
            print(f"    {name:<24}" + "".join(f" {int(p * 100):>3}%={cur[p]['mean']:.3f}" for p in grid))
