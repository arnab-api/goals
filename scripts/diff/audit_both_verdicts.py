"""Audit: how much do 'both' verdicts inflate causal-shift / readout-shift? (before 2026-10-13, judge_causal / eval_readouts counted both as a hit).
Reads the stored oracle labels only; "with both" = old rule, "without" = current rule."""
import json, glob, os, collections as C
from pathlib import Path
import numpy as np

# run from anywhere: paths are relative to results/diff/<model>
os.chdir(Path(__file__).resolve().parents[2] / "results" / "diff" / "Qwen2.5-Coder-32B-Instruct")
ARMS = {  # paper arm -> glob of per-cell results.json
    "KV-transfer (Fig 5, Tab kv)": "kv_transfer/multi_pred/ph-goal_250__ro-goal_250__abl-off__a2__pad/*/results.json",
    "KV multi-hierarchy (Fig 21)": "kv_transfer/multi_hier/ph-goal_250__ro-goal_250__a2__pad/*/results.json",
    "Q gsr order-diff (Fig 6, Tab 9)": "diff_trigger/ph-goal_250__a2__piv-eq__cl-ppgazerestrict-anceq__seq/order_diff/*/results.json",
    "Q gsr order-matched (Tab 8)": "diff_trigger/ph-goal_250__a2__piv-eq__cl-ppgazerestrict-anceq__seq/order_matched/*/results.json",
    "Q random-head control (re-clamp)": "diff_trigger/ph-control250_250__a2__piv-eq__seq/order_diff/*/results.json",
}
for name, g in ARMS.items():
    cells = sorted(glob.glob(g))
    lab_c, lab_r = C.Counter(), C.Counter()
    rate = {"causal": [[], []], "readout": [[], []]}   # [with both, without both] per cell
    deltas = []
    for f in cells:
        R = json.load(open(f)); R = R if isinstance(R, list) else R.get("records", R)
        cl = [r["causal"]["label"] for r in R if r.get("causal")]
        rl = [r["readout"]["patched"] for r in R if r.get("readout")]
        lab_c.update(cl); lab_r.update(rl)
        for key, L in (("causal", cl), ("readout", rl)):
            if not L: continue
            w = sum(l in ("src", "both") for l in L) / len(L); wo = sum(l == "src" for l in L) / len(L)
            rate[key][0].append(w); rate[key][1].append(wo)
            if key == "causal": deltas.append(w - wo)
    n = sum(lab_c.values())
    print(f"\n== {name}: {len(cells)} cells, {n} records")
    print(f"   causal labels  {dict(lab_c)}")
    print(f"   readout labels {dict(lab_r)}")
    for key in rate:
        if rate[key][0]:
            a, b = np.mean(rate[key][0]), np.mean(rate[key][1])
            print(f"   {key:8s} with both {a:.4f} · without {b:.4f} · Δ {a-b:+.4f}")
    if deltas: print(f"   max per-cell causal Δ {max(deltas):.3f}")
