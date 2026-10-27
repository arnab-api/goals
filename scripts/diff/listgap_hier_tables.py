"""Sub-sub-goal (hier) list+gap vs old task-list span, on the same instances (I1 off, I2 off, goal-250, V 1/2/4)."""
import collections, glob, json, os
R = "results/diff/Qwen2.5-Coder-32B-Instruct/path_patch_hier"
H = "a1__th-goal_250__gh-goal_250__ro-goal_250"
tag = lambda v, new: f"I1-off__I2-off__I3-on{'' if v == 1 else f'-V{v}'}__win-span__span-{'list+gap__cppnote-sys' if new else 'list'}__{H}"
rows = collections.defaultdict(lambda: collections.defaultdict(list))   # (v, cell) -> {"new": [...], "old": [...]}
for v in (1, 2, 4):
    for fp in sorted(glob.glob(f"{R}/{tag(v, True)}/*/results.json")):
        cell = fp.split("/")[-2]
        new = json.load(open(fp)); keys = {r["key"] for r in new}
        ofp = f"{R}/{tag(v, False)}/{cell}/results.json"
        old = [r for r in json.load(open(ofp)) if r["key"] in keys] if os.path.exists(ofp) else []
        rows[(v, cell)]["new"], rows[(v, cell)]["old"] = new, old
def agg(rs):
    n = len(rs)
    return (sum(r["causal"]["triggered"] for r in rs) / n, sum(bool(r["readout"] and r["readout"]["shifted"]) for r in rs) / n, n) if n else None
f = lambda a: "—" if not a else f"{a[0]:.2f} / {a[1]:.2f}"
print("| cell | n | V1 new · old | V2 new · old | V4 new · old |"); print("|---|---|---|---|---|")
for cell in sorted({c for _, c in rows}):
    n = len(rows[(1, cell)]["new"]) if (1, cell) in rows else 0
    print(f"| {cell} | {n} | " + " | ".join(f"**{f(agg(rows[(v, cell)]['new']))}** · {f(agg(rows[(v, cell)]['old']))}" if (v, cell) in rows else "—" for v in (1, 2, 4)) + " |")
for v in (1, 2, 4):
    N = [r for (vv, c), d in rows.items() if vv == v for r in d["new"]]; O = [r for (vv, c), d in rows.items() if vv == v for r in d["old"]]
    print(f"pooled V{v}: new {f(agg(N))} (n={len(N)}) · old {f(agg(O))} (n={len(O)})")
    for o in ("match", "diff"):
        N = [r for (vv, c), d in rows.items() if vv == v and f"__{o}__" in c for r in d["new"]]; O = [r for (vv, c), d in rows.items() if vv == v and f"__{o}__" in c for r in d["old"]]
        if N: print(f"   order {o}: new {f(agg(N))} (n={len(N)}) · old {f(agg(O))}")
