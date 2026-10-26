"""Markdown tables for the list+gap (task span + closing \\n, C++ note in system prompt) runs vs the old task-list span."""
import json, os, collections
R = "/disk/u/arnab/Codes/Projects/goals/results/diff/Qwen2.5-Coder-32B-Instruct/path_patch"
H = "a1__th-goal_250__gh-goal_250__ro-goal_250"
def tag(v, new): return f"I1-off__I2-off__I3-on{'' if v == 1 else f'-V{v}'}__win-span__span-{'list+gap__cppnote-sys' if new else 'list'}__{H}"
DIRS = [("python-python", "py→py"), ("python-cpp", "py→cpp"), ("cpp-python", "cpp→py"), ("cpp-cpp", "cpp→cpp")]
DTS = [("int", "int"), ("str", "str"), ("list_int", "list[int]"), ("list_str", "list[str]")]
def load(v, new, dt, d):
    fp = f"{R}/{tag(v, new)}/{dt}__{d}/results.json"
    return json.load(open(fp)) if os.path.exists(fp) else []
def agg(rs):
    n = len(rs)
    if not n: return None
    c = sum(r["causal"]["triggered"] for r in rs) / n
    ro = sum(bool(r["readout"] and r["readout"]["shifted"]) for r in rs) / n
    val = sum(bool(r["readout"] and r["readout"]["valid"]) for r in rs) / n
    return c, ro, val, n
f = lambda a: "—" if a is None else f"{a[0]:.2f} / {a[1]:.2f}"
fv = lambda a: "—" if a is None else f"{a[0]:.2f} / {a[1]:.2f} [{a[2]:.2f}]" + ("" if a[3] == 32 else f" (n={a[3]})")
cells = {(v, new, dt, d): load(v, new, dt, d) for v in (1, 2, 4) for new in (0, 1) for dt, _ in DTS for d, _ in DIRS}
# completeness
print("complete cells (new):", sum(len(cells[(v, 1, dt, d)]) == 32 for v in (1, 2, 4) for dt, _ in DTS for d, _ in DIRS), "/ 48")
print("\n**Pooled and by language direction** (new · old)\n")
print("| V | pooled | " + " | ".join(n for _, n in DIRS) + " |"); print("|---" * 6 + "|")
for v in (1, 2, 4):
    row = []
    for sel in [None] + [d for d, _ in DIRS]:
        a = agg([r for (vv, nw, dt, d), rs in cells.items() if vv == v and nw == 1 and (sel is None or d == sel) for r in rs])
        b = agg([r for (vv, nw, dt, d), rs in cells.items() if vv == v and nw == 0 and (sel is None or d == sel) for r in rs])
        row.append(f"**{f(a)}** · {f(b)}" + (f" (n={a[3]})" if a and a[3] not in (512, 128) else ""))
    print(f"| ×{v} | " + " | ".join(row) + " |")
print("\n**By datatype** (new · old)\n")
print("| V | " + " | ".join(n for _, n in DTS) + " |"); print("|---" * 5 + "|")
for v in (1, 2, 4):
    row = []
    for sel, _ in DTS:
        a = agg([r for (vv, nw, dt, d), rs in cells.items() if vv == v and nw == 1 and dt == sel for r in rs])
        b = agg([r for (vv, nw, dt, d), rs in cells.items() if vv == v and nw == 0 and dt == sel for r in rs])
        row.append(f"**{f(a)}** · {f(b)}" + (f" (n={a[3]})" if a and a[3] != 128 else ""))
    print(f"| ×{v} | " + " | ".join(row) + " |")
for v in (1, 2, 4):
    print(f"\n*×{v}*\n")
    print("| datatype | " + " | ".join(n for _, n in DIRS) + " | all |"); print("|---" * 6 + "|")
    for sel, nm in DTS + [(None, "all")]:
        row = [fv(agg([r for (vv, nw, dt, d), rs in cells.items() if vv == v and nw == 1 and (sel is None or dt == sel) and d == dd for r in rs])) for dd, _ in DIRS]
        row.append(fv(agg([r for (vv, nw, dt, d), rs in cells.items() if vv == v and nw == 1 and (sel is None or dt == sel) for r in rs])))
        print(f"| {nm} | " + " | ".join(row) + " |")
