"""Old (task list only) vs new (task list + its closing \\n, C++ note in the system prompt) path-patching values on the SAME
instances, every axis: sub-goal (I2 off/on x V 1/2/4, span+progress) and sub-sub-goal (I2 off x V 1/2/4), plus controls."""
import glob, json, collections
M = "results/diff/Qwen2.5-Coder-32B-Instruct"
H = "a1__th-goal_250__gh-goal_250__ro-goal_250"
NEW, OLD = "list+gap__cppnote-sys", "list"
DIRS = [("python-python", "py→py"), ("python-cpp", "py→cpp"), ("cpp-python", "cpp→py"), ("cpp-cpp", "cpp→cpp")]
def tag(i2, v, win, span, heads=H):
    return (f"I1-off__{'I2-on-res' if i2 else 'I2-off'}__I3-on{'' if v == 1 else f'-V{v}'}__win-{'span' if win == 'span' else 'span+prog'}"
            f"__span-{span}__{heads}")
def load(root, t):
    return {(f.split("/")[-2], r["key"]): r for f in glob.glob(f"{M}/{root}/{t}/*/results.json") for r in json.load(open(f))}
def agg(rs, e=False):
    n = len(rs)
    if not n: return "—"
    c = sum(r["causal"]["triggered"] for r in rs) / n
    ro = sum(bool(r.get("readout") and r["readout"]["shifted"]) for r in rs) / n
    s = f"{c:.2f} / {ro:.2f}"
    if e:
        rr = [r["readout_res"] for r in rs if r.get("readout_res")]
        s += f" · e {sum(x['shifted_to_res'] for x in rr) / len(rr):.2f}"
    return s
def pair(root, i2, v, win):
    N, O = load(root, tag(i2, v, win, NEW)), load(root, tag(i2, v, win, OLD))
    ks = sorted(set(N) & set(O)); return [N[k] for k in ks], [O[k] for k in ks], ks
out = []
cfgs = [(False, v, "span") for v in (1, 2, 4)] + [(True, v, "span") for v in (1, 2, 4)] + [(False, 1, "span+progress"), (True, 1, "span+progress")]
lab = lambda i2, v, w: f"{'task span' if w == 'span' else 'task span + progress'} · I2 {'on' if i2 else 'off'} · V×{v}"
for title, sel in [("pooled + by language direction", "dir"), ("by datatype", "dt")]:
    cols = [("all", None)] + ([(lbl, d) for d, lbl in DIRS] if sel == "dir" else [(d, d) for d in ("int", "str", "list_int", "list_str")])
    out.append(f"\n**Sub-goal · {title}** — each cell: **new** · old (causal / readout; with I2 on, · e = readout → p_res's frontier e)\n")
    out.append("| config | " + " | ".join(c[0] for c in cols) + " |")
    out.append("|---" * (len(cols) + 1) + "|")
    for i2, v, w in cfgs:
        N, O, ks = pair("path_patch", i2, v, w)
        row = []
        for name, key in cols:
            f = (lambda k: True) if key is None else ((lambda k: k[0].endswith("__" + key)) if sel == "dir" else (lambda k: k[0].startswith(key + "__")))
            idx = [i for i, k in enumerate(ks) if f(k)]
            row.append(f"**{agg([N[i] for i in idx], i2)}** · {agg([O[i] for i in idx], i2)}")
        out.append(f"| {lab(i2, v, w)} | " + " | ".join(row) + " |")
# sub-sub-goal
out.append("\n**Sub-sub-goal (connective pivot) · I2 off** — each cell: **new** · old (causal / readout)\n")
groups = [("all", lambda c: True)] + [(n, (lambda d: lambda c: c.endswith("__" + d))(d)) for d, n in DIRS] + \
         [(d, (lambda d: lambda c: c.startswith(d + "__"))(d)) for d in ("list_int", "list_str")] + \
         [(o, (lambda o: lambda c: f"__{o}__" in c)(o)) for o in ("match", "diff")] + [(k, (lambda k: lambda c: f"__{k}__" in c)(k)) for k in ("and", "or")]
out.append("| V | " + " | ".join(g[0] for g in groups) + " |"); out.append("|---" * (len(groups) + 1) + "|")
for v in (1, 2, 4):
    N, O, ks = pair("path_patch_hier", False, v, "span")
    out.append(f"| ×{v} | " + " | ".join(f"**{agg([N[i] for i, k in enumerate(ks) if g(k[0])])}** · {agg([O[i] for i, k in enumerate(ks) if g(k[0])])}" for _, g in groups) + " |")
# controls
out.append("\n**Control-250 (V×4, no readout)** — causal-shift, new · old\n")
CH = "a1__th-control250_250__gh-control250_250__ro-off"
for root, lvl in (("path_patch", "sub-goal"), ("path_patch_hier", "sub-sub-goal")):
    N, O = load(root, tag(False, 4, "span", NEW, CH)), load(root, tag(False, 4, "span", OLD, CH))
    cn = collections.Counter(r["causal"]["label"] for r in N.values()); co = collections.Counter(r["causal"]["label"] for r in O.values())
    out.append(f"- {lvl}: **{cn['src']}/{len(N)}** ({dict(cn)}) · {co['src']}/{len(O)} ({dict(co)})")
print("\n".join(out))
