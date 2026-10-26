"""Compare the smoke cache (fresh, scratch root) with the notebook's real cache (list[str] has all 14 old configs)."""
import glob, json, os, sys
SMOKE = sys.argv[1]
REAL = "/disk/u/arnab/Codes/Projects/goals/results/notebook_caches/task_vs_progress/Qwen2.5-Coder-32B-Instruct/list_str__python-python__seed1123__mx64__ro0__task+gap"
for fp in sorted(glob.glob(f"{SMOKE}/*.json")):
    s = json.load(open(fp)); r = json.load(open(os.path.join(REAL, os.path.basename(fp))))
    print(f"== {os.path.basename(fp)}  clean gen_dst equal: {s['clean']['gen_dst'] == r['clean']['gen_dst']}  gen_src equal: {s['clean']['gen_src'] == r['clean']['gen_src']}")
    for name, row in s["configs"].items():
        old = r["configs"].get(name)
        same = "n/a (new config)" if old is None else ("gen identical" if old["gen"] == row["gen"] else f"gen DIFFERS (verdict {old['causal']} -> {row['causal']})")
        print(f"   {name:34s} {row['causal']:8s} {same}")
    for s_ in ("task", "progress", "task+progress"):
        a, b = s["configs"][f"kv·all · {s_} · V×1"], s["configs"][f"residual · {s_} · V×1"]
        print(f"   kv·all V1 == residual on {s_:14s}: gen {'identical' if a['gen'] == b['gen'] else 'differs'} ({a['causal']} / {b['causal']})")
