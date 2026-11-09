"""Re-generate the transfers that FAILED to compile in ``exec_check_transfers.py`` with a longer
budget (default 256 tokens instead of 64), then re-run the compile + execution check on them.

Same instances, same intervention, same greedy decoding as the reported runs; only the patched generation's token
budget changes (KV: ``run_patched(max_new=...)``; Q: ``run_q_transfer_intervention(gsr_max_new=...)``). No readout and
no judge: the set of successful transfers stays the one judged on the 64-token generations. Each record also says
whether the first part of the new generation reproduces the reported 64-token generation (``prefix_match``).
Reads results/ only; writes ``<out>/regen_cpp.json``.

    python scripts/nrgs/exec_check_regen_cpp.py --cuda 7 \
        --exec-dir agent_reports/claude/exec_check_cross_language_transfers --out <dir>
"""
import argparse
import json
import os
import sys
import time

sys.path.append(".")


def _preparse_cuda(argv):
    for i, a in enumerate(argv):
        if a == "--cuda" and i + 1 < len(argv):
            return argv[i + 1]
    return None


if _preparse_cuda(sys.argv[1:]):
    os.environ["CUDA_VISIBLE_DEVICES"] = _preparse_cuda(sys.argv[1:])

import src.nrgs.data as DATA  # noqa: E402
import src.nrgs.functional as FUNC  # noqa: E402
import src.nrgs.kv_transfer as PP  # noqa: E402
from src.nrgs import exec_check as XC  # noqa: E402
from src.utils.experiment_utils import set_seed  # noqa: E402

sys.path.append("scripts/nrgs")
import exec_check_transfers as ECT  # noqa: E402

G250 = "head_sets/Qwen2.5-Coder-32B-Instruct/goal_topk/goal__lam20_top250.json"


def _failed(rows, directions):
    return [r for r in rows if r["direction"] in directions and not r["compiles"]]


def _prefix_match(old, new):
    o = old.split("</code")[0].rstrip()
    return new.startswith(o)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cuda", default=None)
    ap.add_argument("--exec-dir", required=True, help="dir with kv.json / q.json from exec_check_transfers.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--heads", default=G250, help="JSON head set of the run being re-generated (patch = readout heads)")
    ap.add_argument("--kv-dir", default=None, help="KV run dir (default: the paper's)")
    ap.add_argument("--q-dir", default=None, help="Q GSR run dir (default: the paper's)")
    ap.add_argument("--directions", nargs="+", default=["python-cpp"], choices=["python-cpp", "cpp-python"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fp = os.path.join(args.out, "regen_cpp.json")
    done = {(r["arm"], r["variant"], r["datatype"], r["direction"], r["key"]): r for r in (json.load(open(fp)) if os.path.exists(fp) else [])}

    kv_fail = _failed(json.load(open(os.path.join(args.exec_dir, "kv.json")))["rows"], args.directions)
    q_fail = _failed(json.load(open(os.path.join(args.exec_dir, "q.json")))["rows"], args.directions)
    print(f"failed: kv {len(kv_fail)}, q {len(q_fail)}")
    heads = [tuple(h) for h in json.load(open(args.heads))]
    ECT.KV, ECT.Q = args.kv_dir or ECT.KV, args.q_dir or ECT.Q

    import torch
    ctx = FUNC.setup("Qwen/Qwen2.5-Coder-32B-Instruct", device_map={"": 0} if torch.cuda.device_count() == 1 else "auto")
    ctx.readout_heads, ctx.gsp = heads, None

    def save():
        json.dump(list(done.values()), open(fp, "w"), indent=1, default=str)

    # ── KV (V x4): kv_transfer_eval.py sets the C++ note in the system prompt
    DATA.CPP_NOTE_IN_SYSTEM = True
    by_dt = {}
    for r in kv_fail:
        by_dt.setdefault((r["datatype"], r["direction"]), []).append(r)
    for (dt, d), rs in by_dt.items():
        sl, dl = ECT.DIRS[d]
        insts, seed = PP.sample_multi_instances(dt, sl, dl, n_samples=32, seed=1123)
        set_seed(seed)
        insts = {PP.inst_key(i): i for i in insts}
        rec_by_key = {rec["key"]: rec for (_, _, rec) in
                      ((a, b, c) for (a, b, c) in ECT.kv_jobs() if a == dt and b == d)}
        for r in rs:
            k = ("kv_v4", "-", dt, d, r["key"])
            if k in done:
                continue
            t0 = time.time()
            inst, rec = insts[r["key"]], rec_by_key[r["key"]]
            g = PP.build_geometry(inst, span="task", ctx=ctx)
            assert g["dst_prompt"] == rec["raw"]["dst_prompt"], "DST prompt did not reproduce"
            gen = PP.run_patched(g, goal_heads=heads, readout_heads=heads, v_scale=4.0, max_new=args.max_new,
                                 readout=False, ctx=ctx)["gen_patched"]
            shared = r["key"].split("shared=")[1].split("|")[0].split("-")
            c = XC.check(rec["raw"]["dst_prompt"], gen, datatype=dt, targets=[r["target"], r["other"]], listed=shared)
            done[k] = dict(arm="kv_v4", variant="-", datatype=dt, direction=d, key=r["key"],
                           target=r["target"], other=r["other"], gen64=r["gen"], gen=gen,
                           prefix_match=_prefix_match(r["gen"], gen), was_truncated=r["truncated"],
                           compile_error64=r["compile_error"], **c)
            save()
            print(f"kv {dt} {d} {r['key']}: compiles={c['compiles']} prefix={done[k]['prefix_match']} ({time.time() - t0:.0f}s)")

    # ── Q (GSR): q_transfer_eval.py leaves the C++ note off (module default)
    DATA.CPP_NOTE_IN_SYSTEM = False
    want = {(r["variant"], r["datatype"], r["direction"], r["key"]): r for r in q_fail}
    for variant, dt, d, rec, inst in ECT.q_jobs():
        r = want.get((variant, dt, d, rec["key"]))
        if r is None:
            continue
        k = ("q_gsr", variant, dt, d, rec["key"])
        if k in done:
            continue
        t0 = time.time()
        raw = FUNC.run_q_transfer_intervention(ctx.mt, inst, heads, readout_heads=heads, max_new=96, alpha=2.0,
                                               compute_readout=False, gsr_max_new=args.max_new)
        gen = raw["gen_patched"]
        pf = DATA._finalize_prefill(inst.dst_parts()[1])
        prefix = pf.split("<code>\n", 1)[1] if "<code>\n" in pf else pf
        c = XC.check(prefix, gen, datatype=dt, targets=[r["target"], r["other"], r["C"]],
                     listed=[p.id for p in inst.dst_order])
        done[k] = dict(arm="q_gsr", variant=variant, datatype=dt, direction=d, key=rec["key"],
                       target=r["target"], other=r["other"], C=r["C"], gen64=r["gen"], gen=gen,
                       prefix_match=_prefix_match(r["gen"], gen), was_truncated=r["truncated"],
                       compile_error64=r["compile_error"], **c)
        save()
        print(f"q {variant} {dt} {d} {rec['key']}: compiles={c['compiles']} prefix={done[k]['prefix_match']} "
              f"({time.time() - t0:.0f}s)")
    print(f"done: {len(done)} records -> {fp}")


if __name__ == "__main__":
    main()
