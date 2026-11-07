"""Compile + execution check of the SUCCESSFUL cross-language NRGS transfers (Coder-32B, goal 250 heads), via
:mod:`src.nrgs.exec_check`: does the patched destination code compile, and does its frontier statement implement the
transferred sub-task completely (same value as the bank's reference implementation on every test input)?

  kv : KV transfer of the task span, V x4 (the paper's arm); success = the causal judge's ``triggered``
       (the destination implements SRC's next sub-task); target = SRC's next sub-task
  q  : Q transfer, gaze-select-restrict (a2, __seq), order_diff and order_matched; success = ``causal.triggered``
       (the destination implements D, the sub-task pending under SRC's progress); target = D. Q records do not
       store the DST prompt: instances are rebuilt from the bank with the eval's per-cell seed and matched by key.

Directions: python->cpp and cpp->python. Pure parsing + g++ / python subprocesses; reads results/ only.

    python scripts/nrgs/exec_check_transfers.py --out agent_reports/claude/data/exec_check   # writes {kv,q}.json
"""
import argparse
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.append(".")

import src.nrgs.data as DATA  # noqa: E402
from src.nrgs import exec_check as XC  # noqa: E402
from src.utils.env_utils import DEFAULT_RESULTS_DIR  # noqa: E402
from src.utils.experiment_utils import set_seed  # noqa: E402

M = "Qwen2.5-Coder-32B-Instruct"
KV = (f"{DEFAULT_RESULTS_DIR}/diff/{M}/path_patch/"
      "I1-off__I2-off__I3-on-V4__win-span__span-list+gap__cppnote-sys__a1__th-goal_250__gh-goal_250__ro-goal_250")
Q = f"{DEFAULT_RESULTS_DIR}/diff/{M}/diff_trigger/ph-goal_250__a2__piv-eq__cl-ppgazerestrict-anceq__seq"
DTS = ["int", "str", "list[int]", "list[str]"]
DIRS = {"python-cpp": ("python", "cpp"), "cpp-python": ("cpp", "python")}


def kv_jobs():
    for dt in DTS:
        dk = dt.replace("[", "_").replace("]", "")
        for d in DIRS:
            for rec in json.load(open(f"{KV}/{dk}__{d}/results.json")):
                if rec["causal"]["triggered"]:
                    yield dt, d, rec


def kv_one(job):
    dt, d, rec = job
    shared = re.search(r"shared=([^|]+)", rec["key"]).group(1).split("-")
    r = XC.check(rec["raw"]["dst_prompt"], rec["raw"]["gen_patched"], datatype=dt,
                 targets=[rec["expected_next"], rec["dst_next"]], listed=shared)
    return dict(arm="kv_v4", variant="-", datatype=dt, direction=d, key=rec["key"], target=rec["expected_next"],
                other=rec["dst_next"], gen=rec["raw"]["gen_patched"], **r)


def _key(inst):
    ids = lambda seq: "-".join(p.id for p in seq)
    return (f"{inst.meta['variant']}|src={ids(inst.src_order)}|dst={ids(inst.dst_order)}"
            f"|prog_src={'-'.join(inst.src_progress)}|prog_dst={'-'.join(inst.dst_progress)}")


def q_jobs(seed=0):
    for variant in ["order_diff", "order_matched"]:
        for dt in DTS:
            for d, (sl, dl) in DIRS.items():
                tag = f"{dt.replace('[', '(').replace(']', ')')}__{sl}-{dl}"
                cfg_seed = (seed + int(hashlib.md5(f"{variant}/{tag}".encode()).hexdigest()[:8], 16)) % (2**32)
                set_seed(cfg_seed)        # = q_transfer_eval.py's per-cell seed
                insts = {_key(i): i for i in DATA.diff_trigger_setups(dt, variant=variant, n_samples=32, seed=cfg_seed,
                                                                       src_lang=sl, dst_lang=dl)}
                recs = json.load(open(f"{Q}/{variant}/{tag}/results.json"))
                for rec in (recs if isinstance(recs, list) else recs.values()):
                    assert rec["key"] in insts, f"instance did not reproduce: {rec['key']}"
                    if rec["causal"]["triggered"]:
                        yield variant, dt, d, rec, insts[rec["key"]]


def q_one(job):
    variant, dt, d, rec, inst = job
    pf = DATA._finalize_prefill(inst.dst_parts()[1])
    prefix = pf.split("<code>\n", 1)[1] if "<code>\n" in pf else pf
    B = next(p.id for p in inst.dst_order if p.id not in inst.dst_progress)       # DST's own next sub-task
    C, D = inst.src_target, inst.diff_target
    r = XC.check(prefix, rec["gen_patched"], datatype=dt, targets=[D, B, C], listed=[p.id for p in inst.dst_order])
    return dict(arm="q_gsr", variant=variant, datatype=dt, direction=d, key=rec["key"], target=D, other=B, C=C,
                gen=rec["gen_patched"], **r)


def summarize(rows):
    out = {}
    for (arm, variant, d) in sorted({(r["arm"], r["variant"], r["direction"]) for r in rows}):
        rs = [r for r in rows if (r["arm"], r["variant"], r["direction"]) == (arm, variant, d)]
        f = lambda s, k: sum(r["compiles"] and r["targets"][r["target"]][k] for r in s)
        cell = lambda s: dict(n=len(s), compiles=sum(r["compiles"] for r in s), complete=f(s, "complete"),
                              complete_typical=f(s, "complete_typical"),
                              cut_off=sum(r["truncated"] and not r["compiles"] for r in s))
        out[f"{arm}|{variant}|{d}"] = dict(all=cell(rs), **{dt: cell([r for r in rs if r["datatype"] == dt]) for dt in DTS})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", nargs="+", default=["kv", "q"], choices=["kv", "q"])
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for arm in args.arms:
        jobs, fn = (list(kv_jobs()), kv_one) if arm == "kv" else (list(q_jobs()), q_one)
        with ThreadPoolExecutor(args.workers) as ex:
            rows = list(ex.map(fn, jobs))
        json.dump(dict(summary=summarize(rows), rows=rows), open(os.path.join(args.out, f"{arm}.json"), "w"),
                  indent=1, default=str)
        for k, v in summarize(rows).items():
            a = v["all"]
            print(f"{k:32s} n={a['n']:3d}  compiles {a['compiles'] / a['n']:.3f}  complete {a['complete'] / a['n']:.3f}"
                  f"  complete(typical) {a['complete_typical'] / a['n']:.3f}  cut off {a['cut_off']}")


if __name__ == "__main__":
    main()
