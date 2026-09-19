#!/usr/bin/env python
"""Re-score existing ``c_in_dst`` diff-trigger results with the JOINT B/C/D classifier.

The original run scored two INDEPENDENT binary oracle calls (causal = D-vs-B, literalC = C-vs-B), whose
marginals can both fire on one generation and need not sum to <=1 (they double-count when C and D
compete — exactly the ``c_in_dst`` case). This utility re-judges each SAVED patched generation with a
single mutually-exclusive verdict (``src.diff.eval.judge_joint``) → one of {B (no-op), C (src-target),
D (progress), neither}, and writes ``record['joint']`` + joint rates back into ``results.json`` /
``summary.json``. No GPU / no re-generation (greedy decoding is deterministic; generations are on disk).

``task_desc`` isn't stored per-record, so instances are reconstructed deterministically from
``diff_trigger_setups`` (same seed/n/naming) and matched by ``key``.

Usage:
    conda run -n goals python scripts/diff/rescore_c_in_dst_joint.py --dry     # verify key match only
    conda run -n goals python scripts/diff/rescore_c_in_dst_joint.py           # re-judge (oracle calls)
"""
import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

sys.path.append(".")

import src.diff.data as D
import src.diff.eval as E

MODEL_DIR = "results/diff/Qwen2.5-Coder-32B-Instruct/diff_trigger"


def _inst_key(inst):
    ids = lambda seq: "-".join(p.id for p in seq)
    return (f"{inst.meta['variant']}|src={ids(inst.src_order)}|dst={ids(inst.dst_order)}"
            f"|prog_src={'-'.join(inst.src_progress)}|prog_dst={'-'.join(inst.dst_progress)}")


def _cfg_seed(variant, tag, seed):
    return (seed + int(hashlib.md5(f"{variant}/{tag}".encode()).hexdigest()[:8], 16)) % (2**32)


def rescore_config(results_fp, args):
    tag = results_fp.parent.name                       # e.g. "list(int)__python-python"
    dt_safe, langs = tag.split("__")
    sl, dl = langs.split("-")
    dt = dt_safe.replace("(", "[").replace(")", "]")   # "list(int)" -> "list[int]"
    cfg_seed = _cfg_seed("c_in_dst", tag, args.seed)

    setups = D.diff_trigger_setups(
        dt, variant="c_in_dst", n_samples=args.n_samples, seed=cfg_seed,
        naming=args.naming, letter_stub="var_eq", src_lang=sl, dst_lang=dl,
        distinct_families=True, nontrivial_frontier=True,
    )
    by_key = {_inst_key(i): i for i in setups}

    records = json.loads(results_fp.read_text())
    matched = sum(1 for r in records if r["key"] in by_key)
    print(f"  {results_fp.parent.parent.name}/{tag}: {len(records)} records, "
          f"{matched}/{len(records)} reconstructed")
    if args.dry:
        return None

    counts = {"native": 0, "src_target": 0, "diff_target": 0, "neither": 0, "parse_error": 0}
    for r in records:
        inst = by_key.get(r["key"])
        if inst is None:
            r["joint"] = None
            continue
        joint = E.judge_joint(inst, r["gen_patched"], model_name=args.oracle_model)
        r["joint"] = joint
        counts[joint["label"]] = counts.get(joint["label"], 0) + 1
    results_fp.write_text(json.dumps(records, indent=2))

    n = len(records)
    summ_fp = results_fp.parent / "summary.json"
    summ = json.loads(summ_fp.read_text()) if summ_fp.exists() else {}
    summ.update(
        joint_native_rate=counts["native"] / n,
        joint_c_rate=counts["src_target"] / n,
        joint_d_rate=counts["diff_target"] / n,
        joint_neither_rate=counts["neither"] / n,
    )
    summ_fp.write_text(json.dumps(summ, indent=2))
    print(f"    JOINT: B(no-op)={counts['native']/n:.3f} C(src-target)={counts['src_target']/n:.3f} "
          f"D(progress)={counts['diff_target']/n:.3f} neither={counts['neither']/n:.3f} "
          f"(parse_err={counts['parse_error']}) n={n}")
    return dict(config=results_fp.parent.parent.name, tag=tag, n=n, **{
        "B": counts["native"] / n, "C": counts["src_target"] / n,
        "D": counts["diff_target"] / n, "neither": counts["neither"] / n})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry", action="store_true", help="reconstruct + report key match only, no oracle calls")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-samples", type=int, default=32)
    ap.add_argument("--naming", default="letters")
    ap.add_argument("--oracle-model", default="claude-sonnet-4-6")
    ap.add_argument("--glob", default=f"{MODEL_DIR}/*/c_in_dst/*/results.json")
    args = ap.parse_args()

    fps = sorted(Path(p) for p in glob.glob(args.glob))
    print(f"found {len(fps)} c_in_dst results.json")
    rows = []
    for fp in fps:
        row = rescore_config(fp, args)
        if row:
            rows.append(row)
    if rows:
        print("\n=== JOINT (exclusive) summary ===")
        for r in rows:
            print(f"  {r['config']:42s} {r['tag']:24s} B={r['B']:.3f} C={r['C']:.3f} "
                  f"D={r['D']:.3f} neither={r['neither']:.3f}")


if __name__ == "__main__":
    main()
