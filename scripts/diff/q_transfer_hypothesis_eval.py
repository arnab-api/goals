#!/usr/bin/env python
"""Q-transfer generation eval for the query-content hypotheses (``src/diff/hyp_eval.py``).

For each hypothesis config (``src/diff/nrgs_prefix.py::CONFIGS``: h1c_*, h2_*, h3_*, and the H1 set), sample
instances on the 16-predicate q-transfer banks (4 datatypes), transplant SRC's pivot query into DST over
``--patch-heads`` (pre-RoPE, + OV boost ``--alpha``), greedy-decode clean and patched DST, and judge each
continuation with the exclusive multi-letter judge (which DST pending task / SRC frontier / neither).
Every judged letter is scored against EVERY hypothesis's prediction for that config, so one run gives
clean rate, patched rate and shift per hypothesis.

``--n-samples`` is per (hypothesis, datatype), split evenly over that hypothesis's configs
(default 64 -> 32 per config for the two-config hypotheses; 4 datatypes -> 256 per hypothesis).

Results: ``results/diff/<model>/q_transfer_hyp/<run_tag>/<config>/<dt>__<sl>-<dl>/{results,summary}.json``
and ``<run_tag>/summary.json`` (per hypothesis, pooled and by datatype).
Usage:
    conda run -n goals python scripts/diff/q_transfer_hypothesis_eval.py --cuda 0 --configs H1c H2 H3 \\
        --patch-heads head_sets/Qwen2.5-Coder-32B-Instruct/hyp-h3-lam20__nrgs_h3_thr0.9.json
    conda run -n goals python scripts/diff/q_transfer_hypothesis_eval.py --cuda 0 --patch-heads goal --max-instances 2
"""
import argparse
import collections
import json
import logging
import math
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _preparse_cuda(argv):
    for i, a in enumerate(argv):
        if a == "--cuda" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--cuda="):
            return a.split("=", 1)[1]
    return None


_cuda = _preparse_cuda(sys.argv[1:])
if _cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = _cuda
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import src.diff.functional as FUNC  # noqa: E402
import src.diff.hyp_eval as HE  # noqa: E402
import src.diff.nrgs_prefix as N  # noqa: E402
from src.utils import env_utils, logging_utils  # noqa: E402

logger = logging.getLogger("q_transfer_hypothesis_eval")
LANG_TRANSFERS = {"py-py": ("python", "python"), "py-cpp": ("python", "cpp"),
                  "cpp-py": ("cpp", "python"), "cpp-cpp": ("cpp", "cpp")}
HYPS = ("H0", "H1", "H1c", "H2", "H3", HE.SRC_TARGET, "neither")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    ap.add_argument("--cuda", default=None)
    ap.add_argument("--configs", nargs="+", default=["H1c", "H2", "H3"],
                    help="hypothesis names (H1 / H1c / H2 / H3) and/or config names")
    ap.add_argument("--datatypes", nargs="+", default=["int", "str", "list[int]", "list[str]"])
    ap.add_argument("--lang-transfers", nargs="+", default=["py-py"], choices=list(LANG_TRANSFERS))
    ap.add_argument("--n-samples", type=int, default=64, help="per (hypothesis, datatype), split over its configs")
    ap.add_argument("--patch-heads", default="goal", help="'goal' | head-set JSON [[layer, head], ...]")
    ap.add_argument("--alpha", type=float, default=2.0, help="OV boost of the patch heads at the decision token")
    ap.add_argument("--pivot", default="eq", choices=["eq", "indent"])
    ap.add_argument("--n-steer", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--oracle-model", default="claude-sonnet-4-6")
    ap.add_argument("--allow-family-overlap", action="store_true")
    ap.add_argument("--max-instances", type=int, default=None, help="smoke test: cap per (config, datatype)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="re-run instances already in results.json")
    ap.add_argument("--tag-suffix", default="")
    return ap.parse_args()


def _resolve_heads(model_short, spec):
    """'goal' = the model's saved goal head set (head_sets/<model>/goal__lam20_thr0.9_band*.json, the
    same selection FUNC.setup makes); otherwise a head-set JSON [[layer, head], ...]."""
    if spec == "goal":
        fps = sorted(Path("head_sets", model_short).glob("goal__lam20_thr0.9_band*.json"))
        assert len(fps) == 1, f"expected one goal head set for {model_short}, found {fps}"
        spec = str(fps[0])
    return [(int(l), int(h)) for l, h in json.loads(Path(spec).read_text())]


def _heads_tag(spec, n):
    return spec if spec == "goal" else f"{Path(spec).stem.split('__')[0]}_{n}"


def _rates(records, key):
    n = len(records)
    return {h: (sum(r[key][h] for r in records) / n if n else None) for h in HYPS} | {"n": n}


def summarize(records):
    clean, patched = _rates(records, "score_clean"), _rates(records, "score_patched")
    shift = {h: (patched[h] - clean[h]) if records else None for h in HYPS}
    labels = collections.Counter(r["judge_patched"]["label"] for r in records)
    return dict(n=len(records), clean=clean, patched=patched, shift=shift, patched_labels=dict(labels))


def run_config(ctx, args, cfg_name, dt, sl, dl, n, patch_heads, out_root):
    outdir = out_root / cfg_name / f"{dt.replace('[', '(').replace(']', ')')}__{sl}-{dl}"
    outdir.mkdir(parents=True, exist_ok=True)
    fp = outdir / "results.json"
    records = {} if args.force or not fp.exists() else {r["key"]: r for r in json.loads(fp.read_text())}
    insts = HE.hypothesis_setups(dt, cfg_name, n_samples=n, seed=args.seed,
                                 letter_stub="var_eq" if args.pivot == "eq" else "indent",
                                 src_lang=sl, dst_lang=dl, distinct_families=not args.allow_family_overlap)
    if args.max_instances:
        insts = insts[: args.max_instances]
    if len(insts) < n and not args.max_instances:
        logger.warning(f"[{cfg_name}/{dt}] only {len(insts)} / {n} instances satisfy the constraints")
    todo = [x for x in insts if HE.inst_key(x) not in records]
    logger.info(f"[{cfg_name}/{dt} {sl}->{dl}] {N.CONFIGS[cfg_name].notation} · {len(insts)} instances, "
                f"{len(insts) - len(todo)} resumed, {len(todo)} to run")
    for j, inst in enumerate(todo, 1):
        raw = FUNC.run_q_transfer_intervention(ctx.mt, inst, patch_heads, readout_heads=patch_heads, n_steer=args.n_steer,
                                               max_new=args.max_new, alpha=args.alpha, n_pivot=1,
                                               compute_gaze=False, compute_readout=False, ctx=ctx)
        jc = HE.judge_letters(inst, raw["gen_dst"], model_name=args.oracle_model)
        jp = HE.judge_letters(inst, raw["gen_patched"], model_name=args.oracle_model)
        sc, sp = HE.score(inst, jc["label"]), HE.score(inst, jp["label"])
        P = inst.meta["predictions"]
        hit = [h for h in ("H0", "H1", "H1c", "H2", "H3") if sp[h]]
        logger.info(f"[{cfg_name}/{dt}] ({j}/{len(todo)}) clean={jc['label']} patched={jp['label']} "
                    f"-> {','.join(hit) or ('SRC' if sp[HE.SRC_TARGET] else '-')}   (predictions {P})")
        key = HE.inst_key(inst)
        records[key] = dict(key=key, config=cfg_name, hypothesis=inst.meta["hypothesis"], datatype=dt,
                            src_lang=sl, dst_lang=dl, roles=inst.meta["roles"], predictions=P,
                            gen_src=raw["gen_src"], gen_dst=raw["gen_dst"], gen_patched=raw["gen_patched"],
                            judge_clean=jc, judge_patched=jp, score_clean=sc, score_patched=sp)
        tmp = fp.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(list(records.values()), indent=1))
        tmp.replace(fp)
    recs = [records[HE.inst_key(x)] for x in insts if HE.inst_key(x) in records]
    summ = summarize(recs)
    (outdir / "summary.json").write_text(json.dumps(summ, indent=1))
    own = N.CONFIGS[cfg_name].hypothesis
    logger.info(f"[{cfg_name}/{dt}] n={summ['n']}  {own}: clean {summ['clean'][own]:.2f} -> patched "
                f"{summ['patched'][own]:.2f} (shift {summ['shift'][own]:+.2f}) · labels {summ['patched_labels']}")
    return recs


def main():
    args = parse_args()
    logging_utils.configure(level=logging.INFO, force=True)
    logger.info(f"config: {json.dumps(vars(args))}")
    cfgs = N.resolve_configs(args.configs)
    by_h = collections.defaultdict(list)
    for c in cfgs:
        by_h[N.CONFIGS[c].hypothesis].append(c)
    n_per = {c: math.ceil(args.n_samples / len(by_h[N.CONFIGS[c].hypothesis])) for c in cfgs}
    logger.info(f"configs {cfgs} · per (config, datatype) n = {n_per}")

    ctx = N.load_ctx(args.model)                     # model-agnostic (FUNC.setup is 32B-specific)
    patch_heads = _resolve_heads(args.model.split("/")[-1], args.patch_heads)
    run_tag = f"ph-{_heads_tag(args.patch_heads, len(patch_heads))}__a{args.alpha:g}__piv-{args.pivot}{args.tag_suffix}"
    out_root = Path(env_utils.DEFAULT_RESULTS_DIR) / "diff" / args.model.split("/")[-1] / "q_transfer_hyp" / run_tag
    out_root.mkdir(parents=True, exist_ok=True)
    logger.info(f"patch heads {args.patch_heads} (n={len(patch_heads)}) -> {out_root}")

    allrec = []
    for lt in args.lang_transfers:
        sl, dl = LANG_TRANSFERS[lt]
        for c in cfgs:
            for dt in args.datatypes:
                allrec += run_config(ctx, args, c, dt, sl, dl, n_per[c], patch_heads, out_root)

    summary = {}
    for h, cs in by_h.items():
        rs = [r for r in allrec if r["config"] in cs]
        summary[h] = dict(configs=cs, pooled=summarize(rs),
                          by_datatype={dt: summarize([r for r in rs if r["datatype"] == dt]) for dt in args.datatypes},
                          by_config={c: summarize([r for r in rs if r["config"] == c]) for c in cs})
        s = summary[h]["pooled"]
        logger.info(f"== {h} ({'/'.join(cs)}) n={s['n']}: own-letter clean {s['clean'][h]:.2f} -> patched "
                    f"{s['patched'][h]:.2f} (shift {s['shift'][h]:+.2f}) · patched rates "
                    + " ".join(f"{k}={v:.2f}" for k, v in s["patched"].items() if k != "n" and v is not None))
    (out_root / "summary.json").write_text(json.dumps(dict(args=vars(args), n_patch_heads=len(patch_heads),
                                                           hypotheses=summary), indent=1))
    logger.info(f"summary -> {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
