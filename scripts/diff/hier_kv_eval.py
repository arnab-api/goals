#!/usr/bin/env python
"""Batch multi-hierarchy kv-transfer — the kv-(task-)transfer setup one level down (sub-SUB-goals).

Instances come from :func:`src.diff.hier_data.hier_kv_setups`: the frontier sub-goal is a list filter
whose condition is ``x <and|or> y``; ``x`` is prefilled up to the connective (the pivot), and SRC vs DST
differ only in the second condition (z vs y)::

    p_dst      : a'(x'y) -> y
    p_src_match: a'(x'z) -> z        (--orders match)
    p_src_diff : a'(zx') -> z        (--orders diff: SRC's spec lists z before x)

Mechanism and eval are the UNCHANGED kv-transfer ones (sibling of ``kv_transfer_eval.py``, same record /
summary schema): :func:`src.diff.functional.run_kv_transfer_intervention` (SRC task-list k+v -> DST) +
:func:`src.diff.eval.judge_causal` / :func:`src.diff.eval.eval_readouts`. Each record additionally gets
``rule`` = :func:`src.diff.hier_data.rule_verdict` (execute the first written condition on a pool) so its
agreement with the oracle can be checked before relying on it.

Results: ``results/diff/<model>/kv_transfer/multi_hier/<run_tag>/<dt>__<conn>__<order>__python-python/``
(``results.json`` written per instance — resumable — and ``meta.json``); ``summary.json`` per run.

    conda run -n goals python scripts/diff/hier_kv_eval.py --cuda 0 \\
        --patch-heads head_sets/Qwen2.5-Coder-32B-Instruct/goal_topk/goal__lam20_top250.json
"""

import argparse
import collections
import dataclasses
import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Importing the kv eval applies --cuda at MODULE LEVEL (before torch) and pulls in DATA / E / FUNC.
from scripts.diff import kv_transfer_eval as K  # noqa: E402

import src.diff.hier_data as H  # noqa: E402
import src.diff.nrgs_prefix as N  # noqa: E402

E, FUNC = K.E, K.FUNC
env_utils, logging_utils, set_seed = K.env_utils, K.logging_utils, K.set_seed
logger = logging.getLogger("hier_kv_eval")

SETUP = "multi_hier"  # results: diff/<model>/kv_transfer/multi_hier/<run_tag>/<config_tag>/
GOAL250 = "head_sets/Qwen2.5-Coder-32B-Instruct/goal_topk/goal__lam20_top250.json"


def _resolve_heads(model_short, spec):
    """'auto' = goal-250 for Coder-32B (the paper's kv-transfer config), else the model's saved goal set;
    'goal' = head_sets/<model>/goal__lam20_thr0.9_band*.json; anything else = a JSON [[layer, head], ...]."""
    if spec == "auto":
        spec = GOAL250 if model_short == "Qwen2.5-Coder-32B-Instruct" else "goal"
    if spec == "goal":
        fps = sorted(Path("head_sets", model_short).glob("goal__lam20_thr0.9_band*.json"))
        assert fps, f"no goal head set for {model_short} under head_sets/"
        spec = str(fps[0])
    return spec, [tuple(h) for h in json.loads(Path(spec).read_text())]


@dataclass
class HierRecord(K.InstanceRecord):
    hier: dict = dataclasses.field(default_factory=dict)  # a / x / y / z / connective / order
    rule: Optional[dict] = None  # execution verdict on the patched generation
    rule_clean: Optional[dict] = None  # ... on the clean SRC / DST generations (sanity of the setup)
    clean_oracle: Optional[dict] = None  # c++ DST (no rule check): judge_causal on clean SRC / DST gens


def _rule_summary(recs: list[dict]) -> dict:
    """Rule-based rates + agreement with the oracle on the patched generation."""
    rule = [r["rule"]["label"] for r in recs if r.get("rule")]
    out = dict(
        rule_src_rate=(sum(l == "src" for l in rule) / len(rule)) if rule else None,
        rule_label_counts=dict(collections.Counter(rule)),
        clean_src_ok=None,
        clean_dst_ok=None,
    )
    cl = [r["rule_clean"] for r in recs if r.get("rule_clean")] or [r["clean_oracle"] for r in recs if r.get("clean_oracle")]
    if cl:  # rule verdicts (python DST) or oracle verdicts (c++ DST) on the clean generations
        out["clean_src_ok"] = sum(c["src"] == "src" for c in cl) / len(cl)
        out["clean_dst_ok"] = sum(c["dst"] == "dst" for c in cl) / len(cl)
    co = [r for r in recs if r.get("clean_oracle") and r.get("causal")]
    if co:
        ok = [r for r in co if r["clean_oracle"]["src"] == "src" and r["clean_oracle"]["dst"] == "dst"]
        out["causal_trigger_rate_clean_valid"] = (sum(r["causal"]["triggered"] for r in ok) / len(ok)) if ok else None
        out["n_clean_valid"] = len(ok)
    pairs = [(r["causal"]["label"], r["rule"]["label"]) for r in recs if r.get("causal") and r.get("rule")]
    if pairs:
        # oracle {src, both} = triggered; rule "src" = the written condition IS z (execution-equal)
        out["oracle_rule_agree_trigger"] = sum((o in ("src", "both")) == (ru == "src") for o, ru in pairs) / len(pairs)
        out["oracle_rule_confusion"] = {f"{o}|{ru}": n for (o, ru), n in collections.Counter(pairs).items()}
    return out


def run_config(ctx, args, dt, conn, order, sl, dl, patch_heads, readout_heads, out_root):
    mt = ctx.mt
    tag = f"{K._safe(dt)}__{conn}__{order}__{sl}-{dl}"
    outdir = out_root / tag
    outdir.mkdir(parents=True, exist_ok=True)
    results_fp = outdir / "results.json"
    done = {}
    if results_fp.exists() and args.resume and not args.force:
        done = {r["key"]: r for r in json.loads(results_fp.read_text())}
        logger.info(f"[{tag}] resume: {len(done)} instances already done")

    # the (a, x, y, z) sample depends on datatype + connective only, so `match` and `diff` share it
    # py-py seed key unchanged; every direction involving c++ shares ONE sample (lambda conditions excluded;
    # key kept = the original py-cpp key) -> py-cpp / cpp-py / cpp-cpp are paired, and match / diff too
    _sk = f"{dt}|{conn}" if (sl, dl) == ("python", "python") else f"{dt}|{conn}|python-cpp"
    cfg_seed = (args.seed + int(hashlib.md5(_sk.encode()).hexdigest()[:8], 16)) % (2**32)
    set_seed(cfg_seed)
    setups = H.hier_kv_setups(dt, n_samples=args.n_samples, seed=cfg_seed, connective=conn, order=order,
                              src_lang=sl, dst_lang=dl)
    if args.max_instances:
        setups = setups[: args.max_instances]
    records = dict(done)
    todo = [i for i in setups if H.inst_key(i) not in records or args.force]
    logger.info(f"[{tag}] sampled {len(setups)} (seed={cfg_seed}) — {len(setups) - len(todo)} resumed, {len(todo)} to run")

    for j, inst in enumerate(todo, 1):
        key = H.inst_key(inst)
        logger.info(f"[{tag}] ({j}/{len(todo)}) a={inst.meta['a']} x={inst.meta['x']} | "
                    f"SRC z={inst.meta['z']!r} vs DST y={inst.meta['y']!r}")
        raw = FUNC.run_kv_transfer_intervention(
            mt, inst, patch_heads, readout_heads=readout_heads, max_new=args.max_new,
            alpha=args.alpha, align=args.align, ctx=ctx)
        rule = rule_clean = None
        if dl == "python":  # execution check is Python-only (patched / clean DST code is python)
            rule = H.rule_verdict(inst, raw["gen_patched"])
        if (sl, dl) == ("python", "python"):  # both clean runs are python -> rule-check them
            rule_clean = dict(src=H.rule_verdict(inst, raw["gen_src"])["label"],
                              dst=H.rule_verdict(inst, raw["gen_dst"])["label"])
        causal = readout = clean_oracle = None
        if not args.no_oracle:
            causal = E.judge_causal(inst, raw["gen_patched"], model_name=args.oracle_model)
            if (sl, dl) != ("python", "python"):  # a c++ side has no rule check -> the oracle validates the clean runs
                clean_oracle = {k: E.judge_causal(inst, raw[f"gen_{k}"], model_name=args.oracle_model)["label"]
                                for k in ("src", "dst")}
            readout = E.eval_readouts(inst, raw["ro_src"], raw["ro_dst"], raw["ro_patched"],
                                      model_name=args.oracle_model)
        _c = f"{'HIT' if causal['triggered'] else 'miss'}({causal['label']})" if causal else "off"
        _r = (f"{readout['source']}>{readout['dest']}>{readout['patched']}"
              f"[{'valid' if readout['valid'] else 'inv'},{'shift' if readout['shifted'] else 'no'}]"
              if readout else "off")
        _ru = (rule["label"] if rule else "n/a") + (
            f" (clean rule {rule_clean['src']}/{rule_clean['dst']})" if rule_clean else
            f" (clean oracle {clean_oracle['src']}/{clean_oracle['dst']})" if clean_oracle else "")
        logger.info(f"[{tag}]   => causal={_c} · rule={_ru} "
                    f"· readout={_r} · patched: {raw['gen_patched'].split(chr(10))[0]!r}")
        rec = HierRecord(
            key=key, setup=SETUP, datatype=dt, group=f"{inst.meta['x']}|{conn}", naming="letters",
            letter_stub="var_eq", presentation="listed", src_lang=sl, dst_lang=dl,
            var_name=inst.var_name, expected_next=inst.meta["z"], expected_next_desc=inst.expected_next_desc,
            dst_next=inst.meta["y"], dst_next_desc=inst.dst_next_desc, raw=raw, causal=causal,
            readout=readout, hier=dict(inst.meta), rule=rule, rule_clean=rule_clean, clean_oracle=clean_oracle)
        records[key] = dataclasses.asdict(rec)
        tmp = outdir / "results.json.tmp"
        tmp.write_text(json.dumps(list(records.values()), indent=2))
        tmp.replace(results_fp)

    recs = list(records.values())
    base = K._summarize([K.InstanceRecord(**{k: v for k, v in r.items() if k not in ("hier", "rule", "rule_clean", "clean_oracle")})
                         for r in recs])
    meta = dict(model=args.model, setup=SETUP, datatype=dt, connective=conn, order=order,
                src_lang=sl, dst_lang=dl,
                patch_heads=K._patch_heads_tag(args.patch_heads, len(patch_heads)),
                readout_heads=K._patch_heads_tag(args.readout_heads, len(readout_heads)),
                align=args.align, alpha=args.alpha, oracle_model=args.oracle_model, n_instances=len(recs),
                **base, **_rule_summary(recs))
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2))
    logger.info(f"[{tag}] SUMMARY: causal={meta['causal_trigger_rate']} rule_src={meta['rule_src_rate']} "
                f"agree={meta.get('oracle_rule_agree_trigger')} readout_shift|valid={meta['readout_shift_rate_valid']} "
                f"clean_ok={meta['clean_src_ok']}/{meta['clean_dst_ok']} n={len(recs)}")
    return meta


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    p.add_argument("--cuda", default=None, help="CUDA_VISIBLE_DEVICES (applied before torch import)")
    p.add_argument("--datatypes", nargs="+", default=list(H.HIER_DATATYPES), choices=list(H.HIER_DATATYPES))
    p.add_argument("--connectives", nargs="+", default=["and", "or"], choices=["and", "or"])
    p.add_argument("--orders", nargs="+", default=["match", "diff"], choices=["match", "diff"],
                   help="match: SRC spec 'x <conn> z' ; diff: SRC spec 'z <conn> x' (code still writes x first)")
    p.add_argument("--lang-transfers", nargs="+", default=["py-py"], choices=["py-py", "py-cpp", "cpp-py", "cpp-cpp"],
                   help="a C++ side renders the filter as a range-for loop (guards prefilled); see src.diff.hier_data")
    p.add_argument("--patch-heads", default="auto", help="'auto' (goal-250 on Coder-32B, else goal) | 'goal' | JSON [[layer, head], ...]")
    p.add_argument("--readout-heads", default="auto", help="GoalScope readout heads (same choices)")
    p.add_argument("--align", default="pad", choices=["right", "pad"])
    p.add_argument("--alpha", type=float, default=2.0, help="OV boost of the patch heads at the pivot (1.0 = off)")
    p.add_argument("--n-samples", type=int, default=32, help="instances per (datatype, connective, order)")
    p.add_argument("--seed", type=int, default=1123)
    p.add_argument("--max-instances", type=int, default=None, help="further cap instances/config (smoke)")
    p.add_argument("--max-new", type=int, default=64)
    p.add_argument("--oracle-model", default="claude-sonnet-4-6")
    p.add_argument("--no-oracle", action="store_true", help="skip causal + readout oracle (rule verdict only)")
    p.add_argument("--tag-suffix", default="", help="appended to the run tag")
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-resume", dest="resume", action="store_false")
    return p.parse_args()


def main():
    args = parse_args()
    logging_utils.configure(level=logging.INFO, force=True)
    model_short = args.model.split("/")[-1]
    args.patch_heads, patch_heads = _resolve_heads(model_short, args.patch_heads)
    args.readout_heads, readout_heads = _resolve_heads(model_short, args.readout_heads)
    # one visible GPU -> pin the whole model to it: "auto" can offload the last layers to CPU/meta, and the
    # GoalScope readout reads o_proj.weight OUTSIDE a forward pass ("Cannot copy out of meta tensor")
    import torch
    device_map = {"": 0} if torch.cuda.device_count() == 1 else "auto"
    ctx = N.load_ctx(args.model, device_map=device_map)  # model-agnostic (FUNC.setup is 32B-specific)
    ctx.readout_heads, ctx.union_heads, ctx.gsp = readout_heads, None, None
    ph = K._patch_heads_tag(args.patch_heads, len(patch_heads))
    ro = K._patch_heads_tag(args.readout_heads, len(readout_heads))
    run_tag = f"ph-{ph}__ro-{ro}__a{args.alpha:g}__{args.align}{args.tag_suffix}"
    out_root = Path(env_utils.DEFAULT_RESULTS_DIR) / "diff" / args.model.split("/")[-1] / K.MECHANISM / SETUP / run_tag
    out_root.mkdir(parents=True, exist_ok=True)
    logger.info(f"run_tag={run_tag} · patch={ph}(n={len(patch_heads)}) readout={ro}(n={len(readout_heads)})")

    metas = []
    for lt in args.lang_transfers:
        sl, dl = K.LANG_TRANSFERS[lt]
        for dt in args.datatypes:
            for conn in args.connectives:
                for order in args.orders:
                    logger.info(f"═══ {lt} / {dt} / {conn} / {order} ═══")
                    metas.append(run_config(ctx, args, dt, conn, order, sl, dl, patch_heads, readout_heads, out_root))
    # per-invocation summary name, so parallel jobs over different datatypes / connectives don't clobber it
    sub = ("__".join(K._safe(d) for d in args.datatypes) + "__" + "-".join(args.connectives) + "__"
           + "-".join(args.orders) + ("" if args.lang_transfers == ["py-py"] else "__" + "-".join(args.lang_transfers)))
    summ = out_root / f"summary__{sub}.json"
    summ.write_text(json.dumps(metas, indent=2))
    logger.info(f"all done — {len(metas)} configs; summary at {summ}")


if __name__ == "__main__":
    main()
