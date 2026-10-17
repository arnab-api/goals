#!/usr/bin/env python
"""Batch path patching on the multi-predicate kv-transfer setup (``src.diff.path_patch``; notebook
``notebooks/diff/05_kv_transfer.ipynb`` §Path patching / §Path-patching variants).

    p_src: a'b'c -> c (k/v donor)    p_dst: a'b'd -> d (patched)    p_res: a'b'e -> e (I2 residual donor)

I1 = ``--task-heads`` read SRC's window k/v (rule ``--i1-query``); I2 = restore ``<|im_end|> .. {var}`` (pivot excluded)
from p_res (``--i2-donor res``) or clean DST, every layer; I3 = ``--goal-heads`` read SRC's window k/v from ` =` on.
Config axes (each a separate run dir): ``--i1`` × ``--i2`` × ``--i3`` × ``--window`` (``span`` | ``span+progress``)
× ``--i3-v-scale`` (I3 value boost; 1.0 = off).
Defaults = the 4 runs I1=off, I3=on, I2 ∈ {on, off}, window ∈ {span, span+progress}. Instances (``multi_predicate_setups``,
k=3, letters / var_eq) are sampled once per (datatype, language pair) and shared by every config; the clean SRC / DST
runs are computed once per instance and token sequence. Eval = the kv-transfer oracle (``judge_causal`` +
``eval_readouts``; same record / summary schema as ``kv_transfer_eval.py``). With I2 on and the p_res donor, the
patched readout is also judged 3-way over {d, c, e} (``readout_res``; ``res_readout_rate``), against the clean p_res
readout as the validity baseline (``res_valid``).

Results: ``results/diff/<model>/path_patch/<run_tag>/<dt>__<src>-<dst>/`` (``results.json`` per instance — resumable —
and ``meta.json``); ``summary__<...>.json`` per invocation under ``results/diff/<model>/path_patch/``.

    conda run -n goals python scripts/diff/path_patch_eval.py --cuda 6                      # 4 configs x 4 datatypes x 32
    conda run -n goals python scripts/diff/path_patch_eval.py --cuda 6 --datatypes list[str] --max-instances 2   # smoke
"""

import argparse
import collections
import dataclasses
import hashlib
import itertools
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Importing the kv eval applies --cuda at MODULE LEVEL (before torch) and pulls in DATA / E / FUNC.
from scripts.diff import kv_transfer_eval as K  # noqa: E402

import src.diff.nrgs_prefix as N  # noqa: E402
import src.diff.path_patch as PP  # noqa: E402

DATA, E, FUNC = K.DATA, K.E, K.FUNC
env_utils, logging_utils, set_seed = K.env_utils, K.logging_utils, K.set_seed
logger = logging.getLogger("path_patch_eval")

MECHANISM = "path_patch"
GOAL250 = "head_sets/Qwen2.5-Coder-32B-Instruct/goal_topk/goal__lam20_top250.json"
ONOFF = {"on": True, "off": False}


def _resolve_heads(model_short, spec):
    """'auto' = goal-250 for Coder-32B, else the model's saved goal set; 'goal' = head_sets/<model>/goal__lam20_thr0.9_band*.json;
    anything else = a JSON [[layer, head], ...]."""
    if spec == "auto":
        spec = GOAL250 if model_short == "Qwen2.5-Coder-32B-Instruct" else "goal"
    if spec == "goal":
        fps = sorted(Path("head_sets", model_short).glob("goal__lam20_thr0.9_band*.json"))
        assert fps, f"no goal head set for {model_short} under head_sets/"
        spec = str(fps[0])
    return spec, [tuple(h) for h in json.loads(Path(spec).read_text())]


def _query_arg(v):
    return v if v in ("all", "last") else int(v)


@dataclass
class PathPatchRecord(K.InstanceRecord):
    config: dict = dataclasses.field(default_factory=dict)     # i1 / i2 / i3 / window / donor / ...
    geometry: dict = dataclasses.field(default_factory=dict)   # windows, pads, p_res frontier
    readout_res: Optional[dict] = None   # I2 with the p_res donor: 3-way {d, c, e} verdicts on the patched + clean p_res readouts


def _run_tag(cfg, args, heads_tag):
    seg = [f"I1-{'on' if cfg['i1'] else 'off'}" + (f"-q{args.i1_query}" if cfg["i1"] and args.i1_query != "all" else ""),
           f"I2-{'on-' + args.i2_donor if cfg['i2'] else 'off'}" + (f"-{args.i2_from}" if cfg["i2"] and args.i2_from != "im_end" else ""),
           f"I3-{'on' if cfg['i3'] else 'off'}" + ("-blind" if cfg["i3"] and args.blind_self else "")
           + (f"-q{args.i3_query}" if cfg["i3"] and args.i3_query != "last" else "")
           + (f"-V{cfg['v_scale']:g}" if cfg["i3"] and cfg["v_scale"] != 1.0 else ""),
           "win-" + ("span" if cfg["window"] == "span" else "span+prog" + ("" if args.include_pivot else "-nopiv")),
           f"span-{args.span}", f"a{args.alpha:g}", heads_tag]
    return "__".join(seg) + args.tag_suffix


def _load_done(fp, args):
    if fp.exists() and args.resume and not args.force:
        return {r["key"]: r for r in json.loads(fp.read_text())}
    return {}


def _write(fp, records):
    tmp = fp.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(list(records.values()), indent=2))
    tmp.replace(fp)


def _meta(args, cfg, dt, sl, dl, recs, heads_tags):
    base = K._summarize([K.InstanceRecord(**{k: v for k, v in r.items() if k not in ("config", "geometry", "readout_res")})
                         for r in recs])
    rr = [r["readout_res"] for r in recs if r.get("readout_res")]
    rr_valid = [x for x in rr if x["res_valid"]]
    res_rates = dict(
        res_readout_rate=(sum(x["shifted_to_res"] for x in rr) / len(rr)) if rr else None,
        res_readout_rate_valid=(sum(x["shifted_to_res"] for x in rr_valid) / len(rr_valid)) if rr_valid else None,
        res_valid_rate=(len(rr_valid) / len(rr)) if rr else None,
        res_patched_label_counts=dict(collections.Counter(x["patched"] for x in rr)) if rr else None)
    pads = [r["geometry"].get("pad_src", 0) + r["geometry"].get("pad_dst", 0) for r in recs]
    return dict(model=args.model, mechanism=MECHANISM, datatype=dt, src_lang=sl, dst_lang=dl, **cfg,
                span=args.span, include_pivot=args.include_pivot, i1_query=args.i1_query, i3_query=args.i3_query,
                blind_self=args.blind_self, i2_donor=args.i2_donor, i2_from=args.i2_from, alpha=args.alpha,
                **heads_tags, oracle_model=args.oracle_model, n_instances=len(recs),
                mean_pad_tokens=(sum(pads) / len(pads)) if pads else None, **base, **res_rates)


def run_datatype(ctx, args, dt, sl, dl, configs, heads, out_dirs):
    task_heads, goal_heads, readout_heads = heads
    sample_tag = f"{K._safe(dt)}__{sl}-{dl}"
    insts, seed = PP.sample_multi_instances(dt, sl, dl, n_samples=args.n_samples, seed=args.seed)
    set_seed(seed)
    if args.max_instances:
        insts = insts[: args.max_instances]
    fps = {name: out_dirs[name] / sample_tag / "results.json" for name in out_dirs}
    for fp in fps.values():
        fp.parent.mkdir(parents=True, exist_ok=True)
    records = {name: _load_done(fps[name], args) for name in out_dirs}
    logger.info(f"[{sample_tag}] {len(insts)} instances (seed={seed}) · "
                + " · ".join(f"{n}: {len(records[n])} done" for n in out_dirs))

    for j, inst in enumerate(insts, 1):
        key = PP.inst_key(inst)
        todo = [(name, cfg) for name, cfg in configs if key not in records[name] or args.force]
        if not todo:
            continue
        t0 = time.time()
        logger.info(f"[{sample_tag}] ({j}/{len(insts)}) SRC c={inst.src_frontier.id} · DST d={inst.dst_frontier.id} · "
                    f"{len(todo)} config(s)")
        geoms, cleans, ro_res = {}, {}, None
        for name, cfg in todo:
            w = cfg["window"]
            if w not in geoms:
                geoms[w] = PP.build_geometry(inst, window=w, span=args.span, include_pivot=args.include_pivot,
                                             i2_from=args.i2_from, res_frontier=args.res_frontier, res_seed=args.seed,
                                             ctx=ctx)
            g = geoms[w]
            ck = (tuple(PP._ids(g["src_tok"])), tuple(PP._ids(g["dst_tok"])))   # clean runs shared per token sequence
            if ck not in cleans:
                cleans[ck] = PP.clean_runs(g, readout_heads=readout_heads, max_new=args.max_new, ctx=ctx)
            clean = cleans[ck]
            pat = PP.run_patched(g, task_heads=task_heads, goal_heads=goal_heads, readout_heads=readout_heads,
                                 i1=cfg["i1"], i2=cfg["i2"], i3=cfg["i3"], i1_query=args.i1_query, i3_query=args.i3_query,
                                 blind_self=args.blind_self, v_scale=cfg["v_scale"], alpha=args.alpha,
                                 donor=args.i2_donor, max_new=args.max_new,
                                 ctx=ctx)
            use_res = cfg["i2"] and args.i2_donor == "res"
            if use_res and ro_res is None:   # clean p_res readout at ` =` (p_res is the same for every config)
                ro_res = PP.res_readout(g, readout_heads=readout_heads, ctx=ctx)
            raw = dict(src_prompt=g["src_prompt"], dst_prompt=g["dst_prompt"], res_prompt=g["res_prompt"], **clean, **pat,
                       **({"ro_res": ro_res} if use_res else {}))
            causal = readout = readout_res = None
            if not args.no_oracle:
                causal, readout, readout_res = PP.judge(inst, raw, e_desc=g["e_desc"], model_name=args.oracle_model)
            _c = f"{'HIT' if causal['triggered'] else 'miss'}({causal['label']})" if causal else "off"
            _r = (f"{readout['source']}>{readout['dest']}>{readout['patched']}"
                  f"[{'valid' if readout['valid'] else 'inv'},{'shift' if readout['shifted'] else 'no'}]" if readout else "off")
            _rr = (f" · readout3={readout_res['patched']}(p_res clean={readout_res['res_clean']})" if readout_res else "")
            logger.info(f"[{sample_tag}]   {name}: causal={_c} · readout={_r}{_rr} · "
                        f"patched: {raw['gen_patched'].strip().split(chr(10))[0]!r}")
            rec = PathPatchRecord(
                key=key, setup="multi_pred", datatype=dt, group="shared:" + "-".join(p.id for p in inst.shared),
                naming="letters", presentation="listed", src_lang=sl, dst_lang=dl, var_name=inst.var_name,
                expected_next=inst.src_frontier.id, expected_next_desc=inst.src_frontier.description,
                dst_next=inst.dst_frontier.id, dst_next_desc=inst.dst_frontier.description, letter_stub="var_eq",
                raw=raw, causal=causal, readout=readout, config=dict(cfg), geometry=PP.geometry_info(g),
                readout_res=readout_res)
            records[name][key] = dataclasses.asdict(rec)
            _write(fps[name], records[name])
        logger.info(f"[{sample_tag}]   ({time.time() - t0:.0f}s)")

    metas = []
    for name, cfg in configs:
        recs = list(records[name].values())
        meta = _meta(args, cfg, dt, sl, dl, recs, heads_tags=args.heads_tags)
        (fps[name].parent / "meta.json").write_text(json.dumps(meta, indent=2))
        logger.info(f"[{sample_tag}] SUMMARY {name}: causal={meta['causal_trigger_rate']} "
                    f"readout_shift|valid={meta['readout_shift_rate_valid']} "
                    f"readout->e={meta['res_readout_rate']} (|p_res valid={meta['res_readout_rate_valid']}) n={len(recs)}")
        metas.append(dict(run_tag=name, **meta))
    return metas


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    p.add_argument("--cuda", default=None, help="CUDA_VISIBLE_DEVICES (applied before torch import)")
    p.add_argument("--datatypes", nargs="+", default=list(DATA.QTASKS), choices=list(DATA.QTASKS))
    p.add_argument("--lang-transfers", nargs="+", default=["py-py"], choices=list(K.LANG_TRANSFERS))
    p.add_argument("--n-samples", type=int, default=32, help="instances per datatype (x language pair)")
    p.add_argument("--max-instances", type=int, default=None, help="cap instances per datatype (smoke)")
    p.add_argument("--seed", type=int, default=1123)
    # interventions — each a config AXIS (space-separated on/off values)
    p.add_argument("--i1", nargs="+", default=["off"], choices=list(ONOFF))
    p.add_argument("--i2", nargs="+", default=["on", "off"], choices=list(ONOFF))
    p.add_argument("--i3", nargs="+", default=["on"], choices=list(ONOFF))
    p.add_argument("--window", nargs="+", default=["span", "span+progress"], choices=["span", "span+progress"],
                   help="transplanted SRC window: the task span, or task span + progress up to ` =` (A1 padding)")
    # fixed knobs
    p.add_argument("--span", default="list", choices=["list", "frontier"], help="task span = whole task list | frontier item")
    p.add_argument("--exclude-pivot", dest="include_pivot", action="store_false",
                   help="span+progress window stops before ` =` (default: includes ` =`)")
    p.add_argument("--task-heads", default="auto", help="I1 heads: 'auto' (goal-250 on Coder-32B) | 'goal' | JSON path")
    p.add_argument("--goal-heads", default="auto", help="I3 heads (same choices)")
    p.add_argument("--readout-heads", default="auto", help="GoalScope readout heads (same choices)")
    p.add_argument("--i1-query", default="all", type=_query_arg, help="I1 query rule: all | last | int n")
    p.add_argument("--i3-query", default="last", type=_query_arg, help="I3 query rule: last (default) | all | int n")
    p.add_argument("--blind-self", action="store_true", help="I3 heads cannot attend to the current token's own key")
    p.add_argument("--i3-v-scale", nargs="+", type=float, default=[1.0],
                   help="config AXIS: x the transplanted SRC window VALUES read by the I3 heads (attention pattern unchanged); "
                        "a value != 1 tags its run '...I3-on-V<a>'")
    p.add_argument("--i2-donor", default="res", choices=["res", "dst"], help="I2 residual donor: p_res (a'b'e) | clean DST")
    p.add_argument("--i2-from", default="im_end", choices=["im_end", "span_end"], help="I2 window start")
    p.add_argument("--res-frontier", default=None, help="fix p_res's frontier id (default: seeded per instance)")
    p.add_argument("--alpha", type=float, default=1.0, help="OV boost of the I3 heads at -1 (1.0 = off)")
    p.add_argument("--max-new", type=int, default=64)
    p.add_argument("--oracle-model", default="claude-sonnet-4-6")
    p.add_argument("--no-oracle", action="store_true")
    p.add_argument("--tag-suffix", default="")
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-resume", dest="resume", action="store_false")
    return p.parse_args()


def main():
    args = parse_args()
    logging_utils.configure(level=logging.INFO, force=True)
    model_short = args.model.split("/")[-1]
    resolved = {k: _resolve_heads(model_short, getattr(args, f"{k}_heads")) for k in ("task", "goal", "readout")}
    heads = tuple(resolved[k][1] for k in ("task", "goal", "readout"))
    args.heads_tags = {f"{k}_heads": K._patch_heads_tag(resolved[k][0], len(resolved[k][1])) for k in resolved}
    tags = args.heads_tags
    heads_tag = (f"th-{tags['task_heads']}__gh-{tags['goal_heads']}__ro-{tags['readout_heads']}")

    configs = []
    for i1, i2, i3, w, vs in itertools.product(args.i1, args.i2, args.i3, args.window, args.i3_v_scale):
        if not ONOFF[i3] and vs != args.i3_v_scale[0]:
            continue   # v_scale only acts on I3: with I3 off one config is enough
        cfg = dict(i1=ONOFF[i1], i2=ONOFF[i2], i3=ONOFF[i3], window=w, v_scale=vs)
        configs.append((_run_tag(cfg, args, heads_tag), cfg))
    root = Path(env_utils.DEFAULT_RESULTS_DIR) / "diff" / model_short / MECHANISM
    out_dirs = {name: root / name for name, _ in configs}
    for name, _ in configs:
        logger.info(f"config: {name}")

    import torch
    device_map = {"": 0} if torch.cuda.device_count() == 1 else "auto"   # one GPU: pin (GoalScope reads o_proj weights)
    ctx = N.load_ctx(args.model, device_map=device_map)
    ctx.readout_heads, ctx.union_heads, ctx.gsp = heads[2], None, None

    metas = []
    for lt in args.lang_transfers:
        sl, dl = K.LANG_TRANSFERS[lt]
        for dt in args.datatypes:
            logger.info(f"═══ {lt} / {dt} ═══")
            metas += run_datatype(ctx, args, dt, sl, dl, configs, heads, out_dirs)
    # the config AXES are part of the name, so invocations over the same datatype / direction with different
    # configs (e.g. the V-boost grid vs the V x1 runs) never overwrite each other's summary
    axes = ("cfg-i1" + "-".join(args.i1) + "_i2" + "-".join(args.i2) + "_i3" + "-".join(args.i3)
            + "_win" + "-".join("sp" if w == "span+progress" else "s" for w in args.window)
            + "_v" + "-".join(f"{v:g}" for v in args.i3_v_scale))
    sub = "__".join(K._safe(d) for d in args.datatypes) + "__" + "-".join(args.lang_transfers) + "__" + axes + args.tag_suffix
    summ = root / f"summary__{sub}.json"
    summ.write_text(json.dumps(metas, indent=2))
    logger.info(f"all done — {len(metas)} (config, datatype) cells; summary at {summ}")


if __name__ == "__main__":
    main()
