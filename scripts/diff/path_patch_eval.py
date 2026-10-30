#!/usr/bin/env python
"""Batch pivot-onwards KV transfer (``src.diff.path_patch``) on the multi-predicate setup (``--setup multi``) or one
level down, the multi-hierarchy setup (``--setup hier``).

    multi:  p_src: a'b'c -> c (k/v donor)    p_dst: a'b'd -> d (patched)
    hier:   p_src: a'(x'z) -> z              p_dst: a'(x'y) -> y      pivot = the connective
            (instances = ``H.seeded_setups``; axes ``--connectives`` x ``--orders``)

``--goal-heads`` read SRC's window k/v from the pivot on. Config axes (each a separate run dir): ``--span``
(``task`` | ``task+progress``) x ``--i3-v-scale`` (value boost of the transplanted window; 1.0 = off). Defaults = the
paper's main configuration: task span, V x 4, the 250 goal heads. Instances (``multi_predicate_setups``, k=3,
letters / var_eq) are sampled once per (datatype, language pair) and shared by every config; the clean SRC / DST runs
are computed once per instance and token sequence. Eval = ``judge_causal`` + ``eval_readouts``. C++ prompts carry
the library note in the system prompt (``DATA.CPP_NOTE_IN_SYSTEM``).

Results: ``results/diff/<model>/path_patch/<run_tag>/<dt>__<src>-<dst>/`` (multi) or
``results/diff/<model>/path_patch_hier/<run_tag>/<dt>__<conn>__<order>__<src>-<dst>/`` (hier) — ``results.json`` per
instance (resumable) and ``meta.json``; ``summary__<...>.json`` per invocation under that root. hier records also carry the
instance's (a, x, y, z) and ``H.rule_verdict`` (execution of the first written condition; Python destinations).

    conda run -n goals python scripts/diff/path_patch_eval.py --cuda 6                      # 4 datatypes x 32
    conda run -n goals python scripts/diff/path_patch_eval.py --cuda 6 --datatypes list[str] --max-instances 2   # smoke
    conda run -n goals python scripts/diff/path_patch_eval.py --cuda 6 --setup hier --connectives and --orders match
"""

import argparse
import collections
import dataclasses
import itertools
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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


_cuda = _preparse_cuda(sys.argv[1:])   # CUDA_VISIBLE_DEVICES must be set before torch is imported
if _cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = _cuda
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import src.diff.data as DATA  # noqa: E402
import src.diff.functional as FUNC  # noqa: E402
import src.diff.hier_data as H  # noqa: E402
import src.diff.path_patch as PP  # noqa: E402
from src.utils import env_utils, logging_utils  # noqa: E402
from src.utils.experiment_utils import set_seed  # noqa: E402

logger = logging.getLogger("path_patch_eval")

MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"
MECHANISM = "path_patch"
ROOTS = {"multi": "path_patch", "hier": "path_patch_hier"}      # results/diff/<model>/<root>/
GOAL250 = "head_sets/Qwen2.5-Coder-32B-Instruct/goal_topk/goal__lam20_top250.json"
MAX_NEW = 64
ORACLE_MODEL = "claude-sonnet-4-6"
# --span -> the run-tag / summary / record.config window names of the original runs (an I1 / I2 / I3 x window x
# span grid). They are kept verbatim so re-runs land in, and can be compared with, the cached results.
TAG_SPAN = {"task": "win-span__span-list+gap", "task+progress": "win-span+prog__span-list+gap"}
SUMMARY_SPAN = {"task": "s", "task+progress": "sp"}
WINDOW = {"task": "span", "task+progress": "span+progress"}


def _load_heads(spec):
    return spec, [tuple(h) for h in json.loads(Path(spec).read_text())]


@dataclass
class PathPatchRecord(PP.InstanceRecord):
    config: dict = dataclasses.field(default_factory=dict)     # i1 / i2 / i3 / window / v_scale
    geometry: dict = dataclasses.field(default_factory=dict)   # windows, pads


@dataclass
class HierPathPatchRecord(PathPatchRecord):
    hier: dict = dataclasses.field(default_factory=dict)   # a / x / y / z / connective / order / languages
    rule: Optional[dict] = None          # H.rule_verdict on the patched generation (Python destinations)
    rule_clean: Optional[dict] = None    # ... on the clean SRC / DST generations (py-py)


_BASE_FIELDS = {f.name for f in dataclasses.fields(PP.InstanceRecord)}   # cached records may carry more keys


def _run_tag(span, v_scale, heads_tag):
    return (f"I1-off__I2-off__I3-on{f'-V{v_scale:g}' if v_scale != 1.0 else ''}__{TAG_SPAN[span]}__cppnote-sys__a1__"
            + heads_tag)


def _load_done(fp, args):
    if fp.exists() and not args.force:
        return {r["key"]: r for r in json.loads(fp.read_text())}
    return {}


def _write(fp, records):
    tmp = fp.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(list(records.values()), indent=2))
    tmp.replace(fp)


def _meta(cfg, dt, sl, dl, recs, heads_tags, conn=None, order=None):
    base = PP.summarize([PP.InstanceRecord(**{k: v for k, v in r.items() if k in _BASE_FIELDS}) for r in recs])
    rule = [r["rule"]["label"] for r in recs if r.get("rule")]
    hier = (dict(setup="hier", connective=conn, order=order,
                 rule_src_rate=(sum(l == "src" for l in rule) / len(rule)) if rule else None,
                 rule_label_counts=dict(collections.Counter(rule)) if rule else None) if conn else {})
    pads = [r["geometry"].get("pad_src", 0) + r["geometry"].get("pad_dst", 0) for r in recs]
    return dict(model=MODEL, mechanism=MECHANISM, datatype=dt, src_lang=sl, dst_lang=dl, **cfg,
                **heads_tags, oracle_model=ORACLE_MODEL, n_instances=len(recs),
                mean_pad_tokens=(sum(pads) / len(pads)) if pads else None, **base, **hier)


def run_datatype(ctx, args, dt, sl, dl, configs, heads, out_dirs, conn=None, order=None):
    """One sample (multi: datatype x direction; hier: datatype x connective x order x direction) over every config."""
    goal_heads, readout_heads = heads
    hier = args.setup == "hier"
    if hier:
        sample_tag = f"{PP._safe(dt)}__{conn}__{order}__{sl}-{dl}"
        insts, seed = PP.sample_hier_instances(dt, conn, order, sl, dl, n_samples=args.n_samples, seed=args.seed)
    else:
        sample_tag = f"{PP._safe(dt)}__{sl}-{dl}"
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
        logger.info(f"[{sample_tag}] ({j}/{len(insts)}) " + (
            f"a={inst.meta['a']} x={inst.meta['x']} · SRC z={inst.meta['z']} · DST y={inst.meta['y']}" if hier
            else f"SRC c={inst.src_frontier.id} · DST d={inst.dst_frontier.id}") + f" · {len(todo)} config(s)")
        geoms, cleans = {}, {}
        for name, cfg in todo:
            s = cfg["span"]
            if s not in geoms:
                geoms[s] = PP.build_geometry(inst, span=s, ctx=ctx)
            g = geoms[s]
            ck = (tuple(PP._ids(g["src_tok"])), tuple(PP._ids(g["dst_tok"])))   # clean runs shared per token sequence
            if ck not in cleans:
                cleans[ck] = PP.clean_runs(g, readout_heads=readout_heads, max_new=MAX_NEW,
                                           readout=not args.no_readout, ctx=ctx)
            clean = cleans[ck]
            pat = PP.run_patched(g, goal_heads=goal_heads, readout_heads=readout_heads, v_scale=cfg["v_scale"],
                                 max_new=MAX_NEW, readout=not args.no_readout, ctx=ctx)
            raw = dict(src_prompt=g["src_prompt"], dst_prompt=g["dst_prompt"], **clean, **pat)
            causal, readout = PP.judge(inst, raw, model_name=ORACLE_MODEL)
            _c = f"{'HIT' if causal['triggered'] else 'miss'}({causal['label']})"
            _r = (f"{readout['source']}>{readout['dest']}>{readout['patched']}"
                  f"[{'valid' if readout['valid'] else 'inv'},{'shift' if readout['shifted'] else 'no'}]" if readout else "off")
            logger.info(f"[{sample_tag}]   {name}: causal={_c} · readout={_r} · "
                        f"patched: {raw['gen_patched'].strip().split(chr(10))[0]!r}")
            rec_cfg = dict(i1=False, i2=False, i3=True, window=WINDOW[s], v_scale=cfg["v_scale"])
            if hier:
                rule = H.rule_verdict(inst, raw["gen_patched"]) if dl == "python" else None
                rule_clean = ({k: H.rule_verdict(inst, raw[f"gen_{k}"])["label"] for k in ("src", "dst")}
                              if (sl, dl) == ("python", "python") else None)
                rec = HierPathPatchRecord(
                    key=key, setup="multi_hier", datatype=dt, group=f"{inst.meta['x']}|{conn}", naming="letters",
                    presentation="listed", src_lang=sl, dst_lang=dl, var_name=inst.var_name,
                    expected_next=inst.meta["z"], expected_next_desc=inst.expected_next_desc,
                    dst_next=inst.meta["y"], dst_next_desc=inst.dst_next_desc, letter_stub="var_eq",
                    raw=raw, causal=causal, readout=readout, config=rec_cfg, geometry=PP.geometry_info(g),
                    hier=dict(inst.meta), rule=rule, rule_clean=rule_clean)
            else:
                rec = PathPatchRecord(
                    key=key, setup="multi_pred", datatype=dt, group="shared:" + "-".join(p.id for p in inst.shared),
                    naming="letters", presentation="listed", src_lang=sl, dst_lang=dl, var_name=inst.var_name,
                    expected_next=inst.src_frontier.id, expected_next_desc=inst.src_frontier.description,
                    dst_next=inst.dst_frontier.id, dst_next_desc=inst.dst_frontier.description, letter_stub="var_eq",
                    raw=raw, causal=causal, readout=readout, config=rec_cfg, geometry=PP.geometry_info(g))
            records[name][key] = dataclasses.asdict(rec)
            _write(fps[name], records[name])
        logger.info(f"[{sample_tag}]   ({time.time() - t0:.0f}s)")

    metas = []
    for name, cfg in configs:
        recs = list(records[name].values())
        meta = _meta(dict(i1=False, i2=False, i3=True, window=WINDOW[cfg["span"]], v_scale=cfg["v_scale"]),
                     dt, sl, dl, recs, heads_tags=args.heads_tags, conn=conn, order=order)
        (fps[name].parent / "meta.json").write_text(json.dumps(meta, indent=2))
        logger.info(f"[{sample_tag}] SUMMARY {name}: causal={meta['causal_trigger_rate']} "
                    f"readout_shift|valid={meta['readout_shift_rate_valid']} n={len(recs)}")
        metas.append(dict(run_tag=name, **meta))
    return metas


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cuda", default=None, help="CUDA_VISIBLE_DEVICES (applied before torch import)")
    p.add_argument("--setup", default="multi", choices=["multi", "hier"],
                   help="multi = multi-predicate a'b'c vs a'b'd; hier = multi-hierarchy a'(x'z) vs a'(x'y) (pivot = connective)")
    p.add_argument("--datatypes", nargs="+", default=None, choices=sorted(set(DATA.QTASKS) | set(H.HIER_DATATYPES)),
                   help="default: every datatype of the setup (multi: the q-bank; hier: list[str], list[int])")
    p.add_argument("--connectives", nargs="+", default=["and", "or"], choices=["and", "or"], help="hier only: config axis")
    p.add_argument("--orders", nargs="+", default=["match", "diff"], choices=["match", "diff"],
                   help="hier only: SRC spec order (match: x before z; diff: z before x)")
    p.add_argument("--lang-transfers", nargs="+", default=["py-py"], choices=list(PP.LANG_TRANSFERS))
    p.add_argument("--n-samples", type=int, default=32, help="instances per datatype (x language pair)")
    p.add_argument("--max-instances", type=int, default=None, help="cap instances per datatype (smoke)")
    p.add_argument("--seed", type=int, default=1123)
    p.add_argument("--span", nargs="+", default=["task"], choices=list(TAG_SPAN),
                   help="config AXIS, the transplanted SRC window: task = `(1)` .. the user turn's <|im_end|> (excluded); "
                        "task+progress = task start .. the pivot ` =` (included)")
    p.add_argument("--goal-heads", default=GOAL250, help="JSON [[layer, head], ...]: the heads that read SRC's window")
    p.add_argument("--readout-heads", default=GOAL250, help="GoalScope readout heads (JSON)")
    p.add_argument("--i3-v-scale", nargs="+", type=float, default=[4.0],
                   help="config AXIS: x the transplanted SRC window VALUES read by the goal heads (attention pattern "
                        "unchanged); a value != 1 tags its run '...I3-on-V<a>'")
    p.add_argument("--no-readout", action="store_true",
                   help="skip every GoalScope readout (clean + patched; run tag 'ro-off'): causal-shift only")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    logging_utils.configure(level=logging.INFO, force=True)
    DATA.CPP_NOTE_IN_SYSTEM = True
    bank = list(H.HIER_DATATYPES) if args.setup == "hier" else list(DATA.QTASKS)
    args.datatypes = args.datatypes or bank
    bad = [d for d in args.datatypes if d not in bank]
    assert not bad, f"datatype(s) {bad} not available for --setup {args.setup} (choose from {bank})"
    model_short = MODEL.split("/")[-1]
    resolved = {k: _load_heads(getattr(args, f"{k}_heads")) for k in ("goal", "readout")}
    heads = tuple(resolved[k][1] for k in ("goal", "readout"))
    args.heads_tags = {f"{k}_heads": PP.patch_heads_tag(resolved[k][0], len(resolved[k][1])) for k in resolved}
    gh = args.heads_tags["goal_heads"]
    # `th-` (the original runs' task-onwards heads) was always the goal head set
    heads_tag = f"th-{gh}__gh-{gh}__ro-" + ("off" if args.no_readout else args.heads_tags["readout_heads"])

    configs = []
    for s, vs in itertools.product(args.span, args.i3_v_scale):
        configs.append((_run_tag(s, vs, heads_tag), dict(span=s, v_scale=vs)))
    root = Path(env_utils.DEFAULT_RESULTS_DIR) / "diff" / model_short / ROOTS[args.setup]
    out_dirs = {name: root / name for name, _ in configs}
    for name, _ in configs:
        logger.info(f"config: {name}")

    import torch
    device_map = {"": 0} if torch.cuda.device_count() == 1 else "auto"   # one GPU: pin (GoalScope reads o_proj weights)
    ctx = FUNC.setup(MODEL, device_map=device_map)
    ctx.readout_heads, ctx.gsp = heads[1], None

    metas = []
    for lt in args.lang_transfers:
        sl, dl = PP.LANG_TRANSFERS[lt]
        for dt in args.datatypes:
            for conn, order in (itertools.product(args.connectives, args.orders) if args.setup == "hier" else [(None, None)]):
                logger.info(f"═══ {lt} / {dt}" + (f" / {conn} / {order}" if conn else "") + " ═══")
                metas += run_datatype(ctx, args, dt, sl, dl, configs, heads, out_dirs, conn=conn, order=order)
    # the config AXES are part of the name, so invocations over the same datatype / direction with different
    # configs (e.g. the V-boost grid vs the task+progress runs) never overwrite each other's summary
    axes = ("cfg-i1off_i2off_i3on_win" + "-".join(SUMMARY_SPAN[s] for s in args.span)
            + "_v" + "-".join(f"{v:g}" for v in args.i3_v_scale) + "_span-list+gap_cppnote-sys")
    hier_seg = ("__" + "-".join(args.connectives) + "__" + "-".join(args.orders)) if args.setup == "hier" else ""
    sub = "__".join(PP._safe(d) for d in args.datatypes) + hier_seg + "__" + "-".join(args.lang_transfers) + "__" + axes
    summ = root / f"summary__{sub}.json"
    summ.write_text(json.dumps(metas, indent=2))
    logger.info(f"all done — {len(metas)} (config, datatype) cells; summary at {summ}")


if __name__ == "__main__":
    main()
