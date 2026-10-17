#!/usr/bin/env python
"""Batch pre-commitment swap (``src.diff.precommit_swap``; notebook ``notebooks/precommitment/replace_context.ipynb``).

    p_src: a'b'c -> c (residual donor)    p_dst: a'b'd -> d (patched)

Over ``<|im_end|> .. {var}`` (the eq pivot `` =`` excluded) every decoder layer's OUTPUT residual of DST is replaced with
SRC's on the prompt pass. Patched implements c -> pre-commitment effect; d -> no-op (the pre-commitment is redundant).
Config axis ``--src-front on off``: ``on`` also transfers the frontier description (SRC's c text -> DST's d text,
``\\n``-padded) — a positive control for the intervention logic (DST's task list then carries c).

Instances = the path-patching batch sample (``PP.sample_multi_instances``: k=3, letters / var_eq, per-(datatype,
direction) seed), so instance i here is instance i in ``path_patch_eval.py``. Eval = the kv-transfer oracle
(``PP.judge``: ``judge_causal`` on the first computation, BOTH is a miss; ``eval_readouts`` source -> destination ->
patched at `` =``). Every record also carries ``PS.pivot_sanity``: pre-window positions bit-identical to clean DST, the
window equal to SRC's residuals, and the per-layer relative change of the (clean) pivot.

Results: ``results/diff/<model>/precommit_swap/<run_tag>/<dt>__<src>-<dst>/`` — ``results.json`` per instance (resumable),
``meta.json``; ``summary__<...>.json`` per invocation under the root.

    conda run -n goals python scripts/diff/precommit_swap_eval.py --cuda 7 --max-instances 2      # smoke
    conda run -n goals python scripts/diff/precommit_swap_eval.py --cuda 7                        # 4 datatypes x 32, py-py, src-front on/off
"""

import argparse
import collections
import dataclasses
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Importing the kv eval applies --cuda at MODULE LEVEL (before torch) and pulls in DATA / E / FUNC.
from scripts.diff import kv_transfer_eval as K  # noqa: E402
from scripts.diff import path_patch_eval as PPE  # noqa: E402

import src.diff.nrgs_prefix as N  # noqa: E402
import src.diff.path_patch as PP  # noqa: E402
import src.diff.precommit_swap as PS  # noqa: E402

DATA = K.DATA
env_utils, logging_utils, set_seed = K.env_utils, K.logging_utils, K.set_seed
logger = logging.getLogger("precommit_swap_eval")

MECHANISM = "precommit_swap"
ONOFF = {"on": True, "off": False}
CAUSAL_LABELS = ("src", "dst", "both", "neither")   # src = c (pre-commitment effect), dst = d (no-op)


@dataclass
class PrecommitSwapRecord(K.InstanceRecord):
    config: dict = dataclasses.field(default_factory=dict)     # src_front / layers
    geometry: dict = dataclasses.field(default_factory=dict)   # windows, pads
    sanity: dict = dataclasses.field(default_factory=dict)     # PS.pivot_sanity


_EXTRA = ("config", "geometry", "sanity")


def _run_tag(cfg, args):
    return (f"I2-src__front-{'on' if cfg['src_front'] else 'off'}__layers-{args.layers_tag}__ro-{args.readout_tag}"
            + args.tag_suffix)


def _meta(args, cfg, dt, sl, dl, recs):
    base = K._summarize([K.InstanceRecord(**{k: v for k, v in r.items() if k not in _EXTRA}) for r in recs])
    n = len(recs)
    lab = collections.Counter(r["causal"]["label"] for r in recs if r.get("causal"))
    san = [r["sanity"] for r in recs if r.get("sanity")]
    return dict(model=args.model, mechanism=MECHANISM, datatype=dt, src_lang=sl, dst_lang=dl, **cfg,
                layers=args.layers_tag, readout_heads=args.readout_tag, oracle_model=args.oracle_model, n_instances=n,
                **base,
                causal_label_rates={k: (lab[k] / sum(lab.values())) if lab else None for k in CAUSAL_LABELS},
                causal_label_counts=dict(lab),
                sanity_pre_window_untouched=(sum(s["pre_max_abs"] == 0 for s in san) / len(san)) if san else None,
                sanity_window_is_src=(sum(s["win_max_abs"] < 1e-2 for s in san) / len(san)) if san else None,
                sanity_pivot_affected=(sum(s["pivot_affected"] for s in san) / len(san)) if san else None,
                sanity_pivot_l0_untouched=(sum(s["pivot_l0_delta"] == 0 for s in san) / len(san)) if san else None,
                sanity_mean_pivot_max_rel_delta=(sum(s["pivot_max_rel_delta"] for s in san) / len(san)) if san else None,
                sanity_top1_patched_is_src=(sum(s["top1_patched_is_src"] for s in san) / len(san)) if san else None,
                sanity_top1_patched_is_dst=(sum(s["top1_patched_is_dst"] for s in san) / len(san)) if san else None)


def run_datatype(ctx, args, dt, sl, dl, configs, layers, readout_heads, out_dirs):
    """One sample (datatype x direction) over every config."""
    sample_tag = f"{K._safe(dt)}__{sl}-{dl}"
    insts, seed = PP.sample_multi_instances(dt, sl, dl, n_samples=args.n_samples, seed=args.seed)
    set_seed(seed)
    if args.max_instances:
        insts = insts[: args.max_instances]
    fps = {name: out_dirs[name] / sample_tag / "results.json" for name in out_dirs}
    for fp in fps.values():
        fp.parent.mkdir(parents=True, exist_ok=True)
    records = {name: PPE._load_done(fps[name], args) for name in out_dirs}
    logger.info(f"[{sample_tag}] {len(insts)} instances (seed={seed}) · "
                + " · ".join(f"{n}: {len(records[n])} done" for n in out_dirs))

    for j, inst in enumerate(insts, 1):
        key = PP.inst_key(inst)
        todo = [(name, cfg) for name, cfg in configs if key not in records[name] or args.force]
        if not todo:
            continue
        t0 = time.time()
        logger.info(f"[{sample_tag}] ({j}/{len(insts)}) SRC c={inst.src_frontier.id} · DST d={inst.dst_frontier.id}"
                    f" · {len(todo)} config(s)")
        cleans = {}
        for name, cfg in todo:
            g = PS.swap_geometry(inst, include_src_front=cfg["src_front"], ctx=ctx)
            cache = PS.swap_cache(g, layers, ctx=ctx)
            sanity = PS.pivot_sanity(g, cache, layers, ctx=ctx)
            ck = (tuple(PP._ids(g["src_tok"])), tuple(PP._ids(g["dst_tok"])))   # clean runs shared per token sequence
            out = PS.runs(g, cache, layers, readout_heads=readout_heads, max_new=args.max_new, ctx=ctx,
                          clean=cleans.get(ck))
            cleans[ck] = {k: out[k] for k in ("gen_src", "gen_dst", "ro_src", "ro_dst")}
            raw = dict(src_prompt=g["src_prompt"], dst_prompt=g["dst_prompt"], **out)
            causal = readout = None
            if not args.no_oracle:
                causal, readout, _ = PP.judge(inst, raw, model_name=args.oracle_model)
            _c = causal["label"] if causal else "off"
            _r = (f"{readout['source']}>{readout['dest']}>{readout['patched']}"
                  f"[{'valid' if readout['valid'] else 'inv'},{'shift' if readout['shifted'] else 'no'}]" if readout else "off")
            logger.info(f"[{sample_tag}]   {name}: causal={_c} · readout={_r} · sanity: pre={sanity['pre_max_abs']:.2g} "
                        f"win={sanity['win_max_abs']:.2g} pivotΔ L0={sanity['pivot_l0_delta']:.2g} "
                        f"first={sanity['first_layer_moved']} max={sanity['pivot_max_rel_delta']:.2f} · "
                        f"patched: {raw['gen_patched'].strip().split(chr(10))[0]!r}")
            rec = PrecommitSwapRecord(
                key=key, setup="multi_pred", datatype=dt, group="shared:" + "-".join(p.id for p in inst.shared),
                naming="letters", presentation="listed", src_lang=sl, dst_lang=dl, var_name=inst.var_name,
                expected_next=inst.src_frontier.id, expected_next_desc=inst.src_frontier.description,
                dst_next=inst.dst_frontier.id, dst_next_desc=inst.dst_frontier.description, letter_stub="var_eq",
                raw=raw, causal=causal, readout=readout, config=dict(cfg, layers=args.layers_tag),
                geometry=PS.geometry_info(g), sanity=sanity)
            records[name][key] = dataclasses.asdict(rec)
            PPE._write(fps[name], records[name])
        logger.info(f"[{sample_tag}]   ({time.time() - t0:.0f}s)")

    metas = []
    for name, cfg in configs:
        recs = list(records[name].values())
        meta = _meta(args, cfg, dt, sl, dl, recs)
        (fps[name].parent / "meta.json").write_text(json.dumps(meta, indent=2))
        logger.info(f"[{sample_tag}] SUMMARY {name}: c={meta['causal_label_rates']['src']} "
                    f"d={meta['causal_label_rates']['dst']} readout_shift={meta['readout_shift_rate']} "
                    f"valid={meta['readout_valid_rate']} pivot_affected={meta['sanity_pivot_affected']} n={len(recs)}")
        metas.append(dict(run_tag=name, **meta))
    return metas


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    p.add_argument("--cuda", default=None, help="CUDA_VISIBLE_DEVICES (applied before torch import)")
    p.add_argument("--datatypes", nargs="+", default=None, choices=sorted(DATA.QTASKS), help="default: all four")
    p.add_argument("--lang-transfers", nargs="+", default=["py-py"], choices=list(K.LANG_TRANSFERS))
    p.add_argument("--n-samples", type=int, default=32, help="instances per datatype (x language pair)")
    p.add_argument("--max-instances", type=int, default=None, help="cap instances per datatype (smoke)")
    p.add_argument("--seed", type=int, default=1123)
    p.add_argument("--src-front", nargs="+", default=["off", "on"], choices=list(ONOFF),
                   help="config AXIS: also transfer the frontier description (SRC c-span -> DST d-span, '\\n'-padded); "
                        "'on' is the positive control")
    p.add_argument("--layers", default="all", help="'all' or comma-separated layer indices whose OUTPUT residual is swapped")
    p.add_argument("--readout-heads", default="auto", help="GoalScope readout heads: 'auto' (goal-250 on Coder-32B) | 'goal' | JSON path")
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
    args.datatypes = args.datatypes or list(DATA.QTASKS)
    model_short = args.model.split("/")[-1]
    spec, readout_heads = PPE._resolve_heads(model_short, args.readout_heads)
    args.readout_tag = K._patch_heads_tag(spec, len(readout_heads))

    import torch
    device_map = {"": 0} if torch.cuda.device_count() == 1 else "auto"   # one GPU: pin (GoalScope reads o_proj weights)
    ctx = N.load_ctx(args.model, device_map=device_map)
    ctx.readout_heads, ctx.union_heads, ctx.gsp = readout_heads, None, None
    layers = list(range(ctx.n_layer)) if args.layers == "all" else sorted(int(x) for x in args.layers.split(","))
    args.layers_tag = "all" if args.layers == "all" else f"{layers[0]}-{layers[-1]}n{len(layers)}"

    configs = [(_run_tag(cfg, args), cfg) for cfg in (dict(src_front=ONOFF[v]) for v in args.src_front)]
    root = Path(env_utils.DEFAULT_RESULTS_DIR) / "diff" / model_short / MECHANISM
    out_dirs = {name: root / name for name, _ in configs}
    for name, _ in configs:
        logger.info(f"config: {name}")

    metas = []
    for lt in args.lang_transfers:
        sl, dl = K.LANG_TRANSFERS[lt]
        for dt in args.datatypes:
            logger.info(f"═══ {lt} / {dt} ═══")
            metas += run_datatype(ctx, args, dt, sl, dl, configs, layers, readout_heads, out_dirs)
    sub = ("__".join(K._safe(d) for d in args.datatypes) + "__" + "-".join(args.lang_transfers)
           + "__front-" + "-".join(args.src_front) + f"__layers-{args.layers_tag}" + args.tag_suffix)
    summ = root / f"summary__{sub}.json"
    summ.write_text(json.dumps(metas, indent=2))
    logger.info(f"all done — {len(metas)} (config, datatype) cells; summary at {summ}")


if __name__ == "__main__":
    main()
