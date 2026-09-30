#!/usr/bin/env python
"""Mine the NRGS common-prefix contrastive dataset for one or more contrast configs
(``src/diff/nrgs_prefix.py::CONFIGS``; notation ``p: a'bc' -> b`` = spec order, x' prefilled, arrow = frontier):

    order_diff_1     src a'cb' -> c    | dst a'bd -> b   | target d
    order_diff_2     src a'bc'd' -> b  | dst a'dec -> d  | target e
    order_diff_3     src a'b'cd' -> c  | dst a'bde -> b  | target e
    order_matched_1  src a'b'c -> c    | dst a'bd -> b   | target d
    order_matched_2  src a'b'c'd -> d  | dst a'bce -> b  | target e
    diff_vs_prog_1   src a'b'cd -> c   | dst a'bdc -> b  | target d

For every goal-head localization pivot (DST native, target, shared prefix) and config, the other letters
are drawn from the bank; the divergence gate runs in the NRGS context (DST top-1 after the prefix = the
native token, DST* top-1 = the target token); the passing instances are saved with the model's own
``b_id`` (native) / ``d_id`` (target) loss targets and a (native, target)-pair-disjoint train/val split,
shared across configs. Resumable: gate rows are cached by key (the key includes the config).

Everything goes to ONE file; each record carries ``config`` / ``group``, so a shard is a filter:
``N.load_dataset(path, configs=["order_diff_2"])`` or ``train_nrgs_mask.py --configs order_diff_2``.

Outputs (``results/nrgs_prefix/<model>/<out_tag>__<sl>-<dl>/``; out_tag = ``mixed`` for >1 config, else the
config name):
    gate.json       every gated instance (passed or not), resumable cache
    dataset.json    {args, configs: {name: notation + counts}, instances: [record + b_id, d_id, split]}
Usage:
    conda run -n goals python scripts/diff/mine_nrgs_prefix.py --cuda 0 --configs all --n-per-pivot 4
    conda run -n goals python scripts/diff/mine_nrgs_prefix.py --cuda 0 --configs order_diff_2 order_diff_3
"""
import argparse
import json
import logging
import os
import sys
import time
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

import src.diff.nrgs_prefix as N  # noqa: E402
from src.utils import env_utils, logging_utils  # noqa: E402

logger = logging.getLogger(__name__)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    ap.add_argument("--cuda", default=None, help="CUDA_VISIBLE_DEVICES (applied before torch import)")
    ap.add_argument("--configs", nargs="+", default=["all"],
                    help=f"contrast configs to mine (default all): {sorted(N.CONFIGS)}; aliases {sorted(N.ALIASES)}")
    ap.add_argument("--variant", default=None, choices=N.VARIANTS, help="(deprecated) one config; same as --configs X")
    ap.add_argument("--out-tag", default=None, help="output dir tag (default: 'mixed' for >1 config, else its name)")
    ap.add_argument("--datatypes", nargs="+", default=["str", "int", "list[int]"])
    ap.add_argument("--n-per-pivot", type=int, default=4, help="assignments of the config's other letters per (pivot, config)")
    ap.add_argument("--all-prefixes", action="store_true",
                    help="keep head_loc wrapper augmentations (bool(..., True and ..., ...) as separate pivots")
    ap.add_argument("--allow-family-overlap", action="store_true",
                    help="let A / C be near-duplicates of the other roles (e.g. is_even vs is_odd)")
    ap.add_argument("--src-lang", default="python")
    ap.add_argument("--dst-lang", default="python")
    ap.add_argument("--gate", default="strict", choices=["strict", "relaxed"],
                    help="strict: diverge AND both top-1s equal the head_loc tokens; relaxed: diverge only")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-instances", type=int, default=None,
                    help="smoke test: gate a stratified sample of N (balanced config x datatype, spread over pairs)")
    return ap.parse_args()


def main():
    args = parse_args()
    logging_utils.configure(level=logging.INFO, force=True)
    logger.info(f"config: {json.dumps(vars(args))}")
    cfgs = N.resolve_configs([args.variant] if args.variant else args.configs)
    out_tag = args.out_tag or ("mixed" if len(cfgs) > 1 else cfgs[0])
    out_dir = Path(env_utils.DEFAULT_RESULTS_DIR) / "nrgs_prefix" / args.model.split("/")[-1] / \
        f"{out_tag}__{args.src_lang}-{args.dst_lang}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for c in cfgs:
        logger.info(f"  {c:16s} {N.CONFIGS[c].notation}")

    pivots = N.load_pivots(args.model, datatypes=args.datatypes, core_only=not args.all_prefixes)
    insts = N.build_instances(pivots, configs=cfgs, n_per_pivot=args.n_per_pivot, seed=args.seed,
                              distinct_families=not args.allow_family_overlap,
                              src_lang=args.src_lang, dst_lang=args.dst_lang)
    if args.max_instances:
        insts = N.stratified_sample(insts, args.max_instances, seed=args.seed)
    logger.info(f"{len(pivots)} pivots x {len(cfgs)} configs -> {len(insts)} instances ({args.datatypes}) -> {out_dir}")

    ctx = N.load_ctx(args.model)
    gate_fp = out_dir / "gate.json"
    done = {r["key"]: r for r in json.loads(gate_fp.read_text())} if gate_fp.exists() else {}
    todo = [x for x in insts if x.key not in done]
    logger.info(f"gate: {len(done)} cached, {len(todo)} to run -> {gate_fp}")
    chunk = 8 * args.batch_size
    t0 = time.time()
    for s in range(0, len(todo), chunk):
        for r in N.gate(ctx.mt, ctx, todo[s: s + chunk], batch_size=args.batch_size):
            done[r["key"]] = r
        tmp = gate_fp.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(list(done.values()), indent=1))
        tmp.replace(gate_fp)
        n_done, el = min(s + chunk, len(todo)), time.time() - t0
        logger.info(f"gate: {n_done}/{len(todo)}  ({el:.0f}s, ETA {el / n_done * (len(todo) - n_done):.0f}s)")

    ok_key = "passed" if args.gate == "strict" else "diverge"
    kept = [x for x in insts if done[x.key][ok_key]]
    train, val = N.split_by_pair(kept, val_frac=args.val_frac, seed=args.seed)     # pair-disjoint across configs
    split = {x.key: "train" for x in train} | {x.key: "val" for x in val}
    records = [N.to_record(x) | {"b_id": done[x.key]["b_id"], "d_id": done[x.key]["d_id"],
                                 "split": split[x.key]} for x in kept]
    meta = {}
    for c in cfgs:
        rc = [r for r in records if r["config"] == c]
        cf = N.CONFIGS[c]
        meta[c] = dict(group=cf.group, notation=cf.notation, note=cf.note,
                       has_src_ctrl=cf.src_ctrl_prefill is not None,
                       n_gated=sum(x.config == c for x in insts), n_kept=len(rc),
                       n_train=sum(r["split"] == "train" for r in rc), n_val=sum(r["split"] == "val" for r in rc))
    (out_dir / "dataset.json").write_text(json.dumps({"args": vars(args), "configs": meta, "instances": records}, indent=1))

    rate = lambda g, f: sum(r[f] for r in g) / max(1, len(g))  # noqa: E731
    for c in cfgs + ["ALL"]:
        for dt in args.datatypes + ["all"]:
            g = [done[x.key] for x in insts if (c == "ALL" or x.config == c) and (dt == "all" or x.datatype == dt)]
            k = [r for r in records if (c == "ALL" or r["config"] == c) and (dt == "all" or r["datatype"] == dt)]
            if dt != "all" and c != "ALL":
                continue                                                       # per-config lines: totals only
            logger.info(f"{c:16s} {dt:9s} n={len(g):5d}  diverge={rate(g, 'diverge'):.2f}  match_b={rate(g, 'match_b'):.2f}  "
                        f"match_d={rate(g, 'match_d'):.2f}  kept={len(k)} "
                        f"(train {sum(r['split'] == 'train' for r in k)} / val {sum(r['split'] == 'val' for r in k)})")
    logger.info(f"dataset -> {out_dir / 'dataset.json'}")


if __name__ == "__main__":
    main()
