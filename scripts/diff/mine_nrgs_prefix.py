#!/usr/bin/env python
"""Mine the NRGS common-prefix contrastive dataset (see ``src/diff/nrgs_prefix.py``).

Builds (A, B, C, D, prefix) instances from the goal-head localization pivots, runs the divergence gate
in the NRGS context (DST top-1 after the prefix = b, DST* top-1 = d), and saves the gate-passing
instances — with the model's own ``b_id`` / ``d_id`` loss targets and a (b, d)-pair-disjoint
train/val split — for ``scripts/diff/train_nrgs_mask.py``. Resumable: gate rows are cached by key.

Outputs (``results/nrgs_prefix/<model>/<variant>__<sl>-<dl>/``):
    gate.json       every gated instance (passed or not), resumable cache
    dataset.json    {config, instances: [record + b_id, d_id, split]} for the passing instances
Usage:
    conda run -n goals python scripts/diff/mine_nrgs_prefix.py --cuda 0 \\
        --variant order_diff --datatypes str int "list[int]" --n-per-pivot 4
"""
import argparse
import json
import logging
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

import src.diff.nrgs_prefix as N  # noqa: E402
from src.utils import env_utils, logging_utils  # noqa: E402

logger = logging.getLogger(__name__)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    ap.add_argument("--cuda", default=None, help="CUDA_VISIBLE_DEVICES (applied before torch import)")
    ap.add_argument("--variant", default="order_diff", choices=N.VARIANTS)
    ap.add_argument("--datatypes", nargs="+", default=["str", "int", "list[int]"])
    ap.add_argument("--n-per-pivot", type=int, default=4, help="(A, C) orderings per (b, d, prefix) pivot")
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
                    help="smoke test: gate a stratified sample of N (balanced datatypes, spread over pairs)")
    return ap.parse_args()


def main():
    args = parse_args()
    logging_utils.configure(level=logging.INFO, force=True)
    logger.info(f"config: {json.dumps(vars(args))}")
    out_dir = Path(env_utils.DEFAULT_RESULTS_DIR) / "nrgs_prefix" / args.model.split("/")[-1] / \
        f"{args.variant}__{args.src_lang}-{args.dst_lang}"
    out_dir.mkdir(parents=True, exist_ok=True)

    pivots = N.load_pivots(args.model, datatypes=args.datatypes, core_only=not args.all_prefixes)
    insts = N.build_instances(pivots, variant=args.variant, n_per_pivot=args.n_per_pivot, seed=args.seed,
                              distinct_families=not args.allow_family_overlap,
                              src_lang=args.src_lang, dst_lang=args.dst_lang)
    if args.max_instances:
        insts = N.stratified_sample(insts, args.max_instances, seed=args.seed)
    logger.info(f"{len(pivots)} pivots -> {len(insts)} instances ({args.variant}, {args.datatypes})")

    ctx = N.load_ctx(args.model)
    gate_fp = out_dir / "gate.json"
    done = {r["key"]: r for r in json.loads(gate_fp.read_text())} if gate_fp.exists() else {}
    todo = [x for x in insts if x.key not in done]
    logger.info(f"gate: {len(done)} cached, {len(todo)} to run -> {gate_fp}")
    chunk = 8 * args.batch_size
    for s in range(0, len(todo), chunk):
        for r in N.gate(ctx.mt, ctx, todo[s: s + chunk], batch_size=args.batch_size):
            done[r["key"]] = r
        tmp = gate_fp.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(list(done.values()), indent=1))
        tmp.replace(gate_fp)
        logger.info(f"gate: {min(s + chunk, len(todo))}/{len(todo)}")

    ok_key = "passed" if args.gate == "strict" else "diverge"
    kept = [x for x in insts if done[x.key][ok_key]]
    train, val = N.split_by_pair(kept, val_frac=args.val_frac, seed=args.seed)
    split = {x.key: "train" for x in train} | {x.key: "val" for x in val}
    records = [N.to_record(x) | {"b_id": done[x.key]["b_id"], "d_id": done[x.key]["d_id"],
                                 "split": split[x.key]} for x in kept]
    (out_dir / "dataset.json").write_text(json.dumps({"config": vars(args), "instances": records}, indent=1))

    for dt in args.datatypes + ["ALL"]:
        g = [done[x.key] for x in insts if dt == "ALL" or x.datatype == dt]
        k = [r for r in records if dt == "ALL" or r["datatype"] == dt]
        rate = lambda f: sum(r[f] for r in g) / max(1, len(g))  # noqa: E731
        logger.info(f"{dt:10s} n={len(g):5d}  diverge={rate('diverge'):.2f}  match_b={rate('match_b'):.2f}  "
                    f"match_d={rate('match_d'):.2f}  kept={len(k)} "
                    f"(train {sum(r['split'] == 'train' for r in k)} / val {sum(r['split'] == 'val' for r in k)})")
    logger.info(f"dataset -> {out_dir / 'dataset.json'}")


if __name__ == "__main__":
    main()
