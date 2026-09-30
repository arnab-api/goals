#!/usr/bin/env python
"""Train a sparse query-head mask on the mined NRGS common-prefix dataset (``src/diff/nrgs_mask.py``).

Input: ``results/nrgs_prefix/<model>/<variant>__<sl>-<dl>/dataset.json`` from ``mine_nrgs_prefix.py``.
The per-instance cache (SRC / SRC' queries for ALL layers, DST* target log-probs, clean margins) is built
once and reused across runs (e.g. a lambda sweep).

Defaults reproduce the recipe behind the 122 goal heads (results/localization/<model>/lam20 + FUNC.setup):
all 64 layers as candidates, lambda 20, lr 1e-2, 50 epochs, 64 instances per optimizer step; selection =
mask > 0.9 within layers [25, 56). The full-depth mask is always saved, so any other selection is post-hoc.

Outputs (``results/nrgs_mask/<model>/<run_tag>/``):
    mask.pt          {mask_values (full depth), layers, selected_heads, history, config, final_eval}
    final_eval.json  val flip / margin gain / KL: learned (soft, binarized) vs goal / control heads,
                     under SRC's and (order_diff) SRC''s query
and the selected heads as a head set: ``head_sets/<model>/nrgs/progress__<run_tag>.json``.
Usage:
    conda run -n goals python scripts/diff/train_nrgs_mask.py --cuda 0,1,2 --variant order_diff \\
        --lambda-sparse 5 --epochs 30 --batch-size 8
    # smoke: 2 epochs on 16 train instances
    conda run -n goals python scripts/diff/train_nrgs_mask.py --cuda 0,1 --epochs 2 --max-train 16 --tag smoke
"""
import argparse
import hashlib
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
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

import src.diff.nrgs_mask as NM  # noqa: E402
import src.diff.nrgs_prefix as N  # noqa: E402
from src.utils import env_utils, logging_utils  # noqa: E402

logger = logging.getLogger(__name__)
GOAL_HEADS = "head_sets/{model}/goal_topk/goal__lam20_top250.json"
CTRL_HEADS = "head_sets/{model}/control/control250__laterhalf.json"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    ap.add_argument("--cuda", default=None, help="CUDA_VISIBLE_DEVICES (applied before torch import)")
    ap.add_argument("--variant", default="order_diff",
                    help="legacy dataset tag (results/nrgs_prefix/<model>/<tag>__<sl>-<dl>/); see --data-tag")
    ap.add_argument("--data-tag", default=None,
                    help="dataset dir tag from mine_nrgs_prefix.py, e.g. 'mixed' (default: --variant)")
    ap.add_argument("--configs", nargs="+", default=None,
                    help="train on these config shards of the dataset only (default: every config in it)")
    ap.add_argument("--src-lang", default="python")
    ap.add_argument("--dst-lang", default="python")
    ap.add_argument("--dataset", default=None, help="override the dataset.json path")
    ap.add_argument("--layers", default="0-64", help="candidate layer band lo-hi (half-open)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--lambda-sparse", type=float, default=20.0, help="weight on mean(mask)")
    ap.add_argument("--w-kl", type=float, default=1.0, help="weight on KL(P_patched || P_DST*)")
    ap.add_argument("--w-dir", type=float, default=1.0, help="weight on the directional d-over-b term")
    ap.add_argument("--w-ctrl", type=float, default=0.0,
                    help="weight on 'SRC\\' query must not flip' (order_diff only; 0 = off)")
    ap.add_argument("--dir-loss", default="logistic", choices=["logistic", "margin"])
    ap.add_argument("--batch-size", type=int, default=64, help="instances per optimizer step")
    ap.add_argument("--micro-batch", type=int, default=8,
                    help="instances per forward/backward; gradients accumulate to --batch-size")
    ap.add_argument("--grad-ckpt", action="store_true", help="gradient checkpointing (less memory, slower)")
    ap.add_argument("--threshold", type=float, default=0.9, help="mask value above which a head is selected")
    ap.add_argument("--select-band", default="25-56",
                    help="keep selected heads with layer in [lo, hi) (after training, as for the 122 goal heads); "
                         "'none' = all candidate layers")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-train", type=int, default=None, help="smoke test: first N train instances")
    ap.add_argument("--tag", default="", help="suffix for the run tag")
    ap.add_argument("--goal-heads", default=GOAL_HEADS, help="fixed head set for the final-eval comparison "
                    "({model} = model short name); skipped with a warning if missing")
    ap.add_argument("--ctrl-heads", default=CTRL_HEADS, help="matched control head set for the final eval")
    return ap.parse_args()


SHORT = {"order_diff_1": "od1", "order_diff_2": "od2", "order_diff_3": "od3", "order_matched_1": "om1",
         "order_matched_2": "om2", "diff_vs_prog_1": "dvp1"}


def _data_label(args):
    tag = args.data_tag or args.variant
    return tag + ("+" + "-".join(SHORT.get(c, c) for c in N.resolve_configs(args.configs)) if args.configs else "")


def _run_tag(args, lo, hi):
    t = (f"{_data_label(args)}__{args.src_lang}-{args.dst_lang}__L{lo}-{hi}__lam{args.lambda_sparse:g}"
         f"__kl{args.w_kl:g}_dir{args.w_dir:g}_ctrl{args.w_ctrl:g}__{args.dir_loss}__s{args.seed}")
    return t + (f"__{args.tag}" if args.tag else "")


def main():
    args = parse_args()
    logging_utils.configure(level=logging.INFO, force=True)
    logger.info(f"config: {json.dumps(vars(args))}")
    model_short = args.model.split("/")[-1]
    lo, hi = (int(x) for x in args.layers.split("-"))
    ds_fp = Path(args.dataset) if args.dataset else (
        Path(env_utils.DEFAULT_RESULTS_DIR) / "nrgs_prefix" / model_short
        / f"{args.data_tag or args.variant}__{args.src_lang}-{args.dst_lang}" / "dataset.json")
    records = N.load_dataset(ds_fp, configs=args.configs)
    import collections
    by_cfg = collections.Counter(N.canonical(r.get("config", r.get("variant"))) for r in records)
    logger.info(f"dataset {ds_fp}: {len(records)} instances"
                + (f" (shards {N.resolve_configs(args.configs)})" if args.configs else "") + f" · per config {dict(by_cfg)}")

    ctx = N.load_ctx(args.model)
    mt = ctx.mt
    layers = list(range(lo, min(hi, mt.n_layer)))

    # ── cache (all layers; shared across runs on the same dataset) ──
    ds_hash = hashlib.md5(ds_fp.read_bytes()).hexdigest()[:10]
    cache_fp = Path(env_utils.DEFAULT_RESULTS_DIR) / "nrgs_mask" / model_short / "cache" / \
        f"{_data_label(args)}__{args.src_lang}-{args.dst_lang}__{ds_hash}.pt"
    if cache_fp.exists():
        caches = [NM.NRGSCache(**c) for c in torch.load(cache_fp, weights_only=False)]
        logger.info(f"cache hit: {cache_fp} ({len(caches)})")
    else:
        caches = NM.build_caches(mt, ctx, records, batch_size=args.micro_batch)
        cache_fp.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_fp.with_suffix(f".tmp{os.getpid()}")      # atomic: parallel runs may build the same cache
        torch.save([c.__dict__ for c in caches], tmp)
        os.replace(tmp, cache_fp)
        logger.info(f"cache -> {cache_fp}")
    train = [c for c in caches if c.split == "train"]
    val = [c for c in caches if c.split == "val"]
    if args.max_train:
        train = train[: args.max_train]
    logger.info(f"train {len(train)} / val {len(val)} · layers {lo}-{hi} ({len(layers)}) · heads {mt.n_heads}")

    if args.grad_ckpt:
        mt._model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        mt._model.train()   # checkpointing is active only in train mode; Qwen2 has no dropout

    run_tag = _run_tag(args, lo, hi)
    out_dir = Path(env_utils.DEFAULT_RESULTS_DIR) / "nrgs_mask" / model_short / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"run_tag={run_tag} -> {out_dir}")
    config = vars(args) | {"dataset": str(ds_fp), "cache": str(cache_fp), "layers_used": layers,
                           "n_train": len(train), "n_val": len(val)}

    band = None if args.select_band.lower() == "none" else tuple(int(x) for x in args.select_band.split("-"))

    def select(mask):
        allh = mask.selected_heads(layers, args.threshold)
        return [h for h in allh if band is None or band[0] <= h[0] < band[1]], allh

    def save(mask, hist, final_eval=None):
        full = mask.to_full_depth(layers, mt.n_layer)
        sel, allh = select(mask)
        torch.save(dict(mask_values=full.values.detach().cpu(), layers=layers, selected_heads=sel,
                        selected_heads_all_layers=allh, select_band=band, threshold=args.threshold,
                        history=hist.__dict__, config=config, final_eval=final_eval), out_dir / "mask.pt")

    mask, hist = NM.train_nrgs_mask(
        mt, train, val, layers, n_epochs=args.epochs, lr=args.lr, lambda_sparse=args.lambda_sparse,
        w_kl=args.w_kl, w_dir=args.w_dir, w_ctrl=args.w_ctrl, dir_loss=args.dir_loss,
        batch_size=args.batch_size, micro_batch=args.micro_batch, seed=args.seed, threshold=args.threshold,
        on_epoch=lambda e, m, h: save(m, h),
    )
    if args.grad_ckpt:
        mt._model.gradient_checkpointing_disable()
        mt._model.eval()

    # ── final eval on val: learned vs fixed head sets, SRC vs SRC' query ──
    selected, selected_all = select(mask)
    logger.info(f"selected {len(selected)} heads (mask > {args.threshold} in band {band}); "
                f"{len(selected_all)} above threshold across all candidate layers")
    dev = mask.values.device
    variants = {
        "learned_soft": mask.values.detach(),
        "learned_bin": NM.mask_from_heads(selected, layers, mt.n_heads, dev),        # = the exported head set
    }
    for name, spec in (("goal", args.goal_heads), ("ctrl", args.ctrl_heads)):
        fp = Path(spec.format(model=model_short))
        if not fp.exists():
            logger.warning(f"final eval: {name} head set {fp} not found — skipped")
            continue
        heads = [(int(l), int(h)) for l, h in json.loads(fp.read_text())]
        variants[f"{name}{len(heads)}"] = NM.mask_from_heads(heads, layers, mt.n_heads, dev)
    roles = [False, True] if any(c.q_ctrl is not None for c in val) else [False]
    final_eval = {}
    for name, mv in variants.items():
        for use_ctrl in roles:
            k = f"{name}|{'src_ctrl' if use_ctrl else 'src'}"
            final_eval[k] = NM.evaluate(mt, mv, layers, val, args.micro_batch, use_ctrl=use_ctrl)
            e = final_eval[k]
            logger.info(f"final  {k:24s} n={e['n']:4d}  flip={e['flip']:.3f}  margin_gain={e['margin_gain']:+.2f}  kl={e['kl']:.2f}")
            for cf, ec in e["by_config"].items():
                logger.info(f"         {'':24s}   {cf:16s} n={ec['n']:4d}  flip={ec['flip']:.3f}  "
                            f"margin_gain={ec['margin_gain']:+.2f}  kl={ec['kl']:.2f}")
    save(mask, hist, final_eval)
    (out_dir / "final_eval.json").write_text(json.dumps(final_eval, indent=1))

    hs_fp = Path("head_sets") / model_short / "nrgs" / f"progress__{run_tag}.json"
    hs_fp.parent.mkdir(parents=True, exist_ok=True)
    hs_fp.write_text(json.dumps([list(h) for h in selected]))
    logger.info(f"{len(selected)} heads -> {hs_fp} · mask -> {out_dir / 'mask.pt'}")


if __name__ == "__main__":
    main()
