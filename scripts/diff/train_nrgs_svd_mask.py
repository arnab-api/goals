#!/usr/bin/env python
"""Train a sub-head SVD-component query mask on a mined NRGS dataset (``src/diff/nrgs_svd_mask.py``;
the filter paper's App. I recipe). Separate from ``train_nrgs_mask.py`` (head-level) but on the same data,
objective and per-instance cache (``results/nrgs_mask/<model>/cache/``, shared with the head-level runs).

Each candidate head's query transplant is split into its d_head W_QK singular components; the mask
m in [0, 1]^(layers x heads x d_head) chooses which components carry SRC's query:

    q_patch = q_dst + (q_src - q_dst) M diag(m) M^-1        (pre-RoPE, at DST's last token)

``--restrict-heads`` limits the trainable components to a head set (e.g. the goal heads, as in the paper's
"only filter heads" row); all other heads stay un-patched.

Outputs (``results/nrgs_svd_mask/<model>/<run_tag>/``):
    mask.pt          {mask_values (full depth, n_layer x n_heads x d_head), singular values, allowed, history,
                      config, final_eval}
    final_eval.json  val flip / margin gain / KL: learned (soft; components > threshold; whole heads that keep
                     >= 1 component) vs goal / control head sets (all components = the head-level transplant)
    components.json  {"L.H": [component indices > threshold]} (index 0 = largest singular value)
Usage:
    conda run -n goals python scripts/diff/train_nrgs_svd_mask.py --cuda 0,1 --data-tag h3 --epochs 10
    conda run -n goals python scripts/diff/train_nrgs_svd_mask.py --cuda 0,1 --data-tag h3 --restrict-heads goal
    # smoke (14B): math checks + 2 epochs
    conda run -n goals python scripts/diff/train_nrgs_svd_mask.py --model Qwen/Qwen2.5-Coder-14B-Instruct \\
        --data-tag order_diff --layers 0-48 --epochs 2 --batch-size 8 --check --tag smoke
"""
import argparse
import collections
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

import baukit  # noqa: E402
import torch  # noqa: E402

import src.diff.nrgs_mask as NM  # noqa: E402
import src.diff.nrgs_prefix as N  # noqa: E402
import src.diff.nrgs_svd_mask as SM  # noqa: E402
from src.utils import env_utils, logging_utils  # noqa: E402

logger = logging.getLogger(__name__)
GOAL_HEADS = "head_sets/{model}/goal_topk/goal__lam20_top250.json"
CTRL_HEADS = "head_sets/{model}/control/control250__laterhalf.json"
SHORT = {"order_diff_1": "od1", "order_diff_2": "od2", "order_diff_3": "od3", "order_matched_1": "om1",
         "order_matched_2": "om2", "diff_vs_prog_1": "dvp1",
         "h1c_1": "h1c1", "h1c_2": "h1c2", "h2_1": "h2a", "h2_2": "h2b", "h3_1": "h3a", "h3_2": "h3b"}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    ap.add_argument("--cuda", default=None, help="CUDA_VISIBLE_DEVICES (applied before torch import)")
    ap.add_argument("--data-tag", required=True, help="dataset dir tag from mine_nrgs_prefix.py ('h3', 'h1c', 'mixed', ...)")
    ap.add_argument("--configs", nargs="+", default=None, help="train on these config / hypothesis shards only")
    ap.add_argument("--src-lang", default="python")
    ap.add_argument("--dst-lang", default="python")
    ap.add_argument("--dataset", default=None, help="override the dataset.json path")
    ap.add_argument("--layers", default="0-64", help="candidate layer band lo-hi (half-open)")
    ap.add_argument("--restrict-heads", default=None,
                    help="only these heads carry trainable components: 'goal' (= --goal-heads) or a head-set JSON")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--lambda-sparse", type=float, default=1.0,
                    help="weight on mean(component mask) over ALL candidate components (head-level runs use 20)")
    ap.add_argument("--w-kl", type=float, default=1.0)
    ap.add_argument("--w-dir", type=float, default=1.0)
    ap.add_argument("--w-ctrl", type=float, default=0.0, help="weight on 'SRC\\' query must not flip' (0 = off)")
    ap.add_argument("--dir-loss", default="logistic", choices=["logistic", "margin"])
    ap.add_argument("--batch-size", type=int, default=64, help="instances per optimizer step")
    ap.add_argument("--micro-batch", type=int, default=8, help="instances per forward/backward")
    ap.add_argument("--threshold", type=float, default=0.9, help="component mask value above which it is selected")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-train", type=int, default=None, help="smoke test: first N train instances")
    ap.add_argument("--tag", default="", help="suffix for the run tag")
    ap.add_argument("--goal-heads", default=GOAL_HEADS)
    ap.add_argument("--ctrl-heads", default=CTRL_HEADS)
    ap.add_argument("--check", action="store_true",
                    help="before training: verify the SVD basis on real weights and the m=1 / m=0 equivalences")
    return ap.parse_args()


def _data_label(args):
    return args.data_tag + ("+" + "-".join(SHORT.get(c, c) for c in N.resolve_configs(args.configs)) if args.configs else "")


def _heads(spec, model_short):
    fp = Path(spec.format(model=model_short))
    return [(int(l), int(h)) for l, h in json.loads(fp.read_text())] if fp.exists() else None


@torch.no_grad()
def run_checks(mt, layers, bases, caches):
    """(1) U = W_Q M orthonormal and U S V^T = W_Q W_K^T (V^T = S^-1 M^-1 W_K^T) on a few heads;
    (2) m = 1 on a head set == the head-level transplant; (3) m = 0 == clean DST."""
    H, d, g = mt.n_heads, mt.head_dim, mt.n_heads // mt.n_kv_heads
    for l in (layers[0], layers[len(layers) // 2], layers[-1]):
        wq = baukit.get_module(mt._model, SM._proj(mt, l, "q")).weight.float()
        wk = baukit.get_module(mt._model, SM._proj(mt, l, "k")).weight.float().to(wq.device)
        for h in (0, H // 2, H - 1):
            Wq = wq[h * d:(h + 1) * d].T                        # (d_embd, d)
            Wk = wk[(h // g) * d:(h // g + 1) * d].T
            b = bases[l]
            M, Minv, S = b["M"][h], b["Minv"][h], b["S"][h]
            U = Wq @ M
            Vt = torch.diag(1 / S) @ Minv @ Wk.T
            WQK = Wq @ Wk.T
            orth_u = float((U.T @ U - torch.eye(d, device=U.device)).abs().max())
            orth_v = float((Vt @ Vt.T - torch.eye(d, device=U.device)).abs().max())
            rec = float((U @ torch.diag(S) @ Vt - WQK).norm() / WQK.norm())
            inv = float((M @ Minv - torch.eye(d, device=U.device)).abs().max())
            logger.info(f"check basis L{l} H{h}: |U^T U - I|={orth_u:.1e}  |V^T V - I|={orth_v:.1e}  "
                        f"rel|U S V^T - W_QK|={rec:.1e}  |M M^-1 - I|={inv:.1e}  s0/s_last={float(S[0] / S[-1]):.1f}")
            assert orth_u < 1e-2 and orth_v < 1e-2 and rec < 1e-3 and inv < 1e-2, "SVD basis check failed"
    cs = caches[: min(4, len(caches))]
    heads = [(layers[len(layers) // 2], h) for h in range(0, H, 3)] + [(layers[-1], 1)]
    lp_head = NM.patched_logprobs(mt, NM.mask_from_heads(heads, layers, H, None), layers, cs)
    lp_svd = SM.patched_logprobs(mt, SM.comp_mask_from_heads(heads, layers, H, d), layers, bases, cs)
    lp_clean = NM.dst_logprobs(mt, cs)
    lp_zero = SM.patched_logprobs(mt, torch.zeros(len(layers), H, d), layers, bases, cs)
    d1 = float((lp_head - lp_svd).abs().max())
    d0 = float((lp_clean - lp_zero).abs().max())
    a1 = float((lp_head.argmax(-1) == lp_svd.argmax(-1)).float().mean())
    logger.info(f"check m=1 vs head-level transplant: max|dlogp|={d1:.3f} (top-1 agree {a1:.2f}) · "
                f"m=0 vs clean DST: max|dlogp|={d0:.3f}  (bf16 tolerance)")
    assert a1 == 1.0 and d0 < 0.5, "m=1 / m=0 equivalence check failed"


def main():
    args = parse_args()
    logging_utils.configure(level=logging.INFO, force=True)
    logger.info(f"config: {json.dumps(vars(args))}")
    model_short = args.model.split("/")[-1]
    lo, hi = (int(x) for x in args.layers.split("-"))
    ds_fp = Path(args.dataset) if args.dataset else (
        Path(env_utils.DEFAULT_RESULTS_DIR) / "nrgs_prefix" / model_short
        / f"{args.data_tag}__{args.src_lang}-{args.dst_lang}" / "dataset.json")
    records = N.load_dataset(ds_fp, configs=args.configs)
    by_cfg = collections.Counter(N.canonical(r.get("config", r.get("variant"))) for r in records)
    logger.info(f"dataset {ds_fp}: {len(records)} instances · per config {dict(by_cfg)}")

    ctx = N.load_ctx(args.model)
    mt = ctx.mt
    layers = list(range(lo, min(hi, mt.n_layer)))

    # ── per-instance cache: the SAME file as train_nrgs_mask.py (shared across head- and component-level runs) ──
    ds_hash = hashlib.md5(ds_fp.read_bytes()).hexdigest()[:10]
    cache_fp = Path(env_utils.DEFAULT_RESULTS_DIR) / "nrgs_mask" / model_short / "cache" / \
        f"{_data_label(args)}__{args.src_lang}-{args.dst_lang}__{ds_hash}.pt"
    if cache_fp.exists():
        caches = [NM.NRGSCache(**c) for c in torch.load(cache_fp, weights_only=False)]
        logger.info(f"cache hit: {cache_fp} ({len(caches)})")
    else:
        caches = NM.build_caches(mt, ctx, records, batch_size=args.micro_batch)
        cache_fp.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_fp.with_suffix(f".tmp{os.getpid()}")
        torch.save([c.__dict__ for c in caches], tmp)
        os.replace(tmp, cache_fp)
        logger.info(f"cache -> {cache_fp}")
    train = [c for c in caches if c.split == "train"]
    val = [c for c in caches if c.split == "val"]
    if args.max_train:
        train = train[: args.max_train]

    bases = SM.svd_bases(mt, layers)
    logger.info(f"SVD bases: {len(layers)} layers x {mt.n_heads} heads x {mt.head_dim} components "
                f"(GQA group {mt.n_heads // mt.n_kv_heads}) · train {len(train)} / val {len(val)}")
    if args.check:
        run_checks(mt, layers, bases, val or train)

    allowed, restrict = None, None
    if args.restrict_heads:
        restrict = _heads(args.goal_heads if args.restrict_heads == "goal" else args.restrict_heads, model_short)
        assert restrict, f"--restrict-heads {args.restrict_heads}: head set not found"
        allowed = SM.allowed_from_heads(restrict, layers, mt.n_heads)
        logger.info(f"restricting components to {int(allowed.sum())} heads ({args.restrict_heads})")

    rtag = ("" if not args.restrict_heads else
            f"__only-{'goal' if args.restrict_heads == 'goal' else Path(args.restrict_heads).stem.split('__')[0]}")
    run_tag = (f"{_data_label(args)}__{args.src_lang}-{args.dst_lang}__L{lo}-{hi}{rtag}__lam{args.lambda_sparse:g}"
               f"__kl{args.w_kl:g}_dir{args.w_dir:g}_ctrl{args.w_ctrl:g}__{args.dir_loss}__s{args.seed}"
               + (f"__{args.tag}" if args.tag else ""))
    out_dir = Path(env_utils.DEFAULT_RESULTS_DIR) / "nrgs_svd_mask" / model_short / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"run_tag={run_tag} -> {out_dir}")
    config = vars(args) | {"dataset": str(ds_fp), "cache": str(cache_fp), "layers_used": layers,
                           "n_train": len(train), "n_val": len(val)}

    def save(mask, hist, final_eval=None):
        full = torch.zeros(mt.n_layer, mt.n_heads, mt.head_dim)
        full[layers] = mask.values.detach().cpu()
        allow = torch.zeros(mt.n_layer, mt.n_heads)
        allow[layers] = mask.allowed.squeeze(-1).cpu()
        torch.save(dict(mask_values=full, layers=layers, allowed=allow, threshold=args.threshold,
                        singular_values={l: bases[l]["S"].cpu() for l in layers},
                        history=hist.__dict__, config=config, final_eval=final_eval), out_dir / "mask.pt")

    mask, hist = SM.train_svd_mask(
        mt, train, val, layers, bases, allowed=allowed, n_epochs=args.epochs, lr=args.lr,
        lambda_sparse=args.lambda_sparse, w_kl=args.w_kl, w_dir=args.w_dir, w_ctrl=args.w_ctrl,
        dir_loss=args.dir_loss, batch_size=args.batch_size, micro_batch=args.micro_batch, seed=args.seed,
        threshold=args.threshold, on_epoch=lambda e, m, h: save(m, h))

    # ── final eval on val ──
    vals = mask.values.detach()
    comp_bin = (vals > args.threshold).float()
    head_any = (comp_bin.sum(-1, keepdim=True) > 0).float().expand_as(comp_bin).contiguous()
    n_comp, n_head = int(comp_bin.sum()), int(head_any[..., 0].sum())
    logger.info(f"selected {n_comp} components (mask > {args.threshold}) in {n_head} heads "
                f"(mean {n_comp / max(1, n_head):.1f} of {mt.head_dim} per head)")
    variants = {"learned_soft": vals, f"learned_comp{n_comp}": comp_bin, f"learned_heads{n_head}": head_any}
    for name, spec in (("goal", args.goal_heads), ("ctrl", args.ctrl_heads)):
        hs = _heads(spec, model_short)
        if hs is None:
            logger.warning(f"final eval: {name} head set {spec} not found — skipped")
            continue
        variants[f"{name}{len(hs)}"] = SM.comp_mask_from_heads(hs, layers, mt.n_heads, mt.head_dim, vals.device)
    roles = [False, True] if any(c.q_ctrl is not None for c in val) else [False]
    final_eval = {}
    for name, mv in variants.items():
        for use_ctrl in roles:
            k = f"{name}|{'src_ctrl' if use_ctrl else 'src'}"
            final_eval[k] = e = SM.evaluate(mt, mv, layers, bases, val, args.micro_batch, use_ctrl=use_ctrl)
            logger.info(f"final  {k:26s} n={e['n']:4d}  flip={e['flip']:.3f}  margin_gain={e['margin_gain']:+.2f}  kl={e['kl']:.2f}")
            for cf, ec in e["by_config"].items():
                logger.info(f"         {'':26s}   {cf:16s} n={ec['n']:4d}  flip={ec['flip']:.3f}  "
                            f"margin_gain={ec['margin_gain']:+.2f}  kl={ec['kl']:.2f}")
    save(mask, hist, final_eval)
    (out_dir / "final_eval.json").write_text(json.dumps(final_eval, indent=1))
    comps = {f"{layers[li]}.{h}": [int(i) for i in torch.nonzero(comp_bin[li, h]).flatten()]
             for li in range(len(layers)) for h in range(mt.n_heads) if comp_bin[li, h].any()}
    (out_dir / "components.json").write_text(json.dumps(comps))
    per_layer = collections.Counter(int(k.split(".")[0]) // 8 * 8 for k in comps)
    logger.info(f"components.json: {len(comps)} heads · heads per 8-layer block "
                f"{ {f'L{k}-{k + 7}': v for k, v in sorted(per_layer.items())} } · -> {out_dir}")


if __name__ == "__main__":
    main()
