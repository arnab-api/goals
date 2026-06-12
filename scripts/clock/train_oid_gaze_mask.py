#!/usr/bin/env python
"""Jointly train a sparse binary mask over the ``< layer_boundary`` heads for the order-ID gaze
experiment.

Objective (per same-spec oid pair):

    L = -(T(target_span) - T(orig_span)) + lambda_sparse * mean(mask)

where ``T`` is the raw Q·K gaze affinity on the locked late-layer gaze heads, read at the
destination's ``=`` pivot, and the masked ``< layer_boundary`` heads' pivot o_proj outputs are
transplanted source→dest (``m*src + (1-m)*cascade``).  Reuses ``HeadMask`` + projected GD (the
goalscope recipe): all-ones init, clamp to ``[0,1]`` after each step, sparsity = ``λ*mean(mask)``.

Pairs are processed B=1 with gradient accumulation (``--accum``) ⇒ no padding, low attention memory.

Typical usage
─────────────
    conda run -n goals python scripts/clock/train_oid_gaze_mask.py \\
        --n-pairs 60 --n-epochs 20 --cuda 4,5,6,7

    conda run -n goals python scripts/clock/train_oid_gaze_mask.py \\
        --normalize --lambda-sparse 5e-2 --lr 3e-2 --orig-idx 0 --target-idx 2

Writes ``mask.pt`` (mask values + config + history + selected heads), ``history.png`` under
``results/multistep_clock/<version>/<model>/oid_gaze/mask/<tag>/``.
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.models import ModelandTokenizer  # noqa: E402
from src.goals.head_loc_attn import HeadMask  # noqa: E402
from src.utils import logging_utils  # noqa: E402
from src.goals.oid_gaze import (  # noqa: E402
    GazeScorer,
    build_oid_pairs,
    cache_root,
    load_gaze_heads,
    mask_patch_hooks,
)

logger = logging.getLogger("train_oid_gaze_mask")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # model / data
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    p.add_argument("--version", default="conditions")
    p.add_argument("--modularity", default="monolithic", choices=["monolithic", "modular"])
    p.add_argument("--cuda", default=None, help="CUDA_VISIBLE_DEVICES (e.g. '4,5,6,7')")
    # pivots
    p.add_argument("--orig-idx", type=int, default=None, help="pin dest pivot condition; if unset, vary over all valid indices < target")
    p.add_argument("--target-idx", type=int, default=None, help="pin redirect-target condition; if unset, vary over all valid indices > orig")
    # gaze readout heads
    p.add_argument("--layer-boundary", type=int, default=40)
    p.add_argument("--gaze-topn", type=int, default=100)
    p.add_argument("--gaze-q-span", type=int, default=15)
    p.add_argument("--gaze-compare-with", default="mean", choices=["mean", "max"])
    # affinity knobs
    p.add_argument("--normalize", action="store_true", help="cosine similarity (default: raw dot product)")
    p.add_argument("--include-rope", action="store_true")
    p.add_argument("--gaze-device", default="cuda:0")
    # training
    p.add_argument("--n-pairs", type=int, default=500)
    p.add_argument("--n-epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=5e-2)
    p.add_argument("--lambda-sparse", type=float, default=2e-2)
    p.add_argument("--accum", type=int, default=8, help="pairs per optimizer step (grad accumulation)")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--sel-threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    logging_utils.configure(level=logging.INFO)

    mt = ModelandTokenizer(
        model_key=args.model, dtype=torch.bfloat16, device_map="auto", attn_implementation="eager"
    )
    nH = mt.model.config.num_attention_heads
    hd = mt.n_embd // nH
    layers_below = list(range(0, args.layer_boundary))

    gaze_heads = load_gaze_heads(
        args.model, args.version, args.gaze_q_span, args.gaze_compare_with, args.gaze_topn, args.layer_boundary
    )
    logger.info(f"gaze readout heads (L>{args.layer_boundary}): {len(gaze_heads)}")
    scorer = GazeScorer(
        mt, gaze_heads, normalize=args.normalize, include_rope=args.include_rope, device=args.gaze_device
    )

    pairs = build_oid_pairs(
        mt, args.model, args.version, args.modularity, layers_below, args.n_pairs,
        orig_idx=args.orig_idx, target_idx=args.target_idx,
    )
    _orig_desc = "any" if args.orig_idx is None else f"cond{args.orig_idx}"
    _tgt_desc = "any" if args.target_idx is None else f"cond{args.target_idx}"
    logger.info(f"built {len(pairs)} oid pairs (orig={_orig_desc}, target={_tgt_desc}, orig<target)")
    assert len(pairs) >= 4, "need a few pairs to train"

    # sample-disjoint train/val split (multiple pairs can share a sample)
    seen = list(dict.fromkeys(p.sample for p in pairs))
    n_val_s = max(1, int(args.val_frac * len(seen)))
    val_samples = set(seen[:n_val_s])
    val_pairs = [p for p in pairs if p.sample in val_samples]
    train_pairs = [p for p in pairs if p.sample not in val_samples]
    assert train_pairs, "all pairs landed in val; need more samples or a smaller --val-frac"
    logger.info(f"train {len(train_pairs)} pairs / val {len(val_pairs)} pairs  "
                f"({len(seen) - n_val_s}/{len(seen)} samples in train)")

    for prm in mt._model.parameters():
        prm.requires_grad_(False)
    mask = HeadMask(len(layers_below), nH).to(args.gaze_device)
    opt = AdamW(mask.parameters(), lr=args.lr)

    def swap_delta(pair):
        T = scorer.compute_T(
            pair.dest_inputs,
            [pair.span_orig, pair.span_target],
            pre_hooks=mask_patch_hooks(mt, layers_below, pair.src_hv, mask, nH, hd),
        )[0]
        return T[1] - T[0]  # T(target) - T(orig)

    rng = random.Random(args.seed)
    hist = {"train_dT": [], "val_dT": [], "mean_mask": [], "n_active": []}

    for epoch in range(args.n_epochs):
        order = train_pairs[:]
        rng.shuffle(order)
        opt.zero_grad()
        ep_dt = 0.0
        for i, pair in enumerate(order):
            d = swap_delta(pair)
            loss = (-d + args.lambda_sparse * mask.values.mean()) / args.accum
            loss.backward()
            ep_dt += float(d.detach())
            if (i + 1) % args.accum == 0 or (i + 1) == len(order):
                opt.step()
                mask.clamp_()
                opt.zero_grad()
        with torch.no_grad():
            vds = [float(swap_delta(p).detach()) for p in val_pairs]
        hist["train_dT"].append(ep_dt / len(order))
        hist["val_dT"].append(float(np.mean(vds)))
        hist["mean_mask"].append(float(mask.values.mean()))
        hist["n_active"].append(mask.n_active(args.sel_threshold))
        logger.info(
            f"epoch {epoch:2d}  train ΔT={hist['train_dT'][-1]:+.4f}  val ΔT={hist['val_dT'][-1]:+.4f}  "
            f"mean(mask)={hist['mean_mask'][-1]:.3f}  n_active(>{args.sel_threshold})={hist['n_active'][-1]}"
        )

    decision_heads = mask.selected_heads(layers_below, threshold=args.sel_threshold)
    mv = mask.values.detach().cpu().numpy()
    ranked = sorted(decision_heads, key=lambda lh: mv[layers_below.index(lh[0]), lh[1]], reverse=True)
    logger.info(f"selected {len(decision_heads)} decision heads (mask>{args.sel_threshold}); top:")
    for (l, h) in ranked[:20]:
        logger.info(f"  L{l:2d} H{h:2d}  mask={mv[layers_below.index(l), h]:.3f}")

    # ── outputs ──
    tag = (
        f"orig{'any' if args.orig_idx is None else args.orig_idx}"
        f"_tgt{'any' if args.target_idx is None else args.target_idx}"
        f"_b{args.layer_boundary}_top{args.gaze_topn}"
        f"_{'cos' if args.normalize else 'dot'}{'_rope' if args.include_rope else ''}"
        f"_lam{args.lambda_sparse}_n{len(pairs)}"
    )
    out_dir = args.out or os.path.join(cache_root(args.model, args.version), "oid_gaze", "mask", tag)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(
        {
            "mask_values": mask.values.detach().cpu(),
            "layers": layers_below,
            "gaze_heads": gaze_heads,
            "decision_heads": decision_heads,
            "history": hist,
            "config": vars(args),
        },
        os.path.join(out_dir, "mask.pt"),
    )
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(
            {
                **vars(args),
                "n_pairs_used": len(pairs),
                "n_decision_heads": len(decision_heads),
                "final_val_dT": hist["val_dT"][-1],
                "decision_heads": ranked,
            },
            f,
            indent=2,
        )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ep = range(args.n_epochs)
    axes[0].plot(ep, hist["train_dT"], "o-", label="train ΔT")
    axes[0].plot(ep, hist["val_dT"], "s--", label="val ΔT")
    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("T(target) - T(orig)")
    axes[0].set_title("gaze swap-delta")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    ax2 = axes[0].twinx()
    ax2.plot(ep, hist["n_active"], "^:", color="tab:gray")
    ax2.set_ylabel("n_active heads", color="tab:gray")
    im = axes[1].imshow(mv.T, aspect="auto", cmap="viridis", vmin=0, vmax=1, origin="lower")
    axes[1].set_xlabel(f"layer (0..{args.layer_boundary - 1})")
    axes[1].set_ylabel("head")
    axes[1].set_title(f"trained mask ({len(decision_heads)} heads > {args.sel_threshold})")
    fig.colorbar(im, ax=axes[1], shrink=0.8, label="mask value")
    fig.suptitle(f"oid gaze mask — {tag}", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "history.png"), dpi=130, bbox_inches="tight")
    logger.info(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
