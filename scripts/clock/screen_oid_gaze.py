#!/usr/bin/env python
"""Per-head screen for the order-ID gaze experiment.

For each same-spec oid pair (dest cut at ``orig`` pivot, src cut at ``target`` pivot), transplant
each ``< layer_boundary`` head's pivot o_proj output source→dest *one at a time* and measure the
change in the gaze swap-delta ``T(target_span) - T(orig_span)`` (raw Q·K affinity on the locked
late-layer gaze heads).  A head scores high if moving *it alone* redirects the late gaze from the
destination's condition toward the source's.

Heads are batched within each source layer ⇒ ``layer_boundary`` forwards/pair (+ baseline + joint).

Typical usage
─────────────
    conda run -n goals python scripts/clock/screen_oid_gaze.py \\
        --n-pairs 20 --cuda 4,5,6,7

    # cosine affinity, pin origin cond-0 -> target cond-2, more gaze heads:
    conda run -n goals python scripts/clock/screen_oid_gaze.py \\
        --normalize --orig-idx 0 --target-idx 2 --gaze-topn 500 --n-pairs 30

Writes ``screen.npz`` (per-pair deltaT + baselines), ``screen.png``, and ``meta.json`` under
``results/multistep_clock/<version>/<model>/oid_gaze/screen/<tag>/``.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.models import ModelandTokenizer  # noqa: E402
from src.utils import logging_utils  # noqa: E402
from src.goals.oid_gaze import (  # noqa: E402
    GazeScorer,
    add_patch_hooks,
    build_oid_pairs,
    cache_root,
    load_gaze_heads,
    replicate_inputs,
)

logger = logging.getLogger("screen_oid_gaze")


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
    p.add_argument("--layer-boundary", type=int, default=40, help="intervene <boundary, read >boundary")
    p.add_argument("--gaze-topn", type=int, default=100)
    p.add_argument("--gaze-q-span", type=int, default=15)
    p.add_argument("--gaze-compare-with", default="mean", choices=["mean", "max"])
    # affinity knobs
    p.add_argument("--normalize", action="store_true", help="cosine similarity (default: raw dot product)")
    p.add_argument("--include-rope", action="store_true", help="apply rotary to q,k before the dot")
    p.add_argument("--gaze-device", default="cuda:0")
    # screen
    p.add_argument("--n-pairs", type=int, default=150)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--out", default=None, help="output dir (default: results/.../oid_gaze/screen/<tag>)")
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
    assert pairs, "no valid oid pairs found"

    deltaT = np.full((len(pairs), args.layer_boundary, nH), np.nan, dtype=np.float32)
    base_hl = np.zeros(len(pairs), dtype=np.float32)
    joint_hl = np.zeros(len(pairs), dtype=np.float32)

    for pi, pair in enumerate(pairs):
        with torch.no_grad():
            T_base = scorer.compute_T(pair.dest_inputs, pair.spans)[0].cpu().numpy()
        bh = float(T_base[pair.target_idx] - T_base[pair.orig_idx])
        base_hl[pi] = bh

        dest_B = replicate_inputs(pair.dest_inputs, nH)  # one batch element per head
        for li, ly in enumerate(layers_below):
            diff = pair.src_hv[li] - pair.dest_hv[li]  # (n_embd,) source minus dest
            add = torch.zeros(nH, mt.n_embd, dtype=torch.float32)
            for b in range(nH):
                add[b, b * hd:(b + 1) * hd] = diff[b * hd:(b + 1) * hd]
            with torch.no_grad():
                T = scorer.compute_T(dest_B, pair.spans, pre_hooks=add_patch_hooks(mt, ly, add)).cpu().numpy()
            deltaT[pi, ly] = (T[:, pair.target_idx] - T[:, pair.orig_idx]) - bh

        joint_hooks = []
        for li, ly in enumerate(layers_below):
            joint_hooks += add_patch_hooks(mt, ly, (pair.src_hv[li] - pair.dest_hv[li]).unsqueeze(0))
        with torch.no_grad():
            Tj = scorer.compute_T(pair.dest_inputs, pair.spans, pre_hooks=joint_hooks)[0].cpu().numpy()
        joint_hl[pi] = float((Tj[pair.target_idx] - Tj[pair.orig_idx]) - bh)
        logger.info(
            f"[{pi + 1}/{len(pairs)}] {pair.sample}  base={bh:+.4f}  joint ΔT={joint_hl[pi]:+.4f}  "
            f"single-head max={np.nanmax(deltaT[pi]):+.4f}"
        )

    mean_dT = np.nanmean(deltaT, axis=0)  # (n_layers_below, nH)
    std_dT = np.nanstd(deltaT, axis=0)
    flat = [((ly, h), float(mean_dT[ly, h])) for ly in layers_below for h in range(nH) if not np.isnan(mean_dT[ly, h])]
    top = sorted(flat, key=lambda x: x[1], reverse=True)[: args.topk]

    logger.info(f"baseline swap-delta: {base_hl.mean():+.4f} ± {base_hl.std():.4f}")
    logger.info(f"joint (all <L{args.layer_boundary}) swap-delta: {joint_hl.mean():+.4f} ± {joint_hl.std():.4f}")
    logger.info(f"top-{args.topk} redirecting heads (mean ΔT over {len(pairs)} pairs):")
    for (ly, h), v in top:
        logger.info(f"  L{ly:2d} H{h:2d}   ΔT={v:+.4f} ± {std_dT[ly, h]:.4f}")

    # ── outputs ──
    tag = (
        f"orig{'any' if args.orig_idx is None else args.orig_idx}"
        f"_tgt{'any' if args.target_idx is None else args.target_idx}"
        f"_b{args.layer_boundary}_top{args.gaze_topn}"
        f"_{'cos' if args.normalize else 'dot'}{'_rope' if args.include_rope else ''}_n{len(pairs)}"
    )
    out_dir = args.out or os.path.join(cache_root(args.model, args.version), "oid_gaze", "screen", tag)
    os.makedirs(out_dir, exist_ok=True)

    np.savez_compressed(
        os.path.join(out_dir, "screen.npz"),
        deltaT=deltaT,
        base_hl=base_hl,
        joint_hl=joint_hl,
        layers_below=np.array(layers_below),
        gaze_heads=np.array(gaze_heads),
        samples=np.array([p.sample for p in pairs]),
    )
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(
            {
                **vars(args),
                "n_pairs_used": len(pairs),
                "n_gaze_heads": len(gaze_heads),
                "baseline_mean": float(base_hl.mean()),
                "joint_mean": float(joint_hl.mean()),
                "top_heads": [{"layer": ly, "head": h, "mean_dT": v} for (ly, h), v in top],
            },
            f,
            indent=2,
        )

    fig, (axh, axb) = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw={"width_ratios": [2.4, 1]})
    vlim = np.nanmax(np.abs(mean_dT)) or 1e-6
    im = axh.imshow(mean_dT.T, aspect="auto", cmap="RdBu_r", vmin=-vlim, vmax=vlim, origin="lower")
    axh.set_xlabel(f"source layer (<L{args.layer_boundary})")
    axh.set_ylabel("head")
    axh.set_title(f"mean ΔT(target-orig), single-head transplant  (N={len(pairs)})")
    fig.colorbar(im, ax=axh, shrink=0.8, label="mean ΔT (+ = redirect to target)")
    labels_k = [f"L{ly}H{h}" for (ly, h), _ in top]
    axb.barh(range(len(top)), [v for _, v in top], xerr=[std_dT[ly, h] for (ly, h), _ in top], color="tab:red")
    axb.set_yticks(range(len(top)))
    axb.set_yticklabels(labels_k, fontsize=8)
    axb.invert_yaxis()
    axb.axvline(0, color="k", lw=0.6)
    axb.axvline(joint_hl.mean(), color="tab:green", ls="--", lw=1, label=f"joint ({joint_hl.mean():+.3f})")
    axb.set_xlabel("mean ΔT")
    axb.set_title(f"top-{args.topk} heads")
    axb.legend(fontsize=7)
    fig.suptitle(f"oid gaze screen — {tag}", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "screen.png"), dpi=130, bbox_inches="tight")
    logger.info(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
