#!/usr/bin/env python
"""Does the locked gaze head bank *selectively* look at the predicate being implemented?

Pure observation — **no intervention**.  For every sample and every condition ``k`` with a
findable ``=`` pivot (``<var_k> =``) and a non-empty instruction-key span, we cut the generation
right after that pivot and read the raw Q·K gaze affinity (on the ``k=topn`` locked late-layer
gaze heads) from the pivot query to **each** instruction span.  We call the pivot a *success* iff
its own instruction span wins the argmax:

    gaze(span_k) > max( gaze(span_j) for j != k with a non-empty span )

i.e. while writing code for condition ``k`` the gaze bank looks at instruction ``k`` more than at
any other instruction.  We report the overall hit-rate, the per-condition-index hit-rate, and the
margin distribution ``gaze(span_k) - max_{j!=k} gaze(span_j)``.

Typical usage
─────────────
    conda run -n goals python scripts/clock/eval_gaze_selectivity.py \\
        --n-gaze-heads 100 --cuda 4,5,6,7

    conda run -n goals python scripts/clock/eval_gaze_selectivity.py \\
        --n-gaze-heads 100 --normalize --max-samples 200

Writes ``selectivity.npz`` (per-pivot gaze matrix + win flags), ``selectivity.png``, and
``meta.json`` under ``results/multistep_clock/<version>/<model>/oid_gaze/selectivity/<tag>/``.
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
from src.tokens import prepare_input  # noqa: E402
from src.utils import logging_utils  # noqa: E402
from src.goals.oid_gaze import (  # noqa: E402
    GazeScorer,
    PIVOT_VARS,
    build_prefill,
    cache_root,
    load_gaze_heads,
    _key_idx,
)

logger = logging.getLogger("eval_gaze_selectivity")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # model / data
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    p.add_argument("--version", default="conditions")
    p.add_argument("--modularity", default="monolithic", choices=["monolithic", "modular"])
    p.add_argument("--cuda", default=None, help="CUDA_VISIBLE_DEVICES (e.g. '4,5,6,7')")
    p.add_argument("--max-samples", type=int, default=None, help="cap #samples (default: all)")
    # gaze readout heads
    p.add_argument("--layer-boundary", type=int, default=40, help="read gaze heads in layers > boundary")
    p.add_argument("--n-gaze-heads", type=int, default=100, help="k: top-k late-layer gaze heads")
    p.add_argument("--gaze-topn", type=int, default=500, help="candidate pool before the layer filter")
    p.add_argument("--gaze-q-span", type=int, default=15)
    p.add_argument("--gaze-compare-with", default="mean", choices=["mean", "max"])
    # affinity knobs
    p.add_argument("--normalize", action="store_true", help="cosine similarity (default: raw dot product)")
    p.add_argument("--include-rope", action="store_true", help="apply rotary to q,k before the dot")
    p.add_argument("--gaze-device", default="cuda:0")
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

    gaze_heads = load_gaze_heads(
        args.model, args.version, args.gaze_q_span, args.gaze_compare_with,
        args.gaze_topn, args.layer_boundary,
    )[: args.n_gaze_heads]
    logger.info(f"using {len(gaze_heads)} gaze readout heads (L>{args.layer_boundary})")
    scorer = GazeScorer(
        mt, gaze_heads, normalize=args.normalize, include_rope=args.include_rope, device=args.gaze_device
    )

    croot = cache_root(args.model, args.version)
    mdir = os.path.join(croot, args.modularity)
    samples = sorted(os.listdir(mdir))
    if args.max_samples:
        samples = samples[: args.max_samples]

    MAXC = 8  # condition slots tracked for the per-index breakdown
    rows = []  # (sample, k, n_cand, win, margin, gaze_k, gaze_best_other, winner, T_padded)
    per_idx_hit = np.zeros(MAXC, dtype=np.int64)
    per_idx_tot = np.zeros(MAXC, dtype=np.int64)
    n_ambiguous = 0  # pivots dropped because own span overlaps a competitor span

    for s in samples:
        kr_path = os.path.join(mdir, s, "key_ranges.json")
        if not (os.path.exists(kr_path) and os.path.exists(os.path.join(mdir, s, "labels.json"))):
            continue
        kr = json.load(open(kr_path))["step_key_token_ranges"]
        n_cond = len(kr)

        for k in range(n_cond):
            if not (kr[k] and kr[k][0]):
                continue
            try:
                prefill = build_prefill(croot, args.modularity, s, PIVOT_VARS[k])
            except Exception:
                continue
            di = prepare_input(prompts=[prefill], tokenizer=mt)
            seq = di["input_ids"].shape[1]
            if int(di["attention_mask"].sum()) != seq:  # B=1 must not pad
                continue
            spans = [_key_idx(kr[i], seq) for i in range(n_cond)]
            if spans[k].size == 0:
                continue
            cand = [i for i in range(n_cond) if spans[i].size > 0]
            if len(cand) < 2:  # need ≥1 competitor for the argmax to be meaningful
                continue
            # drop ambiguous pivots: own span must be disjoint from every competitor span
            own = set(spans[k].tolist())
            if any(own & set(spans[j].tolist()) for j in cand if j != k):
                n_ambiguous += 1
                continue

            with torch.no_grad():
                T = scorer.compute_T(di, spans)[0].cpu().numpy()  # (n_cond,)
            cand_others = [j for j in cand if j != k]
            best_other = max(cand_others, key=lambda j: T[j])  # index of the strongest competitor
            margin = float(T[k] - T[best_other])
            win = margin > 0
            winner = int(cand[int(np.argmax([T[j] for j in cand]))])  # who the gaze actually pointed at
            T_pad = np.full(MAXC, np.nan, dtype=np.float32)
            T_pad[: min(n_cond, MAXC)] = T[: min(n_cond, MAXC)]
            rows.append((s, k, len(cand), bool(win), margin, float(T[k]), float(T[best_other]), winner, T_pad))
            if k < MAXC:
                per_idx_tot[k] += 1
                per_idx_hit[k] += int(win)
        if len(rows) and len(rows) % 50 == 0:
            hr = np.mean([r[3] for r in rows])
            logger.info(f"...{len(rows)} pivots scored  running hit-rate={hr:.3f}")

    assert rows, "no scorable pivots found"
    wins = np.array([r[3] for r in rows], dtype=bool)
    margins = np.array([r[4] for r in rows], dtype=np.float32)
    n = len(rows)
    hit_rate = float(wins.mean())
    # baseline: a random instruction span winning = mean(1/n_cand)
    chance = float(np.mean([1.0 / r[2] for r in rows]))

    logger.info("=" * 64)
    logger.info(f"pivots scored: {n}  over {len({r[0] for r in rows})} samples  "
                f"({n_ambiguous} dropped: own span overlapped a competitor)")
    logger.info(f"SELECTIVITY hit-rate: {hit_rate:.3f}   (chance ≈ {chance:.3f})")
    logger.info(f"margin gaze(k)-max_other: mean={margins.mean():+.4f}  median={np.median(margins):+.4f}")
    logger.info("per-condition-index hit-rate:")
    for k in range(MAXC):
        if per_idx_tot[k]:
            logger.info(f"  cond{k}: {per_idx_hit[k]}/{per_idx_tot[k]} = {per_idx_hit[k] / per_idx_tot[k]:.3f}")

    # ── outputs ──
    tag = (
        f"k{len(gaze_heads)}_b{args.layer_boundary}"
        f"_{'cos' if args.normalize else 'dot'}{'_rope' if args.include_rope else ''}_n{n}"
    )
    out_dir = args.out or os.path.join(croot, "oid_gaze", "selectivity", tag)
    os.makedirs(out_dir, exist_ok=True)

    np.savez_compressed(
        os.path.join(out_dir, "selectivity.npz"),
        samples=np.array([r[0] for r in rows]),
        cond_idx=np.array([r[1] for r in rows]),
        n_cand=np.array([r[2] for r in rows]),
        win=wins,
        margin=margins,
        gaze_k=np.array([r[5] for r in rows], dtype=np.float32),
        gaze_best_other=np.array([r[6] for r in rows], dtype=np.float32),
        winner_idx=np.array([r[7] for r in rows], dtype=np.int64),
        gaze_T=np.stack([r[8] for r in rows]),  # (n, MAXC) padded with nan
        gaze_heads=np.array(gaze_heads),
        per_idx_hit=per_idx_hit,
        per_idx_tot=per_idx_tot,
    )
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(
            {
                **vars(args),
                "n_gaze_heads_used": len(gaze_heads),
                "n_pivots": n,
                "n_ambiguous_dropped": n_ambiguous,
                "hit_rate": hit_rate,
                "chance": chance,
                "margin_mean": float(margins.mean()),
                "margin_median": float(np.median(margins)),
                "per_idx": {
                    int(k): {"hit": int(per_idx_hit[k]), "tot": int(per_idx_tot[k])}
                    for k in range(MAXC) if per_idx_tot[k]
                },
            },
            f,
            indent=2,
        )

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.6))
    axes[0].axhline(chance, color="k", ls="--", lw=1, label=f"chance ≈ {chance:.3f}")
    ks = [k for k in range(MAXC) if per_idx_tot[k]]
    axes[0].bar(ks, [per_idx_hit[k] / per_idx_tot[k] for k in ks], color="tab:blue")
    axes[0].axhline(hit_rate, color="tab:red", lw=1.2, label=f"overall {hit_rate:.3f}")
    axes[0].set_xlabel("condition index k"); axes[0].set_ylabel("hit-rate")
    axes[0].set_ylim(0, 1); axes[0].set_title("selectivity by condition"); axes[0].legend(fontsize=8)
    for k in ks:
        axes[0].text(k, per_idx_hit[k] / per_idx_tot[k] + 0.02, str(per_idx_tot[k]), ha="center", fontsize=7)

    axes[1].hist(margins, bins=40, color="tab:purple", alpha=0.8)
    axes[1].axvline(0, color="k", lw=0.8)
    axes[1].set_xlabel("gaze(k) - max_other"); axes[1].set_ylabel("# pivots")
    axes[1].set_title(f"margin (mean {margins.mean():+.3f})")

    gk = np.array([r[5] for r in rows]); go = np.array([r[6] for r in rows])
    axes[2].scatter(go, gk, s=8, alpha=0.4, c=np.where(wins, "tab:green", "tab:gray"))
    lim = [min(go.min(), gk.min()), max(go.max(), gk.max())]
    axes[2].plot(lim, lim, "k--", lw=0.8)
    axes[2].set_xlabel("best other-span gaze"); axes[2].set_ylabel("own-span gaze")
    axes[2].set_title(f"own vs best-other  ({wins.sum()}/{n} above diag)")
    fig.suptitle(f"oid gaze selectivity — {tag}", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "selectivity.png"), dpi=130, bbox_inches="tight")
    logger.info(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
