#!/usr/bin/env python
"""Gaze-objective variant of ``train_completion_mask.py``.

Same transplant setup — the SAME contrastive (source, dest) completion pairs, the SAME per-head
o_proj-input transplant ``m*src + (1-m)*dest`` at ``--patch-tokens`` (default ``-2``, the
``"    \\n"`` pivot), the SAME ``HeadMask`` (all-ones init, clamp ``[0,1]``, sparsity
``λ*mean(mask)``), the SAME legs (ml→sl / sl→ml) and ``--same`` minimal-pair option.

The ONLY difference is the readout. Instead of the verdict-logit contrast
``logp[target_ans] - logp[dst_ans]``, we maximise a **gaze contrast** — how much the decision token
(last position, ``-1``) attends to the *target* condition's spec span versus the *orig* condition's
spec span, measured on a fixed set of top-``--n-gaze`` "step-lookback" gaze heads:

    score(dest) = gaze(T_target) - gaze(T_orig)

The transplant pushes the dest toward its FLIPPED verdict, and the verdict ⟺ which condition the
model gazes at (current vs. next variable). So per leg (dst clean style in parens):

  ml→sl  (clean = moved-on)  flip → STAY    : target span = condition[target_slot]
                                              orig   span = condition[target_slot + 1]
  sl→ml  (clean = staying)   flip → MOVE-ON : target span = condition[target_slot + 1]
                                              orig   span = condition[target_slot]

Gaze is read at ``-1`` (NOT the patched ``-2``: some heads' output AT ``-2`` is replaced by the
transplant, which would make a gaze read there deceptive). The gaze affinity itself is the raw
Q·K dot product on the gaze heads' own ``q_proj``/``k_proj`` (``GazeScorer.compute_T_masked``),
computed from the residual latent the mask-patched forward produces — so it is differentiable
through the mask. Objective per pair:

    L = -(gaze(T_target) - gaze(T_orig)) + lambda_sparse * mean(mask)

Gaze heads are loaded from disk (``head_scores/qspan-<q>_cmp-<c>/aggregated.npz``, written by
``scripts/clock/score_condition_heads.py``) — so that scoring must already have been run for this
model/version.

Condition spans are localised from each context's ``ConditionSpec`` (``spec_texts`` + ``id_substrs``,
the same machinery the attention viz uses), bounded to each condition's instruction region — no
oracle. Spans are precomputed once per dest prompt and offset per-row to the left-padded batch.

Writes ``mask.pt`` + ``history.png`` under
``results/multistep_clock/<version>/<model>/completion_mask_gaze/<tag>/``.

Typical usage
─────────────
    conda run -n goals python scripts/clock/train_completion_mask_gaze.py \\
        --same --direction both --n-gaze 100 --lambda-sparse 5e-2 --cuda 4,5,6,7
"""

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch.optim import AdamW

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.attention import _condition_token_spans  # noqa: E402  (span localiser, shared w/ viz)
from src.models import ModelandTokenizer  # noqa: E402
from src.tokens import prepare_input  # noqa: E402
from src.goals.completion_pairs import (  # noqa: E402
    LEG_STYLES,
    build_completion_pairs,
    legs_for,
)
from src.goals.head_loc_attn import HeadMask  # noqa: E402
from src.goals.oid_gaze import GazeScorer, cache_root  # noqa: E402
from src.utils import logging_utils  # noqa: E402

# reuse the transplant machinery verbatim from the logit-objective trainer
from scripts.clock.train_completion_mask import (  # noqa: E402
    _extract_src_hv,
    _slice_inputs,
    completion_patch_hooks,
)
from scripts.clock.score_condition_heads import load_head_scores  # noqa: E402

logger = logging.getLogger("train_completion_mask_gaze")


# ──────────────────────────────────────────────────────────────────────────────
# Gaze span bookkeeping.
# ──────────────────────────────────────────────────────────────────────────────


def _leg_cond_indices(leg: str, target_slot: int) -> tuple[int, int]:
    """(target_cond, orig_cond) condition indices for a leg. The transplant flips the dest's
    verdict; the flipped verdict's condition is the gaze TARGET, the clean one is ORIG."""
    if leg == "ml->sl":   # clean = moved-on (next); flip → stay (current)
        return target_slot, target_slot + 1
    return target_slot + 1, target_slot  # sl->ml: clean = stay; flip → move-on (next)


def precompute_gaze_spans(mt, pairs, direction):
    """Map each dest prompt string → (target_idx, orig_idx) RELATIVE token-index arrays (within the
    dest's own, un-padded tokenization). Keyed by prompt string, which already disambiguates the leg
    (the two legs render the dest in different styles → different strings)."""
    span_map: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    n_empty = 0
    for leg in legs_for(direction):
        ds = LEG_STYLES[leg]["dst_style"]
        for p in pairs:
            dest = p.dest
            prompt = dest.prompt(ds)
            if prompt in span_map:
                continue
            tgt_c, org_c = _leg_cond_indices(leg, int(dest.target_slot))
            ids = prepare_input(prompts=[prompt], tokenizer=mt).input_ids[0]
            tok_strs = [mt.tokenizer.decode([int(i)]) for i in ids]
            spans = _condition_token_spans(tok_strs, dest.spec.spec_texts, dest.spec.id_substrs)
            tgt = np.asarray(spans[tgt_c] if tgt_c < len(spans) else [], dtype=np.int64)
            org = np.asarray(spans[org_c] if org_c < len(spans) else [], dtype=np.int64)
            if tgt.size == 0 or org.size == 0:
                n_empty += 1
            span_map[prompt] = (tgt, org)
    if n_empty:
        logger.warning(f"{n_empty} dest prompt(s) had an empty target/orig span (skipped at train time)")
    return span_map


# ──────────────────────────────────────────────────────────────────────────────


def iter_leg_batches_gaze(pairs, batch_size, direction):
    """Like ``completion_pairs.iter_leg_batches`` but yields ``(src_prompts, dst_prompts, leg)`` so
    the caller can look up each dest's precomputed gaze spans (keyed by dst prompt string)."""
    for leg in legs_for(direction):
        ss = LEG_STYLES[leg]["src_style"]
        ds = LEG_STYLES[leg]["dst_style"]
        for i in range(0, len(pairs), batch_size):
            chunk = pairs[i : i + batch_size]
            yield (
                [p.source.prompt(ss) for p in chunk],
                [p.dest.prompt(ds) for p in chunk],
                leg,
            )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # model / data
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    p.add_argument("--version", default="conditions")
    p.add_argument("--mode", default="chat", choices=["chat", "fim"])
    p.add_argument("--cuda", default=None, help="CUDA_VISIBLE_DEVICES (e.g. '4,5,6,7')")
    # patch / mask geometry
    p.add_argument("--patch-tokens", default="-2",
                   help="comma-separated token indices to transplant (e.g. '-2' or '-2,-1')")
    p.add_argument("--layer-lo", type=int, default=0, help="first layer included in the mask")
    p.add_argument("--layer-hi", type=int, default=None, help="one past the last layer (default: n_layer)")
    p.add_argument("--direction", default="both", choices=["ml->sl", "sl->ml", "both"])
    # gaze heads (loaded from disk) + affinity
    p.add_argument("--n-gaze", type=int, default=100, help="top-K step-lookback gaze heads to read gaze on")
    p.add_argument("--gaze-q-span", type=int, default=15, help="q_span the head scores were computed with")
    p.add_argument("--gaze-compare-with", default="mean", choices=["mean", "max"],
                   help="compare_with the head scores were computed with")
    p.add_argument("--gaze-normalize", action="store_true", help="cosine gaze affinity instead of raw Q·K")
    p.add_argument("--gaze-rope", action="store_true", help="apply rotary to q/k before the gaze dot")
    # pairing
    p.add_argument("--same", action="store_true",
                   help="source & dest share the same predicate spec (minimal pair); ignores the two --match flags")
    p.add_argument("--match-target-slot", action="store_true", help="force matched target_slot (ignored if --same)")
    p.add_argument("--match-n-cond", action="store_true", help="force matched n_conditions (ignored if --same)")
    p.add_argument("--target-slots", default="1,2,3,4", help="comma-separated target_slot choices (each in {1,2,3,4})")
    p.add_argument("--max-conditions", type=int, default=6, help="upper bound when sampling n_conditions")
    # training
    p.add_argument("--n-pairs", type=int, default=600)
    p.add_argument("--n-epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=5e-2)
    p.add_argument("--lambda-sparse", type=float, default=2e-1)
    p.add_argument("--batch-size", type=int, default=32,
                   help="pairs per forward; sources+dests are tokenized together (left-padded to a shared length)")
    p.add_argument("--accum", type=int, default=1, help="batches per optimizer step (grad accumulation)")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--sel-threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mask-device", default="cuda:0")
    p.add_argument("--out", default=None)
    logging_utils.add_logging_args(p)  # -v -> DEBUG, -q -> WARNING (default INFO)
    return p.parse_args()


def main():
    args = parse_args()
    if args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    logging_utils.configure(args)

    mt = ModelandTokenizer(
        model_key=args.model, dtype=torch.bfloat16, device_map="auto", attn_implementation="eager"
    )
    nH = mt.model.config.num_attention_heads
    hd = getattr(mt.model.config, "head_dim", mt.n_embd // nH)
    layer_hi = args.layer_hi if args.layer_hi is not None else mt.n_layer
    layers = list(range(args.layer_lo, layer_hi))
    patch_tokens = [int(x) for x in args.patch_tokens.split(",")]

    # ── gaze heads from disk (top-n_gaze step-lookback heads) ──
    try:
        scores = load_head_scores(
            model=args.model, version=args.version,
            q_span=args.gaze_q_span, compare_with=args.gaze_compare_with,
        )["combined"]
    except FileNotFoundError as e:
        raise SystemExit(
            f"gaze head scores not found — run scripts/clock/score_condition_heads.py first "
            f"(--q_span {args.gaze_q_span} --compare_with {args.gaze_compare_with}).\n{e}"
        )
    flat = np.argsort(scores, axis=None)[::-1][: args.n_gaze]
    gaze_heads = [(int(l), int(h)) for l, h in (np.unravel_index(i, scores.shape) for i in flat)]
    logger.info(
        f"layers {layers[0]}..{layers[-1]} ({len(layers)} masked) · patch_tokens={patch_tokens} "
        f"· direction={args.direction} · same={args.same} · n_gaze={len(gaze_heads)} "
        f"(qspan={args.gaze_q_span}, cmp={args.gaze_compare_with}) · batch_size={args.batch_size}"
    )

    gaze_scorer = GazeScorer(
        mt, gaze_heads, normalize=args.gaze_normalize, include_rope=args.gaze_rope,
        device=args.mask_device,
    )

    pairs = build_completion_pairs(
        mt,
        n_pairs=args.n_pairs,
        target_slots=[int(x) for x in args.target_slots.split(",")],
        direction=args.direction,
        same=args.same,
        match_target_slot=args.match_target_slot,
        match_n_cond=args.match_n_cond,
        max_conditions=args.max_conditions,
        mode=args.mode,
        seed=args.seed,
        patch_tokens=patch_tokens,
        log=logger.info,
    )
    assert len(pairs) >= 4, "need a few pairs to train"
    span_map = precompute_gaze_spans(mt, pairs, args.direction)

    # pair-disjoint train/val split
    prng = random.Random(args.seed)
    prng.shuffle(pairs)
    n_val = max(1, int(args.val_frac * len(pairs)))
    val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]
    assert train_pairs, "all pairs landed in val; lower --val-frac or raise --n-pairs"
    logger.info(f"train {len(train_pairs)} / val {len(val_pairs)} pairs")

    for prm in mt._model.parameters():
        prm.requires_grad_(False)
    mask = HeadMask(len(layers), nH).to(args.mask_device)
    opt = AdamW(mask.parameters(), lr=args.lr)

    def batch_score(src_prompts, dst_prompts):
        """Tokenize sources+dests together (left-pad → aligned -2), transplant the masked source
        o_proj inputs, then read gaze(T_target) - gaze(T_orig) at the dest's last position. Returns
        ``(score (B,), valid (B,) bool)`` — invalid rows have an empty target/orig span."""
        B = len(src_prompts)
        combined = prepare_input(prompts=src_prompts + dst_prompts, tokenizer=mt, padding_side="left")
        src_inputs = _slice_inputs(combined, slice(0, B))
        dst_inputs = _slice_inputs(combined, slice(B, 2 * B))
        src_hv = _extract_src_hv(mt, src_inputs, layers, patch_tokens)
        hooks = completion_patch_hooks(mt, layers, src_hv, mask, nH, hd, patch_tokens)

        seq = dst_inputs["input_ids"].shape[1]
        am = dst_inputs["attention_mask"]
        key_masks = torch.zeros(2, B, seq)            # [0]=target, [1]=orig
        valid = torch.zeros(B, dtype=torch.bool)
        for r in range(B):
            off = seq - int(am[r].sum())              # left-pad offset for this row
            tgt, org = span_map[dst_prompts[r]]
            if tgt.size and org.size:
                valid[r] = True
                for k in tgt:
                    if 0 <= off + k < seq:
                        key_masks[0, r, off + k] = 1.0
                for k in org:
                    if 0 <= off + k < seq:
                        key_masks[1, r, off + k] = 1.0
        T = gaze_scorer.compute_T_masked(dst_inputs, key_masks, pre_hooks=hooks)  # (B, 2)
        return T[:, 0] - T[:, 1], valid

    rng = random.Random(args.seed)
    hist = {"train_score": [], "val_score": [], "mean_mask": [], "n_active": []}

    for epoch in range(args.n_epochs):
        order = train_pairs[:]
        rng.shuffle(order)
        batches = list(iter_leg_batches_gaze(order, args.batch_size, args.direction))
        opt.zero_grad()
        ep_score, n_rows = 0.0, 0
        for bi, (sp, dp, _leg) in enumerate(batches):
            scores, valid = batch_score(sp, dp)
            vmask = valid.to(scores.device)
            if not bool(vmask.any()):
                continue
            sv = scores[vmask]
            target_loss = -sv.mean()
            sparsity_loss = args.lambda_sparse * mask.values.mean()
            loss = (target_loss + sparsity_loss) / args.accum
            loss.backward()
            logger.debug(
                f"epoch {epoch} batch {bi + 1}/{len(batches)} (B={int(vmask.sum())}/{scores.numel()}): "
                f"target_loss={float(target_loss):+.4f}  sparsity_loss={float(sparsity_loss):.4f}  "
                f"gaze_contrast={float(sv.mean().detach()):+.4f}"
            )
            ep_score += float(sv.sum().detach())
            n_rows += int(vmask.sum())
            if (bi + 1) % args.accum == 0 or (bi + 1) == len(batches):
                opt.step()
                mask.clamp_()
                opt.zero_grad()

        with torch.no_grad():
            vsum, vn = 0.0, 0
            for (sp, dp, _leg) in iter_leg_batches_gaze(val_pairs, args.batch_size, args.direction):
                s, valid = batch_score(sp, dp)
                vm = valid.to(s.device)
                if bool(vm.any()):
                    vsum += float(s[vm].sum())
                    vn += int(vm.sum())
        hist["train_score"].append(ep_score / max(n_rows, 1))
        hist["val_score"].append(vsum / max(vn, 1))
        hist["mean_mask"].append(float(mask.values.mean()))
        hist["n_active"].append(mask.n_active(args.sel_threshold))
        logger.info(
            f"epoch {epoch:2d}  train gaze={hist['train_score'][-1]:+.4f}  "
            f"val gaze={hist['val_score'][-1]:+.4f}  mean(mask)={hist['mean_mask'][-1]:.3f}  "
            f"n_active(>{args.sel_threshold})={hist['n_active'][-1]}"
        )

    decision_heads = mask.selected_heads(layers, threshold=args.sel_threshold)
    mv = mask.values.detach().cpu().numpy()
    ranked = sorted(decision_heads, key=lambda lh: mv[layers.index(lh[0]), lh[1]], reverse=True)
    logger.info(f"selected {len(decision_heads)} decision heads (mask>{args.sel_threshold}); top:")
    for (l, h) in ranked[:20]:
        logger.info(f"  L{l:2d} H{h:2d}  mask={mv[layers.index(l), h]:.3f}")

    # ── outputs ──
    pt_tag = "_".join(str(t) for t in patch_tokens)
    dir_tag = args.direction.replace("->", "2")  # filesystem-safe ("ml->sl" -> "ml2sl")
    tag = (
        f"{'same' if args.same else 'diff'}"
        f"{'_mts' if (not args.same and args.match_target_slot) else ''}"
        f"{'_mnc' if (not args.same and args.match_n_cond) else ''}"
        f"_pt{pt_tag}_{dir_tag}_L{args.layer_lo}-{layer_hi}"
        f"_gaze{args.n_gaze}_lam{args.lambda_sparse}_n{len(pairs)}"
    )
    out_dir = args.out or os.path.join(cache_root(args.model, args.version), "completion_mask_gaze", tag)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(
        {
            "mask_values": mask.values.detach().cpu(),
            "layers": layers,
            "patch_tokens": patch_tokens,
            "decision_heads": decision_heads,
            "gaze_heads": gaze_heads,
            "history": hist,
            "meta": {
                "objective": "gaze_contrast",  # gaze(target) - gaze(orig) at -1
                "model": args.model, "version": args.version, "mode": args.mode,
                "direction": args.direction, "same": args.same,
                "match_target_slot": args.match_target_slot, "match_n_cond": args.match_n_cond,
                "target_slots": args.target_slots, "max_conditions": args.max_conditions,
                "n_gaze": args.n_gaze, "gaze_q_span": args.gaze_q_span,
                "gaze_compare_with": args.gaze_compare_with,
                "gaze_normalize": args.gaze_normalize, "gaze_rope": args.gaze_rope,
                "n_pairs": len(pairs), "n_epochs": args.n_epochs, "lr": args.lr,
                "batch_size": args.batch_size, "accum": args.accum,
                "lambda_sparse": args.lambda_sparse, "sel_threshold": args.sel_threshold,
            },
        },
        os.path.join(out_dir, "mask.pt"),
    )

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ep = range(len(hist["train_score"]))
    ax[0].plot(ep, hist["train_score"], "o-", label="train")
    ax[0].plot(ep, hist["val_score"], "s--", label="val")
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("gaze(target) - gaze(orig)  @ -1")
    ax[0].set_title("transplant gaze contrast"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(ep, hist["n_active"], "^:", color="tab:gray")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel(f"n_active (>{args.sel_threshold})")
    ax[1].set_title("mask sparsity"); ax[1].grid(alpha=0.3)
    fig.suptitle(tag)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "history.png"), dpi=120)
    logger.info(f"saved mask.pt + history.png → {out_dir}")


if __name__ == "__main__":
    main()
