#!/usr/bin/env python
"""Jointly train a sparse binary mask over attention heads that *carry/calculate* the
multiline "completion" (stop-vs-continue) signal.

Hypothesis
──────────
At the target condition's pivot, some attention heads write a *completion* signal into the
``"    \\n"`` token (index ``-2``); the last token (``-1``) then reads it to decide whether to
**stay** on the current variable (condition incomplete → "continue") or **move on** to the next
variable (condition complete → "moved_on"). This script asks: which heads' OV (o_proj-input)
contributions at the patched position(s) carry that signal?

Setup (mirrors scripts/clock/train_oid_gaze_mask.py + scripts/train_head_mask.py)
────────────────────────────────────────────────────────────────────────────────
Each pair = (source = multiline/"continue", destination = singleline/"moved_on"). We transplant
the masked heads' per-head o_proj inputs source→dest at ``--patch-tokens`` with the projected-GD
recipe ``m*src + (1-m)*dest`` (mask=1 ⇒ take source, mask=0 ⇒ keep dest), and read the verdict at
the destination's last position as a logit contrast:

    score(dest) = logp[src_ans] - logp[dst_ans]          # push dest toward the source's verdict

where ``src_ans = argmax`` of the source's next token (the transplant-target verdict token) and
``dst_ans = argmax`` of the destination's own next token. Objective per pair:

    L = -score(dest_patched) + lambda_sparse * mean(mask)

``HeadMask`` (all-ones init, clamp ``[0,1]`` after each step, sparsity ``λ*mean(mask)``). Each
batch (``--batch-size`` pairs) tokenizes its sources+destinations TOGETHER, left-padded to one
shared length so every row's patch positions map to the same ``-2`` token from the right (no
positional confound); ``--accum`` batches per optimizer step.

Pairing (variable-name scheme is ``naming="letters"`` so verdicts are single tokens)
────────────────────────────────────────────────────────────────────────────────────
Per example: ``target_slot ∈ {1,2,3,4}`` (never 0 — need a prior so staying is in-context) and
``n_conditions ≥ target_slot + 2`` (need a *next* var for "moved_on").

  --same              source & dest share the SAME predicate list/spec (same seed + same
                      target_slot/n_conditions; only style differs) → minimal pair. When set,
                      --match-target-slot / --match-n-cond are ignored (implied).
  (default)           source & dest use DIFFERENT predicate lists; target_slot / n_conditions
                      are sampled independently unless pinned by:
  --match-target-slot force the same target_slot on both sides.
  --match-n-cond      force the same n_conditions on both sides.

Directions
──────────
  --direction ml->sl  train on ml→sl only (transplant continue into moved_on).  [default]
  --direction sl->ml  train on sl→ml only (transplant moved_on into continue).
  --direction both    train on both legs (symmetric).

Typical usage
─────────────
    conda run -n goals python scripts/clock/train_completion_mask.py \\
        --n-pairs 200 --n-epochs 20 --cuda 4,5,6,7

    conda run -n goals python scripts/clock/train_completion_mask.py \\
        --same --patch-tokens -2 --direction both --lambda-sparse 5e-2

Writes ``mask.pt`` (mask values + config + history + selected heads) and ``history.png`` under
``results/multistep_clock/<version>/<model>/completion_mask/<tag>/``.
"""

import argparse
import logging
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.optim import AdamW

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.models import ModelandTokenizer  # noqa: E402
from src.tokens import prepare_input  # noqa: E402
from src.functional import get_module_nnsight  # noqa: E402
from src.goals.head_loc_attn import HeadMask  # noqa: E402
from src.goals.multiline_prompts import (  # noqa: E402
    build_example,
    replace_in_code_newline_with_tabnewline,
)
from src.goals.oid_gaze import cache_root  # noqa: E402
from src.utils import logging_utils  # noqa: E402

logger = logging.getLogger("train_completion_mask")


# ──────────────────────────────────────────────────────────────────────────────
# Pair construction.
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class CompletionPair:
    """One (multiline=continue, singleline=moved_on) contrastive pair, token-aligned at the patch
    positions. Prompts are stored as strings; each training batch re-tokenizes its sources +
    destinations TOGETHER (left-padded to a shared length) so every row's patch positions map to
    the same ``-2`` token from the right (no positional confound)."""

    key: str
    ml_prompt: str
    sl_prompt: str
    ml_ans: int  # multiline argmax  → the "continue" verdict token
    sl_ans: int  # singleline argmax → the "moved_on" verdict token


def _build_prompt(mt, rng, target_slot, n_conditions, mode, style):
    ex = build_example(
        mt=mt, rng=rng, target_slot=target_slot, n_conditions=n_conditions,
        mode=mode, style=style, naming="letters",
    )
    return replace_in_code_newline_with_tabnewline(ex["full_prompt"]) + "   "


@torch.no_grad()
def _argmax_next(mt, inputs):
    out = mt._model(**dict(inputs), use_cache=False)
    return int(out.logits[0, -1, :].argmax())


@torch.no_grad()
def _extract_src_hv(mt, inputs, layers, patch_tokens):
    """Source o_proj inputs at ``patch_tokens`` → (B, len(layers), len(patch_tokens), n_embd), CPU."""
    saved: dict[int, torch.Tensor] = {}
    handles = []
    try:
        for li, L in enumerate(layers):
            op = get_module_nnsight(mt._model, mt.attn_module_name_format.format(L) + ".o_proj")

            def mk(li_):
                def pre(module, args):
                    x = args[0]  # (B, seq, n_embd)
                    saved[li_] = torch.stack(
                        [x[:, t, :] for t in patch_tokens], dim=1
                    ).detach().float().cpu()  # (B, len(patch_tok), n_embd)
                    return None
                return pre

            handles.append(op.register_forward_pre_hook(mk(li)))
        mt._model(**dict(inputs), use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return torch.stack([saved[li] for li in range(len(layers))], dim=1)  # (B, n_layers, P, n_embd)


def build_completion_pairs(mt, args, layers, patch_tokens):
    """Sample up to ``args.n_pairs`` validated pairs.

    Kept iff (a) ml & sl share the token at every patch position (token-aligned replace) and
    (b) their next-token argmaxes differ (a real continue↔moved_on contrast)."""
    target_slots = [int(x) for x in args.target_slots.split(",")]
    assert all(1 <= ts <= 4 for ts in target_slots), "target_slot must be in {1,2,3,4}"
    n_preds_max = 19  # len(MULTILINE_PREDS); n_conditions cannot exceed it

    def _sample_ncond(rng, ts):
        lo = ts + 2  # need a NEXT var for moved_on
        hi = min(max(lo, args.max_conditions), n_preds_max)
        return rng.randint(lo, hi)

    prng = random.Random(args.seed)
    pairs: list[CompletionPair] = []
    attempts = 0
    max_attempts = 50 * args.n_pairs
    while len(pairs) < args.n_pairs and attempts < max_attempts:
        attempts += 1
        ts = prng.choice(target_slots)
        nc = _sample_ncond(prng, ts)

        if args.same:
            seed = prng.randrange(10 ** 9)
            ml_prompt = _build_prompt(mt, random.Random(seed), ts, nc, args.mode, "multiline")
            sl_prompt = _build_prompt(mt, random.Random(seed), ts, nc, args.mode, "singleline")
        else:
            ts_sl = ts if args.match_target_slot else prng.choice(target_slots)
            nc_sl = nc if args.match_n_cond else _sample_ncond(prng, ts_sl)
            ml_prompt = _build_prompt(mt, random.Random(prng.randrange(10 ** 9)), ts, nc, args.mode, "multiline")
            sl_prompt = _build_prompt(mt, random.Random(prng.randrange(10 ** 9)), ts_sl, nc_sl, args.mode, "singleline")

        # gate with B=1 tokenization (argmax token ids are padding-invariant; the train-time
        # forward re-tokenizes sources+dests together)
        ml = prepare_input(ml_prompt, tokenizer=mt, padding_side="left")
        sl = prepare_input(sl_prompt, tokenizer=mt, padding_side="left")

        # (a) token-aligned at every patch position
        aligned = True
        for t in patch_tokens:
            a = mt.tokenizer.decode(ml["input_ids"][0, t].item())
            b = mt.tokenizer.decode(sl["input_ids"][0, t].item())
            if a != b:
                aligned = False
                break
        if not aligned:
            continue

        # (b) a real contrast
        ml_ans = _argmax_next(mt, ml)
        sl_ans = _argmax_next(mt, sl)
        if ml_ans == sl_ans:
            continue

        pairs.append(
            CompletionPair(
                key=f"p{len(pairs)}_ts{ts}_nc{nc}",
                ml_prompt=ml_prompt, sl_prompt=sl_prompt, ml_ans=ml_ans, sl_ans=sl_ans,
            )
        )
        if len(pairs) % 25 == 0:
            logger.info(f"  built {len(pairs)}/{args.n_pairs} pairs ({attempts} attempts)")
    logger.info(f"built {len(pairs)} pairs from {attempts} attempts")
    return pairs


# ──────────────────────────────────────────────────────────────────────────────
# Patch hooks + readout (generalised mask_patch_hooks: arbitrary patch positions).
# ──────────────────────────────────────────────────────────────────────────────


def completion_patch_hooks(mt, layers, src_hv, mask, n_q, head_dim, patch_tokens):
    """Mask-weighted per-head o_proj-input transplant at every ``patch_tokens`` position,
    broadcast over the batch: ``m*src + (1-m)*cascade``. ``src_hv``: (B, len(layers),
    len(patch_tokens), n_embd). Differentiable through ``mask.values``."""
    n_embd = n_q * head_dim
    hooks = []
    for li, L in enumerate(layers):
        op = get_module_nnsight(mt._model, mt.attn_module_name_format.format(L) + ".o_proj")

        def mk(li_):
            def pre(module, args):
                x = args[0].clone()
                dev, dt, B_ = x.device, x.dtype, x.shape[0]
                m = mask.values[li_].to(dev).view(1, n_q, 1)  # broadcast over batch
                for pj, t in enumerate(patch_tokens):
                    src3 = src_hv[:, li_, pj].to(dev).view(B_, n_q, head_dim)
                    cas3 = x[:, t, :].float().view(B_, n_q, head_dim)
                    x[:, t, :] = (m * src3 + (1 - m) * cas3).view(B_, n_embd).to(dt)
                return (x,) + tuple(args[1:])
            return pre

        hooks.append((op, mk(li)))
    return hooks


def completion_score(mt, dest_inputs, src_ans, dst_ans, pre_hooks):
    """Per-row logp[src_ans] - logp[dst_ans] at the destination's last position, under the patch
    hooks → (B,). Differentiable through any mask in ``pre_hooks``."""
    handles = [mod.register_forward_pre_hook(fn) for mod, fn in pre_hooks]
    try:
        out = mt._model(**dict(dest_inputs), use_cache=False)
    finally:
        for h in handles:
            h.remove()
    logp = out.logits[:, -1, :].float().log_softmax(-1)  # (B, vocab)
    rows = torch.arange(logp.shape[0], device=logp.device)
    sa = torch.as_tensor(src_ans, device=logp.device)
    da = torch.as_tensor(dst_ans, device=logp.device)
    return logp[rows, sa] - logp[rows, da]  # (B,)


def _slice_inputs(inputs, sl):
    return {k: (v[sl] if torch.is_tensor(v) else v) for k, v in inputs.items()}


def iter_leg_batches(pairs, batch_size, direction):
    """Yield (src_prompts, dst_prompts, src_ans, dst_ans) chunks of ≤ batch_size — one stream per
    leg implied by ``direction`` (ml→sl and/or sl→ml)."""
    legs = []
    if direction in ("ml->sl", "both"):
        legs.append("ml->sl")
    if direction in ("sl->ml", "both"):
        legs.append("sl->ml")
    for leg in legs:
        for i in range(0, len(pairs), batch_size):
            chunk = pairs[i:i + batch_size]
            if leg == "ml->sl":  # transplant continue → moved_on
                yield ([p.ml_prompt for p in chunk], [p.sl_prompt for p in chunk],
                       [p.ml_ans for p in chunk], [p.sl_ans for p in chunk])
            else:                # transplant moved_on → continue
                yield ([p.sl_prompt for p in chunk], [p.ml_prompt for p in chunk],
                       [p.sl_ans for p in chunk], [p.ml_ans for p in chunk])


# ──────────────────────────────────────────────────────────────────────────────


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
    logging_utils.configure(args)  # reads args.log_level (-v -> DEBUG)

    mt = ModelandTokenizer(
        model_key=args.model, dtype=torch.bfloat16, device_map="auto", attn_implementation="eager"
    )
    nH = mt.model.config.num_attention_heads
    hd = getattr(mt.model.config, "head_dim", mt.n_embd // nH)
    layer_hi = args.layer_hi if args.layer_hi is not None else mt.n_layer
    layers = list(range(args.layer_lo, layer_hi))
    patch_tokens = [int(x) for x in args.patch_tokens.split(",")]
    logger.info(f"layers {layers[0]}..{layers[-1]} ({len(layers)} masked) · patch_tokens={patch_tokens} "
                f"· direction={args.direction} · same={args.same} · batch_size={args.batch_size} · accum={args.accum}")

    pairs = build_completion_pairs(mt, args, layers, patch_tokens)
    assert len(pairs) >= 4, "need a few pairs to train"

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

    def batch_score(src_prompts, dst_prompts, src_ans, dst_ans):
        """Tokenize sources+dests TOGETHER (left-pad → shared length → aligned -2), extract the
        source o_proj inputs, then run the mask-patched forward on the dests → per-row score (B,)."""
        B = len(src_prompts)
        combined = prepare_input(prompts=src_prompts + dst_prompts, tokenizer=mt, padding_side="left")
        src_inputs = _slice_inputs(combined, slice(0, B))
        dst_inputs = _slice_inputs(combined, slice(B, 2 * B))
        src_hv = _extract_src_hv(mt, src_inputs, layers, patch_tokens)  # (B, n_layers, P, n_embd)
        hooks = completion_patch_hooks(mt, layers, src_hv, mask, nH, hd, patch_tokens)
        return completion_score(mt, dst_inputs, src_ans, dst_ans, hooks)

    rng = random.Random(args.seed)
    hist = {"train_score": [], "val_score": [], "mean_mask": [], "n_active": []}

    for epoch in range(args.n_epochs):
        order = train_pairs[:]
        rng.shuffle(order)
        batches = list(iter_leg_batches(order, args.batch_size, args.direction))
        opt.zero_grad()
        ep_score, n_rows = 0.0, 0
        for bi, (sp, dp, sa, da) in enumerate(batches):
            scores = batch_score(sp, dp, sa, da)  # (B,)
            target_loss = -scores.mean()
            sparsity_loss = args.lambda_sparse * mask.values.mean()
            loss = (target_loss + sparsity_loss) / args.accum
            loss.backward()
            logger.debug(
                f"epoch {epoch} batch {bi + 1}/{len(batches)} (B={scores.numel()}): "
                f"target_loss={float(target_loss):+.4f}  sparsity_loss={float(sparsity_loss):.4f}  "
                f"score={float(scores.mean().detach()):+.4f}"
            )
            ep_score += float(scores.sum().detach())
            n_rows += scores.numel()
            if (bi + 1) % args.accum == 0 or (bi + 1) == len(batches):
                opt.step()
                mask.clamp_()
                opt.zero_grad()

        with torch.no_grad():
            vsum, vn = 0.0, 0
            for (sp, dp, sa, da) in iter_leg_batches(val_pairs, args.batch_size, args.direction):
                s = batch_score(sp, dp, sa, da).detach()
                vsum += float(s.sum())
                vn += s.numel()
        hist["train_score"].append(ep_score / max(n_rows, 1))
        hist["val_score"].append(vsum / max(vn, 1))
        hist["mean_mask"].append(float(mask.values.mean()))
        hist["n_active"].append(mask.n_active(args.sel_threshold))
        logger.info(
            f"epoch {epoch:2d}  train score={hist['train_score'][-1]:+.4f}  "
            f"val score={hist['val_score'][-1]:+.4f}  mean(mask)={hist['mean_mask'][-1]:.3f}  "
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
        f"_lam{args.lambda_sparse}_n{len(pairs)}"
    )
    out_dir = args.out or os.path.join(cache_root(args.model, args.version), "completion_mask", tag)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(
        {
            "mask_values": mask.values.detach().cpu(),
            "layers": layers,
            "patch_tokens": patch_tokens,
            "decision_heads": decision_heads,
            "history": hist,
            "meta": {
                "model": args.model, "version": args.version, "mode": args.mode,
                "direction": args.direction, "same": args.same,
                "match_target_slot": args.match_target_slot, "match_n_cond": args.match_n_cond,
                "target_slots": args.target_slots, "max_conditions": args.max_conditions,
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
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("logp(src_ans) - logp(dst_ans)")
    ax[0].set_title("transplant score"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(ep, hist["n_active"], "^:", color="tab:gray")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel(f"n_active (>{args.sel_threshold})")
    ax[1].set_title("mask sparsity"); ax[1].grid(alpha=0.3)
    fig.suptitle(tag)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "history.png"), dpi=120)
    logger.info(f"saved mask.pt + history.png → {out_dir}")


if __name__ == "__main__":
    main()
