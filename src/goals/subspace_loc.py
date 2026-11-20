"""Subspace localization: the head-localization objective, with a gated residual subspace instead of a head mask.

Same data, target and loss as :func:`src.goals.head_loc_attn.train_head_mask` (the goal-head localization):
oracle-verified pivot pairs, a source prompt (C++ / another language) about to implement sub-task A, a Python
destination prompt about to implement a different sub-task B, and the target = the LM's next-token distribution
on a Python prompt whose next sub-task is A. At the destination's pivot (its last token) we patch, at every layer
``l`` of a band (cascading),

    h_dst <- h_dst + W_lᵀ diag(g_l) W_l (h_src - h_dst)

i.e. an interchange of the residual's component in a learned subspace (``W_l``: ``r`` orthonormal rows) whose
directions are gated by ``g_l ∈ [0, 1]^r``. Loss = KL(P_patched ‖ P_target) on the next token + a sparsity price on
the gates. ``r`` is only a ceiling: the gates decide how many directions per layer are used (the learned rank).

The sparsity price is the same per open direction for every ``r``: ``λ · Σ g / (L · PRICE_UNIT)`` with a fixed
``PRICE_UNIT = 128`` (one attention head's width). With ``mean(g)`` the price of a direction would scale with
``1 / r``, so the learned rank would depend on the ceiling.
"""

from __future__ import annotations

import logging
import os
import random
from contextlib import contextmanager
from dataclasses import dataclass, field

import baukit
import torch
import torch.nn.functional as F
from torch import nn

from src.functional import free_gpu_cache
from src.goals.head_loc_attn import PivotCache, augment_pivot_caches
from src.tokens import prepare_input
from src.utils.metrics import recall as recall_metric

logger = logging.getLogger(__name__)

PRICE_UNIT = 128  # fixed normaliser of the sparsity price (one head's width), independent of r


class GatedSubspaceMask(nn.Module):
    """``W ∈ R^{L×r×d}`` (orthonormal rows per layer) and gates ``g ∈ [0,1]^{L×r}`` (init 1 = all directions open)."""

    def __init__(self, layers: list[int], d_model: int, r: int, seed: int = 0) -> None:
        super().__init__()
        self.layers = list(layers)
        gen = torch.Generator().manual_seed(seed)
        W = torch.stack([torch.linalg.qr(torch.randn(d_model, r, generator=gen))[0].T for _ in self.layers])
        self.W = nn.Parameter(W.float())                              # (L, r, d)
        self.g = nn.Parameter(torch.ones(len(self.layers), r))        # (L, r)

    @property
    def r(self) -> int:
        return self.W.shape[1]

    def sparsity(self) -> torch.Tensor:
        """λ-free sparsity price: Σ g / (L · PRICE_UNIT) — the same per open direction for every r."""
        return self.g.sum() / (len(self.layers) * PRICE_UNIT)

    @torch.no_grad()
    def project_(self) -> None:
        """After each optimizer step: gates back into [0, 1], each W_l back onto orthonormal rows (QR)."""
        self.g.clamp_(0.0, 1.0)
        for k in range(self.W.shape[0]):
            Q, _ = torch.linalg.qr(self.W[k].T)
            self.W[k].copy_(Q.T)

    @torch.no_grad()
    def n_active(self, thr: float = 0.5) -> list[int]:
        """Open directions per layer (gate > thr): the learned rank."""
        return (self.g > thr).sum(dim=1).tolist()

    def delta(self, k: int, diff: torch.Tensor) -> torch.Tensor:
        """``W_kᵀ diag(g_k) W_k diff`` for ``diff`` (B, d) float32."""
        W, g = self.W[k].to(diff.device), self.g[k].to(diff.device)
        return ((diff @ W.T) * g) @ W


def _layer_module(mt, layer: int):
    return baukit.get_module(mt._model, mt.layer_name_format.format(layer))


@torch.inference_mode()
def residuals_at_last_token(mt, prompts: list[str], layers: list[int]) -> torch.Tensor:
    """Each prompt's residual (the decoder layer's OUTPUT) at its last token, per layer: (B, L, d) float32 CPU.
    ``prepare_input`` left-pads, so the last token is position -1 for every row."""
    out = {}
    hs = []
    for l in layers:
        def hook(mod, inp, o, l=l):
            h = o[0] if isinstance(o, tuple) else o
            out[l] = h[:, -1].float().cpu()
        hs.append(_layer_module(mt, l).register_forward_hook(hook))
    try:
        mt._model(**prepare_input(prompts, tokenizer=mt))
    finally:
        for h in hs:
            h.remove()
    return torch.stack([out[l] for l in layers], dim=1)


@contextmanager
def interchange_hooks(mt, sm: GatedSubspaceMask, src: torch.Tensor):
    """Patch the last token at every layer of ``sm.layers``: h <- h + W_lᵀ diag(g_l) W_l (h_src - h). Cascades
    (each layer sees the earlier layers' patches). ``src``: (B, L, d). Differentiable in ``sm.W`` and ``sm.g``."""
    hs = []
    for k, l in enumerate(sm.layers):
        def hook(mod, inp, o, k=k):
            h = o[0] if isinstance(o, tuple) else o
            last = h[:, -1].float()
            new = last + sm.delta(k, src[:, k].to(h.device) - last)
            h = torch.cat([h[:, :-1], new.to(h.dtype).unsqueeze(1)], dim=1)
            return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
        hs.append(_layer_module(mt, l).register_forward_hook(hook))
    try:
        yield
    finally:
        for h in hs:
            h.remove()


def forward_with_interchange(mt, dest_prompts: list[str], src: torch.Tensor, sm: GatedSubspaceMask) -> torch.Tensor:
    """Destination forward with the gated interchange at the pivot: next-token logits (B, V) float32."""
    with interchange_hooks(mt, sm, src):
        logits = mt._model(**prepare_input(dest_prompts, tokenizer=mt)).logits[:, -1].float()
    return logits


@dataclass
class SubspaceTrainingHistory:
    loss_kl: list[float] = field(default_factory=list)
    loss_sparse: list[float] = field(default_factory=list)
    n_active: list[list[int]] = field(default_factory=list)   # per epoch: open directions per layer
    val_recall: list[list[float]] = field(default_factory=list)
    val_epochs: list[int] = field(default_factory=list)
    baseline_closed: list[float] = field(default_factory=list)      # val recall with no patch
    baseline_open_init: list[float] = field(default_factory=list)   # val recall with the random init, all gates open


@torch.no_grad()
def evaluate_val(mt, val_caches: list[PivotCache], sm: GatedSubspaceMask, *, batch_size: int, k: int = 5,
                 aug_rng=None, aug_cache_batch_size: int = 64) -> list[float]:
    """recall@1..k of the patched destination's top tokens against ``target_top_token`` (as in the head mask)."""
    if aug_rng is not None:
        val_caches = augment_pivot_caches(mt, val_caches, cache_batch_size=aug_cache_batch_size, rng=aug_rng)
    preds, targets = [], []
    for i in range(0, len(val_caches), batch_size):
        grp = val_caches[i:i + batch_size]
        src = residuals_at_last_token(mt, [c.src_prompt for c in grp], sm.layers)
        logits = forward_with_interchange(mt, [c.dest_prompt for c in grp], src, sm)
        for row, c in zip(logits.topk(k).indices.tolist(), grp):
            preds.append([mt.tokenizer.decode([t]) for t in row])
            targets.append(c.pivot_instance.target_top_token)
    return recall_metric(preds, targets)


def train_subspace_mask(mt, pivot_caches: list[PivotCache], layers: list[int], *, r: int, n_epochs: int,
                        lr_W: float, lr_g: float, lambda_sparse: float, batch_size: int, checkpoint_every: int,
                        checkpoint_dir: str | None, augment: bool = True, augment_seed: int | None = None,
                        augment_cache_batch_size: int = 64, include_docstring: bool = True, seed: int = 0
                        ) -> tuple[GatedSubspaceMask, SubspaceTrainingHistory]:
    """Projected gradient descent on (W, g): L = KL(P_patched ‖ P_target) + λ · Σ g / (L · PRICE_UNIT).
    Mirrors ``train_head_mask``'s augment path: an 80/20 train/val split, every instance re-augmented at the start
    of each epoch (src residuals recomputed per batch), val recall + a checkpoint every ``checkpoint_every`` epochs."""
    for p in mt._model.parameters():
        p.requires_grad_(False)
    sm = GatedSubspaceMask(layers, mt.n_embd, r, seed=seed).to(mt.device)
    opt = torch.optim.AdamW([{"params": [sm.W], "lr": lr_W}, {"params": [sm.g], "lr": lr_g}])
    history = SubspaceTrainingHistory()
    rng = random.Random(augment_seed)

    val_size = max(1, int(len(pivot_caches) * 0.2))
    train_caches, val_caches = pivot_caches[: len(pivot_caches) - val_size], pivot_caches[len(pivot_caches) - val_size:]
    logger.info(f"train_subspace_mask: layers {layers[0]}-{layers[-1]} ({len(layers)}) × r={r} (≤{len(layers) * r} "
                f"directions) | train={len(train_caches)} val={val_size} | batch={batch_size} | {n_epochs} epochs | "
                f"lr W={lr_W} g={lr_g} | λ={lambda_sparse} (price per open direction λ/{len(layers) * PRICE_UNIT}) | "
                f"augment={augment}")

    # reference points for the val recall, before any training: no patch (all gates closed = the clean destination)
    # and the random initial subspace with every gate open
    with torch.no_grad():
        g0 = sm.g.detach().clone()
        sm.g.zero_()
        history.baseline_closed = evaluate_val(mt, val_caches, sm, batch_size=batch_size)
        sm.g.copy_(g0)
        history.baseline_open_init = evaluate_val(mt, val_caches, sm, batch_size=batch_size)
    logger.info("  val recall, gates closed (clean dest): "
                + "  ".join(f"@{k + 1}={v:.3f}" for k, v in enumerate(history.baseline_closed)))
    logger.info("  val recall, random init, gates open:    "
                + "  ".join(f"@{k + 1}={v:.3f}" for k, v in enumerate(history.baseline_open_init)))

    for epoch in range(n_epochs):
        caches = (augment_pivot_caches(mt, train_caches, include_docstring=include_docstring,
                                       cache_batch_size=augment_cache_batch_size, rng=rng)
                  if augment else train_caches)
        tot_kl = tot_sp = 0.0
        n = 0
        for i in range(0, len(caches), batch_size):
            grp = caches[i:i + batch_size]
            src = residuals_at_last_token(mt, [c.src_prompt for c in grp], layers)
            log_p_target = torch.stack([c.log_p_target for c in grp])
            opt.zero_grad()
            logits = forward_with_interchange(mt, [c.dest_prompt for c in grp], src, sm)
            log_p = F.log_softmax(logits, dim=-1)
            # KL(P_patched ‖ P_target), batchmean — exactly the head-mask loss
            kl = F.kl_div(log_p_target.to(log_p.device), log_p, reduction="batchmean", log_target=True)
            sp = lambda_sparse * sm.sparsity()
            (kl + sp.to(kl.device)).backward()
            opt.step()
            sm.project_()
            tot_kl += kl.item(); tot_sp += sp.item(); n += 1
            del logits, log_p, kl
        free_gpu_cache()
        act = sm.n_active()
        history.loss_kl.append(tot_kl / n); history.loss_sparse.append(tot_sp / n); history.n_active.append(act)
        logger.info(f"  epoch {epoch + 1:3d}/{n_epochs}: KL={tot_kl / n:.4f} sparse={tot_sp / n:.4f} "
                    f"open directions={sum(act)} per layer={act}")

        if checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
            rec = evaluate_val(mt, val_caches, sm, batch_size=batch_size,
                               aug_rng=rng if augment else None, aug_cache_batch_size=augment_cache_batch_size)
            history.val_recall.append(rec); history.val_epochs.append(epoch + 1)
            logger.info("  val recall: " + "  ".join(f"@{k + 1}={v:.3f}" for k, v in enumerate(rec)))
            if checkpoint_dir is not None:
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(dict(epoch=epoch + 1, W=sm.W.detach().cpu(), g=sm.g.detach().cpu(), layers=layers,
                                history=history.__dict__),
                           os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1:04d}.pt"))
    return sm, history
