"""Learned sparse head mask for attention head localization.

Learns a continuous mask over (n_layers_tested, n_heads) attention heads whose
cross-language interchange (C++ source → Python destination) minimizes
KL(P_patched ∥ P_target).

Based on arxiv:2510.26784 (filter heads via differentiable causal masking),
adapted for attention output patching and the cross-language interchange design
from claude_reports/brainstorm_attention_head_localization.md.

Pipeline:
    1. Cache o_proj inputs for source (C++) and destination (Python) prompts.
    2. Precompute per-head analytical deltas: W_o_h @ (src_h - dest_h).
    3. Train mask: differentiable patched forward pass via baukit eager hooks.
       attn_out[l][:,-1,:] += Σ_h mask[l,h] * delta[l,h]
    4. Post-training threshold (mask > 0.5) → selected heads.

Mask parameterization (from paper):
    - nn.Parameter initialized to all ones
    - Clamped to [0,1] after each optimizer.step() (box-constrained GD)
    - No sigmoid, no temperature — raw values as interpolation weights
    - Selection: heads where values > 0.5 after training
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import baukit
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW

from src.functional import get_module_nnsight
from src.goals.head_loc_data import HeadLocDataset, PredicatePair, PivotInstance
from src.goals.head_loc_pivot import (
    build_cpp_prompt,
    build_python_prompt,
    get_next_token_probs,
)
from src.tokens import prepare_input

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Learnable mask
# ---------------------------------------------------------------------------


class HeadMask(nn.Module):
    """Real-valued mask over (n_layers_tested, n_heads) attention heads.

    Initialized to all ones so every head is active at the start.  L1 sparsity
    pressure drives unimportant heads to zero; the task loss keeps causally
    important heads high.  Box constraint [0, 1] enforced by clamping after
    each optimizer step (no sigmoid — raw values used as interpolation weights).
    """

    def __init__(self, n_layers_tested: int, n_heads: int) -> None:
        super().__init__()
        self.values = nn.Parameter(torch.ones(n_layers_tested, n_heads))

    def clamp_(self) -> None:
        """Enforce [0, 1] box constraint. Call after every optimizer.step()."""
        with torch.no_grad():
            self.values.clamp_(0.0, 1.0)

    def selected_heads(
        self, layers: list[int], threshold: float = 0.5
    ) -> list[tuple[int, int]]:
        """Return (layer, head) pairs where mask value > threshold."""
        vals = self.values.detach()
        return [
            (layers[li], h)
            for li in range(vals.shape[0])
            for h in range(vals.shape[1])
            if vals[li, h].item() > threshold
        ]

    def n_active(self, threshold: float = 0.5) -> int:
        return int((self.values.detach() > threshold).sum().item())


# ---------------------------------------------------------------------------
# Utility: o_proj weight
# ---------------------------------------------------------------------------


def _get_o_proj_weight(mt, layer: int) -> torch.Tensor:
    """Return o_proj weight (n_embd, n_heads * head_dim) as float32 CPU tensor."""
    path = mt.attn_module_name_format.format(layer) + ".o_proj"
    module = get_module_nnsight(mt._model, path)
    return module.weight.detach().cpu().float()


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------


@torch.inference_mode()
def get_o_proj_inputs(mt, prompt: str, layers: list[int]) -> dict[int, torch.Tensor]:
    """Extract o_proj input at last token for each layer in a single trace.

    Returns {layer: (n_embd,) float32 CPU tensor}.
    """
    inputs = prepare_input(prompts=[prompt], tokenizer=mt)
    saves: dict[int, Any] = {}

    with mt.trace(inputs, scan=False, validate=False):
        for layer in layers:
            attn_name = mt.attn_module_name_format.format(layer)
            o_proj = get_module_nnsight(mt, attn_name + ".o_proj")
            # o_proj.input is the tensor directly: (batch, seq, n_embd)
            saves[layer] = o_proj.input[0, -1, :].save()

    return {layer: saves[layer].float().cpu() for layer in layers}


@torch.no_grad()
def precompute_per_head_deltas(
    mt,
    src_o_proj_inputs: dict[int, torch.Tensor],
    dest_o_proj_inputs: dict[int, torch.Tensor],
    layers: list[int],
) -> torch.Tensor:
    """Compute W_o_h @ (src_h - dest_h) for every (layer, head) pair.

    Returns float32 CPU tensor of shape (n_layers_tested, n_heads, n_embd).
    This is precomputed once per (src, dest) pair and reused at every training
    step — the mask then linearly weights these fixed deltas.
    """
    n_heads = mt.model.config.num_attention_heads
    head_dim = mt.n_embd // n_heads
    n_layers_tested = len(layers)

    deltas = torch.zeros(n_layers_tested, n_heads, mt.n_embd)

    for li, layer in enumerate(layers):
        W_o = _get_o_proj_weight(mt, layer)  # (n_embd, n_embd)
        diff = src_o_proj_inputs[layer] - dest_o_proj_inputs[layer]  # (n_embd,)
        for h in range(n_heads):
            W_o_h = W_o[:, h * head_dim : (h + 1) * head_dim]  # (n_embd, head_dim)
            diff_h = diff[h * head_dim : (h + 1) * head_dim]  # (head_dim,)
            deltas[li, h] = W_o_h @ diff_h  # (n_embd,)

    return deltas


# ---------------------------------------------------------------------------
# Cached data per pivot instance
# ---------------------------------------------------------------------------


@dataclass
class PivotCache:
    """Pre-computed tensors for one PivotInstance.  Cached once, reused every epoch."""

    pivot_instance: PivotInstance
    dest_inputs: Any  # TokenizerOutput for the dest prompt
    log_p_target: torch.Tensor  # (vocab_size,) float32 CPU — log of target dist
    per_head_deltas: torch.Tensor  # (n_layers_tested, n_heads, n_embd) float32 CPU


@torch.inference_mode()
def cache_pivot_data(
    mt,
    pivot_instances: list[tuple[PredicatePair, PivotInstance]],
    ds: HeadLocDataset,
    layers: list[int],
    include_docstring: bool = False,
) -> list[PivotCache]:
    """Cache all data needed for mask training.

    Heavy upfront cost (2 forward passes per instance); training steps are then
    cheap (one patched forward pass + backward through the mask only).
    """
    caches: list[PivotCache] = []

    for i, (pair, pi) in enumerate(pivot_instances):
        logger.info(
            f"  Caching {i + 1}/{len(pivot_instances)}: "
            f"{pair.destination_predicate} ← {pair.source_predicate} "
            f"prefix={pi.pivot_prefix!r}"
        )

        src_sample = ds.get_sample(pair.source_predicate)
        dest_sample = ds.get_sample(pair.destination_predicate)
        targ_sample = ds.get_sample(pair.source_predicate)  # target == source

        for name, sample in [
            (pair.source_predicate, src_sample),
            (pair.destination_predicate, dest_sample),
        ]:
            if sample is None:
                raise ValueError(f"Sample not found: {name!r}")

        src_prompt = build_cpp_prompt(
            mt, src_sample, include_docstring=include_docstring
        )
        dest_prompt = build_python_prompt(
            mt,
            dest_sample,
            pivot_prefix=pi.pivot_prefix,
            include_docstring=include_docstring,
        )
        targ_prompt = build_python_prompt(
            mt,
            targ_sample,
            pivot_prefix=pi.pivot_prefix,
            include_docstring=include_docstring,
        )

        src_o_proj = get_o_proj_inputs(mt, src_prompt, layers)
        dest_o_proj = get_o_proj_inputs(mt, dest_prompt, layers)
        per_head_deltas = precompute_per_head_deltas(
            mt, src_o_proj, dest_o_proj, layers
        )

        p_target = get_next_token_probs(mt, targ_prompt)
        log_p_target = torch.log(p_target.clamp(min=1e-10))

        dest_inputs = prepare_input(prompts=[dest_prompt], tokenizer=mt)

        caches.append(
            PivotCache(
                pivot_instance=pi,
                dest_inputs=dest_inputs,
                log_p_target=log_p_target,
                per_head_deltas=per_head_deltas,
            )
        )

    logger.info(f"cache_pivot_data: cached {len(caches)} pivot instances")
    return caches


# ---------------------------------------------------------------------------
# Differentiable patched forward pass
# ---------------------------------------------------------------------------


def forward_with_masked_patch(
    mt,
    dest_inputs: Any,
    per_head_deltas: torch.Tensor,
    mask: HeadMask,
    layers: list[int],
) -> torch.Tensor:
    """Run dest_inputs with mask-weighted head patching via baukit eager hooks.

    For each layer l (index li in layers):
        attn_out[l][:, -1, :] += Σ_h mask.values[li, h] * per_head_deltas[li, h]

    Differentiable w.r.t. mask.values — gradients flow through the linear
    combination of precomputed deltas.

    Args:
        mt: ModelandTokenizer (frozen).
        dest_inputs: Tokenized destination prompt.
        per_head_deltas: (n_layers_tested, n_heads, n_embd) float32 CPU.
        mask: HeadMask with learnable values (requires_grad=True).
        layers: Layer indices corresponding to per_head_deltas axis 0.

    Returns:
        (vocab_size,) float32 logits, differentiable w.r.t. mask.values.
    """
    device = mt.device
    attn_names = [mt.attn_module_name_format.format(layer) for layer in layers]
    layer_to_li = {name: li for li, name in enumerate(attn_names)}
    mask_vals = mask.values  # (n_layers_tested, n_heads)
    deltas = per_head_deltas.to(device)  # (n_layers_tested, n_heads, n_embd)

    def edit_fn(output, layer_name):
        li = layer_to_li[layer_name]
        is_tuple = isinstance(output, tuple)
        attn_out = output[0] if is_tuple else output  # (batch, seq, n_embd)
        device = attn_out.device
        dtype = attn_out.dtype

        # (n_heads,) @ (n_heads, n_embd) = (n_embd,) — differentiable w.r.t. mask_vals
        delta = mask_vals[li].to(dtype).to(device) @ deltas[li].to(dtype).to(device)

        # Non-in-place addition at last token position only
        batch, seq, _ = attn_out.shape
        last_pos = torch.zeros(batch, seq, 1, device=device, dtype=dtype)
        last_pos[:, -1, :] = 1.0
        new_attn_out = attn_out + last_pos * delta.to(attn_out.dtype).view(1, 1, -1)

        return (new_attn_out,) + output[1:] if is_tuple else new_attn_out

    with baukit.TraceDict(
        module=mt._model,
        layers=attn_names,
        edit_output=edit_fn,
    ):
        output = mt._model(**dest_inputs)

    return output.logits[0, -1, :].float()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass
class MaskTrainingHistory:
    loss_causal: list[float] = field(default_factory=list)
    loss_sparse: list[float] = field(default_factory=list)
    loss_total: list[float] = field(default_factory=list)
    n_active: list[int] = field(default_factory=list)


def train_head_mask(
    mt,
    pivot_caches: list[PivotCache],
    layers: list[int],
    n_epochs: int = 20,
    lr: float = 1e-2,
    lambda_sparse: float = 1e-2,
) -> tuple[HeadMask, MaskTrainingHistory]:
    """Learn a sparse head mask via projected gradient descent.

    Objective:
        L = KL(P_patched ∥ P_target) + lambda_sparse * Σ mask.values

    KL direction: KL(P_patched ∥ P_target) = Σ P_patched * log(P_patched/P_target)
    — pushes the patched destination distribution toward the Python target.

    Args:
        mt: ModelandTokenizer (frozen — only mask params are updated).
        pivot_caches: Pre-cached data from cache_pivot_data().
        layers: Layer indices (must match pivot_caches[*].per_head_deltas axis 0).
        n_epochs: Training epochs (default 20, from paper).
        lr: AdamW learning rate (default 1e-2, from paper).
        lambda_sparse: L1 sparsity coefficient (default 1e-2, from paper).

    Returns:
        (trained HeadMask, MaskTrainingHistory)
    """
    n_layers_tested = len(layers)
    n_heads = mt.model.config.num_attention_heads

    mask = HeadMask(n_layers_tested, n_heads)
    optimizer = AdamW(mask.parameters(), lr=lr)
    history = MaskTrainingHistory()

    # Freeze model parameters so backward() doesn't accumulate grads for them
    for param in mt._model.parameters():
        param.requires_grad_(False)

    logger.info(
        f"train_head_mask: {n_layers_tested}L × {n_heads}H = "
        f"{n_layers_tested * n_heads} mask params | "
        f"{len(pivot_caches)} instances | {n_epochs} epochs | "
        f"lr={lr} λ={lambda_sparse}"
    )

    for epoch in range(n_epochs):
        epoch_kl = 0.0
        epoch_sparse = 0.0

        for cache in pivot_caches:
            optimizer.zero_grad()

            logits = forward_with_masked_patch(
                mt, cache.dest_inputs, cache.per_head_deltas, mask, layers
            )

            # KL(P_patched ∥ P_target):
            # F.kl_div(log_p_target, log_p_patched, log_target=True)
            # = Σ exp(log_p_patched) * (log_p_patched - log_p_target)
            # = Σ p_patched * log(p_patched / p_target)  ✓
            log_p_patched = F.log_softmax(logits, dim=-1)
            loss_kl = F.kl_div(
                cache.log_p_target.to(logits.device),
                log_p_patched,
                reduction="sum",
                log_target=True,
            )

            loss_sparse = lambda_sparse * mask.values.sum()
            loss = loss_kl + loss_sparse

            loss.backward()
            optimizer.step()
            mask.clamp_()

            epoch_kl += loss_kl.item()
            epoch_sparse += loss_sparse.item()

        n = len(pivot_caches)
        avg_kl = epoch_kl / n
        avg_sparse = epoch_sparse / n
        avg_total = avg_kl + avg_sparse
        n_act = mask.n_active()

        history.loss_causal.append(avg_kl)
        history.loss_sparse.append(avg_sparse)
        history.loss_total.append(avg_total)
        history.n_active.append(n_act)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"  Epoch {epoch + 1:3d}/{n_epochs}: "
                f"KL={avg_kl:.4f}  sparse={avg_sparse:.4f}  "
                f"total={avg_total:.4f}  n_active={n_act}"
            )

    selected = mask.selected_heads(layers)
    logger.info(f"train_head_mask done: {len(selected)} active heads → {selected}")
    return mask, history
