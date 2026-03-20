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
       Sparsity: lambda * mean(mask) — scale-invariant to number of layers tested.
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

from src.functional import free_gpu_cache, get_module_nnsight
from src.goals.head_loc_augment import augment_pair, augment_source
from src.goals.augment import _VAR_VOCAB
from src.goals.head_loc_data import HeadLocDataset, PivotInstance, PredicatePair
from src.goals.head_loc_pivot import (
    build_cpp_prompt,
    build_python_prompt,
    get_next_token_probs_batch,
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
def get_o_proj_inputs_batch(
    mt, prompts: list[str], layers: list[int]
) -> dict[int, torch.Tensor]:
    """Extract o_proj input at last token for each layer, for a batch of prompts.

    Left-padding (from prepare_input) ensures ``[:, -1, :]`` is the last real
    token for every sequence in the batch.

    Returns:
        ``{layer: (batch, n_embd) float32 CPU tensor}``
    """
    inputs = prepare_input(prompts=prompts, tokenizer=mt)
    saves: dict[int, Any] = {}

    with mt.trace(inputs, scan=False, validate=False):
        for layer in layers:
            attn_name = mt.attn_module_name_format.format(layer)
            o_proj = get_module_nnsight(mt, attn_name + ".o_proj")
            saves[layer] = o_proj.input[:, -1, :].save()  # (batch, n_embd)

    return {layer: saves[layer].float().cpu() for layer in layers}


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
def precompute_per_head_deltas_batch(
    mt,
    src_o_proj_inputs: dict[int, torch.Tensor],
    dest_o_proj_inputs: dict[int, torch.Tensor],
    layers: list[int],
) -> torch.Tensor:
    """Compute per-head deltas for a batch of (src, dest) pairs.

    Args:
        src_o_proj_inputs:  ``{layer: (batch, n_embd)}`` from get_o_proj_inputs_batch.
        dest_o_proj_inputs: ``{layer: (batch, n_embd)}`` from get_o_proj_inputs_batch.
        layers: Layer indices.

    Returns:
        Float32 CPU tensor of shape ``(batch, n_layers_tested, n_heads, n_embd)``.
    """
    n_heads = mt.model.config.num_attention_heads
    head_dim = mt.n_embd // n_heads
    batch = src_o_proj_inputs[layers[0]].shape[0]

    deltas = torch.zeros(batch, len(layers), n_heads, mt.n_embd)

    for li, layer in enumerate(layers):
        W_o = _get_o_proj_weight(mt, layer)  # (n_embd, n_embd)
        diff = src_o_proj_inputs[layer] - dest_o_proj_inputs[layer]  # (batch, n_embd)
        for h in range(n_heads):
            W_o_h = W_o[:, h * head_dim : (h + 1) * head_dim]  # (n_embd, head_dim)
            diff_h = diff[:, h * head_dim : (h + 1) * head_dim]  # (batch, head_dim)
            deltas[:, li, h] = diff_h @ W_o_h.T  # (batch, n_embd)

    return deltas


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
    """Pre-computed tensors for one PivotInstance.  Cached once, reused every epoch.

    ``per_head_deltas`` may be ``None`` when ``cache_pivot_data`` is called with
    ``precompute_deltas=False``.  In that case ``src_prompt`` is populated and
    ``train_head_mask`` recomputes deltas lazily per batch (saves ~5 GB CPU RAM
    for N=100, at the cost of 2 extra forward passes per optimizer step per epoch).
    """

    pivot_instance: PivotInstance
    dest_prompt: str  # raw string — batched with src_prompt for o_proj extraction
    log_p_target: torch.Tensor  # (vocab_size,) float32 CPU — log of target dist
    per_head_deltas: torch.Tensor | None  # (n_layers_tested, n_heads, n_embd) or None
    src_prompt: str | None = None  # populated when precompute_deltas=False


@torch.inference_mode()
def cache_pivot_data(
    mt,
    pivot_instances: list[tuple[PredicatePair, PivotInstance]],
    ds: HeadLocDataset,
    layers: list[int],
    include_docstring: bool = False,
    precompute_deltas: bool = True,
    cache_batch_size: int = 64,
) -> list[PivotCache]:
    """Cache all data needed for mask training.

    Builds all prompts first, then runs forward passes in chunks of
    ``cache_batch_size`` to bound peak GPU activation memory.

    **``precompute_deltas=True``** (default — fastest training, most memory):
      - src + dest o_proj: chunked combined passes (src and dest in same chunk
        so they share sequence length → no positional shift in the delta).
      - target next-token probs: chunked pass.
      - CPU RAM: ~5 GB for N=100, Qwen-32B.

    **``precompute_deltas=False``** (memory-efficient — lazily recomputed in training):
      - Only target next-token probs are cached (chunked).
      - ``src_prompt`` is stored; deltas are recomputed per batch per epoch inside
        ``train_head_mask`` (2 extra forward passes per optimizer step).
      - CPU RAM: negligible (only log_p_target per instance).

    Args:
        cache_batch_size: Max prompts per forward pass during caching.  Each
            targ-prob chunk runs ``cache_batch_size`` prompts; each delta chunk
            runs ``2 × cache_batch_size`` (src + dest interleaved).  Default 64
            keeps peak GPU activation memory to ~8 GB on Qwen-32B (A100 safe).
    """
    src_prompts: list[str] = []
    dest_prompts: list[str] = []
    targ_prompts: list[str] = []

    for pair, pi in pivot_instances:
        src_sample = ds.get_sample(pair.source_predicate)
        dest_sample = ds.get_sample(pair.destination_predicate)
        targ_sample = ds.get_sample(pair.source_predicate)  # target == source

        for name, sample in [
            (pair.source_predicate, src_sample),
            (pair.destination_predicate, dest_sample),
        ]:
            if sample is None:
                raise ValueError(f"Sample not found: {name!r}")

        src_prompts.append(
            build_cpp_prompt(mt, src_sample, include_docstring=include_docstring)
        )
        dest_prompts.append(
            build_python_prompt(
                mt,
                dest_sample,
                pivot_prefix=pi.pivot_prefix,
                include_docstring=include_docstring,
            )
        )
        targ_prompts.append(
            build_python_prompt(
                mt,
                targ_sample,
                pivot_prefix=pi.pivot_prefix,
                include_docstring=include_docstring,
            )
        )

    N = len(pivot_instances)

    logger.info(
        f"cache_pivot_data: {N} instances | precompute_deltas={precompute_deltas} | "
        f"cache_batch_size={cache_batch_size}"
    )

    # Target next-token log-probs — always cached, processed in chunks.
    log_p_chunks: list[torch.Tensor] = []
    for i in range(0, N, cache_batch_size):
        chunk = targ_prompts[i : i + cache_batch_size]
        p = get_next_token_probs_batch(mt, chunk)  # (chunk, vocab) CPU
        log_p_chunks.append(torch.log(p.clamp(min=1e-10)))
        free_gpu_cache()
        logger.debug(f"  targ probs: {min(i + cache_batch_size, N)}/{N}")
    log_p_target = torch.cat(log_p_chunks, dim=0)  # (N, vocab) CPU float32

    # Per-head deltas — optionally precomputed, also chunked.
    all_deltas: torch.Tensor | None = None
    if precompute_deltas:
        delta_chunks: list[torch.Tensor] = []
        for i in range(0, N, cache_batch_size):
            src_chunk = src_prompts[i : i + cache_batch_size]
            dest_chunk = dest_prompts[i : i + cache_batch_size]
            B = len(src_chunk)
            # Combine src+dest so they share sequence length → no positional shift.
            combined_o = get_o_proj_inputs_batch(
                mt, src_chunk + dest_chunk, layers
            )  # {layer: (2B, n_embd)}
            src_o = {layer: combined_o[layer][:B] for layer in layers}
            dest_o = {layer: combined_o[layer][B:] for layer in layers}
            delta_chunks.append(
                precompute_per_head_deltas_batch(mt, src_o, dest_o, layers)
            )  # (B, L, H, E)
            del combined_o, src_o, dest_o
            free_gpu_cache()
            logger.debug(f"  deltas: {min(i + cache_batch_size, N)}/{N}")
        all_deltas = torch.cat(delta_chunks, dim=0)  # (N, L, H, E)
        logger.info("cache_pivot_data: deltas precomputed and cached")

    caches: list[PivotCache] = []
    for i, (pair, pi) in enumerate(pivot_instances):
        logger.debug(
            f"  Building cache {i + 1}/{N}: "
            f"{pair.destination_predicate} ← {pair.source_predicate} "
            f"prefix={pi.pivot_prefix!r}"
        )
        caches.append(
            PivotCache(
                pivot_instance=pi,
                dest_prompt=dest_prompts[i],
                log_p_target=log_p_target[i],
                per_head_deltas=all_deltas[i] if all_deltas is not None else None,
                src_prompt=src_prompts[i] if not precompute_deltas else None,
            )
        )

    logger.info(f"cache_pivot_data: cached {len(caches)} pivot instances")
    return caches


@torch.inference_mode()
def augment_pivot_caches(
    mt,
    pivot_caches: list[PivotCache],
    pivot_instances: list[tuple[PredicatePair, PivotInstance]],
    ds: HeadLocDataset,
    layers: list[int],
    include_docstring: bool = False,
    cache_batch_size: int = 64,
) -> list[PivotCache]:
    """Generate semantics-preserving augmented variants of existing pivot caches.

    Applies surface-form augmentations that are guaranteed not to change the pivot
    position or invalidate the target token, then recomputes prompts, log_p_target,
    and per_head_deltas for each variant.

    Safe augmentations applied:

    ``toggle_type_hints`` (all instances)
        Strips parameter type annotations from the Python ``def`` line of dest and
        target.  The C++ src is also toggled (Python-side only).  Never touches
        the ``return <pivot_prefix>`` line, so the pivot position is unchanged.

    ``variable_rename`` (pivot_prefix == "" only)
        Renames the variable throughout src, dest, and target using the same new
        name, so the pivot prefix tokens stay identical across dest and target.
        Skipped for non-empty pivot prefixes because the prefix may contain the
        old variable name.

    The returned caches always have ``per_head_deltas`` precomputed (not lazy),
    regardless of whether the originals are lazy.  If you mix augmented caches
    with lazy originals in ``train_head_mask``, convert all originals to eager
    first by re-running ``cache_pivot_data`` with ``precompute_deltas=True``.

    Args:
        mt: ModelandTokenizer (frozen).
        pivot_caches: Original caches from ``cache_pivot_data()``.
        pivot_instances: Corresponding ``(PredicatePair, PivotInstance)`` list;
            must have the same length and order as ``pivot_caches``.
        ds: HeadLocDataset for sample look-up.
        layers: Layer indices for delta precomputation.
        include_docstring: Passed to prompt builders.
        cache_batch_size: Max prompts per forward pass (bounds GPU activation
            memory the same way as in ``cache_pivot_data``).

    Returns:
        List of additional ``PivotCache`` objects (augmented variants only —
        originals are not repeated).
    """
    assert len(pivot_caches) == len(pivot_instances), (
        "pivot_caches and pivot_instances must have the same length"
    )

    aug_src_prompts:  list[str]           = []
    aug_dest_prompts: list[str]           = []
    aug_targ_prompts: list[str]           = []
    aug_pis:          list[PivotInstance] = []

    for _cache, (pair, pi) in zip(pivot_caches, pivot_instances):
        src_sample  = ds.get_sample(pair.source_predicate)
        dest_sample = ds.get_sample(pair.destination_predicate)
        targ_sample = ds.get_sample(pair.source_predicate)  # target == source

        if src_sample is None or dest_sample is None or targ_sample is None:
            logger.warning(
                f"augment_pivot_caches: missing sample for "
                f"{pair.source_predicate} → {pair.destination_predicate} — skipping"
            )
            continue

        def _add(aug_src, aug_dest, aug_targ):
            aug_src_prompts.append(
                build_cpp_prompt(mt, aug_src, include_docstring=include_docstring)
            )
            aug_dest_prompts.append(
                build_python_prompt(
                    mt, aug_dest, pivot_prefix=pi.pivot_prefix,
                    include_docstring=include_docstring,
                )
            )
            aug_targ_prompts.append(
                build_python_prompt(
                    mt, aug_targ, pivot_prefix=pi.pivot_prefix,
                    include_docstring=include_docstring,
                )
            )
            aug_pis.append(pi)

        # 1. toggle_type_hints — always safe (only changes the def line)
        aug_dest_th, aug_targ_th = augment_pair(dest_sample, targ_sample, "toggle_type_hints")
        aug_src_th = augment_source(src_sample, "toggle_type_hints")
        _add(aug_src_th, aug_dest_th, aug_targ_th)

        # 2. variable_rename — only safe when pivot_prefix is empty
        if pi.pivot_prefix == "":
            old_var = dest_sample.variable_name
            for new_var in _VAR_VOCAB.get(old_var, [])[:3]:
                if new_var == old_var:
                    continue
                cfg = {"new_var": new_var}
                aug_dest_v, aug_targ_v = augment_pair(
                    dest_sample, targ_sample, "variable_rename", cfg
                )
                aug_src_v = augment_source(src_sample, "variable_rename", cfg)
                _add(aug_src_v, aug_dest_v, aug_targ_v)

    N = len(aug_src_prompts)
    if N == 0:
        logger.info("augment_pivot_caches: no variants generated")
        return []

    logger.info(f"augment_pivot_caches: computing {N} augmented caches")

    # Target log-probs — chunked to bound GPU memory
    log_p_chunks: list[torch.Tensor] = []
    for i in range(0, N, cache_batch_size):
        chunk = aug_targ_prompts[i : i + cache_batch_size]
        p = get_next_token_probs_batch(mt, chunk)
        log_p_chunks.append(torch.log(p.clamp(min=1e-10)))
        free_gpu_cache()
        logger.debug(f"  aug targ probs: {min(i + cache_batch_size, N)}/{N}")
    log_p_target = torch.cat(log_p_chunks, dim=0)  # (N, vocab)

    # Per-head deltas — always precomputed, chunked
    delta_chunks: list[torch.Tensor] = []
    for i in range(0, N, cache_batch_size):
        src_chunk  = aug_src_prompts[i : i + cache_batch_size]
        dest_chunk = aug_dest_prompts[i : i + cache_batch_size]
        B = len(src_chunk)
        combined_o = get_o_proj_inputs_batch(mt, src_chunk + dest_chunk, layers)
        src_o  = {l: combined_o[l][:B] for l in layers}
        dest_o = {l: combined_o[l][B:] for l in layers}
        delta_chunks.append(precompute_per_head_deltas_batch(mt, src_o, dest_o, layers))
        del combined_o, src_o, dest_o
        free_gpu_cache()
        logger.debug(f"  aug deltas: {min(i + cache_batch_size, N)}/{N}")
    all_deltas = torch.cat(delta_chunks, dim=0)  # (N, L, H, E)

    aug_caches = [
        PivotCache(
            pivot_instance=aug_pis[i],
            dest_prompt=aug_dest_prompts[i],
            log_p_target=log_p_target[i],
            per_head_deltas=all_deltas[i],
            src_prompt=None,
        )
        for i in range(N)
    ]

    logger.info(f"augment_pivot_caches: done — {N} augmented caches generated")
    return aug_caches


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
        attn_out[l][:, -1, :] += Σ_h mask.values[li, h] * per_head_deltas[:, li, h]

    Differentiable w.r.t. mask.values — gradients flow through the linear
    combination of precomputed deltas.

    Args:
        mt: ModelandTokenizer (frozen).
        dest_inputs: Tokenized destination prompt(s).
        per_head_deltas: Either:
            - 3D ``(n_layers_tested, n_heads, n_embd)`` for a single instance →
              returns ``(vocab_size,)`` logits (backward-compatible).
            - 4D ``(batch, n_layers_tested, n_heads, n_embd)`` for a batch →
              returns ``(batch, vocab_size)`` logits.
        mask: HeadMask with learnable values (requires_grad=True).
        layers: Layer indices corresponding to per_head_deltas axis -3.

    Returns:
        ``(vocab_size,)`` float32 logits for 3D input (single instance), or
        ``(batch, vocab_size)`` float32 logits for 4D input (batch).
        Always differentiable w.r.t. mask.values.
    """
    # Normalise to 4D; remember whether we need to squeeze the output
    if per_head_deltas.dim() == 3:
        per_head_deltas = per_head_deltas.unsqueeze(0)  # (1, L, H, E)
        squeeze_output = True
    else:
        squeeze_output = False

    attn_names = [mt.attn_module_name_format.format(layer) for layer in layers]
    layer_to_li = {name: li for li, name in enumerate(attn_names)}
    mask_vals = mask.values  # (n_layers_tested, n_heads)
    batch_deltas = per_head_deltas  # (batch, n_layers_tested, n_heads, n_embd) — CPU

    def edit_fn(output, layer_name):
        li = layer_to_li[layer_name]
        is_tuple = isinstance(output, tuple)
        attn_out = output[0] if is_tuple else output  # (batch, seq, n_embd)
        dev = attn_out.device
        dtype = attn_out.dtype

        # mask_vals[li]: (n_heads,)
        # batch_deltas[:, li]: (batch, n_heads, n_embd)
        # Weighted sum kept in float32 so gradients w.r.t. mask_vals are
        # computed at full precision — bfloat16 would corrupt them.
        delta_f32 = (
            mask_vals[li].to(dev).view(1, -1, 1)  # float32, requires_grad
            * batch_deltas[:, li].to(dev)  # float32 (precomputed CPU)
        ).sum(dim=1)  # (batch, n_embd) float32

        # Patch only the last token position.  Avoid allocating a full
        # (batch, seq, n_embd) float32 tensor — that would keep ~97 MB per
        # layer live in the autograd graph until backward() (~3 GB total for
        # 34 layers at batch=16, seq=200, n_embd=5120).
        # torch.cat preserves gradient flow: CatBackward routes the upstream
        # gradient slice [:, -1:, :] back through new_last → delta_f32 → mask_vals.
        new_last = attn_out[:, -1:, :] + delta_f32.to(dtype).unsqueeze(
            1
        )  # (batch, 1, n_embd)
        new_attn_out = torch.cat(
            [attn_out[:, :-1, :], new_last], dim=1
        )  # (batch, seq, n_embd)

        return (new_attn_out,) + output[1:] if is_tuple else new_attn_out

    with baukit.TraceDict(
        module=mt._model,
        layers=attn_names,
        edit_output=edit_fn,
    ):
        output = mt._model(**dest_inputs)

    logits = output.logits[:, -1, :].float()  # (batch, vocab)
    return logits[0] if squeeze_output else logits


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
    batch_size: int = 8,
    checkpoint_dir: str | None = None,
    checkpoint_every: int = 10,
) -> tuple[HeadMask, MaskTrainingHistory]:
    """Learn a sparse head mask via projected gradient descent.

    Objective:
        L = KL(P_patched ∥ P_target) + lambda_sparse * mean(mask.values)

    KL direction: KL(P_patched ∥ P_target) = Σ P_patched * log(P_patched/P_target)
    — pushes the patched destination distribution toward the Python target.

    Instances are processed in batches of ``batch_size`` per optimizer step
    (``reduction="batchmean"`` keeps the KL loss scale consistent with the old
    per-instance ``reduction="sum"`` regardless of batch size).

    If ``pivot_caches[0].per_head_deltas is None`` (i.e., ``cache_pivot_data`` was
    called with ``precompute_deltas=False``), deltas are recomputed lazily per batch
    per epoch using one combined src+dest forward pass — same positional alignment
    as the eager path, at the cost of extra compute.

    Args:
        mt: ModelandTokenizer (frozen — only mask params are updated).
        pivot_caches: Pre-cached data from cache_pivot_data().
        layers: Layer indices (must match pivot_caches[*].per_head_deltas axis 0).
        n_epochs: Training epochs (default 20, from paper).
        lr: AdamW learning rate (default 1e-2, from paper).
        lambda_sparse: L1 sparsity coefficient (default 1e-2, from paper).
        batch_size: Instances per optimizer step (default 8).

    Returns:
        (trained HeadMask, MaskTrainingHistory)
    """
    n_layers_tested = len(layers)
    n_heads = mt.model.config.num_attention_heads
    N = len(pivot_caches)

    mask = HeadMask(n_layers_tested, n_heads)
    optimizer = AdamW(mask.parameters(), lr=lr)
    history = MaskTrainingHistory()

    # Freeze model parameters so backward() doesn't accumulate grads for them
    for param in mt._model.parameters():
        param.requires_grad_(False)

    lazy_deltas = pivot_caches[0].per_head_deltas is None
    batch_groups = [pivot_caches[i : i + batch_size] for i in range(0, N, batch_size)]

    # Pre-build everything that is cheap to cache outside the epoch loop.
    # When lazy_deltas=True, src_prompts are stored per batch instead of stacked
    # delta tensors — deltas are recomputed on every optimizer step.
    batch_log_p_list = [
        torch.stack([c.log_p_target for c in grp])
        for grp in batch_groups  # (B, vocab) CPU
    ]
    if lazy_deltas:
        batch_src_prompts_list = [[c.src_prompt for c in grp] for grp in batch_groups]
        batch_dest_prompts_list = [[c.dest_prompt for c in grp] for grp in batch_groups]
    else:
        batch_inputs_list = [
            prepare_input([c.dest_prompt for c in grp], tokenizer=mt)
            for grp in batch_groups
        ]
        # Do NOT pre-stack deltas here: torch.stack would copy all N × (L, H, E)
        # float32 tensors that already live in pivot_caches[*].per_head_deltas,
        # doubling CPU RAM (~14 GB for N=512 on Qwen-32B).  Stack on-the-fly per
        # batch inside the epoch loop instead (negligible cost: 16 tensors × 28 MB).

    logger.info(
        f"train_head_mask: {n_layers_tested}L × {n_heads}H = "
        f"{n_layers_tested * n_heads} mask params | "
        f"{N} instances | batch_size={batch_size} ({len(batch_groups)} batches/epoch) | "
        f"{n_epochs} epochs | lr={lr} λ={lambda_sparse} | "
        f"lazy_deltas={lazy_deltas}"
    )

    for epoch in range(n_epochs):
        epoch_kl = 0.0
        epoch_sparse = 0.0
        n_steps = 0

        for bi, batch_log_p in enumerate(batch_log_p_list):
            if lazy_deltas:
                # Recompute deltas for this batch: one combined forward pass so
                # src and dest share the same sequence length (no positional shift).
                src_ps = batch_src_prompts_list[bi]
                dest_ps = batch_dest_prompts_list[bi]
                B = len(src_ps)
                with torch.inference_mode():
                    combined_o = get_o_proj_inputs_batch(mt, src_ps + dest_ps, layers)
                src_o_b = {layer: combined_o[layer][:B] for layer in layers}
                dest_o_b = {layer: combined_o[layer][B:] for layer in layers}
                batch_deltas = precompute_per_head_deltas_batch(
                    mt, src_o_b, dest_o_b, layers
                )
                del (
                    combined_o,
                    src_o_b,
                    dest_o_b,
                )  # free CPU tensors before forward pass
                free_gpu_cache()  # flush nnsight trace residuals before the grad pass
                batch_inputs = prepare_input(dest_ps, tokenizer=mt)
            else:
                batch_inputs = batch_inputs_list[bi]
                # Stack on-the-fly — avoids the ~14 GB duplicate that pre-stacking would create.
                batch_deltas = torch.stack(
                    [c.per_head_deltas for c in batch_groups[bi]]
                )  # (B, L, H, E) CPU float32

            optimizer.zero_grad()

            logits = forward_with_masked_patch(
                mt, batch_inputs, batch_deltas, mask, layers
            )  # (B, vocab)

            # KL(P_patched ∥ P_target) — batchmean: sum over vocab / batch_size
            # Consistent loss scale with old per-instance reduction="sum"
            log_p_patched = F.log_softmax(logits, dim=-1)
            loss_kl = F.kl_div(
                batch_log_p.to(logits.device),
                log_p_patched,
                reduction="batchmean",
                log_target=True,
            )

            loss_sparse = lambda_sparse * mask.values.mean()
            loss = loss_kl + loss_sparse

            loss.backward()
            optimizer.step()
            mask.clamp_()

            epoch_kl += loss_kl.item()
            epoch_sparse += loss_sparse.item()
            n_steps += 1

            if logger.isEnabledFor(logging.DEBUG):
                mem_alloc = {
                    i: f"{torch.cuda.memory_allocated(i) / 1e9:.2f}GB"
                    for i in range(torch.cuda.device_count())
                }
                logger.debug(
                    f"  e{epoch + 1:03d} b{bi + 1:03d}/{len(batch_log_p_list)}"
                    f"  KL={loss_kl.item():.4f}  sparse={loss_sparse.item():.4f}"
                    f"  n_active={mask.n_active()}"
                    f"  gpu_alloc={mem_alloc}"
                )

        avg_kl = epoch_kl / n_steps
        avg_sparse = epoch_sparse / n_steps
        avg_total = avg_kl + avg_sparse
        n_act = mask.n_active()

        history.loss_causal.append(avg_kl)
        history.loss_sparse.append(avg_sparse)
        history.loss_total.append(avg_total)
        history.n_active.append(n_act)

        free_gpu_cache()

        logger.debug("-" * 100)
        logger.info(
            f"  Epoch {epoch + 1:3d}/{n_epochs}: "
            f"KL={avg_kl:.4f}  sparse={avg_sparse:.4f}  "
            f"total={avg_total:.4f}  n_active={n_act}"
        )
        logger.debug("-" * 100)

        if checkpoint_dir is not None and (epoch + 1) % checkpoint_every == 0:
            import os

            os.makedirs(checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(
                checkpoint_dir, f"checkpoint_epoch_{epoch + 1:04d}.pt"
            )
            torch.save(
                {
                    "epoch": epoch + 1,
                    "mask_values": mask.values.detach().cpu(),
                    "layers": layers,
                    "history": {
                        "loss_causal": history.loss_causal,
                        "loss_sparse": history.loss_sparse,
                        "loss_total": history.loss_total,
                        "n_active": history.n_active,
                    },
                },
                ckpt_path,
            )
            logger.info(f"  Checkpoint saved → {ckpt_path}")

    selected = mask.selected_heads(layers)
    logger.info(f"train_head_mask done: {len(selected)} active heads → {selected}")
    return mask, history
