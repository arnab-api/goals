"""Learned sparse head mask for attention head localization.

Learns a continuous mask over (n_layers_tested, n_heads) attention heads whose
cross-language interchange (C++ source → Python destination) minimizes
KL(P_patched ∥ P_target).

Based on arxiv:2510.26784 (filter heads via differentiable causal masking),
adapted for the cross-language interchange design from
claude_reports/brainstorm_attention_head_localization.md.

Pipeline (current — o_proj pre-hook replacement):
    1. Cache o_proj inputs for source (C++) prompts per layer.
    2. Train mask: per-head interpolation via register_forward_pre_hook on o_proj.
       For each head h at layer l:
           o_proj_input[:, -1, h*H:(h+1)*H] = mask[l,h]*src_h + (1-mask[l,h])*cascade_h
       This gives true replacement semantics: mask=1 → pure source, mask=0 → pure cascade.
       Sparsity: lambda * mean(mask) — scale-invariant to number of layers tested.
    3. Post-training threshold (mask > 0.5) → selected heads.

Pipeline (obsolete — additive delta approach, functions renamed with _obsolete suffix):
    1. Cache o_proj inputs for both source and destination prompts.
    2. Precompute per-head analytical deltas: W_o_h @ (src_h - dest_h).
    3. Train mask: attn_out[l][:,-1,:] += Σ_h mask[l,h] * delta[l,h]  (via baukit hooks)
    Note: additive delta does not give true replacement when cascade effects are present.

Mask parameterization:
    - nn.Parameter initialized to all ones
    - Clamped to [0,1] after each optimizer.step() (box-constrained GD)
    - No sigmoid, no temperature — raw values as interpolation weights
    - Selection: heads where values > 0.5 after training
"""

from __future__ import annotations

import dataclasses
import logging
import random
from dataclasses import dataclass, field
from typing import Any

import baukit
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW

from src.functional import free_gpu_cache, get_module_nnsight
from src.goals.augment import _VAR_VOCAB, adapt_text_var
from src.goals.head_loc_augment import augment_pair, augment_source
from src.goals.head_loc_data import (
    HeadLocDataset,
    PivotInstance,
    PredicatePair,
    SinglePredicateSample,
)
from src.goals.head_loc_pivot import (
    build_cpp_prompt,
    build_haskell_prompt,
    build_java_prompt,
    build_prolog_prompt,
    build_python_prompt,
    build_rust_prompt,
    get_next_token_probs_batch,
)
from src.tokens import TokenizerOutput, prepare_input
from src.utils.metrics import recall as recall_metric

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Augmentation constants
# ---------------------------------------------------------------------------

# Source language builders (excluding Python, which is always the destination).
# "cpp" is the original — alts are the other four.
_SOURCE_LANG_BUILDERS: dict[str, Any] = {
    "cpp": build_cpp_prompt,
    "java": build_java_prompt,
    "rust": build_rust_prompt,
    "haskell": build_haskell_prompt,
    "prolog": build_prolog_prompt,
}
_SOURCE_LANG_ALTS = [k for k in _SOURCE_LANG_BUILDERS if k != "cpp"]

_AUG_TYPES = ["toggle_type_hints", "variable_rename", "source_lang"]
_AUG_WEIGHTS = [0.10, 0.35, 0.55]


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
# Utility: o_proj weight (obsolete — used only by delta-based functions)
# ---------------------------------------------------------------------------


def _get_o_proj_weight_obsolete(mt, layer: int) -> torch.Tensor:
    """Return o_proj weight (n_embd, n_heads * head_dim) as float32 CPU tensor.

    .. deprecated::
        Used only by the obsolete delta-based patching path
        (``precompute_per_head_deltas_batch_obsolete`` /
        ``forward_with_masked_patch_obsolete``).
    """
    path = mt.attn_module_name_format.format(layer) + ".o_proj"
    module = get_module_nnsight(mt._model, path)
    return module.weight.detach().cpu().float()


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------


@torch.inference_mode()
def get_o_proj_inputs_batch(
    mt,
    prompts: list[str] | TokenizerOutput,
    layers: list[int],
    token_positions: list[int] | None = None,
) -> dict[int, torch.Tensor]:
    """Extract o_proj input for each layer, for a batch of prompts.

    Left-padding (from prepare_input) ensures ``[:, -1, :]`` is the last real
    token for every sequence in the batch.

    Args:
        token_positions: Token positions to extract.  When ``None`` (default),
            extracts only the last position, returning ``(batch, n_embd)``.
            When a list of ints is given, extracts those positions and returns
            ``(batch, len(token_positions), n_embd)``.

    Returns:
        ``{layer: tensor}`` where tensor shape is ``(batch, n_embd)`` for the
        default single-position case, or ``(batch, n_positions, n_embd)`` when
        ``token_positions`` is specified.
    """
    if isinstance(prompts, list) and all(isinstance(p, str) for p in prompts):
        inputs = prepare_input(prompts=prompts, tokenizer=mt)
    elif isinstance(prompts, TokenizerOutput):
        inputs = prompts
    else:
        error_msg = f"prompts must be list of strings or TokenizerOutput.\nGot type {type(prompts)}."
        if isinstance(prompts, list):
            error_msg += f"\nList element types: {[type(p) for p in prompts]}"
        raise ValueError(error_msg)
    saves: dict[int, Any] = {}

    with mt.trace(inputs, scan=False, validate=False):
        for layer in layers:
            attn_name = mt.attn_module_name_format.format(layer)
            o_proj = get_module_nnsight(mt, attn_name + ".o_proj")
            if token_positions is None:
                saves[layer] = o_proj.input[:, -1, :].save()  # (batch, n_embd)
            else:
                saves[layer] = o_proj.input[
                    :, token_positions, :
                ].save()  # (batch, n_pos, n_embd)

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


@torch.inference_mode()
def collect_head_steering_vecs(
    mt,
    src_tokenized: TokenizerOutput,
    selected_heads: list[tuple[int, int]],
    token_positions: list[int],
) -> torch.Tensor:
    """Compute the summed residual-stream contribution of selected heads at each token position.

    For a linear o_proj, head h's contribution to the residual stream is:
        o_proj_input[:, pos, h*hd:(h+1)*hd] @ W_o[:, h*hd:(h+1)*hd].T

    This sums those contributions across all selected heads.

    Args:
        mt: ModelandTokenizer.
        src_tokenized: Tokenized source prompt (batch=1).
        selected_heads: List of (layer, head_idx) pairs identifying which heads to include.
        token_positions: Token positions to extract (absolute indices into src_tokenized).

    Returns:
        Float32 CPU tensor of shape ``(n_positions, n_embd)`` — one steering vector per position.
    """
    from collections import defaultdict

    n_embd = mt.n_embd
    n_heads = mt.model.config.num_attention_heads
    head_dim = n_embd // n_heads

    # Group heads by layer to minimise forward passes
    heads_per_layer: dict[int, list[int]] = defaultdict(list)
    for layer, head in selected_heads:
        heads_per_layer[layer].append(head)

    layers = list(heads_per_layer.keys())

    # Single forward pass — extract o_proj inputs at all requested positions
    o_proj_inputs = get_o_proj_inputs_batch(
        mt, src_tokenized, layers, token_positions=token_positions
    )  # {layer: (1, n_positions, n_embd)}

    n_pos = len(token_positions)
    steering_vecs = torch.zeros(n_pos, n_embd, dtype=torch.float32)

    for layer, heads in heads_per_layer.items():
        attn_name = mt.attn_module_name_format.format(layer)
        o_proj = baukit.get_module(mt._model, attn_name + ".o_proj")
        W_o = o_proj.weight.detach().float().cpu()  # (n_embd, n_embd)

        inp = o_proj_inputs[layer][0]  # (n_positions, n_embd) float32 CPU

        for h in heads:
            h_inp = inp[:, h * head_dim : (h + 1) * head_dim]      # (n_pos, head_dim)
            W_o_h = W_o[:, h * head_dim : (h + 1) * head_dim]      # (n_embd, head_dim)
            steering_vecs += h_inp @ W_o_h.t()                      # (n_pos, n_embd)

    return steering_vecs  # (n_positions, n_embd) float32 CPU


@torch.no_grad()
def precompute_per_head_deltas_batch_obsolete(
    mt,
    src_o_proj_inputs: dict[int, torch.Tensor],
    dest_o_proj_inputs: dict[int, torch.Tensor],
    layers: list[int],
) -> torch.Tensor:
    """[OBSOLETE] Compute per-head deltas for a batch of (src, dest) pairs.

    Superseded by ``forward_with_src_replacement`` which patches o_proj inputs
    directly instead of using additive deltas.

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
        W_o = _get_o_proj_weight_obsolete(mt, layer)  # (n_embd, n_embd)
        diff = src_o_proj_inputs[layer] - dest_o_proj_inputs[layer]  # (batch, n_embd)
        for h in range(n_heads):
            W_o_h = W_o[:, h * head_dim : (h + 1) * head_dim]  # (n_embd, head_dim)
            diff_h = diff[:, h * head_dim : (h + 1) * head_dim]  # (batch, head_dim)
            deltas[:, li, h] = diff_h @ W_o_h.T  # (batch, n_embd)

    return deltas


@torch.no_grad()
def precompute_per_head_deltas_obsolete(
    mt,
    src_o_proj_inputs: dict[int, torch.Tensor],
    dest_o_proj_inputs: dict[int, torch.Tensor],
    layers: list[int],
) -> torch.Tensor:
    """[OBSOLETE] Compute W_o_h @ (src_h - dest_h) for every (layer, head) pair.

    Superseded by ``forward_with_src_replacement``.

    Returns float32 CPU tensor of shape (n_layers_tested, n_heads, n_embd).
    """
    n_heads = mt.model.config.num_attention_heads
    head_dim = mt.n_embd // n_heads
    n_layers_tested = len(layers)

    deltas = torch.zeros(n_layers_tested, n_heads, mt.n_embd)

    for li, layer in enumerate(layers):
        W_o = _get_o_proj_weight_obsolete(mt, layer)  # (n_embd, n_embd)
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

    ``src_head_vals`` is the o_proj input stacked across layers for the source run.
    When ``precompute=True`` (default) it is populated at cache time and
    ``train_head_mask`` uses it directly.  When ``precompute=False`` (lazy) it is
    ``None`` and ``src_prompt`` is set; ``train_head_mask`` re-runs the source
    forward pass once per batch per epoch (saves ~1 GB CPU RAM for N=1480 Qwen-32B
    at the cost of one extra forward pass per optimizer step per epoch).

    ``per_head_deltas`` is retained for backward compatibility with the obsolete
    delta-based training path only.  New code should use ``src_head_vals``.
    """

    pivot_instance: PivotInstance
    dest_prompt: str  # raw string — destination prompt up to pivot prefix
    log_p_target: (
        torch.Tensor
    )  # (vocab_size,) float32 CPU — log of target next-token dist
    per_head_deltas: (
        torch.Tensor | None
    )  # [OBSOLETE] (n_layers_tested, n_heads, n_embd)
    src_sample: (
        SinglePredicateSample  # source predicate sample (targ_sample == src_sample)
    )
    dest_sample: SinglePredicateSample  # destination predicate sample
    src_prompt: str | None = None  # populated when precompute=False (lazy)
    src_head_vals: torch.Tensor | None = None
    # (n_layers_tested, n_embd) float32 CPU — o_proj inputs from the source run,
    # stacked across layers.  ~0.7 MB/instance for Qwen-32B; ~1 GB for N=1480.
    # 40× smaller than the obsolete per_head_deltas (~28 MB/instance, ~42 GB total).

    # Memory note: log_p_target stores the full vocabulary distribution because the
    # training objective is KL(P_patched ∥ P_target), which requires all vocab logits.
    # For Qwen2.5-Coder-32B (vocab=151,936) this is ~0.58 MB per instance, or ~860 MB
    # for N=1480.  An alternative would be cross-entropy against target_top_token only
    # (1 int64 per instance), at the cost of a sparser, harder training signal.
    #
    # src_sample / dest_sample store the original (or augmented) SinglePredicateSample
    # objects so that augment_pivot_caches can rebuild prompts in any language without
    # needing the HeadLocDataset or pivot_instances list.  Memory overhead is negligible
    # (~2-5 KB of strings per instance, ~9 MB total for N=1480).


@torch.inference_mode()
def cache_pivot_data(
    mt,
    pivot_instances: list[tuple[PredicatePair, PivotInstance]],
    ds: HeadLocDataset,
    layers: list[int],
    include_docstring: bool = False,
    precompute: bool = True,
    cache_batch_size: int = 64,
) -> list[PivotCache]:
    """Cache all data needed for mask training.

    Builds all prompts first, then runs forward passes in chunks of
    ``cache_batch_size`` to bound peak GPU activation memory.

    **``precompute=True``** (default — fastest training, low memory):
      - Runs only the **source** forward pass to extract o_proj inputs per layer.
      - target next-token probs: chunked pass.
      - CPU RAM: ~1 GB for N=1480 Qwen-32B (``src_head_vals`` at ~0.7 MB/instance).
      - The destination forward pass happens live inside ``train_head_mask``.

    **``precompute=False``** (lazy — re-runs source forward per batch per epoch):
      - Only target next-token probs are cached.
      - ``src_prompt`` is stored; ``src_head_vals`` are recomputed per batch inside
        ``train_head_mask`` (one extra forward pass per optimizer step per epoch).
      - CPU RAM: negligible (only log_p_target per instance).
      - Required when ``augment=True`` in ``train_head_mask`` (augmented prompts
        change every epoch so precomputed values would be stale).

    Args:
        precompute: Whether to precompute and cache src o_proj inputs (True)
            or store src_prompt for lazy recomputation (False).
        cache_batch_size: Max prompts per forward pass during caching.  Default 64
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
        f"cache_pivot_data: {N} instances | precompute={precompute} | "
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

    # Source o_proj inputs — precomputed when precompute=True.
    # Only the src forward pass is needed (dest is computed live in training).
    all_src_head_vals: torch.Tensor | None = None
    if precompute:
        src_val_chunks: list[torch.Tensor] = []
        for i in range(0, N, cache_batch_size):
            src_chunk = src_prompts[i : i + cache_batch_size]
            src_o = get_o_proj_inputs_batch(
                mt, src_chunk, layers
            )  # {layer: (B, n_embd)}
            # Stack layers into (B, L, n_embd)
            src_val_chunks.append(torch.stack([src_o[ly] for ly in layers], dim=1))
            del src_o
            free_gpu_cache()
            logger.debug(f"  src head vals: {min(i + cache_batch_size, N)}/{N}")
        all_src_head_vals = torch.cat(src_val_chunks, dim=0)  # (N, L, n_embd)
        logger.info(
            f"cache_pivot_data: src_head_vals precomputed and cached "
            f"({all_src_head_vals.numel() * 4 / 1e9:.2f} GB)"
        )

    # Re-collect samples in the same order as prompts (needed for PivotCache fields).
    samples_list: list[tuple[SinglePredicateSample, SinglePredicateSample]] = []
    for pair, _ in pivot_instances:
        samples_list.append(
            (
                ds.get_sample(pair.source_predicate),
                ds.get_sample(pair.destination_predicate),
            )
        )

    caches: list[PivotCache] = []
    for i, (pair, pi) in enumerate(pivot_instances):
        logger.debug(
            f"  Building cache {i + 1}/{N}: "
            f"{pair.destination_predicate} ← {pair.source_predicate} "
            f"prefix={pi.pivot_prefix!r}"
        )
        src_s, dest_s = samples_list[i]
        # Update target_top_token from current model's distribution so that
        # recall eval is correct even when pivot instances were mined on a
        # different model (the full log_p_target was already computed above).
        new_tgt_tok = mt.tokenizer.decode([log_p_target[i].argmax().item()])
        pi = dataclasses.replace(pi, target_top_token=new_tgt_tok)
        caches.append(
            PivotCache(
                pivot_instance=pi,
                dest_prompt=dest_prompts[i],
                log_p_target=log_p_target[i],
                per_head_deltas=None,
                src_sample=src_s,
                dest_sample=dest_s,
                src_prompt=src_prompts[i] if not precompute else None,
                src_head_vals=all_src_head_vals[i]
                if all_src_head_vals is not None
                else None,
            )
        )

    logger.info(f"cache_pivot_data: cached {len(caches)} pivot instances")
    return caches


@torch.inference_mode()
def augment_pivot_caches(
    mt,
    pivot_caches: list[PivotCache],
    include_docstring: bool = False,
    cache_batch_size: int = 64,
    rng: random.Random | None = None,
) -> list[PivotCache]:
    """Return one randomly-augmented PivotCache per input cache (1× expansion).

    For each instance one augmentation type is sampled from
    ``["toggle_type_hints", "variable_rename", "source_lang"]`` with weights
    ``[0.10, 0.35, 0.55]``.

    ``"source_lang"`` (55%)
        Replaces the C++ source prompt with a randomly chosen alternative
        language (Java, Rust, Haskell, or Prolog).  Only ``src_prompt``
        changes — ``dest_prompt``, ``log_p_target``, and samples are reused.

    ``"variable_rename"`` (35%)
        Renames the variable to a random alternative from ``_VAR_VOCAB``.
        Updates ``src_prompt``, ``dest_prompt``, ``pivot_prefix``,
        ``log_p_target``, ``target_top_token``, and stored samples.
        Falls back to ``"source_lang"`` if the variable has no alternatives.

    ``"toggle_type_hints"`` (10%)
        Strips Python param type annotations from the ``def`` line.
        Updates ``src_prompt``, ``dest_prompt``, ``log_p_target``,
        ``target_top_token``, and stored samples.

    The input ``pivot_caches`` is **never mutated**.  Returned caches are
    always lazy (``per_head_deltas=None``, ``src_prompt`` populated) and carry
    the augmented ``src_sample``/``dest_sample`` so they can be re-augmented
    without the original dataset.

    Args:
        mt: ModelandTokenizer (frozen).
        pivot_caches: Original caches — read-only.  Must have ``src_sample``
            and ``dest_sample`` populated (set by ``cache_pivot_data``).
        include_docstring: Passed to all prompt builders.
        cache_batch_size: Max prompts per forward pass for log_p_target
            recomputation (only triggered for variable_rename / toggle_type_hints).
        rng: Optional ``random.Random`` for reproducibility; created fresh if None.

    Returns:
        List of ``PivotCache`` objects, same length as ``pivot_caches``.
    """
    if rng is None:
        rng = random.Random()

    N = len(pivot_caches)

    # Accumulators — one entry per output cache.
    aug_src_prompts: list[str] = []
    aug_dest_prompts: list[str] = []
    aug_pis: list[PivotInstance] = []
    aug_src_samples: list[SinglePredicateSample] = []
    aug_dest_samples: list[SinglePredicateSample] = []
    aug_log_p: list[torch.Tensor | None] = []
    recompute_idx: list[int] = []
    recompute_targ: list[str] = []

    for cache in pivot_caches:
        src_sample = cache.src_sample
        dest_sample = cache.dest_sample
        targ_sample = cache.src_sample  # target == source by construction
        pi = cache.pivot_instance

        aug_type = rng.choices(_AUG_TYPES, weights=_AUG_WEIGHTS, k=1)[0]

        # --- source_lang: only src_prompt changes ---
        if aug_type == "source_lang":
            lang = rng.choice(_SOURCE_LANG_ALTS)
            aug_src_prompts.append(
                _SOURCE_LANG_BUILDERS[lang](
                    mt, src_sample, include_docstring=include_docstring
                )
            )
            aug_dest_prompts.append(cache.dest_prompt)
            aug_pis.append(pi)
            aug_src_samples.append(src_sample)
            aug_dest_samples.append(dest_sample)
            aug_log_p.append(cache.log_p_target)

        # --- variable_rename: all prompts + samples change ---
        elif aug_type == "variable_rename":
            old_var = dest_sample.variable_name
            alts = [v for v in _VAR_VOCAB.get(old_var, []) if v != old_var]
            if not alts:
                # Fallback: source_lang
                lang = rng.choice(_SOURCE_LANG_ALTS)
                aug_src_prompts.append(
                    _SOURCE_LANG_BUILDERS[lang](
                        mt, src_sample, include_docstring=include_docstring
                    )
                )
                aug_dest_prompts.append(cache.dest_prompt)
                aug_pis.append(pi)
                aug_src_samples.append(src_sample)
                aug_dest_samples.append(dest_sample)
                aug_log_p.append(cache.log_p_target)
            else:
                new_var = rng.choice(alts)
                cfg = {"new_var": new_var}
                aug_dest_s, aug_targ_s = augment_pair(
                    dest_sample, targ_sample, "variable_rename", cfg
                )
                aug_src_s = augment_source(src_sample, "variable_rename", cfg)
                aug_prefix = adapt_text_var(pi.pivot_prefix, old_var, new_var)
                aug_src_prompts.append(
                    build_cpp_prompt(mt, aug_src_s, include_docstring=include_docstring)
                )
                aug_dest_prompts.append(
                    build_python_prompt(
                        mt,
                        aug_dest_s,
                        pivot_prefix=aug_prefix,
                        include_docstring=include_docstring,
                    )
                )
                aug_pis.append(
                    dataclasses.replace(pi, pivot_prefix=aug_prefix)
                    if aug_prefix != pi.pivot_prefix
                    else pi
                )
                aug_src_samples.append(aug_src_s)
                aug_dest_samples.append(aug_dest_s)
                aug_log_p.append(None)
                recompute_idx.append(len(aug_log_p) - 1)
                recompute_targ.append(
                    build_python_prompt(
                        mt,
                        aug_targ_s,
                        pivot_prefix=aug_prefix,
                        include_docstring=include_docstring,
                    )
                )

        # --- toggle_type_hints: all prompts + samples change ---
        else:
            aug_dest_s, aug_targ_s = augment_pair(
                dest_sample, targ_sample, "toggle_type_hints"
            )
            aug_src_s = augment_source(src_sample, "toggle_type_hints")
            aug_src_prompts.append(
                build_cpp_prompt(mt, aug_src_s, include_docstring=include_docstring)
            )
            aug_dest_prompts.append(
                build_python_prompt(
                    mt,
                    aug_dest_s,
                    pivot_prefix=pi.pivot_prefix,
                    include_docstring=include_docstring,
                )
            )
            aug_pis.append(pi)
            aug_src_samples.append(aug_src_s)
            aug_dest_samples.append(aug_dest_s)
            aug_log_p.append(None)
            recompute_idx.append(len(aug_log_p) - 1)
            recompute_targ.append(
                build_python_prompt(
                    mt,
                    aug_targ_s,
                    pivot_prefix=pi.pivot_prefix,
                    include_docstring=include_docstring,
                )
            )

    # Batch-recompute log_p_target for variable_rename / toggle_type_hints instances.
    if recompute_targ:
        logger.debug(
            f"augment_pivot_caches: recomputing log_p_target for {len(recompute_targ)}/{N} instances"
        )
        computed: list[torch.Tensor] = []
        for i in range(0, len(recompute_targ), cache_batch_size):
            chunk = recompute_targ[i : i + cache_batch_size]
            p = get_next_token_probs_batch(mt, chunk)
            computed.append(torch.log(p.clamp(min=1e-10)))
            free_gpu_cache()
        log_p_recomputed = torch.cat(computed, dim=0)
        for slot, log_p_row in zip(recompute_idx, log_p_recomputed.unbind(0)):
            aug_log_p[slot] = log_p_row
            # target_top_token may shift after augmentation — overwrite from new distribution.
            new_tgt_tok = mt.tokenizer.decode([log_p_row.argmax().item()])
            aug_pis[slot] = dataclasses.replace(
                aug_pis[slot], target_top_token=new_tgt_tok
            )

    logger.info(
        f"augment_pivot_caches: {N} caches augmented "
        f"({len(recompute_targ)} log_p recomputed, {N - len(recompute_targ)} reused)"
    )

    return [
        PivotCache(
            pivot_instance=aug_pis[i],
            dest_prompt=aug_dest_prompts[i],
            log_p_target=aug_log_p[i],
            per_head_deltas=None,
            src_sample=aug_src_samples[i],
            dest_sample=aug_dest_samples[i],
            src_prompt=aug_src_prompts[i],
        )
        for i in range(N)
    ]


# ---------------------------------------------------------------------------
# Differentiable patched forward pass
# ---------------------------------------------------------------------------


def forward_with_masked_patch_obsolete(
    mt,
    dest_inputs: Any,
    per_head_deltas: torch.Tensor,
    mask: HeadMask,
    layers: list[int],
) -> torch.Tensor:
    """[OBSOLETE] Run dest_inputs with mask-weighted head patching via baukit eager hooks.

    Superseded by ``forward_with_src_replacement`` which intercepts the o_proj
    input for true per-head replacement rather than additive delta injection.

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


def forward_with_src_replacement(
    mt,
    dest_inputs: Any,
    src_head_vals: torch.Tensor,
    mask: HeadMask,
    layers: list[int],
    alpha: float = 1.0,
    return_past_key_values: bool = False,
) -> torch.Tensor:
    """Run dest_inputs with mask-weighted per-head o_proj input replacement.

    For each layer l (index li in layers) and head h at the last token position:
        o_proj_input[:, -1, h*head_dim:(h+1)*head_dim] =
            mask[li,h] * src_h + (1 - mask[li,h]) * cascade_h

    where ``cascade_h`` is the live dest attention output for head h at that layer
    (after cascading patches from all earlier layers), and ``src_h`` is the
    corresponding head output from the source (C++) forward pass.

    True replacement semantics:
        mask=1 → pure source head output (full goal transfer)
        mask=0 → pure cascade dest output (no patching)

    Implemented via ``register_forward_pre_hook`` on o_proj — fully compatible with
    ``torch.autograd`` (unlike nnsight's thread-based SWAP which breaks the graph).

    Args:
        mt: ModelandTokenizer (frozen).
        dest_inputs: Tokenized destination prompt(s).  May include
            ``past_key_values`` and ``use_cache`` keys for KV-cache decoding.
        src_head_vals: Source o_proj inputs stacked across layers.
            Shape ``(batch, n_layers_tested, n_embd)`` for batch, or
            ``(n_layers_tested, n_embd)`` for a single instance.
        mask: HeadMask with learnable values (requires_grad=True).
        layers: Layer indices matching src_head_vals axis -2.
        return_past_key_values: If True, force ``use_cache=True`` and return
            ``(logits, past_key_values)`` so callers can thread the cache into
            the next call.  Patched K/V written into the cache during this
            forward pass will persist there for subsequent decode steps.

    Returns:
        ``(vocab_size,)`` float32 logits for 2D input (single instance), or
        ``(batch, vocab_size)`` float32 logits for 3D input (batch).
        Differentiable w.r.t. mask.values.
        If ``return_past_key_values=True``, returns ``(logits, past_key_values)``.
    """
    if src_head_vals.dim() == 2:
        src_head_vals = src_head_vals.unsqueeze(0)  # (1, L, n_embd)
        squeeze_output = True
    else:
        squeeze_output = False

    n_heads = mt.model.config.num_attention_heads
    n_embd = mt.n_embd
    head_dim = n_embd // n_heads
    mask_vals = mask.values  # (n_layers_tested, n_heads) requires_grad
    src_batch = src_head_vals  # (batch, L, n_embd) float32 CPU

    handles = []
    try:
        for li, layer in enumerate(layers):
            attn_name = mt.attn_module_name_format.format(layer)
            o_proj = get_module_nnsight(mt._model, attn_name + ".o_proj")

            def make_hook(li_: int):
                def pre_hook(module, args):
                    x = args[0]  # (batch, seq, n_embd) bfloat16 — live cascade input
                    dev = x.device
                    dtype = x.dtype
                    B_ = x.shape[0]

                    # Source vals for this layer — float32 for gradient precision
                    src = src_batch[:, li_].to(dev)  # (batch, n_embd) float32
                    src_3d = src.view(B_, n_heads, head_dim)

                    # Cascade (live dest) at last token
                    cas_3d = x[:, -1, :].float().view(B_, n_heads, head_dim)

                    # Per-head interpolation; m has requires_grad through mask_vals
                    m = mask_vals[li_].to(dev).view(1, n_heads, 1)
                    new_last = (alpha * m * src_3d + (1 - m) * cas_3d).view(
                        B_, n_embd
                    )  # (B, n_embd)

                    # Splice into sequence — torch.cat keeps gradient graph lean:
                    # only the last-token slice is live in autograd, not the full seq tensor.
                    new_seq = torch.cat(
                        [x[:, :-1, :], new_last.to(dtype).unsqueeze(1)], dim=1
                    )  # (batch, seq, n_embd)
                    return (new_seq,)

                return pre_hook

            handles.append(o_proj.register_forward_pre_hook(make_hook(li)))

        call_kwargs = dict(dest_inputs)
        if return_past_key_values:
            call_kwargs.setdefault("use_cache", True)
        output = mt._model(**call_kwargs)
    finally:
        for h in handles:
            h.remove()

    logits = output.logits[:, -1, :].float()  # (batch, vocab)
    out_logits = logits[0] if squeeze_output else logits
    if return_past_key_values:
        return out_logits, output.past_key_values
    return out_logits


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass
class MaskTrainingHistory:
    loss_causal: list[float] = field(default_factory=list)
    loss_sparse: list[float] = field(default_factory=list)
    loss_total: list[float] = field(default_factory=list)
    n_active: list[int] = field(default_factory=list)
    val_recall: list[list[float]] = field(
        default_factory=list
    )  # recall@1..K per val epoch
    val_epochs: list[int] = field(default_factory=list)  # epoch indices of val evals


@torch.inference_mode()
def _evaluate_val(
    mt,
    val_caches: list[PivotCache],
    mask: "HeadMask",
    layers: list[int],
    k: int = 5,
    aug_rng: random.Random | None = None,
    aug_cache_batch_size: int = 64,
) -> list[float]:
    """Compute recall@1..k on the val set using the current mask.

    For each instance runs a masked patched forward pass and records the top-k
    predicted tokens.  Compares against ``pivot_instance.target_top_token``.

    Args:
        aug_rng: If provided, val caches are augmented via ``augment_pivot_caches``
            before evaluation (same augmentation policy as training).  The rng is
            advanced in-place so each call produces a different augmentation.

    Returns:
        ``[recall@1, recall@2, ..., recall@k]`` floats.
    """
    if aug_rng is not None:
        val_caches = augment_pivot_caches(
            mt, val_caches, cache_batch_size=aug_cache_batch_size, rng=aug_rng
        )

    all_predictions: list[list[str]] = []
    all_targets: list[str] = []

    for cache in val_caches:
        dest_inputs = prepare_input([cache.dest_prompt], tokenizer=mt)

        if cache.per_head_deltas is not None:
            # Obsolete delta path — kept for backward compat
            logits = forward_with_masked_patch_obsolete(
                mt, dest_inputs, cache.per_head_deltas, mask, layers
            )  # (vocab,)
        else:
            # New src-replacement path (eager or lazy)
            if cache.src_head_vals is not None:
                src_hv = cache.src_head_vals  # (L, n_embd) — precomputed
            else:
                src_o = get_o_proj_inputs_batch(mt, [cache.src_prompt], layers)
                src_hv = torch.stack([src_o[ly][0] for ly in layers])  # (L, n_embd)
            logits = forward_with_src_replacement(
                mt, dest_inputs, src_hv, mask, layers
            )  # (vocab,)

        top_k_ids = logits.topk(k).indices.tolist()
        top_k_toks = [mt.tokenizer.decode([i]) for i in top_k_ids]
        all_predictions.append(top_k_toks)
        all_targets.append(cache.pivot_instance.target_top_token)

    return recall_metric(all_predictions, all_targets)


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
    augment: bool = False,
    augment_include_docstring: bool = False,
    augment_cache_batch_size: int = 64,
    augment_seed: int | None = None,
) -> tuple[HeadMask, MaskTrainingHistory]:
    """Learn a sparse head mask via projected gradient descent.

    Objective:
        L = KL(P_patched ∥ P_target) + lambda_sparse * mean(mask.values)

    KL direction: KL(P_patched ∥ P_target) = Σ P_patched * log(P_patched/P_target)
    — pushes the patched destination distribution toward the Python target.

    Instances are processed in batches of ``batch_size`` per optimizer step
    (``reduction="batchmean"`` keeps the KL loss scale consistent with the old
    per-instance ``reduction="sum"`` regardless of batch size).

    Path detection (automatic, based on first cache):
    - ``per_head_deltas is not None`` → obsolete eager delta path (backward compat)
    - ``src_head_vals is not None``   → new eager path (default; src forward cached)
    - neither                         → new lazy path (src-only forward per batch/epoch)

    The new eager path (``cache_pivot_data(..., precompute=True)``) only runs the
    source forward pass at cache time.  The destination forward happens live inside
    the training loop via ``forward_with_src_replacement``.

    The new lazy path (``cache_pivot_data(..., precompute=False)``) runs a
    src-only forward pass per batch per epoch — one extra forward pass per step.
    Required when ``augment=True`` because source prompts change each epoch.

    When ``augment=True``, a fresh random surface-form augmentation is applied to
    every instance at the start of each epoch (via ``augment_pivot_caches``).
    Requires ``cache_pivot_data(precompute=False)`` — precomputed ``src_head_vals``
    would be stale after per-epoch source-prompt augmentation.

    Args:
        mt: ModelandTokenizer (frozen — only mask params are updated).
        pivot_caches: Pre-cached data from cache_pivot_data().
        layers: Layer indices (must match pivot_caches[*].src_head_vals / per_head_deltas axis 0).
        n_epochs: Training epochs (default 20, from paper).
        lr: AdamW learning rate (default 1e-2, from paper).
        lambda_sparse: L1 sparsity coefficient (default 1e-2, from paper).
        batch_size: Instances per optimizer step (default 8).
        augment: If True, re-augment train and val instances at the start of every
            epoch.  Requires ``cache_pivot_data(precompute=False)`` — src prompts
            change every epoch so precomputed ``src_head_vals`` would be stale.
            pivot_caches must have src_sample/dest_sample populated (always true when
            built by cache_pivot_data).
        augment_include_docstring: Passed to prompt builders during augmentation.
        augment_cache_batch_size: Batch size for log_p_target recomputation during
            augmentation (default 64).
        augment_seed: Seed for the augmentation RNG. If None, non-reproducible.

    Returns:
        (trained HeadMask, MaskTrainingHistory)
    """
    n_layers_tested = len(layers)
    n_heads = mt.model.config.num_attention_heads

    _has_deltas = (
        pivot_caches[0].per_head_deltas is not None
    )  # obsolete eager delta path
    _has_src_vals = pivot_caches[0].src_head_vals is not None  # new eager src-val path
    _use_new_path = not _has_deltas  # True for all new caches

    if augment:
        assert _use_new_path, (
            "augment=True requires the new src-replacement path — call cache_pivot_data "
            "with precompute=False. The old delta path does not support augmentation."
        )
        _aug_rng = random.Random(augment_seed)

    # 80/20 train/val split.  When augment=True both splits are re-augmented each epoch.
    val_size = max(1, int(len(pivot_caches) * 0.2))
    train_size = len(pivot_caches) - val_size
    train_caches = pivot_caches[:train_size]
    val_caches = pivot_caches[train_size:]

    N = train_size  # training set size

    mask = HeadMask(n_layers_tested, n_heads)
    optimizer = AdamW(mask.parameters(), lr=lr)
    history = MaskTrainingHistory()

    # Freeze model parameters so backward() doesn't accumulate grads for them
    for param in mt._model.parameters():
        param.requires_grad_(False)

    batch_groups = [train_caches[i : i + batch_size] for i in range(0, N, batch_size)]

    # Pre-build everything that is cheap to cache outside the epoch loop.
    # When augment=True the batch lists are rebuilt per epoch — skip precomputation.
    if not augment:
        batch_log_p_list = [
            torch.stack([c.log_p_target for c in grp])
            for grp in batch_groups  # (B, vocab) CPU
        ]
        if _has_src_vals:
            # New eager path: precomputed src_head_vals; dest forward runs live.
            batch_src_hv_list = [
                torch.stack([c.src_head_vals for c in grp])
                for grp in batch_groups  # (B, L, n_embd) CPU float32
            ]
            batch_inputs_list = [
                prepare_input([c.dest_prompt for c in grp], tokenizer=mt)
                for grp in batch_groups
            ]
        elif _use_new_path:
            # New lazy path: recompute src_head_vals per batch per epoch.
            batch_src_prompts_list = [
                [c.src_prompt for c in grp] for grp in batch_groups
            ]
            batch_dest_prompts_list = [
                [c.dest_prompt for c in grp] for grp in batch_groups
            ]
        else:
            # Obsolete eager delta path.
            batch_inputs_list = [
                prepare_input([c.dest_prompt for c in grp], tokenizer=mt)
                for grp in batch_groups
            ]
            # Do NOT pre-stack deltas here: torch.stack would copy all N × (L, H, E)
            # float32 tensors that already live in train_caches[*].per_head_deltas,
            # doubling CPU RAM (~14 GB for N=512 on Qwen-32B).  Stack on-the-fly per
            # batch inside the epoch loop instead (negligible cost: 16 tensors × 28 MB).

    _path_desc = (
        "new-eager"
        if _has_src_vals
        else "new-lazy"
        if _use_new_path
        else "delta-obsolete"
    )
    logger.info(
        f"train_head_mask: {n_layers_tested}L × {n_heads}H = "
        f"{n_layers_tested * n_heads} mask params | "
        f"train={N} val={val_size} | batch_size={batch_size} ({len(batch_groups)} batches/epoch) | "
        f"{n_epochs} epochs | lr={lr} λ={lambda_sparse} | "
        f"path={_path_desc} | augment={augment}"
    )

    for epoch in range(n_epochs):
        epoch_kl = 0.0
        epoch_sparse = 0.0
        n_steps = 0

        # Per-epoch augmentation: build fresh batch lists from augmented caches.
        # Always lazy (augment_pivot_caches returns per_head_deltas=None).
        if augment:
            _aug_caches = augment_pivot_caches(
                mt,
                train_caches,
                include_docstring=augment_include_docstring,
                cache_batch_size=augment_cache_batch_size,
                rng=_aug_rng,
            )
            _aug_groups = [
                _aug_caches[i : i + batch_size] for i in range(0, N, batch_size)
            ]
            _batch_log_p_ep = [
                torch.stack([c.log_p_target for c in g]) for g in _aug_groups
            ]
            _batch_src_ep = [[c.src_prompt for c in g] for g in _aug_groups]
            _batch_dest_ep = [[c.dest_prompt for c in g] for g in _aug_groups]
            _batch_src_hv_ep = None  # augmented caches are always lazy
        else:
            _batch_log_p_ep = batch_log_p_list
            _batch_src_hv_ep = batch_src_hv_list if _has_src_vals else None
            _batch_src_ep = (
                batch_src_prompts_list if _use_new_path and not _has_src_vals else None
            )
            _batch_dest_ep = (
                batch_dest_prompts_list if _use_new_path and not _has_src_vals else None
            )

        for bi, batch_log_p in enumerate(_batch_log_p_ep):
            if _batch_src_hv_ep is not None:
                # New eager path: src_head_vals precomputed — just look them up.
                batch_src_hv = _batch_src_hv_ep[bi]  # (B, L, n_embd) CPU float32
                batch_inputs = batch_inputs_list[bi]
            elif _use_new_path:
                # New lazy path: run src-only forward pass per batch.
                src_ps = _batch_src_ep[bi]
                dest_ps = _batch_dest_ep[bi]
                B = len(src_ps)
                with torch.inference_mode():
                    src_o = get_o_proj_inputs_batch(mt, src_ps, layers)
                batch_src_hv = torch.stack(
                    [torch.stack([src_o[ly][i_] for ly in layers]) for i_ in range(B)]
                )  # (B, L, n_embd) float32 CPU
                del src_o
                free_gpu_cache()
                batch_inputs = prepare_input(dest_ps, tokenizer=mt)
            else:
                # Obsolete eager delta path.
                batch_inputs = batch_inputs_list[bi]
                # Stack on-the-fly — avoids the ~14 GB duplicate that pre-stacking would create.
                batch_deltas = torch.stack(
                    [c.per_head_deltas for c in batch_groups[bi]]
                )  # (B, L, H, E) CPU float32

            optimizer.zero_grad()

            if _use_new_path:
                logits = forward_with_src_replacement(
                    mt, batch_inputs, batch_src_hv, mask, layers
                )  # (B, vocab)
            else:
                logits = forward_with_masked_patch_obsolete(
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
                    f"  e{epoch + 1:03d} b{bi + 1:03d}/{len(_batch_log_p_ep)}"
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

        if checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
            # Val evaluation — augmented when augment=True.
            val_rec = _evaluate_val(
                mt,
                val_caches,
                mask,
                layers,
                aug_rng=_aug_rng if augment else None,
                aug_cache_batch_size=augment_cache_batch_size,
            )
            history.val_recall.append(val_rec)
            history.val_epochs.append(epoch + 1)
            recall_str = "  ".join(f"@{k + 1}={v:.3f}" for k, v in enumerate(val_rec))
            logger.info(f"  Val recall: {recall_str}")

            if checkpoint_dir is not None:
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
                            "val_recall": history.val_recall,
                            "val_epochs": history.val_epochs,
                        },
                    },
                    ckpt_path,
                )
                logger.info(f"  Checkpoint saved → {ckpt_path}")

    selected = mask.selected_heads(layers)
    logger.info(f"train_head_mask done: {len(selected)} active heads → {selected}")
    return mask, history
