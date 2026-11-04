"""Learned sparse head mask for attention head localization.

Learns a continuous mask over (n_layers_tested, n_heads) attention heads whose
cross-language interchange (C++ source → Python destination) minimizes
KL(P_patched ∥ P_target).

Based on arxiv:2510.26784 (filter heads via differentiable causal masking),
adapted for a cross-language interchange design.

Pipeline (o_proj input replacement, ``attention_premix`` in an nnterp trace):
    1. Cache o_proj inputs for source (C++) prompts per layer.
    2. Train mask: per-head interpolation of each layer's ``attention_premix`` (the o_proj input,
       ``[batch, seq, num_heads, head_dim]``), assigned back inside one trace of the destination.
       For each head h at layer l:
           attention_premix[:, -1, h] = mask[l,h]*src_h + (1-mask[l,h])*cascade_h
       This gives true replacement semantics: mask=1 → pure source, mask=0 → pure cascade.
       Sparsity: lambda * mean(mask) — scale-invariant to number of layers tested.
    3. Post-training threshold (mask > 0.5) → selected heads.

Mask parameterization:
    - nn.Parameter initialized to all ones
    - Clamped to [0,1] after each optimizer.step() (box-constrained GD)
    - No sigmoid, no temperature — raw values as interpolation weights
    - Selection: heads where values > 0.5 after training
"""

from __future__ import annotations

import dataclasses
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW

from src.functional import free_gpu_cache
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
# Data extraction
# ---------------------------------------------------------------------------


@torch.inference_mode()
def get_o_proj_inputs_batch(
    mt,
    prompts: list[str] | TokenizerOutput,
    layers: list[int],
    token_positions: list[int] | None = None,
    interventions: Optional[Callable] = None,
) -> dict[int, torch.Tensor]:
    """Extract the o_proj input (every head's output side by side) for each layer, for a batch of
    prompts: ``mt.layers[l].self_attn.attention_premix`` with the head axis flattened, in one trace.

    Left-padding (from prepare_input) ensures ``[:, -1, :]`` is the last real
    token for every sequence in the batch.

    Args:
        token_positions: Token positions to extract.  When ``None`` (default),
            extracts only the last position, returning ``(batch, num_heads * head_dim)``.
            When a list of ints is given, extracts those positions and returns
            ``(batch, len(token_positions), num_heads * head_dim)``.
        interventions: optional ``interventions(mt)``, called inside the same trace, so the
            values read reflect it (e.g. a query transplant or an OV boost of other heads, made
            with nnterp values). It runs in the trace's first invoke, which carries the inputs; the
            reads run in a second, empty invoke over the same rows. Each invoke keeps forward
            order on its own, so the intervention may touch any block, deeper or shallower than
            the reads. Composition rule: where both touch the same value (the same block's
            ``attention_premix``), the intervention's edit lands first and the read sees it, as a
            read placed after a forward pre-hook did. ``None`` (default) → a plain trace.

    Returns:
        ``{layer: tensor}`` (float32, CPU) where tensor shape is ``(batch, num_heads * head_dim)``
        for the default single-position case, or ``(batch, n_positions, num_heads * head_dim)`` when
        ``token_positions`` is specified.
    """
    if isinstance(prompts, list) and all(isinstance(p, str) for p in prompts):
        # Left-pad EXPLICITLY. The extraction below uses negative positions ([:, -1] /
        # [:, token_positions]), which only align with the real (right-aligned) tokens under LEFT
        # padding. In practice mt.tokenizer already left-pads (ModelandTokenizer sets it), so
        # this is defensive rather than a fix — but a RAW HF tokenizer defaults to right for many
        # models (e.g. Qwen), so we do not rely on the default. A caller passing a pre-tokenized
        # TokenizerOutput owns its own padding.
        inputs = prepare_input(prompts=prompts, tokenizer=mt, padding_side="left")
    elif isinstance(prompts, TokenizerOutput):
        inputs = prompts
    else:
        error_msg = f"prompts must be list of strings or TokenizerOutput.\nGot type {type(prompts)}."
        if isinstance(prompts, list):
            error_msg += f"\nList element types: {[type(p) for p in prompts]}"
        raise ValueError(error_msg)
    if "offset_mapping" in inputs:
        inputs.pop("offset_mapping")
    # nnsight requires envoys (layers) be accessed in ascending order within a trace, else
    # OutOfOrderError. Callers may pass layers in arbitrary order (e.g. heads drawn by random
    # sampling), so sort defensively — the returned dict is keyed by layer, so order-independent.
    layers = sorted(layers)
    attns = {layer: mt.layers[layer].self_attn for layer in layers}
    saves: dict[int, Any] = {}
    with mt.trace() as tracer:
        with tracer.invoke(dict(inputs)):
            if interventions is not None:
                interventions(mt)
        with tracer.invoke():
            for layer, attn in attns.items():
                premix = attn.attention_premix  # (batch, seq, num_heads, head_dim)
                sel = premix[:, -1] if token_positions is None else premix[:, token_positions]
                saves[layer] = sel.flatten(-2).save()
    return {layer: saves[layer].float().cpu() for layer in layers}


@torch.inference_mode()
def collect_head_steering_vecs(
    mt,
    src_tokenized: TokenizerOutput,
    selected_heads: list[tuple[int, int]],
    token_positions: list[int],
    interventions: Optional[Callable] = None,
) -> torch.Tensor:
    """Compute the summed residual-stream contribution of selected heads at each token position.

    For a linear o_proj, head h's contribution to the residual stream is:
        attention_premix[:, pos, h] @ self_attn.output_weight(h).T

    This sums those contributions across all selected heads, batched over the
    batch dimension of ``src_tokenized``.

    Args:
        mt: ModelandTokenizer.
        src_tokenized: Tokenized source prompt(s); supports batch_size >= 1.
        selected_heads: List of (layer, head_idx) pairs identifying which heads to include.
        token_positions: Token positions to extract (absolute indices into src_tokenized).
        interventions: forwarded to :func:`get_o_proj_inputs_batch` — called inside the
            extraction trace, so the aggregated head write reflects the intervention.

    Returns:
        Float32 CPU tensor of shape ``(batch_size, n_positions, n_embd)`` — one
        steering vector per (batch element, position).
    """
    from collections import defaultdict

    n_embd = mt.hidden_size
    n_heads = mt.num_heads
    head_dim = mt.head_dim  # NOT n_embd // n_heads (Qwen3 decouples head_dim from hidden_size)

    batch_size = src_tokenized["input_ids"].shape[0]

    # Group heads by layer to minimise forward passes
    heads_per_layer: dict[int, list[int]] = defaultdict(list)
    for layer, head in selected_heads:
        heads_per_layer[layer].append(head)

    layers = list(heads_per_layer.keys())

    # Single forward pass — extract o_proj inputs at all requested positions, under
    # `interventions` when given, so the aggregated goal-head write below reflects it.
    o_proj_inputs = get_o_proj_inputs_batch(
        mt, src_tokenized, layers, token_positions=token_positions,
        interventions=interventions,
    )  # {layer: (batch_size, n_positions, num_heads * head_dim)}

    n_pos = len(token_positions)
    steering_vecs = torch.zeros(batch_size, n_pos, n_embd, dtype=torch.float32)

    for layer, heads in heads_per_layer.items():
        attn = mt.layers[layer].self_attn
        dev = attn.device  # the block's device: only the selected heads' weights go to float32

        # (batch_size, n_positions, num_heads, head_dim) float32
        inp = o_proj_inputs[layer].to(dev).unflatten(-1, (n_heads, head_dim))

        for h in heads:
            h_inp = inp[:, :, h]  # (B, n_pos, head_dim)
            # (n_embd, head_dim): nnterp's slice of the output projection for head h, the adapted weight while a
            # PEFT adapter is on
            W_o_h = attn.output_weight(h).detach().float()
            head_outputs = (h_inp @ W_o_h.t()).cpu()  # (B, n_pos, n_embd)
            steering_vecs += head_outputs

    return steering_vecs  # (batch_size, n_positions, n_embd) float32 CPU


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
    """

    pivot_instance: PivotInstance
    dest_prompt: str  # raw string — destination prompt up to pivot prefix
    log_p_target: (
        torch.Tensor
    )  # (vocab_size,) float32 CPU — log of target next-token dist
    src_sample: (
        SinglePredicateSample  # source predicate sample (targ_sample == src_sample)
    )
    dest_sample: SinglePredicateSample  # destination predicate sample
    src_prompt: str | None = None  # populated when precompute=False (lazy)
    src_head_vals: torch.Tensor | None = None
    # (n_layers_tested, num_heads * head_dim) float32 CPU — o_proj inputs from the source run,
    # stacked across layers.  ~0.7 MB/instance for Qwen-32B; ~1 GB for N=1480.

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
            )  # {layer: (B, num_heads * head_dim)}
            # Stack layers into (B, L, num_heads * head_dim)
            src_val_chunks.append(torch.stack([src_o[ly] for ly in layers], dim=1))
            del src_o
            free_gpu_cache()
            logger.debug(f"  src head vals: {min(i + cache_batch_size, N)}/{N}")
        all_src_head_vals = torch.cat(src_val_chunks, dim=0)  # (N, L, num_heads * head_dim)
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
    always lazy (``src_prompt`` populated) and carry
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
            src_sample=aug_src_samples[i],
            dest_sample=aug_dest_samples[i],
            src_prompt=aug_src_prompts[i],
        )
        for i in range(N)
    ]


# ---------------------------------------------------------------------------
# Differentiable patched forward pass
# ---------------------------------------------------------------------------


def _interpolate_heads(
    x: torch.Tensor,
    src: torch.Tensor,
    m: torch.Tensor,
    alpha: float,
    positions: list[int],
) -> torch.Tensor:
    """``x`` with each position ``p`` in ``positions`` replaced by
    ``alpha * m * src + (1 - m) * x[:, p]`` per head, in float32, cast back to ``x.dtype``.

    Args:
        x: ``(batch, seq, num_heads, head_dim)``, the live ``attention_premix`` of one layer.
        src: ``(batch or 1, num_heads, head_dim)`` float32 source head values.
        m: ``(num_heads,)``, the layer's mask row (requires grad during training).
        positions: sorted, de-duplicated, non-negative positions.

    Spliced with ``torch.cat``: the untouched slices are views of ``x`` with no path to ``m``, so
    only the patched positions carry a gradient through the mask.
    """
    src = src.to(x.device)
    m = m.to(x.device).view(1, -1, 1)
    pieces = []
    prev = 0
    for p in positions:
        if p > prev:
            pieces.append(x[:, prev:p])
        cascade = x[:, p].float()  # (batch, num_heads, head_dim)
        pieces.append((alpha * m * src + (1 - m) * cascade).to(x.dtype).unsqueeze(1))
        prev = p + 1
    if prev < x.shape[1]:
        pieces.append(x[:, prev:])
    return torch.cat(pieces, dim=1)


def forward_with_src_replacement(
    mt,
    dest_inputs: Any,
    src_head_inps: torch.Tensor,
    mask: HeadMask,
    mask_layers: Optional[list[int]] = None,
    alpha: float = 1.0,
    patch_positions: Optional[list[int]] = None,
) -> torch.Tensor:
    """Run dest_inputs with mask-weighted per-head o_proj input replacement.

    For each layer l (index li in ``mask_layers``), each head h, and each token position p
    in ``patch_positions``:
        attention_premix[:, p, h] = alpha * mask[li,h] * src_h + (1 - mask[li,h]) * cascade_h

    where ``attention_premix`` is ``mt.layers[l].self_attn.attention_premix`` (the o_proj input,
    ``[batch, seq, num_heads, head_dim]``), ``cascade_h`` is the live dest value of head h at
    position p (after the patches of all earlier layers), and ``src_h`` is the same head's value
    from the source (C++) forward pass.

    True replacement semantics:
        mask=1 → pure source head output (full goal transfer)
        mask=0 → pure cascade dest output (no patching)

    One ``mt.trace`` of the destination. Per layer, in forward order, the interpolation is
    computed in float32, cast back to the model dtype, and the layer's ``attention_premix`` is
    ASSIGNED a ``torch.cat`` of the untouched slices and the patched positions: writing the
    interpolation into the served value in place fails at ``backward()`` ("modified by an
    inplace operation"), and the splice keeps the graph lean, since only the patched positions
    have a path to ``mask.values``.
    Differentiable w.r.t. ``mask.values`` under the caller's grad mode: ``train_head_mask``
    runs it under ``torch.enable_grad()``; under ``no_grad`` / ``inference_mode`` it is a plain
    patched forward.

    Positions: the trace gives a left-padded row mask-derived ``position_ids`` (its first real
    token is position 0), as HF generation does. The version before nnterp called the HF
    module directly, which numbers a padded row from its first pad. RoPE is relative, so the two
    differ only through the rounding of the absolute angles, which bf16 carries through the
    layers (mask gradient: 1e-6 relative on the 0.5B in float32, about 2 % on the 7B in bf16 at
    batch 8); unpadded batches are identical.

    Limitation — single source broadcast across positions:
        ``src_head_inps`` holds a single snapshot (per layer) that is *broadcast* across every
        position in ``patch_positions``. Per-position source variation is not supported. This
        matches the project's working assumption of a position-invariant goal representation.

    Args:
        mt: ModelandTokenizer (frozen).
        dest_inputs: Tokenized destination prompt(s) (``input_ids``, ``attention_mask``).
        src_head_inps: Source o_proj inputs stacked across layers.
            Shape ``(batch, n_layers_tested, num_heads * head_dim)`` for a batch (a batch of
            one broadcasts over the destination rows), or
            ``(n_layers_tested, num_heads * head_dim)`` for a single instance.
        mask: HeadMask with learnable values (requires_grad=True).
        mask_layers: Layer indices matching src_head_inps axis -2.
        alpha: Scale on the source term (1.0 = pure replacement at mask=1).
        patch_positions: Token positions to patch. Negative indices are allowed (``-1`` = last
            position). Duplicates are removed. ``None`` defaults to ``[-1]``.

    Returns:
        ``(vocab_size,)`` float32 logits at the last position for 2D ``src_head_inps`` (single
        instance), or ``(batch, vocab_size)`` for 3D. Differentiable w.r.t. ``mask.values``.
    """
    # `mask_layers` is the ordered key that decodes the mask's ROWS (row li <-> mask_layers[li]).
    # It is NOT "which layers to intervene at": slicing it silently shifts every row.
    # `None` = identity (a full-depth mask).
    if mask_layers is None:
        mask_layers = list(range(mask.values.shape[0]))
    _n_src = src_head_inps.shape[-2]
    if len(mask_layers) != _n_src or len(mask_layers) != mask.values.shape[0]:
        raise ValueError(
            f"mask_layers/source/mask disagree: len(mask_layers)={len(mask_layers)}, "
            f"src layer axis={_n_src}, mask rows={mask.values.shape[0]}. They align POSITIONALLY "
            f"(row li <-> mask_layers[li]), so a mismatch SHIFTS every row rather than failing."
        )
    if len(set(mask_layers)) != len(mask_layers):
        raise ValueError(f"mask_layers has duplicate layers: {list(mask_layers)}")

    squeeze_output = src_head_inps.dim() == 2
    if squeeze_output:
        src_head_inps = src_head_inps.unsqueeze(0)  # (1, L, num_heads * head_dim)

    seq_len = dest_inputs["input_ids"].shape[1]
    pos_arg = [-1] if patch_positions is None else list(patch_positions)
    # Normalize negatives, de-dup, sort
    positions = sorted({p if p >= 0 else seq_len + p for p in pos_arg})
    if positions and (positions[0] < 0 or positions[-1] >= seq_len):
        raise IndexError(
            f"patch_positions out of range for seq_len={seq_len}: {pos_arg} -> {positions}"
        )

    n_heads, head_dim = mt.num_heads, mt.head_dim
    # The attention envoys, bound outside the trace in forward (ascending layer) order, each with
    # its mask row. No positions -> no patching (a plain forward).
    patched = [(li, mt.layers[layer].self_attn) for layer, li in
               sorted((layer, li) for li, layer in enumerate(mask_layers))] if positions else []

    # The served value goes straight into _interpolate_heads rather than being bound to a name
    # here: a tensor bound in a trace body stays referenced until the trace ends (`del` does not
    # release it), which kept one extra (batch, seq, num_heads * head_dim) tensor alive at the
    # forward's memory peak.
    with mt.trace(dict(dest_inputs)):
        for li, attn in patched:
            attn.attention_premix = _interpolate_heads(
                attn.attention_premix,
                src_head_inps[:, li].view(-1, n_heads, head_dim),
                mask.values[li],
                alpha,
                positions,
            )
        logits = mt.logits[:, -1].save()  # (batch, vocab)

    logits = logits.float()
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

        if cache.src_head_vals is not None:
            src_hv = cache.src_head_vals  # (L, num_heads * head_dim) — precomputed
        else:
            src_o = get_o_proj_inputs_batch(mt, [cache.src_prompt], layers)
            src_hv = torch.stack([src_o[ly][0] for ly in layers])  # (L, num_heads * head_dim)
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
    *,
    n_epochs: int,
    lr: float,
    lambda_sparse: float,
    batch_size: int,
    checkpoint_every: int,
    checkpoint_dir: str | None = None,
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

    Two paths:
    - ``augment=False``: needs ``cache_pivot_data(..., precompute=True)``; the source
      forward pass runs only at cache time.  The destination forward happens live
      inside the training loop via ``forward_with_src_replacement``.
    - ``augment=True``: needs ``cache_pivot_data(..., precompute=False)``.  A fresh
      random surface-form augmentation is applied to every instance at the start of
      each epoch (via ``augment_pivot_caches``), and a src-only forward pass runs per
      batch — one extra forward pass per step.

    Args:
        mt: ModelandTokenizer (frozen — only mask params are updated).
        pivot_caches: Pre-cached data from cache_pivot_data().
        layers: Layer indices (must match pivot_caches[*].src_head_vals axis 0).
        n_epochs: Training epochs.
        lr: AdamW learning rate.
        lambda_sparse: L1 sparsity coefficient.
        batch_size: Instances per optimizer step.
        checkpoint_every: Run val recall (and write a checkpoint into
            ``checkpoint_dir``, if given) every K epochs.
        augment: If True, re-augment train and val instances at the start of every
            epoch (see the two paths above).
        augment_include_docstring: Passed to prompt builders during augmentation.
        augment_cache_batch_size: Batch size for log_p_target recomputation during
            augmentation (default 64).
        augment_seed: Seed for the augmentation RNG. If None, non-reproducible.

    Returns:
        (trained HeadMask, MaskTrainingHistory)
    """
    n_layers_tested = len(layers)
    n_heads = mt.num_heads

    if augment:
        _aug_rng = random.Random(augment_seed)
    else:
        assert pivot_caches[0].src_head_vals is not None, (
            "augment=False requires cache_pivot_data(..., precompute=True)"
        )

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
    mt.requires_grad_(False)

    batch_groups = [train_caches[i : i + batch_size] for i in range(0, N, batch_size)]

    # Pre-build everything that is cheap to cache outside the epoch loop.
    # When augment=True the batch lists are rebuilt per epoch — skip precomputation.
    if not augment:
        batch_log_p_list = [
            torch.stack([c.log_p_target for c in grp])
            for grp in batch_groups  # (B, vocab) CPU
        ]
        # Precomputed src_head_vals; dest forward runs live.
        batch_src_hv_list = [
            torch.stack([c.src_head_vals for c in grp])
            for grp in batch_groups  # (B, L, num_heads * head_dim) CPU float32
        ]
        batch_inputs_list = [
            prepare_input([c.dest_prompt for c in grp], tokenizer=mt)
            for grp in batch_groups
        ]

    _path_desc = "lazy+augment" if augment else "precomputed"
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

        # Per-epoch augmentation: build fresh batch lists from augmented caches
        # (always lazy: src head values are recomputed per batch).
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
            _batch_src_hv_ep = batch_src_hv_list

        for bi, batch_log_p in enumerate(_batch_log_p_ep):
            if _batch_src_hv_ep is not None:
                # Precomputed src_head_vals — just look them up.
                batch_src_hv = _batch_src_hv_ep[bi]  # (B, L, num_heads * head_dim) CPU float32
                batch_inputs = batch_inputs_list[bi]
            else:
                # Lazy (augment): run src-only forward pass per batch.
                src_ps = _batch_src_ep[bi]
                dest_ps = _batch_dest_ep[bi]
                B = len(src_ps)
                with torch.inference_mode():
                    src_o = get_o_proj_inputs_batch(mt, src_ps, layers)
                batch_src_hv = torch.stack(
                    [torch.stack([src_o[ly][i_] for ly in layers]) for i_ in range(B)]
                )  # (B, L, num_heads * head_dim) float32 CPU
                del src_o
                free_gpu_cache()
                batch_inputs = prepare_input(dest_ps, tokenizer=mt)

            optimizer.zero_grad()

            # Gradients on explicitly: the eval scripts turn them off process-wide
            # (torch.set_grad_enabled(False)), and the mask needs them here.
            with torch.enable_grad():
                logits = forward_with_src_replacement(
                    mt, batch_inputs, batch_src_hv, mask, layers
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
