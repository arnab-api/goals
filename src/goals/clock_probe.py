"""Clock signal (c) probe utilities — two-signal hypothesis.

Three-stage pipeline (each stage is a separate function):

    Stage 1 — Augmentation
        augment_sample_for_probe(tp, sample, harmless_prefills)
            → list[(MultiPredicateSample, prefill_str)]
        Generates a cross-product of structural augmentations (variable rename,
        type-hint toggle, predicate-order swap) × prefill variants (empty +
        harmless prefills like "True and ", "False or ").

    Stage 2 — Generation + evaluation
        generate_and_evaluate_sample(mt, prompt, sample, prefill, ...)
            → (prompt_ids, gen_ids) | None
        Generates greedily from a pre-built prompt, evaluates the completion
        against unit tests, returns token ids on pass or None on failure/timeout.

    Stage 3 — Latent caching
        cache_latents_for_sequence(mt, prompt_ids, gen_ids, layers, ...)
            → SequenceLatents
        Runs a single nnsight forward pass over the full sequence to extract
        residual / MLP / attn activations at every token position.

Convenience wrapper:
    generate_and_cache_latents(mt, sample, prefill, prompt, layers, ...)
        Calls stages 2 and 3 in sequence; returns SequenceLatents | None.

Probe utilities:
    find_in_progress_mask, collect_probe_examples, train_probe, eval_probe.
"""

from __future__ import annotations

import os
import re
import signal
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, classification_report
from tqdm import tqdm

from src.functional import get_module_nnsight
from src.goals.augment import (
    _FN_PREFIXES,
    _VAR_VOCAB,
    extract_short_descriptions,
    extract_var_names_from_code,
    rename_function,
    rename_variable,
    swap_predicate_order,
    toggle_type_hints,
)
from src.goals.data import SingleCodeEval
from src.goals.eval import evaluate_single_code
from src.goals.predicate_data import MultiPredicateSample, TargetPredicate
from src.tokens import find_token_range, prepare_input

# ---------------------------------------------------------------------------
# Default harmless prefills for the clock probe
# These are appended after "return " in the code stub.
# They do not commit to a specific implementation of either predicate.
# ---------------------------------------------------------------------------

DEFAULT_HARMLESS_PREFILLS: list[str] = [
    "",  # empty — model chooses freely
    "True and ",
    "False or ",
    "bool(",
]


# ---------------------------------------------------------------------------
# SequenceLatents dataclass
# ---------------------------------------------------------------------------


@dataclass
class SequenceLatents:
    """Activations for one greedy-generated sequence.

    Two sets of activation tensors are stored per layer (all float16, CPU):

    Generated tokens  — shape (n_gen, hidden_dim):
        residual, mlp, attn
        Index 0 corresponds to absolute token position ``prompt_len``.

    Code stub tokens  — shape (n_code, hidden_dim), where n_code = prompt_len - code_start:
        code_residual, code_mlp, code_attn
        Index 0 corresponds to absolute token position ``code_start``
        (the ``<code>\\ndef`` token in the prompt).

    Offset summary::

        absolute position = code_start + code_tensor_index       (code span)
        absolute position = prompt_len  + gen_tensor_index        (gen span)

    ``code_start == prompt_len`` is the sentinel for "code pattern not found"
    (produces empty code dicts).  prompt_len and prompt_ids are retained for
    span-labeling bookkeeping (find_token_range offsets).
    """

    prompt_len: int  # number of prompt tokens
    prompt_ids: list[int]  # tokenized prompt (not including generated tokens)
    gen_ids: list[int]  # generated token ids (not including prompt)
    full_text: str  # decoded full sequence (prompt + generated)
    residual: dict[int, torch.Tensor]  # layer → (n_gen, hidden_dim) float16 CPU
    code_start: int = 0  # absolute token index of <code>\ndef; sentinel=prompt_len
    code_residual: dict[int, torch.Tensor] = field(
        default_factory=dict
    )  # (n_code, hidden_dim)
    code_mlp: dict[int, torch.Tensor] = field(default_factory=dict)
    code_attn: dict[int, torch.Tensor] = field(default_factory=dict)
    mlp: dict[int, torch.Tensor] = field(default_factory=dict)
    attn: dict[int, torch.Tensor] = field(default_factory=dict)


# Pattern that uniquely identifies the code stub in the prompt.
# The instruction also contains a bare <code> tag, but only the code stub
# starts with <code>\ndef — so this two-token pattern is unambiguous.
_CODE_SPAN_RE = re.compile(r"<code>\ndef")
# _CODE_SPAN_RE = re.compile(r"<code>")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_code_token_start(prompt_text: str, tokenizer, prompt_len: int) -> int:
    """Return the absolute token index where the code stub begins in the prompt.

    Searches for ``<code>\\ndef``, which uniquely marks the start of the code
    stub (the bare ``<code>`` in the instruction is never followed by ``\\ndef``).
    Returns ``prompt_len`` as a sentinel when the pattern is not found,
    which produces an empty code span (n_code = prompt_len - code_start = 0).

    Must be called with the decoded *prompt only* (not prompt + generation) to
    avoid matching a second ``<code>\\ndef`` that the model may regenerate in
    its completion.

    Args:
        prompt_text: Decoded prompt tokens (NOT including generated tokens).
        tokenizer: Model tokenizer for find_token_range.
        prompt_len: Number of prompt tokens — upper bound and fallback.
    """
    m = _CODE_SPAN_RE.search(prompt_text)
    if m is None:
        return prompt_len - 1
    try:
        tok_start, _ = find_token_range(
            string=prompt_text,
            substring=prompt_text[m.start() : m.end()],
            tokenizer=tokenizer,
            occurrence=-1,
        )
        # print(tok_start, _)
        return min(tok_start, prompt_len - 1)
    except Exception:
        return prompt_len - 1  # immediate left token should be at least in the cache


def _untuple(x: Any) -> torch.Tensor:
    """Return x[0] if x is a tuple/list, else x (handles LLM layer output tuples)."""
    if isinstance(x, (tuple, list)):
        return x[0]
    return x


class _EvalTimeout(Exception):
    pass


def _eval_with_timeout(
    full_code: str,
    unit_tests: list,
    function_name: str,
    timeout_secs: int,
) -> SingleCodeEval:
    """evaluate_single_code with SIGALRM-based timeout (Linux main-thread only)."""

    def _handler(_signum, _frame):
        raise _EvalTimeout()

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout_secs)
    try:
        result = evaluate_single_code(full_code, unit_tests, function_name)
    except _EvalTimeout:
        result = SingleCodeEval(
            code=full_code,
            verdict="TIMEOUT",
            warnings=[],
            unit_test_results=[],
            error_message=f"timed out after {timeout_secs}s",
        )
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
    return result


# ---------------------------------------------------------------------------
# Stage 1 — Augmentation
# ---------------------------------------------------------------------------


def augment_sample_for_probe(
    tp: TargetPredicate,
    sample: MultiPredicateSample,
    harmless_prefills: list[str] | None = None,
) -> list[tuple[MultiPredicateSample, str]]:
    """Generate (augmented_sample, prefill) pairs for probe training data.

    Structural augmentations applied to the sample:
      - Original (no change)
      - Variable rename: up to 3 alternatives from _VAR_VOCAB
      - Type-hint toggle: remove param annotations from def line
      - Predicate-order swap: swap the two predicate clauses in the docstring

    Each structural variant is then combined with every entry in harmless_prefills
    (appended after "return " in the code stub).

    anchor_patterns remain in the TargetPredicate (not the sample), so they are
    valid across all augmented variants — the generated *code* pattern for the
    predicate does not depend on the docstring wording or variable name.

    Args:
        tp: The TargetPredicate owning the sample (needed for extract_short_descriptions).
        sample: The original MultiPredicateSample.
        harmless_prefills: Prefill strings to append after "return ".
            Defaults to DEFAULT_HARMLESS_PREFILLS.

    Returns:
        List of (augmented_sample, prefill) pairs.  Original is always included.
    """
    if harmless_prefills is None:
        harmless_prefills = DEFAULT_HARMLESS_PREFILLS

    if not sample.is_original:
        # Only augment originals to avoid combinatorial blowup
        return [(sample, pf) for pf in harmless_prefills]

    # Ensure short_descriptions are populated (needed for swap_predicate_order)
    extract_short_descriptions(tp)

    structural_variants: list[MultiPredicateSample] = [sample]  # original first

    # Variable rename
    var_names = extract_var_names_from_code(sample.code)
    primary_var = var_names[0] if var_names else None
    if primary_var is not None:
        for new_var in _VAR_VOCAB.get(primary_var, [])[:3]:
            structural_variants.append(rename_variable(sample, primary_var, new_var))

    # Type-hint toggle
    structural_variants.append(toggle_type_hints(sample))

    # Function rename (only if fn starts with "is_")
    fn = sample.function_name or ""
    if fn.startswith("is_"):
        suffix = fn[len("is_") :]
        for prefix in _FN_PREFIXES:
            structural_variants.append(rename_function(sample, prefix + suffix))

    # Predicate-order swap (requires short_descriptions to be set)
    swapped = swap_predicate_order(sample)
    if swapped is not None:
        structural_variants.append(swapped)

    return [(samp, pf) for samp in structural_variants for pf in harmless_prefills]


# ---------------------------------------------------------------------------
# Stage 2 — Generation + evaluation
# ---------------------------------------------------------------------------


def generate_and_evaluate_sample(
    mt,
    prompt: str,
    sample: MultiPredicateSample,
    prefill: str = "",
    max_new_tokens: int = 150,
    eval_timeout: int = 5,
    code_stop_tag: str | None = "</code>",
) -> tuple[list[int], list[int]] | None:
    """Generate greedily from prompt, evaluate against unit tests.

    The generated text is evaluated as:
        full_code = sample.code + (" " if needed) + prefill + stripped_generation

    This matches the logic in run_evaluation_timed (adversarial_persistence.py).

    Args:
        mt: ModelandTokenizer.
        prompt: Full prompt string (already built, e.g. via apply_chat_template).
        sample: MultiPredicateSample — provides unit_tests, code stub, function_name.
        prefill: The prefill string that was appended to sample.code when building
            the prompt. Used to reconstruct full_code for evaluation.
        max_new_tokens: Maximum tokens to generate.
        stop_ids: Token IDs that trigger early stopping (EOS, </code>, etc.).
        eval_timeout: SIGALRM timeout in seconds for evaluate_single_code.
        code_stop_tag: The stop tag to use for extracting the generated code.
    Returns:
        (prompt_ids, gen_ids) if the generation passes all unit tests.
        None if generation fails evaluation or times out.
    """
    prompt_inputs = prepare_input([prompt], tokenizer=mt)
    prompt_ids: list[int] = prompt_inputs["input_ids"][0].tolist()

    input_ids_tensor = torch.tensor([prompt_ids]).to(mt.device)

    with torch.no_grad():
        generated = mt._model.generate(
            input_ids=input_ids_tensor,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=(
                mt.tokenizer.pad_token_id
                if mt.tokenizer.pad_token_id is not None
                else mt.tokenizer.eos_token_id
            ),
        )

    gen_ids: list[int] = generated[0].tolist()[len(prompt_ids) :]
    # print(f"Generated token IDs: {gen_ids}")
    lm_completion = mt.tokenizer.decode(gen_ids, skip_special_tokens=False)
    if code_stop_tag:
        cur_stop_tag = code_stop_tag if code_stop_tag in lm_completion else "\n"
        lm_completion = lm_completion.split(cur_stop_tag)[0].rstrip()

    incomplete_prefix = sample.code + prefill
    if incomplete_prefix.endswith(" ") is False:
        incomplete_prefix += " "
    full_code = incomplete_prefix + lm_completion
    # print(f"Evaluating generated code:\n{full_code}")

    eval_result = _eval_with_timeout(
        full_code,
        list(sample.unit_tests),
        sample.function_name or "func",
        eval_timeout,
    )

    if eval_result.verdict == "PASSED":
        return prompt_ids, gen_ids
    return None


def evaluate_gen_ids(
    gen_ids: list[int],
    sample: MultiPredicateSample,
    prefill: str,
    tokenizer,
    eval_timeout: int = 5,
    code_stop_tag: str | None = "</code>",
) -> bool:
    """Evaluate generated token ids against sample unit tests.

    Mirrors the evaluation logic in generate_and_evaluate_sample, decoupled
    from generation so it can be used after batched generation.

    Args:
        gen_ids: Generated token ids (e.g., from batch_generate).
        sample: MultiPredicateSample providing unit_tests, code, function_name.
        prefill: Prefill string appended to sample.code before generation.
        tokenizer: Model tokenizer for decoding gen_ids.
        eval_timeout: SIGALRM timeout in seconds.
        code_stop_tag: String tag to truncate lm_completion at.

    Returns:
        True if all unit tests pass, False otherwise (including timeouts).
    """
    lm_completion = tokenizer.decode(gen_ids, skip_special_tokens=False)
    if code_stop_tag:
        cur_stop_tag = code_stop_tag if code_stop_tag in lm_completion else "\n"
        lm_completion = lm_completion.split(cur_stop_tag)[0].rstrip()

    incomplete_prefix = sample.code + prefill
    if not incomplete_prefix.endswith(" "):
        incomplete_prefix += " "
    full_code = incomplete_prefix + lm_completion

    eval_result = _eval_with_timeout(
        full_code,
        list(sample.unit_tests),
        sample.function_name or "func",
        eval_timeout,
    )
    return eval_result.verdict == "PASSED"


def batch_generate(
    mt,
    prompts: list[str],
    max_new_tokens: int = 150,
) -> list[tuple[list[int], list[int]]]:
    """Batched greedy generation for a list of prompts.

    Tokenizes all prompts with left-padding (via prepare_input) and runs a
    single batched HuggingFace generate call — much faster than sequential
    single-prompt generation when len(prompts) > 1.

    Args:
        mt: ModelandTokenizer.
        prompts: List of prompt strings.
        max_new_tokens: Maximum new tokens per sequence.

    Returns:
        List of (prompt_ids, gen_ids) one per prompt.  prompt_ids is the
        unpadded tokenization of the prompt.  gen_ids may include the EOS
        token — callers strip it as needed (evaluate_gen_ids does this via
        code_stop_tag string matching).
    """
    if not prompts:
        return []

    # Batch tokenize with left-padding; recover per-sequence real lengths
    # from the attention mask (1 = real token, 0 = left-pad).
    batch_inputs = prepare_input(prompts, tokenizer=mt)
    padded_input_len: int = batch_inputs["input_ids"].shape[1]
    input_lens: list[int] = batch_inputs["attention_mask"].sum(dim=1).tolist()

    input_ids = batch_inputs["input_ids"].to(mt.device)
    attention_mask = batch_inputs["attention_mask"].to(mt.device)

    pad_token_id = (
        mt.tokenizer.pad_token_id
        if mt.tokenizer.pad_token_id is not None
        else mt.tokenizer.eos_token_id
    )

    with torch.no_grad():
        outputs = mt._model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_token_id,
        )

    # outputs: (batch, padded_input_len + n_gen) — slice out gen tokens per seq
    results: list[tuple[list[int], list[int]]] = []
    for i, real_len in enumerate(input_lens):
        # Recover unpadded prompt_ids: left-padding puts real tokens at the end
        start_idx = padded_input_len - real_len
        prompt_ids = batch_inputs["input_ids"][i, start_idx:].tolist()
        gen_ids = outputs[i][padded_input_len:].tolist()
        results.append((prompt_ids, gen_ids))

    return results


# ---------------------------------------------------------------------------
# Stage 3 — Latent caching
# ---------------------------------------------------------------------------


@torch.no_grad()
def cache_latents_for_sequence(
    mt,
    prompt_ids: list[int],
    gen_ids: list[int],
    layers: list[int],
    include_mlp: bool = True,
    include_attn: bool = True,
) -> SequenceLatents:
    """Cache residual / MLP / attn activations for a complete generated sequence.

    Runs a single nnsight forward pass over prompt_ids + gen_ids.  Saves
    activations at every token position for each requested layer.

    Args:
        mt: ModelandTokenizer.
        prompt_ids: Prompt token ids (from prepare_input).
        gen_ids: Generated token ids (from generate_and_evaluate_sample).
        layers: Layer indices to cache.
        include_mlp: Whether to cache MLP module outputs.
        include_attn: Whether to cache attention module outputs.

    Returns:
        SequenceLatents with tensors of shape (seq_len, hidden_dim) per layer.
    """
    full_ids = prompt_ids + gen_ids
    full_text: str = mt.tokenizer.decode(full_ids, skip_special_tokens=False)

    full_inputs = {
        "input_ids": torch.tensor([full_ids]),
        "attention_mask": torch.ones(1, len(full_ids), dtype=torch.long),
    }

    res_saves: dict[int, Any] = {}
    mlp_saves: dict[int, Any] = {}
    attn_saves: dict[int, Any] = {}

    with mt.trace(full_inputs, scan=False, validate=False):
        for layer_idx in layers:
            if include_attn:
                attn_mod = get_module_nnsight(
                    mt, mt.attn_module_name_format.format(layer_idx)
                )
                attn_saves[layer_idx] = attn_mod.output.save()

            if include_mlp:
                mlp_mod = get_module_nnsight(
                    mt, mt.mlp_module_name_format.format(layer_idx)
                )
                mlp_saves[layer_idx] = mlp_mod.output.save()

            layer_mod = get_module_nnsight(mt, mt.layer_name_format.format(layer_idx))
            res_saves[layer_idx] = layer_mod.output.save()

    n_prompt = len(prompt_ids)
    code_start = _find_code_token_start(
        mt.tokenizer.decode(prompt_ids, skip_special_tokens=False),
        mt.tokenizer,
        n_prompt,
    )
    has_code = code_start < n_prompt

    def _slice(saves: dict[int, Any], start: int, end: int) -> dict[int, torch.Tensor]:
        result = {}
        for idx, saved in saves.items():
            t = _untuple(saved)
            if t.ndim == 3:
                t = t.squeeze(0)  # (1, seq, hidden) → (seq, hidden)
            # .half() on a slice creates a new tensor with its own storage
            result[idx] = t[start:end].half().cpu()
        return result

    return SequenceLatents(
        prompt_len=n_prompt,
        prompt_ids=prompt_ids,
        gen_ids=gen_ids,
        full_text=full_text,
        residual=_slice(res_saves, n_prompt, None),
        code_start=code_start,
        code_residual=_slice(res_saves, code_start, n_prompt) if has_code else {},
        code_mlp=_slice(mlp_saves, code_start, n_prompt)
        if (has_code and include_mlp)
        else {},
        code_attn=_slice(attn_saves, code_start, n_prompt)
        if (has_code and include_attn)
        else {},
        mlp=_slice(mlp_saves, n_prompt, None) if include_mlp else {},
        attn=_slice(attn_saves, n_prompt, None) if include_attn else {},
    )


@torch.no_grad()
def cache_latents_batch(
    mt,
    sequences: list[tuple[list[int], list[int]]],
    layers: list[int],
    include_mlp: bool = True,
    include_attn: bool = True,
) -> list[SequenceLatents]:
    """Cache activations for a batch of sequences in a single forward pass.

    Sequences are right-padded to the same length.  Activations at padded
    positions are discarded; only real token positions are returned.
    Right-padding is safe because causal attention prevents pad tokens (at the
    end) from contaminating activations at earlier real token positions.

    Throughput scales roughly linearly with batch size up to memory limits.
    Keep batch size modest (8–16) for full-layer runs (64 layers × 3 types)
    to avoid OOM on the GPU activation saves.

    Args:
        mt: ModelandTokenizer.
        sequences: List of (prompt_ids, gen_ids) pairs.
        layers: Layer indices to cache.
        include_mlp: Whether to cache MLP outputs.
        include_attn: Whether to cache attention outputs.

    Returns:
        List of SequenceLatents, one per input sequence, in the same order.
    """
    if not sequences:
        return []

    pad_id = (
        mt.tokenizer.pad_token_id
        if mt.tokenizer.pad_token_id is not None
        else mt.tokenizer.eos_token_id
    )

    full_ids_list = [p + g for p, g in sequences]
    max_len = max(len(f) for f in full_ids_list)

    padded_ids: list[list[int]] = []
    attention_masks: list[list[int]] = []
    for full_ids in full_ids_list:
        pad_len = max_len - len(full_ids)
        padded_ids.append(full_ids + [pad_id] * pad_len)
        attention_masks.append([1] * len(full_ids) + [0] * pad_len)

    batch_inputs = {
        "input_ids": torch.tensor(padded_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
    }

    res_saves: dict[int, Any] = {}
    mlp_saves: dict[int, Any] = {}
    attn_saves: dict[int, Any] = {}

    with mt.trace(batch_inputs, scan=False, validate=False):
        for layer_idx in layers:
            if include_attn:
                attn_mod = get_module_nnsight(
                    mt, mt.attn_module_name_format.format(layer_idx)
                )
                attn_saves[layer_idx] = attn_mod.output.save()
            if include_mlp:
                mlp_mod = get_module_nnsight(
                    mt, mt.mlp_module_name_format.format(layer_idx)
                )
                mlp_saves[layer_idx] = mlp_mod.output.save()
            layer_mod = get_module_nnsight(mt, mt.layer_name_format.format(layer_idx))
            res_saves[layer_idx] = layer_mod.output.save()

    # Materialise all saves to CPU float16 before splitting per sequence.
    # Shape: (batch, max_len, hidden_dim) — do NOT squeeze the batch dim.
    def _to_cpu(saves: dict[int, Any], layer_idx: int) -> torch.Tensor:
        t = _untuple(saves[layer_idx])
        return t.half().cpu()

    batch_res: dict[int, torch.Tensor] = {li: _to_cpu(res_saves, li) for li in layers}
    batch_mlp: dict[int, torch.Tensor] = (
        {li: _to_cpu(mlp_saves, li) for li in layers} if include_mlp else {}
    )
    batch_attn: dict[int, torch.Tensor] = (
        {li: _to_cpu(attn_saves, li) for li in layers} if include_attn else {}
    )

    latents_list: list[SequenceLatents] = []
    for i, (prompt_ids, gen_ids) in enumerate(sequences):
        n_prompt = len(prompt_ids)
        seq_len = n_prompt + len(gen_ids)
        full_text = mt.tokenizer.decode(prompt_ids + gen_ids, skip_special_tokens=False)

        code_start = _find_code_token_start(
            mt.tokenizer.decode(prompt_ids, skip_special_tokens=False),
            mt.tokenizer,
            n_prompt,
        )
        has_code = code_start < n_prompt

        # .clone() gives each slice its own storage — without it, slices are
        # views of the (batch, max_len, hidden) batch tensor and torch.save
        # would serialize the full batch storage (~8x bloat).
        def _cloned(batch_dict: dict, row: int, start: int, end: int) -> dict:
            return {li: batch_dict[li][row, start:end, :].clone() for li in layers}

        latents_list.append(
            SequenceLatents(
                prompt_len=n_prompt,
                prompt_ids=prompt_ids,
                gen_ids=gen_ids,
                full_text=full_text,
                residual=_cloned(batch_res, i, n_prompt, seq_len),
                code_start=code_start,
                code_residual=_cloned(batch_res, i, code_start, n_prompt)
                if has_code
                else {},
                code_mlp=_cloned(batch_mlp, i, code_start, n_prompt)
                if (has_code and include_mlp)
                else {},
                code_attn=_cloned(batch_attn, i, code_start, n_prompt)
                if (has_code and include_attn)
                else {},
                mlp=_cloned(batch_mlp, i, n_prompt, seq_len) if include_mlp else {},
                attn=_cloned(batch_attn, i, n_prompt, seq_len) if include_attn else {},
            )
        )

    return latents_list


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def generate_and_cache_latents(
    mt,
    sample: MultiPredicateSample,
    prefill: str,
    prompt: str,
    layers: list[int],
    max_new_tokens: int = 150,
    stop_ids: set[int] | None = None,
    eval_timeout: int = 5,
    include_mlp: bool = True,
    include_attn: bool = True,
) -> SequenceLatents | None:
    """Stage 2 + 3 combined: generate, evaluate, then cache latents.

    Returns SequenceLatents if the generation passes unit tests, else None.

    Args:
        sample: Used for unit test evaluation and full_code reconstruction.
        prefill: The prefill string that was appended to sample.code when building prompt.
        prompt: Full prompt string (from apply_chat_template).
        (other args forwarded to generate_and_evaluate_sample / cache_latents_for_sequence)
    """
    ids = generate_and_evaluate_sample(
        mt=mt,
        prompt=prompt,
        sample=sample,
        prefill=prefill,
        max_new_tokens=max_new_tokens,
        eval_timeout=eval_timeout,
    )
    if ids is None:
        return None
    prompt_ids, gen_ids = ids
    return cache_latents_for_sequence(
        mt=mt,
        prompt_ids=prompt_ids,
        gen_ids=gen_ids,
        layers=layers,
        include_mlp=include_mlp,
        include_attn=include_attn,
    )


# ---------------------------------------------------------------------------
# Span labeling
# ---------------------------------------------------------------------------


def find_in_progress_mask(
    latents: SequenceLatents,
    anchor_patterns: list[str],
    tokenizer,
) -> np.ndarray:
    """Boolean mask over generated tokens: True = currently implementing predicate.

    Applies each regex in anchor_patterns to latents.full_text (the decoded
    full sequence).  Matched character spans are mapped to token indices via
    find_token_range, then adjusted to be relative to the generated tokens
    (index 0 = first generated token).

    Args:
        latents: SequenceLatents from cache_latents_for_sequence.
        anchor_patterns: Variable-agnostic regex patterns from PredicateSpec.anchor_patterns.
        tokenizer: The model tokenizer (for find_token_range).

    Returns:
        Boolean np.ndarray of shape (len(latents.gen_ids),).
        True at positions where the model is writing code for the predicate.
    """
    n_gen = len(latents.gen_ids)
    mask = np.zeros(n_gen, dtype=bool)

    for pattern in anchor_patterns:
        for m in re.finditer(pattern, latents.full_text):
            matched = latents.full_text[m.start() : m.end()]
            if not matched:
                continue
            try:
                tok_start, tok_end = find_token_range(
                    string=latents.full_text,
                    substring=matched,
                    tokenizer=tokenizer,
                )
                # Convert absolute token indices → relative to generated tokens
                rel_start = max(tok_start - latents.prompt_len, 0)
                rel_end = min(tok_end - latents.prompt_len, n_gen)
                if rel_start < rel_end:
                    mask[rel_start:rel_end] = True
            except (ValueError, Exception):
                pass

    return mask


# ---------------------------------------------------------------------------
# Oracle-based span labeling
# ---------------------------------------------------------------------------


def find_in_progress_mask_oracle(
    latents: SequenceLatents,
    p_plus_description: str,
    p_minus_description: str,
    tokenizer,
    prefill: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks over generated tokens using Claude as oracle span labeler.

    The oracle is shown the generated text and asked to return the exact
    substrings implementing P+ and P- separately.  When both predicates are
    inseparably interleaved (e.g. ``s.lower() == s[::-1]`` implements both
    "is lowercase" and "is palindrome" as a combined expression), the same
    full span is returned for both.

    This is the preferred labeling method over find_in_progress_mask (regex),
    because anchor_patterns are too broad and produce noisy labels.  Oracle
    calls are cached on disk (via ask_claude's MD5 cache), so re-labeling
    the same text is free.

    Args:
        latents: SequenceLatents from cache_latents_for_sequence.
        p_plus_description: Natural-language description of the target predicate.
        p_minus_description: Description of the other predicate (context for oracle).
        tokenizer: The model tokenizer (for find_token_range).
        prefill: Harmless prefill that was appended before generation (for context).

    Returns:
        (p_plus_mask, p_minus_mask): Two boolean np.ndarrays of shape
        (len(latents.gen_ids),).  p_plus_mask is True at positions where the
        model is writing code for P+; p_minus_mask likewise for P-.
        When P+ and P- are inseparable the same span is marked True in both.
    """
    from src.utils.oracle_llms import ask_claude

    gen_text: str = tokenizer.decode(latents.gen_ids, skip_special_tokens=True)

    prefill_note = (
        f'\nNote: generation was prefixed with "{prefill}".' if prefill else ""
    )

    oracle_prompt = f"""You are a code analysis assistant. A language model generated the following code:{prefill_note}

<generated_code>
{gen_text}
</generated_code>

The function implements two predicates:
- P+ (target): {p_plus_description}
- P- (other):  {p_minus_description}

Task: identify the exact substrings implementing each predicate.

Rules:
1. Wrap P+ implementing code in <p_plus>…</p_plus> tags.
2. Wrap P- implementing code in <p_minus>…</p_minus> tags.
3. If P+ and P- are inseparably interleaved in a single expression
   (e.g. `s.lower() == s[::-1]` simultaneously checks both lowercase and palindrome),
   wrap the SAME full expression in BOTH <p_plus>…</p_plus> AND <p_minus>…</p_minus>.
4. Copy each substring EXACTLY as it appears — character-perfect match required.
5. Focus on the core expression (comparison, membership test, arithmetic check, etc.).
   Do NOT include the `return` keyword, leading indentation, or unrelated code.
6. If a predicate is not implemented in the generated code, omit its tag entirely.

Examples:
  Separate predicates (s == s[::-1] and s.islower()):
    <p_plus>s == s[::-1]</p_plus>
    <p_minus>s.islower()</p_minus>

  Interleaved (single expression covers both):
    <p_plus>s.lower() == s[::-1]</p_plus>
    <p_minus>s.lower() == s[::-1]</p_minus>"""

    response: str = ask_claude(
        oracle_prompt, max_tokens=1024, temperature=0.0, use_cache=True
    )
    print(f"\nOracle response:\n{response}")

    n_gen = len(latents.gen_ids)

    def _spans_to_mask(tag: str) -> np.ndarray:
        mask = np.zeros(n_gen, dtype=bool)
        for span_text in re.findall(rf"<{tag}>(.*?)</{tag}>", response, re.DOTALL):
            span_text = span_text.strip()
            if not span_text:
                continue
            try:
                tok_start, tok_end = find_token_range(
                    string=latents.full_text,
                    substring=span_text,
                    tokenizer=tokenizer,
                )
                rel_start = max(tok_start - latents.prompt_len, 0)
                rel_end = min(tok_end - latents.prompt_len, n_gen)
                if rel_start < rel_end:
                    mask[rel_start:rel_end] = True
            except (ValueError, Exception):
                # Span not found in full_text — search generated portion only
                try:
                    tok_start, tok_end = find_token_range(
                        string=gen_text,
                        substring=span_text,
                        tokenizer=tokenizer,
                    )
                    rel_end = min(tok_end, n_gen)
                    if tok_start < rel_end:
                        mask[tok_start:rel_end] = True
                except (ValueError, Exception):
                    pass
        return mask

    return _spans_to_mask("p_plus"), _spans_to_mask("p_minus")


# ---------------------------------------------------------------------------
# Probe dataset: save / load
# ---------------------------------------------------------------------------


@dataclass
class ProbeRecord:
    """One labeled sequence for clock probe training.

    Stores everything needed to reconstruct the probe example:
    - Predicate metadata (name, description, prefill)
    - The original prompt string
    - The full SequenceLatents (residual / mlp / attn per layer)
    - p_plus_mask / p_minus_mask: oracle-identified token spans per predicate

    in_progress_mask is a computed property: p_plus_mask | p_minus_mask.
    Both masks are required (use all-False arrays when oracle finds nothing).
    """

    predicate_name: str
    predicate_description: str
    prefill: str
    prompt: str
    latents: SequenceLatents
    p_plus_mask: np.ndarray  # shape (n_gen_tokens,), dtype bool
    p_minus_mask: np.ndarray  # shape (n_gen_tokens,), dtype bool

    @property
    def in_progress_mask(self) -> np.ndarray:
        """Union of P+ and P- spans — tokens implementing either predicate."""
        return self.p_plus_mask | self.p_minus_mask


def save_probe_dataset(
    records: list[ProbeRecord],
    directory: str | os.PathLike,
) -> None:
    """Save ProbeRecords as one self-contained .pt file per record.

    Layout::

        <directory>/
            0000.pt     # metadata + masks + float16 tensors for record 0
            0001.pt
            ...

    Each file is a flat dict readable with torch.load; no separate index file.

    Args:
        records: List of ProbeRecord instances.
        directory: Output directory (created if it does not exist).
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    for i, r in enumerate(records):
        torch.save(
            {
                "predicate_name": r.predicate_name,
                "predicate_description": r.predicate_description,
                "prefill": r.prefill,
                "prompt": r.prompt,
                "prompt_len": r.latents.prompt_len,
                "prompt_ids": r.latents.prompt_ids,
                "gen_ids": r.latents.gen_ids,
                "full_text": r.latents.full_text,
                "code_start": r.latents.code_start,
                "p_plus_mask": r.p_plus_mask,
                "p_minus_mask": r.p_minus_mask,
                "residual": r.latents.residual,
                "code_residual": r.latents.code_residual,
                "mlp": r.latents.mlp,
                "code_mlp": r.latents.code_mlp,
                "attn": r.latents.attn,
                "code_attn": r.latents.code_attn,
            },
            directory / f"{i:04d}.pt",
        )


def load_probe_dataset(
    directory: str | os.PathLike, limit: int = None
) -> list[ProbeRecord]:
    """Load ProbeRecords from a directory written by save_probe_dataset.

    Each .pt file is self-contained (metadata + masks + tensors).
    Files are loaded in filename order (0000.pt, 0001.pt, …).

    Returns:
        List of ProbeRecord instances with float16 latent tensors.
    """
    directory = Path(directory)
    pt_files = sorted(directory.glob("*.pt"))
    if limit is not None:
        pt_files = pt_files[:limit]

    records: list[ProbeRecord] = []
    for pt_file in tqdm(pt_files):
        d = torch.load(pt_file, weights_only=False)
        latents = SequenceLatents(
            prompt_len=d["prompt_len"],
            prompt_ids=d.get("prompt_ids", []),
            gen_ids=d["gen_ids"],
            full_text=d.get("full_text", ""),
            residual={int(k): v for k, v in d["residual"].items()},
            code_start=d.get("code_start", d["prompt_len"]),
            code_residual={int(k): v for k, v in d.get("code_residual", {}).items()},
            code_mlp={int(k): v for k, v in d.get("code_mlp", {}).items()},
            code_attn={int(k): v for k, v in d.get("code_attn", {}).items()},
            mlp={int(k): v for k, v in d.get("mlp", {}).items()},
            attn={int(k): v for k, v in d.get("attn", {}).items()},
        )
        records.append(
            ProbeRecord(
                predicate_name=d["predicate_name"],
                predicate_description=d["predicate_description"],
                prefill=d["prefill"],
                prompt=d["prompt"],
                latents=latents,
                p_plus_mask=np.asarray(d["p_plus_mask"], dtype=bool),
                p_minus_mask=np.asarray(d["p_minus_mask"], dtype=bool),
            )
        )

    return records


# ---------------------------------------------------------------------------
# Probe utilities
# ---------------------------------------------------------------------------


def collect_probe_examples(
    latents: SequenceLatents,
    in_progress_mask: np.ndarray,
    layer: int,
    rep_type: str = "residual",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract (X, y) feature-label pairs for one sequence and one layer.

    **Off-by-one correction**: the residual stream at position t is the input
    that produces token t+1.  So to predict whether generated token i is
    "in progress", we should use the latent from position i-1:

        label[i]  →  feature from gen_rep[i-1]   (for i >= 1)
        label[0]  →  feature from code_rep[-1]    (last code-stub token, e.g. `return`)

    If code latents are unavailable (old files), gen_rep[0] is used as a
    fallback for label[0] (slight imprecision for only the very first token).

    Args:
        latents: SequenceLatents from cache_latents_for_sequence.
        in_progress_mask: Boolean array (len = n_gen_tokens) from find_in_progress_mask.
        layer: Layer index to extract features from.
        rep_type: "residual", "mlp", or "attn".

    Returns:
        X: (n_gen_tokens, hidden_dim) float32 numpy array.
        y: (n_gen_tokens,) int32 array (0 = not in progress, 1 = in progress).
    """
    gen_rep = getattr(latents, rep_type)[layer].float()  # (n_gen, hidden)
    y = torch.from_numpy(in_progress_mask.astype(np.int64))

    # Feature for label[0]: last token of the code stub (typically `return` / prefill end)
    code_rep_dict: dict[int, torch.Tensor] = getattr(latents, f"code_{rep_type}")
    if layer in code_rep_dict and len(code_rep_dict[layer]) > 0:
        first_feat = code_rep_dict[layer][-1:].float()  # (1, hidden)
    else:
        # Fallback: use gen_rep[0] itself (no shift for the very first token)
        first_feat = gen_rep[:1]

    # Shift: gen_rep[i-1] predicts label[i] for i in 1..n-1
    X = torch.cat([first_feat, gen_rep[:-1]], dim=0)  # (n_gen, hidden)
    return X, y


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return (start, end) inclusive index pairs for each True run in mask."""
    runs: list[tuple[int, int]] = []
    in_run = False
    start = 0
    for i, v in enumerate(mask):
        if v and not in_run:
            start = i
            in_run = True
        elif not v and in_run:
            runs.append((start, i - 1))
            in_run = False
    if in_run:
        runs.append((start, len(mask) - 1))
    return runs


def collect_regression_examples(
    latents: SequenceLatents,
    p_plus_mask: np.ndarray,
    p_minus_mask: np.ndarray,
    layer: int,
    rep_type: str = "residual",
    rng: np.random.Generator | None = None,
    end_value: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract (X, y) for the regression probe — continuous progress labels.

    For each contiguous run in p_plus_mask and p_minus_mask (handled separately),
    assigns a linearly decreasing label from 1.0 (first token of run) to 0.0
    (last token of run).  Background tokens (outside in_progress_mask) are
    randomly sampled and labeled 0.0.

    Same off-by-one shift as collect_probe_examples: the latent at position t-1
    is used as the feature for gen token t (code_rep[-1] for t=0).

    Args:
        latents: SequenceLatents from cache_latents_for_sequence.
        p_plus_mask: Boolean array (n_gen_tokens,) marking the p_plus span.
        p_minus_mask: Boolean array (n_gen_tokens,) marking the p_minus span.
        layer: Layer index to extract features from.
        rep_type: "residual", "mlp", or "attn".
        rng: Optional random generator for reproducible background sampling.

    Returns:
        X: (n, hidden_dim) float tensor.
        y: (n,) float tensor with values in [0.0, 1.0].
    """
    if rng is None:
        rng = np.random.default_rng()

    gen_rep = getattr(latents, rep_type)[layer].float()  # (n_gen, hidden)

    # Build shifted feature matrix (latent at i-1 predicts gen token i)
    code_rep_dict: dict[int, torch.Tensor] = getattr(latents, f"code_{rep_type}")
    if layer in code_rep_dict and len(code_rep_dict[layer]) > 0:
        first_feat = code_rep_dict[layer][-1:].float()
    else:
        first_feat = gen_rep[:1]
    shifted_rep = torch.cat([first_feat, gen_rep[:-1]], dim=0)  # (n_gen, hidden)

    in_progress_mask = p_plus_mask | p_minus_mask
    X_parts: list[torch.Tensor] = []
    y_parts: list[torch.Tensor] = []
    total_span_tokens = 0

    # Linear 1 → 0 ramp per contiguous run, for p_plus and p_minus independently
    for mask in (p_plus_mask, p_minus_mask):
        for start, end in _contiguous_runs(mask):
            L = end - start + 1
            labels = torch.from_numpy(np.linspace(1.0, end_value, L, dtype=np.float32))
            X_parts.append(shifted_rep[start : end + 1])
            y_parts.append(labels)
            total_span_tokens += L

    if not X_parts:
        return torch.zeros((0, gen_rep.shape[1])), torch.zeros((0,))

    # Random background sample labeled 0.0
    bg_indices = np.where(~in_progress_mask)[0]
    n_bg = max(1, total_span_tokens // 2)
    if len(bg_indices) > 0:
        chosen = rng.choice(bg_indices, size=min(n_bg, len(bg_indices)), replace=False)
        X_parts.append(shifted_rep[chosen])
        y_parts.append(
            torch.zeros(len(chosen), dtype=torch.float32)
        )  #! not inside span, strictly zero label

    return torch.cat(X_parts, dim=0), torch.cat(y_parts, dim=0)


class LinearProbe(nn.Module):
    """Balanced binary linear probe trained with weighted cross-entropy.

    Feature normalization (mean / std) is stored as buffers so the probe
    is fully self-contained — no separate scaler object needed.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        add_bias: bool = False,
        z_normalize: bool = False,
        x_unit_normalize: bool = False,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes, bias=add_bias)
        self.metadata = {"input_dim": input_dim, "num_classes": num_classes}
        self.z_normalize = z_normalize
        self.x_unit_normalize = x_unit_normalize
        nn.init.kaiming_normal_(self.linear.weight)
        self.register_buffer("feat_mean", torch.zeros(input_dim))
        self.register_buffer("feat_std", torch.ones(input_dim))

    def _z_norm(self, x: torch.Tensor) -> torch.Tensor:
        return (x.to(self.feat_mean.device).float() - self.feat_mean) / self.feat_std

    @staticmethod
    def _x_unit_norm(x: torch.Tensor) -> torch.Tensor:
        return x / x.norm(dim=1, keepdim=True).clamp(min=1e-8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits (n, num_classes)."""
        x = x.to(self.feat_mean.device).float()
        if self.x_unit_normalize:
            x = self._x_unit_norm(x)
        if self.z_normalize:
            x = self._z_norm(x)
        return self.linear(x)

    @torch.inference_mode()
    def predict(self, x: torch.Tensor) -> np.ndarray:
        """Return predicted class indices as a numpy int array."""
        self.eval()
        return self.forward(x).argmax(dim=-1).cpu().numpy()


def train_probe(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    n_epochs: int = 300,
    lr: float = 0.01,
    weight_decay: float = 1e-4,
    device: str | torch.device = "cuda",
    add_bias: bool = False,
    z_normalize: bool = False,
    unit_normalize: bool = True,
    lambda_unit: float = 0.1,
) -> LinearProbe:
    """Fit a balanced binary linear probe on GPU.

    Class imbalance is handled via weighted cross-entropy:
        weight[c] = N / (2 * N_c)

    Normalization statistics (mean / std) are computed once on the training
    data and stored in the probe's buffers.

    Args:
        X_train: (n, hidden_dim) float tensor (CPU is fine; moved to device here).
        y_train: (n,) int tensor — 0 = not in progress, 1 = in progress.
        n_epochs: Number of Adam steps.
        lr: Adam learning rate.
        weight_decay: L2 regularization strength.
        device: Training device.
        add_bias: Whether to include a bias term in the linear layer.
        z_normalize: Whether to z-score features before the linear layer.
        unit_normalize: If True, add a penalty that encourages each row of
            the weight matrix to be a unit vector:
                loss_unit = mean((||w_c|| - 1)^2  for c in classes)
        lambda_unit: Weight of the unit-norm penalty (only used when
            unit_normalize=True).

    Returns:
        Trained LinearProbe with normalization buffers populated.
    """
    X = X_train.float()
    y = y_train.long()

    # Compute normalization on CPU before moving to device
    mean = X.mean(dim=0)
    std = X.std(dim=0).clamp(min=1e-8)

    # Balanced class weights
    n = len(y)
    n_pos = int(y.sum().item())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        class_weights = torch.ones(2)
    else:
        class_weights = torch.tensor([n / (2.0 * n_neg), n / (2.0 * n_pos)])

    probe = LinearProbe(
        X.shape[1],
        num_classes=2,
        add_bias=add_bias,
        z_normalize=z_normalize,
        x_unit_normalize=unit_normalize,
    )
    probe.feat_mean.copy_(mean)
    probe.feat_std.copy_(std)
    probe = probe.to(device)

    if unit_normalize:
        X = LinearProbe._x_unit_norm(X)
    X = X.to(device)
    y = y.to(device)
    class_weights = class_weights.to(device)

    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    probe.train()
    for _ in range(n_epochs):
        optimizer.zero_grad()
        loss = criterion(probe(X), y)
        if unit_normalize:
            # Penalise deviation of each weight-row norm from 1
            row_norms = probe.linear.weight.norm(dim=1)  # (num_classes,)
            loss = loss + lambda_unit * ((row_norms - 1) ** 2).mean()
        loss.backward()
        optimizer.step()

    probe.eval()
    return probe


def eval_probe(
    probe: LinearProbe,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
) -> dict:
    """Score a trained probe on test data.

    Returns:
        dict with "balanced_accuracy", "accuracy", "report"
        (classification_report string with per-class precision/recall/F1).
    """
    y_pred = probe.predict(X_test)
    y_np = y_test.numpy() if isinstance(y_test, torch.Tensor) else y_test
    return {
        "balanced_accuracy": balanced_accuracy_score(y_np, y_pred),
        "accuracy": float((y_pred == y_np).mean()),
        "report": classification_report(
            y_np,
            y_pred,
            target_names=["not_in_progress", "in_progress"],
            zero_division=0,
        ),
    }


# ---------------------------------------------------------------------------
# Regression probe — continuous progress signal
# ---------------------------------------------------------------------------


class LinearRegressionProbe(nn.Module):
    """Linear probe that regresses a continuous progress value in [0, 1].

    Labels are 1.0 at the first token of an in-progress span, decrease
    linearly to 0.0 at the last token, and are 0.0 for background tokens.

    Same normalization design as LinearProbe: feat_mean / feat_std stored as
    buffers so the probe is fully self-contained.
    """

    def __init__(
        self,
        input_dim: int,
        add_bias: bool = False,
        z_normalize: bool = False,
        x_unit_normalize: bool = True,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=add_bias)
        nn.init.kaiming_normal_(self.linear.weight)
        self.z_normalize = z_normalize
        self.x_unit_normalize = x_unit_normalize
        self.register_buffer("feat_mean", torch.zeros(input_dim))
        self.register_buffer("feat_std", torch.ones(input_dim))

    def _z_norm(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.feat_mean) / self.feat_std

    @staticmethod
    def _x_unit_norm(x: torch.Tensor) -> torch.Tensor:
        return x / x.norm(dim=1, keepdim=True).clamp(min=1e-8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted progress values (n,)."""
        x = x.to(self.feat_mean.device).float()
        if self.x_unit_normalize:
            x = self._x_unit_norm(x)
        if self.z_normalize:
            x = self._z_norm(x)
        return self.linear(x).squeeze(-1)

    @torch.inference_mode()
    def predict(self, x: torch.Tensor) -> np.ndarray:
        """Return predicted progress values as a numpy float32 array."""
        self.eval()
        return self.forward(x).cpu().numpy()


def train_regression_probe(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    n_epochs: int = 300,
    lr: float = 0.01,
    weight_decay: float = 1e-4,
    device: str | torch.device = "cuda",
    add_bias: bool = False,
    z_normalize: bool = False,
    x_unit_normalize: bool = True,
) -> LinearRegressionProbe:
    """Fit a linear regression probe on GPU using MSE loss.

    Args:
        X_train: (n, hidden_dim) float tensor.
        y_train: (n,) float tensor with values in [0, 1].
        n_epochs: Number of Adam steps.
        lr: Adam learning rate.
        weight_decay: L2 regularization strength.
        device: Training device.
        add_bias: Whether to include a bias term.
        z_normalize: Whether to z-score features before the linear layer.
        x_unit_normalize: Whether to L2-normalise each input vector (row-wise)
            before the linear layer.

    Returns:
        Trained LinearRegressionProbe with normalization buffers populated.
    """
    X = X_train.float()
    y = y_train.float()

    mean = X.mean(dim=0)
    std = X.std(dim=0).clamp(min=1e-8)

    probe = LinearRegressionProbe(
        X.shape[1],
        add_bias=add_bias,
        z_normalize=z_normalize,
        x_unit_normalize=x_unit_normalize,
    )
    probe.feat_mean.copy_(mean)
    probe.feat_std.copy_(std)
    probe = probe.to(device)

    if x_unit_normalize:
        X = LinearRegressionProbe._x_unit_norm(X)
    X = X.to(device)
    y = y.to(device)

    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    probe.train()
    for _ in range(n_epochs):
        optimizer.zero_grad()
        loss = criterion(probe(X), y)
        loss.backward()
        optimizer.step()

    probe.eval()
    return probe


def eval_regression_probe(
    probe: LinearRegressionProbe,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
) -> dict:
    """Score a trained regression probe on test data.

    Returns:
        dict with "mse", "mae", "r2".
    """
    y_pred = probe.predict(X_test)
    y_np = y_test.numpy() if isinstance(y_test, torch.Tensor) else y_test

    mse = float(((y_pred - y_np) ** 2).mean())
    mae = float(np.abs(y_pred - y_np).mean())
    ss_res = float(((y_pred - y_np) ** 2).sum())
    ss_tot = float(((y_np - y_np.mean()) ** 2).sum())
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {"mse": mse, "mae": mae, "r2": r2}
