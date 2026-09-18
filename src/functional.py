import copy
import gc
import logging
import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

import baukit
import numpy as np
import torch

from src.dataset import Relation
from src.models import ModelandTokenizer
from src.tokens import find_token_range, insert_padding_before_pos, prepare_input
from src.utils.typing import ArrayLike, PredictedToken, Tokenizer, TokenizerOutput

logger = logging.getLogger(__name__)


@torch.inference_mode()
def interpret_logits(
    tokenizer: ModelandTokenizer | Tokenizer,
    logits: torch.Tensor,
    k: int = 5,
    interested_tokens: tuple[int] = (),
) -> (
    list[PredictedToken]
    | tuple[list[PredictedToken], dict[int, tuple[int, PredictedToken]]]
):
    tokenizer = unwrap_tokenizer(tokenizer)
    # print(type(tokenizer))
    logits = logits.squeeze()
    probs = torch.nn.functional.softmax(logits, dim=-1).squeeze()
    top_k_indices = logits.topk(dim=-1, k=k).indices.squeeze().tolist()
    if isinstance(top_k_indices, ArrayLike) is False:
        top_k_indices = [top_k_indices]

    candidates = [
        PredictedToken(
            token=tokenizer.decode(t),
            prob=probs[t].item(),
            logit=logits[t].item(),
            token_id=t,
        )
        for t in top_k_indices
    ]

    if len(interested_tokens) > 0:
        rank_tokens = logits.argsort(descending=True).tolist()
        probs = torch.nn.functional.softmax(logits, dim=-1)
        interested_logits = {
            t: (
                rank_tokens.index(t) + 1,
                PredictedToken(
                    token=tokenizer.decode(t),
                    prob=probs[t].item(),
                    logit=logits[t].item(),
                    token_id=t,
                ),
            )
            for t in interested_tokens
        }
        # print(interested_logits)
        # interested_logits = {
        #     k: v
        #     for k, v in sorted(
        #         interested_logits.items(), key=lambda x: x[1][1].prob, reverse=True
        #     )
        # }
        ordered_interested_logits = OrderedDict()
        for k, v in sorted(
            interested_logits.items(), key=lambda x: x[1][1].prob, reverse=True
        ):
            ordered_interested_logits[k] = v
        return candidates, ordered_interested_logits
    return candidates


@torch.inference_mode()
def logit_lens(
    mt: ModelandTokenizer,
    h: torch.Tensor,
    interested_tokens: tuple[int] = (),
    k: int = 5,
    return_logits=False,
) -> (
    list[PredictedToken]
    | tuple[list[PredictedToken], dict[int, tuple[int, PredictedToken]]]
):
    inputs = mt.tokenizer(
        mt.tokenizer.bos_token, add_special_tokens=False, return_tensors="pt"
    )
    with mt.trace(inputs):
        lnf = get_module_nnsight(mt, mt.final_layer_norm_name)
        lnf.input = h.view(1, 1, h.squeeze().shape[0])
        logits = mt.output.logits.save()

    logits = logits.squeeze()
    free_gpu_cache()
    ret = interpret_logits(
        tokenizer=mt, logits=logits, k=k, interested_tokens=interested_tokens
    )

    ret = (logits, ret) if return_logits else ret

    return ret


@torch.inference_mode()
def logit_lens_baukit(
    mt: ModelandTokenizer,
    h: torch.Tensor,
    interested_tokens: tuple[int] = (),
    k: int = 5,
) -> (
    list[PredictedToken]
    | tuple[list[PredictedToken], dict[int, tuple[int, PredictedToken]]]
):
    lnf = baukit.get_module(mt._model, mt.final_layer_norm_name)
    lm_head = baukit.get_module(mt._model, mt.lm_head_name)
    h = untuple(h)
    logits = lm_head(lnf(h)).squeeze()
    free_gpu_cache()
    return interpret_logits(
        tokenizer=mt, logits=logits, k=k, interested_tokens=interested_tokens
    )


@torch.inference_mode()
def forward_pass_to_vocab(
    mt: ModelandTokenizer, h: torch.Tensor, layer_name: str, **kwargs
):
    inputs = mt.tokenizer(
        mt.tokenizer.bos_token, add_special_tokens=False, return_tensors="pt"
    )
    with mt.trace(inputs):
        module = get_module_nnsight(mt, layer_name)
        module.output[0][0, :] = h
        logits = mt.output.logits[0, -1].save()

    free_gpu_cache()

    return interpret_logits(
        tokenizer=mt,
        logits=logits,
        **kwargs,
    )


@torch.no_grad()
def patchscope(
    mt: ModelandTokenizer,
    h: torch.Tensor,
    context: str | None = None,
    placeholder: str = " x",
    context_tokenized: TokenizerOutput | None = None,
    placeholder_idx: int | None = None,
    patch_layers: Optional[list[str]] = None,
    return_logits: bool = False,
    add_orig_latent_to: str | None = None,
    **interpret_kwargs,
) -> (
    list[PredictedToken]
    | tuple[list[PredictedToken], dict[int, tuple[int, PredictedToken]]]
):
    if context is None and context_tokenized is None:
        phrases = [
            " copy",
            " Cat",
            " Java",
            " transistor",
            " python",
            " Leonardo DiCaprio",
            " The Lion King",
            " Washington D.C.",
            " Mount Everest",
            " computer",
        ]
        context = "\n".join([f"{p} >{p}" for p in phrases])
        context = f"{context}\n {placeholder} >"
        # logger.debug(context)

    if context_tokenized is None:
        context_tokenized = prepare_input(
            tokenizer=mt,
            prompts=context,
            return_offsets_mapping=True,
        )

    elif context is None:
        context = mt.tokenizer.decode(
            context_tokenized["input_ids"][0], skip_special_tokens=True
        )
        # logger.debug(f"context: {context}")

    if placeholder_idx is None:
        placeholder_range = find_token_range(
            string=context,
            substring=placeholder,
            tokenizer=mt.tokenizer,
            occurrence=-1,
            offset_mapping=(
                context_tokenized["offset_mapping"][0]
                if "offset_mapping" in context_tokenized
                else None
            ),
        )
        placeholder_idx = placeholder_range[1] - 1
        # print(context_tokenized)
        # logger.debug(
        #     f"placeholder position: {placeholder_idx} | token: \"{mt.tokenizer.decode(context_tokenized['input_ids'][0, placeholder_idx])}\""
        # )

    if "offset_mapping" in context_tokenized:
        context_tokenized.pop("offset_mapping")

    placeholder_ends = mt.tokenizer.decode(
        context_tokenized["input_ids"][0, placeholder_idx]
    )
    if placeholder.strip().endswith(placeholder_ends.strip()) is False:
        logger.warning(
            f"{placeholder=} does not end with {placeholder_ends=} | {placeholder_idx=}"
        )

    patch_layers = (
        [mt.layer_name_format.format(5)] if patch_layers is None else patch_layers
    )
    patches = [
        PatchSpec(
            location=(layer_name, placeholder_idx),
            patch=h,
            clean=None,
            strategy="replace",
        )
        for layer_name in patch_layers
    ]
    if add_orig_latent_to is not None:
        # print("adding")
        patches.append(
            PatchSpec(
                location=(add_orig_latent_to, placeholder_idx),
                patch=h,
                clean=None,
                strategy="add",
            )
        )

    # print(patches)

    logits = get_hs(
        mt=mt,
        input=context_tokenized,
        locations=[(mt.lm_head_name, -1)],
        patches=patches,
        return_dict=False,
    )
    pred = interpret_logits(
        tokenizer=mt,
        logits=logits,
        **interpret_kwargs,
    )

    if return_logits:
        return logits, pred
    return pred


def untuple(object: Any):
    if isinstance(object, tuple) or (
        "LanguageModelProxy" in str(type(object)) and len(object) > 1
    ):
        return object[0]
    return object


def unwrap_model(mt: ModelandTokenizer | torch.nn.Module) -> torch.nn.Module:
    if isinstance(mt, ModelandTokenizer):
        return mt.model
    if isinstance(mt, torch.nn.Module):
        return mt
    raise ValueError("mt must be a ModelandTokenizer or a torch.nn.Module")


def unwrap_tokenizer(mt: ModelandTokenizer | Tokenizer) -> Tokenizer:
    if isinstance(mt, ModelandTokenizer):
        return mt.tokenizer
    return mt


# useful for logging
def bytes_to_human_readable(
    size: int, unit: Literal["B", "KB", "MB", "GB"] = "MB"
) -> str:
    denom = {"B": 1, "KB": 2**10, "MB": 2**20, "GB": 2**30}[unit]
    return f"{size / denom:.3f} {unit}"


def any_is_nontrivial_prefix(predictions: list[str], target: str) -> bool:
    """Return true if any prediction is (case insensitive) prefix of the target."""
    return any(is_nontrivial_prefix(p, target) for p in predictions)


def is_nontrivial_prefix(prediction: str, target: str) -> bool:
    """Return true if prediction is (case insensitive) prefix of the target."""
    target = target.lower().strip()
    prediction = prediction.lower().strip()
    return len(prediction) > 0 and target.startswith(prediction)


def get_tick_marker(value: bool) -> str:
    """Returns a tick or cross marker depending on the value."""
    return "✓" if value else "✗"


def format_whitespace(s: str) -> str:
    """Format whitespace in a string for printing."""
    return s.replace("\n", "\\n").replace("\t", "\\t")


@torch.inference_mode()
def filter_samples_by_model_knowledge(
    mt: ModelandTokenizer, relation: Relation, limit: Optional[int] = None
) -> Relation:
    """Filter samples by model knowledge."""
    logger.debug(f'"{relation.name}" | filtering with {mt.name}')

    filtered_samples = []
    for i in range(len(relation.samples)):
        question, answer = relation[i]
        predictions = predict_next_token(mt, question, k=5)[0]
        top_pred = predictions[0]
        is_known = is_nontrivial_prefix(prediction=top_pred.token, target=answer)
        sample = relation.samples[i]

        logger.debug(
            f"{sample.subject=} -> {answer=} | predicted = '{top_pred.token}'({top_pred.prob:.3f}) ==> ({get_tick_marker(is_known)})"
        )
        if is_known:
            filtered_samples.append(sample)

        if limit is not None and len(filtered_samples) >= limit:
            break

    logger.info(
        f'filtered relation "{relation.name}" to {len(filtered_samples)} samples (with {len(relation._few_shot_samples)}-shots)'
    )

    relation.samples = filtered_samples
    return relation


def release_cache():
    gc.collect()
    if torch.cuda.is_available():
        # before = torch.cuda.memory_allocated()
        torch.cuda.empty_cache()
        # after = torch.cuda.memory_allocated()
        # freed = before - after # ! the effect of empty_cache() is not immediate


@dataclass(frozen=False)
class PatchSpec:
    location: tuple[str, int]
    patch: torch.Tensor
    clean: Optional[torch.Tensor] = None
    strategy: Literal["replace", "add", "add_norm"] = "replace"
    on_pass: Literal["first", "all"] = "all"
    """
    ``strategy="add_norm"`` — norm-relative add: ``patch`` is a UNIT direction
    (scale it by any coefficient before passing it in) and the applied update is
    ``h + ||h|| * patch``, so the injection is a fixed fraction of the local
    residual norm instead of a fixed absolute magnitude.  Only the *direction*
    comes from the caller; the magnitude is read at application time, because
    ``||h||`` varies by layer, by decode step and by batch item.

    ``on_pass="first"`` — apply the patch only during the prefill / first-token
    generation step (the forward pass that consumes the full prompt).

    ``on_pass="all"`` — apply at every autoregressive step.  Because KV-cache
    is enabled by default, only token positions 0 or -1 are valid for
    ``on_pass="all"``; any other fixed position would be out-of-bounds on
    decode steps where the key-dim is 1.
    """


def _apply_patch_to_state(
    current_state: torch.Tensor,
    index: int,
    patch: torch.Tensor,
    strategy: str,
) -> None:
    if patch.dtype != current_state.dtype:
        patch = patch.to(current_state.dtype)
    if patch.device != current_state.device:
        patch = patch.to(current_state.device)
    """In-place patch application (shared by both interfaces)."""
    if strategy == "replace":
        current_state[:, index, :] = patch
    elif strategy == "add":
        current_state[:, index, :] += patch
    elif strategy == "add_norm":
        # Norm-relative add: `patch` is the (already alpha-scaled) UNIT direction, and the
        # magnitude is taken from the state being edited, so the injection is a fixed fraction of
        # the local residual norm rather than a fixed absolute size. This cannot be folded into
        # the patch tensor: ||h|| differs per layer, per decode step and per batch item, so it has
        # to be read here, at application time.
        h = current_state[:, index, :]
        scale = h.float().norm(dim=-1, keepdim=True).to(h.dtype)
        current_state[:, index, :] = h + scale * patch
    else:
        raise ValueError(
            f"strategy must be 'replace', 'add' or 'add_norm', got {strategy!r}"
        )


def _generate_nnsight(
    mt: ModelandTokenizer,
    inputs: TokenizerOutput,
    patches: list["PatchSpec"],
    max_new_tokens: int,
    do_sample: bool,
    n_steer_tokens: Optional[int] = None,
    **kwargs,
) -> torch.Tensor:
    """Run mt.generate() with nnsight patching.

    Uses the shape trick to differentiate the prefill step (seq_len > 1) from
    decode steps (seq_len == 1 with KV-cache).  This allows per-patch
    ``on_pass`` control inside a single ``gen_trace.all()`` context.

    ``n_steer_tokens`` is not supported with the nnsight interface because
    nnsight's proxy graph does not allow mutable Python state inside
    ``gen_trace.all()``.  Pass ``patch_interface="baukit"`` when using
    ``n_steer_tokens``.
    """
    if n_steer_tokens is not None:
        raise NotImplementedError(
            "n_steer_tokens is not supported with patch_interface='nnsight'. "
            "Use patch_interface='baukit' instead."
        )
    with mt.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        output_scores=True,
        return_dict_in_generate=True,
        **kwargs,
    ) as gen_trace:
        if patches:
            with gen_trace.all():
                for cur_patch in patches:
                    module_name, index = cur_patch.location
                    module = get_module_nnsight(mt, module_name)
                    current_state = (
                        module.output[0].save()
                        if "attn" in module_name
                        else module.output.save()
                    )
                    if current_state.ndim == 2:
                        current_state = current_state.unsqueeze(0)

                    # Shape trick: prefill has seq_len > 1; decode steps have
                    # seq_len == 1 when KV-cache is active (default).
                    is_first_pass = current_state.shape[1] > 1
                    should_apply = cur_patch.on_pass == "all" or (
                        cur_patch.on_pass == "first" and is_first_pass
                    )
                    if should_apply:
                        _apply_patch_to_state(
                            current_state, index, cur_patch.patch, cur_patch.strategy
                        )
        gen_out = mt.generator.output.save()

    return gen_out.sequences


def _generate_baukit(
    mt: ModelandTokenizer,
    inputs: TokenizerOutput,
    patches: list["PatchSpec"],
    max_new_tokens: int,
    do_sample: bool,
    n_steer_tokens: Optional[int] = None,
    **kwargs,
) -> torch.Tensor:
    """Run mt._model.generate() with baukit forward-hook patching.

    Baukit hooks receive real tensors (not proxies), so the shape trick is
    unambiguous.  Use this as a fallback if nnsight's proxy graph causes issues.

    ``n_steer_tokens`` limits how many decode steps patches with
    ``on_pass="all"`` are applied.  After that many decode steps the patch is
    silently skipped; ``on_pass="first"`` patches are unaffected.
    """
    if not patches:
        out = mt._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            **kwargs,
        )
        return out.sequences if hasattr(out, "sequences") else out

    modules_to_patches: dict[str, list["PatchSpec"]] = {}
    for p in patches:
        modules_to_patches.setdefault(p.location[0], []).append(p)

    # Use the first (shallowest) hooked layer as a step-boundary sentinel:
    # it fires exactly once per forward pass, so we increment the decode-step
    # counter only when it fires on a decode step (is_first_pass=False).
    sorted_layers = sorted(modules_to_patches.keys())
    sentinel_layer = sorted_layers[0]
    decode_step = [0]  # mutable so the closure can write to it

    # Pre-place each patch on its target module's device/dtype ONCE, here, instead of letting
    # `_apply_patch_to_state` do it on every decode step. With device_map="auto" the steered
    # layers can straddle a GPU boundary, which made that in-hook `.to()` a cross-GPU P2P copy
    # repeated (n_layers x n_steps) times per call. Keyed by id() so caller-owned PatchSpecs are
    # left untouched.
    prepped: dict[int, torch.Tensor] = {}
    for _name, _plist in modules_to_patches.items():
        _mod = baukit.get_module(mt._model, _name)
        _param = next(_mod.parameters(), None)
        for _p in _plist:
            _t = _p.patch
            if _param is not None:
                _t = _t.to(device=_param.device, dtype=_param.dtype)
            prepped[id(_p)] = _t

    def hook(output, layer):
        nonlocal decode_step
        is_tuple = isinstance(output, tuple)
        actual = output[0] if is_tuple else output
        if actual.dim() == 2:
            actual = actual.unsqueeze(0)

        is_first_pass = actual.shape[1] > 1

        # Increment decode-step counter once per decode pass (at the sentinel layer).
        if not is_first_pass and layer == sentinel_layer:
            decode_step[0] += 1

        for p in modules_to_patches.get(layer, []):
            if p.on_pass == "first":
                should_apply = is_first_pass
            elif p.on_pass == "all":
                if is_first_pass:
                    should_apply = True
                elif n_steer_tokens is not None:
                    should_apply = decode_step[0] <= n_steer_tokens
                else:
                    should_apply = True
            else:
                should_apply = False

            if should_apply:
                _apply_patch_to_state(
                    actual, p.location[1], prepped[id(p)], p.strategy
                )

        return (actual,) + output[1:] if is_tuple else actual

    with baukit.TraceDict(
        module=mt._model,
        layers=list(modules_to_patches.keys()),
        edit_output=hook,
    ):
        out = mt._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            **kwargs,
        )

    return out.sequences if hasattr(out, "sequences") else out


def generate_with_patch(
    mt: ModelandTokenizer,
    inputs: str | TokenizerOutput,
    n_gen_per_prompt: int = 5,
    max_new_tokens: int = 16,
    patches: Optional[list[PatchSpec]] = None,
    do_sample: bool = True,
    patch_interface: Literal["nnsight", "baukit"] | callable = "nnsight",
    remove_prefix: bool = False,
    skip_special_tokens: bool = True,
    n_steer_tokens: Optional[int] = None,
    **kwargs,
) -> list[str]:
    """Generate text with optional activation patches.

    Args:
        mt:               Model + tokenizer.
        inputs:           Prompt string or pre-tokenized TokenizerOutput.
        n_gen_per_prompt: Copies of each prompt (only used when inputs is a str).
        max_new_tokens:   Maximum tokens to generate.
        patches:          List of PatchSpec objects.  Each patch controls its own
                          ``strategy`` ("replace"|"add") and ``on_pass``
                          ("first"|"all").  Patches with ``on_pass="all"`` must
                          have ``location[1]`` in {0, -1}.
        do_sample:        Whether to sample (True) or greedy-decode (False).
        patch_interface:  "nnsight" (default) uses nnsight's gen_trace.all()
                          with the shape trick.  Switch to "baukit" if nnsight
                          raises proxy/UnboundLocalError issues.
        remove_prefix:    Strip the input prompt tokens from the returned strings.
        skip_special_tokens: Passed to batch_decode.
        n_steer_tokens:   If set, patches with ``on_pass="all"`` are applied for
                          only the first ``n_steer_tokens`` decode steps; after
                          that the model generates freely.  Requires
                          ``patch_interface="baukit"``.
    """
    if isinstance(inputs, TokenizerOutput):
        if "offset_mapping" in inputs:
            inputs.pop("offset_mapping")
    else:
        inputs = prepare_input(
            prompts=[inputs],
            tokenizer=mt,
            n_gen_per_prompt=n_gen_per_prompt,
        )

    if patches is not None and isinstance(patches, PatchSpec):
        patches = [patches]

    if n_steer_tokens is not None and patch_interface != "baukit":
        raise ValueError(
            "n_steer_tokens requires patch_interface='baukit'. "
            "nnsight does not support mutable step counters inside gen_trace.all()."
        )

    if patches:
        # Validate on_pass="all" constraint: KV-cache makes fixed interior
        # positions invalid after the prefill step.
        for p in patches:
            if p.on_pass == "all" and p.location[1] not in (0, -1):
                raise ValueError(
                    f"PatchSpec with on_pass='all' requires location[1] to be "
                    f"0 or -1 (got {p.location[1]}). With KV-cache, decode steps "
                    f"have seq_len=1, so any other fixed position is out-of-bounds."
                )

        # Sort patches by layer depth so shallower patches apply first.
        try:
            patches = sorted(patches, key=lambda p: int(p.location[0].split(".")[2]))
        except (IndexError, ValueError):
            pass  # Non-standard layer name format — apply in given order.

    start = inputs.input_ids.shape[1] if remove_prefix else 0

    if type(patch_interface) is str:
        if patch_interface == "nnsight":
            sequences = _generate_nnsight(
                mt, inputs, patches or [], max_new_tokens, do_sample, **kwargs
            )
        elif patch_interface == "baukit":
            sequences = _generate_baukit(
                mt, inputs, patches or [], max_new_tokens, do_sample,
                n_steer_tokens=n_steer_tokens, **kwargs
            )
        else:
            raise ValueError(
                f"patch_interface must be 'nnsight' or 'baukit', got {patch_interface!r}"
            )
    else:
        # Custom interface provided by caller.
        sequences = patch_interface(
            mt=mt,
            inputs=inputs,
            patches=patches or [],
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            **kwargs,
        )

    return mt.tokenizer.batch_decode(
        sequences[:, start:], skip_special_tokens=skip_special_tokens
    )


@torch.inference_mode()
def generate_with_beam_search(
    mt: ModelandTokenizer,
    inputs: str | TokenizerOutput,
    num_beams: int = 20,
    num_return_sequences: int = 10,
    max_new_tokens: int = 20,
    no_repeat_ngram_size: int = 3,
    processor: callable = lambda x: x,
) -> list[str]:
    if isinstance(inputs, TokenizerOutput):
        if "offset_mapping" in inputs:
            inputs.pop("offset_mapping")
    else:
        inputs = prepare_input(
            prompts=[inputs],
            tokenizer=mt,
        )

    if no_repeat_ngram_size < 1:
        logger.warning("no_repeat_ngram_size >= 1 recommended for varied generations")

    outputs = mt._model.generate(
        inputs.input_ids,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        max_new_tokens=max_new_tokens,
        no_repeat_ngram_size=no_repeat_ngram_size,
        output_scores=True,
        return_dict_in_generate=True,
    )
    start = inputs.input_ids.shape[1]
    generations = mt.tokenizer.batch_decode(
        outputs.sequences[:, start:], skip_special_tokens=True
    )
    return [processor(g) for g in generations]


@torch.no_grad()
def predict_next_token(
    mt: ModelandTokenizer,
    inputs: Union[str, list[str]] | TokenizerOutput,
    k: int = 5,
    batch_size: int = 8,
    token_of_interest: Optional[Union[Union[str, int], list[Union[str, int]]]] = None,
    patches: Optional[PatchSpec | list[PatchSpec]] = None,
):
    """Predict the next token(s) given the input."""
    if isinstance(inputs, TokenizerOutput):
        if "offset_mapping" in inputs:
            inputs.pop("offset_mapping")
    else:
        inputs = [inputs] if isinstance(inputs, str) else inputs
    if token_of_interest is not None:
        token_of_interest = (
            [token_of_interest]
            if not isinstance(token_of_interest, list)
            else token_of_interest
        )
    if token_of_interest is not None:
        # print(f"{len(token_of_interest)=} | {len(inputs)=}")
        if isinstance(token_of_interest, ArrayLike) is False:
            token_of_interest = [token_of_interest]
        if isinstance(token_of_interest[0], ArrayLike) is False:
            token_of_interest = [token_of_interest]

        assert len(token_of_interest) == (
            len(inputs["input_ids"])
            if isinstance(inputs, TokenizerOutput)
            else len(inputs)
        )
        track_interesting_tokens = []

    is_tokenized = isinstance(inputs, TokenizerOutput)
    total_len = len(inputs["input_ids"]) if is_tokenized else len(inputs)

    if patches is not None and isinstance(patches, PatchSpec):
        patches = [patches]
        if total_len > 1:
            logger.warning(
                "passed `patches`, not supported for batched predictions yet. will give weird results."
            )

    predictions = []
    for i in range(0, total_len, batch_size):
        if is_tokenized is False:
            batch_inputs = prepare_input(
                tokenizer=mt,
                prompts=inputs[i : i + batch_size],
                padding_side="left",
            )
        else:
            batch_inputs = {
                k: v[i : i + batch_size] if isinstance(v, ArrayLike) else v
                for k, v in inputs.items()
            }

        # print(i, i + batch_size, batch_inputs["input_ids"].shape)

        def is_an_attn_head(module_name) -> bool | tuple[int, int]:
            attn_id = mt.attn_module_name_format.split(".")[-1]
            if attn_id not in module_name:
                return False
            if module_name.endswith(attn_id):
                return False

            head_id = module_name.split(".")[-1]
            layer_id = ".".join(module_name.split(".")[:-1])
            return layer_id, int(head_id)

        with mt.trace(batch_inputs, scan=False, validate=False) as tr:  # noqa: F841
            # TODO: patching code is being repeated a couple of times. refactor it.
            if patches is not None:
                for cur_patch in patches:
                    module_name, index = cur_patch.location
                    if is_an_attn_head(module_name) is True:
                        raise NotImplementedError(
                            "patching not supported yet for attn heads"
                        )
                    module = get_module_nnsight(mt, module_name)
                    current_state = (
                        module.output.save()
                        if ("mlp" in module_name or module_name == mt.embedder_name)
                        else module.output[0].save()
                    )
                    if current_state.ndim == 2:
                        current_state = current_state.unsqueeze(0)
                    current_state[:, index, :] = cur_patch.patch
            batch_logits = mt.output.logits.save()

        batch_logits = batch_logits[:, -1, :]
        batch_probs = batch_logits.float().softmax(dim=-1)
        batch_topk = batch_probs.topk(k=k, dim=-1)

        batch_interested_token_indices = None
        if token_of_interest is not None:
            batch_interested_token_indices = []
            for j in range(i, i + batch_inputs["input_ids"].shape[0]):
                batch_interested_token_indices.append(
                    mt.tokenizer(
                        token_of_interest[j],
                        add_special_tokens=False,
                        return_tensors="pt",
                    ).input_ids[:, 0]
                    if type(token_of_interest[j]) is str
                    else token_of_interest[j]
                )

        for batch_order, (token_ids, token_probs) in enumerate(
            zip(batch_topk.indices, batch_topk.values)
        ):
            top_pred = interpret_logits(
                tokenizer=mt,
                logits=batch_logits[batch_order],
                k=k,
                interested_tokens=(
                    batch_interested_token_indices[batch_order]
                    if token_of_interest is not None
                    else []
                ),
            )
            if token_of_interest is not None:
                track_interesting_tokens.append(top_pred[1])
                top_pred = top_pred[0]

            predictions.append(top_pred)

        free_gpu_cache()

    if token_of_interest is not None:
        return predictions, track_interesting_tokens

    return predictions


def get_module_nnsight(model, layer_name):
    layer = model
    for name in layer_name.split("."):
        layer = layer[int(name)] if name.isdigit() else getattr(layer, name)
    return layer


def patch_with_baukit(
    mt: ModelandTokenizer,
    inputs: TokenizerOutput,
    patches: list[PatchSpec] = [],
    model_kwargs: dict = {},
):
    model = mt._model

    batch_size = inputs.input_ids.shape[0]
    seq_len = inputs.input_ids.shape[1]
    # n_heads = mt.config.num_attention_heads
    # head_dim = mt.config.hidden_size // n_heads
    head_dim = get_module_nnsight(
        mt._model, mt.attn_module_name_format.format(0)
    ).head_dim
    # group_size = n_heads // mt.config.num_key_value_heads

    modules_to_patches = {}
    for patch in patches:
        if len(patch.location) == 2:
            layer, idx = patch.location
            head_idx = None
        elif len(patch.location) == 3:
            layer, head_idx, idx = patch.location
        else:
            raise ValueError(f"Invalid patch location: {patch.location}")
        if layer not in modules_to_patches:
            modules_to_patches[layer] = []
        modules_to_patches[layer].append(patch)

    unique_modules = list(modules_to_patches.keys())

    def perform_patch(repr, layer_name):
        # print(layer_name)
        if layer_name not in unique_modules:
            return repr

        current_state = (
            repr
            if (
                "qwen" in mt.name.lower()
                or "llama" in mt.name.lower()
                or (
                    "mlp" in layer_name
                    or layer_name == mt.embedder_name
                    or "_proj" in layer_name
                )
            )
            else repr[0]
        )
        if current_state.dim() == 2:
            current_state = current_state.unsqueeze(0)
        for patch in modules_to_patches[layer_name]:
            # current_state[:, patch.index, :] = patch.patch
            if len(patch.location) == 2:
                _, token_idx = patch.location
                head_idx = None
            elif len(patch.location) == 3:
                _, head_idx, token_idx = patch.location
            else:
                raise ValueError(f"Invalid patch location: {patch.location}")

            if head_idx is None:
                # shared with the nnsight path — also handles "add_norm", which needs ||h|| read
                # here rather than baked into the patch tensor
                _apply_patch_to_state(
                    current_state, token_idx, patch.patch, patch.strategy
                )
            else:
                current_state = current_state.reshape(
                    batch_size, seq_len, -1, head_dim
                ).transpose(1, 2)
                if patch.strategy == "replace":
                    current_state[:, head_idx, token_idx, :] = patch.patch
                elif patch.strategy == "add":
                    current_state[:, head_idx, token_idx, :] += patch.patch
                else:
                    raise ValueError(f"Unknown patch strategy: {patch.strategy}")
                current_state = current_state.transpose(1, 2).reshape(
                    batch_size, seq_len, -1
                )

        return repr

    with baukit.TraceDict(
        module=model, layers=unique_modules, edit_output=perform_patch
    ):
        output = model(**inputs, **model_kwargs)

    return output


def patch_with_nnsight(
    mt: ModelandTokenizer,
    inputs: TokenizerOutput,
    patches: list[PatchSpec] = [],
    model_kwargs: dict = {},
):
    batch_size = inputs.input_ids.shape[0]
    seq_len = inputs.input_ids.shape[1]
    n_heads = mt.config.num_attention_heads
    # head_dim = mt.config.hidden_size // n_heads
    head_dim = get_module_nnsight(
        mt._model, mt.attn_module_name_format.format(0)
    ).head_dim

    with mt.trace(inputs, **model_kwargs) as tracer:
        for cur_patch in patches:
            if len(cur_patch.location) == 2:
                module_name, index = cur_patch.location
                head_idx = None
            elif len(cur_patch.location) == 3:
                module_name, head_idx, index = cur_patch.location

            print(module_name, index, head_idx, cur_patch.strategy)
            module = get_module_nnsight(mt, module_name)
            current_state = (
                module.output
                if (
                    "mlp" in module_name
                    or module_name == mt.embedder_name
                    or "_proj" in module_name
                )
                else module.output[0]
            )
            if current_state.ndim == 2:
                current_state = current_state.unsqueeze(0)

            # current_state = current_state.clone()
            print(
                "log",
                current_state.shape,
                cur_patch.location,
                cur_patch.patch.shape,
            )
            if head_idx is None:
                print("is working")
                tracer.log(">> false")
                if cur_patch.strategy == "replace":
                    current_state[:, index, :] = cur_patch.patch
                elif cur_patch.strategy == "add":
                    current_state[:, index, :] += cur_patch.patch
                else:
                    raise ValueError(
                        f"patch_strategy must be one of 'replace', 'add'. got {cur_patch.strategy}"
                    )
                # module.output[...] = current_state

            else:
                print("-- not working")
                print(">> tracer")
                # print("HI", cur_patch.strategy, cur_patch.location)
                current_state = current_state.clone()
                current_state = current_state.reshape(
                    batch_size, seq_len, n_heads, head_dim
                ).transpose(1, 2)
                print(current_state.shape, head_idx, index)

                if cur_patch.strategy == "replace":
                    current_state[:, head_idx, index, :] = cur_patch.patch
                elif cur_patch.strategy == "add":
                    current_state[:, head_idx, index, :] += cur_patch.patch
                else:
                    raise ValueError(
                        f"patch_strategy must be one of 'replace', 'add'. got {cur_patch.strategy}"
                    )
                current_state = current_state.transpose(1, 2).reshape(
                    batch_size, seq_len, n_heads * head_dim
                )
                print(current_state.shape)
                print(current_state.shape, "Setting")
                module.output[...] = current_state

        output = mt.output.save()
    return output


@torch.no_grad()
def get_hs(
    mt: ModelandTokenizer,
    input: str | TokenizerOutput,
    locations: tuple[str, int] | list[tuple[str, int]],
    patches: Optional[PatchSpec | list[PatchSpec]] = None,
    return_dict: bool = False,
) -> torch.Tensor | dict[tuple[str, int], torch.Tensor]:
    if isinstance(input, TokenizerOutput):
        if "offset_mapping" in input:
            input.pop("offset_mapping")
    else:
        input = prepare_input(prompts=input, tokenizer=mt.tokenizer)

    if isinstance(locations, tuple):
        locations = [locations]
    if patches is not None and isinstance(patches, PatchSpec):
        patches = [patches]
    if patches is not None:
        patches = sorted(
            patches, key=lambda patch: int(patch.location[0].split(".")[2])
        )

    def is_an_attn_head(module_name) -> bool | tuple[int, int]:
        attn_id = mt.attn_module_name_format.split(".")[-1]
        if attn_id not in module_name:
            return False
        if module_name.endswith(attn_id):
            return False

        head_id = module_name.split(".")[-1]
        layer_id = ".".join(module_name.split(".")[:-1])

        return layer_id, int(head_id)

    layer_names = [layer_name for layer_name, _ in locations]
    unique_layer_names = []
    for layer_name in layer_names:
        if layer_name not in unique_layer_names:
            unique_layer_names.append(layer_name)

    layer_states = {layer_name: torch.empty(0) for layer_name in layer_names}
    with mt.trace(input, scan=False):
        if patches is not None:
            for cur_patch in patches:
                module_name, index = cur_patch.location
                if is_an_attn_head(module_name) is True:
                    raise NotImplementedError(
                        "patching not supported yet for attn heads"
                    )
                module = get_module_nnsight(mt, module_name)
                current_state = (
                    module.output.save()
                    if (
                        ("mlp" in module_name or module_name == mt.embedder_name)
                        # or "qwen" in mt.name.lower()
                        # or "llama" in mt.name.lower()
                    )
                    else module.output[0].save()
                )
                # print(current_state.shape)
                if current_state.ndim == 2:
                    current_state = current_state.unsqueeze(0)
                if cur_patch.strategy == "replace":
                    current_state[:, index, :] = cur_patch.patch
                elif cur_patch.strategy == "add":
                    current_state[:, index, :] += cur_patch.patch
                else:
                    raise ValueError(
                        f"patch_strategy must be one of 'replace', 'add'. got {cur_patch.strategy}"
                    )

        for layer_name in unique_layer_names:
            # print(f"Getting state for {layer_name}")
            if is_an_attn_head(layer_name) is False:
                module = get_module_nnsight(mt, layer_name)
                layer_states[layer_name] = module.output.save()
            else:
                attn_module_name, head_idx = is_an_attn_head(layer_name)
                o_proj_name = attn_module_name + ".o_proj"
                head_dim = mt.head_dim  # config head_dim (Qwen3 decouples it from n_embd)
                o_proj = get_module_nnsight(mt, o_proj_name)
                layer_states[layer_name] = o_proj.input[0][0][
                    :, :, head_idx * head_dim : (head_idx + 1) * head_dim
                ].save()

    hs = {}

    for layer_name, index in locations:
        # print(
        #     layer_name, layer_states[layer_name].shape, type(layer_states[layer_name])
        # )
        hs[(layer_name, index)] = (
            untuple(layer_states[layer_name])[:, index, :].squeeze().detach()
        )

    # print(f"==========> {len(hs)=}")
    if len(hs) == 1 and not return_dict:
        return list(hs.values())[0]
    return hs


def extract_rep_at_pos(
    mt: ModelandTokenizer,
    input: str | TokenizerOutput,
    total_length: int,
    locations: tuple[str, int] | list[tuple[str, int]],
    return_dict: bool = False,
):
    if isinstance(input, TokenizerOutput):
        if "offset_mapping" in input:
            input.pop("offset_mapping")
    else:
        input = prepare_input(prompts=input, tokenizer=mt.tokenizer)

    assert total_length >= len(input["input_ids"][0]), (
        "Total length cannot be smaller than the input length"
    )

    input = insert_padding_before_pos(
        inp=input,
        token_position=0,  # prepend pads before the first token
        pad_len=total_length - len(input["input_ids"][0]),
        pad_id=mt.tokenizer.pad_token_id,
    )

    logger.debug(f"{input.input_ids.shape=}")

    return get_hs(mt=mt, input=input, locations=locations, return_dict=return_dict)


@torch.inference_mode
def get_all_module_states(
    mt: ModelandTokenizer,
    input: str | TokenizerOutput,
    kind: Literal["residual", "mlp", "attention"] = "residual",
) -> dict[tuple[str, int], torch.Tensor]:
    if isinstance(input, TokenizerOutput):
        if "offset_mapping" in input:
            input.pop("offset_mapping")
    else:
        input = prepare_input(prompts=input, tokenizer=mt.tokenizer)

    layer_name_format = None
    if kind == "residual":
        layer_name_format = mt.layer_name_format
    elif kind == "mlp":
        layer_name_format = mt.mlp_module_name_format
    elif kind == "attention":
        layer_name_format = mt.attn_module_name_format
    else:
        raise ValueError("kind must be one of 'residual', 'mlp', 'attention'")

    layer_and_index = []
    for layer_idx in range(mt.n_layer):
        for token_idx in range(input.input_ids.shape[1]):
            layer_and_index.append((layer_name_format.format(layer_idx), token_idx))

    return get_hs(mt, input, layer_and_index)


def guess_subject(prompt):
    return re.search(r"(?!Wh(o|at|ere|en|ich|y) )([A-Z]\S*)(\s[A-Z][a-z']*)*", prompt)[
        0
    ].strip()


def patch_linear_subspaces(
    mt: ModelandTokenizer,
    base_input: TokenizerOutput,
    rotator: dict[str, torch.nn.Module],
    patches: list[PatchSpec],
    rotate_dimensions: int = 100,
    with_grad=False,
):
    module_to_tok_indices = {}
    for patch in patches:
        module_name, token_index = patch.location
        if module_name not in module_to_tok_indices:
            module_to_tok_indices[module_name] = []
        module_to_tok_indices[module_name].append(patch)
    for module_name in module_to_tok_indices:
        if module_name not in rotator:
            raise ValueError(
                f"Rotator for module {module_name} not found in rotator dict"
            )
        if rotator[module_name] is None:
            raise ValueError(f"Rotator for module {module_name} is None")

    # print(f">>> {base_input.input_ids.shape=}")
    grad_env = torch.enable_grad if with_grad else torch.no_grad
    with grad_env():
        with mt.trace(base_input) as tracer:  # noqa: F841
            for module_name in module_to_tok_indices:
                module = get_module_nnsight(mt, module_name)

                # get the base representation
                base_state = (
                    module.output
                    if (
                        "qwen" in mt.name.lower()
                        or "llama" in mt.name.lower()
                        or ("mlp" in module_name or module_name == mt.embedder_name)
                    )
                    else module.output[0]
                )

                # logger.debug(f"{module_name} | {base_state.shape=}")

                if base_state.ndim == 2:
                    base_state = base_state.unsqueeze(0)

                if next(rotator[module_name].parameters()).device != base_state.device:
                    rotator[module_name] = rotator[module_name].to(base_state.device)

                #! takes too long
                # try:
                #     device = rotator[module_name].weight.device
                #     dtype = rotator[module_name].weight.dtype
                #     rotator_inverse = torch.linalg.pinv(
                #         rotator[module_name].weight.float()
                #     ).to(device=device, dtype=dtype)
                # except:
                #     print("exception")
                #     rotator_inverse = rotator[module_name].weight.T

                #! becase the rotator should be kept orthogonal
                #! there will be some precision error with bfloat16
                rotator_inverse = rotator[module_name].weight.T
                for patch in module_to_tok_indices[module_name]:
                    _, token_idx = patch.location

                    # print(f"{base_state.shape=}, {token_idx=}")
                    if base_state.ndim == 2:
                        base_state = base_state.unsqueeze(0)
                    base_token_state = base_state[:, token_idx, :].clone()
                    # rotate the base representation
                    rotated_base = rotator[module_name](base_token_state)

                    # rotate the source representation
                    source_state = patch.patch
                    if source_state.ndim == 1:
                        source_state = source_state.unsqueeze(0)
                    # print(f"{base_token_state.shape=}, {source_state.shape=}")
                    rotated_source = rotator[module_name](source_state)
                    # print(f"{rotated_base.shape=}, {rotated_source.shape=}")

                    # patch the first `rotate_dimensions` dimensions of the source representation
                    rotated_patch = torch.cat(
                        [
                            rotated_source[:, :rotate_dimensions],
                            rotated_base[:, rotate_dimensions:],
                        ],
                        dim=1,
                    )

                    # inverse rotate the patched representation
                    patch = torch.matmul(rotated_patch, rotator_inverse.T)

                    # replace the base representation with the patched representation
                    # print(f"{base_state.shape=}, {patch.shape=}")
                    base_state[:, token_idx, :] = patch

            output = mt.output.save()
    return output


def free_gpu_cache():
    torch.cuda.synchronize()
    # before = torch.cuda.memory_allocated()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    # after = torch.cuda.memory_allocated()
    # freed = before - after

    # logger.debug(
    #     f"freed {models.bytes_to_human_readable(freed)} | before={models.bytes_to_human_readable(before)} -> after={models.bytes_to_human_readable(after)}"
    # )


def get_dummy_input(
    tokenizer: ModelandTokenizer | Tokenizer,
):
    dummy_prompt = "The quick brown fox"
    return prepare_input(prompts=dummy_prompt, tokenizer=tokenizer)


# NOTE: low_rank_approx / low_rank_pinv / SVD moved to src.analysis.algo


# useful for saving with jsons
def detensorize(inp: dict[Any, Any] | list[dict[Any, Any]], to_numpy: bool = False):
    if isinstance(inp, list):
        return [detensorize(i) for i in inp]
    if isinstance(inp, dict) is False:
        try:
            cls = type(inp)
            inp = inp.__dict__
        except Exception:
            return inp
    else:
        cls = None

    inp = copy.deepcopy(inp)
    for k in inp:
        if isinstance(inp[k], torch.Tensor):
            if len(inp[k].shape) == 0:
                inp[k] = inp[k].item()
            else:
                inp[k] = inp[k].tolist() if to_numpy is False else inp[k].cpu().numpy()
        else:
            inp[k] = detensorize(inp[k])

    free_gpu_cache()

    if cls is None:
        return inp
    else:
        if cls != TokenizerOutput:
            return cls(**inp)
        else:
            return cls(data=inp)


def save_object(obj, save_path):
    """
    saves an object as a npz file
    assume that the obj is a dictionary
    and does not have another custom object as a value
    """
    if isinstance(obj, dict) is False:
        obj = detensorize(obj)
    np.savez(
        save_path,
        **obj,
    )


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
