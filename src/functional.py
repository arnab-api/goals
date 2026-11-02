import gc
import logging
from dataclasses import dataclass
from typing import Any, Literal, Optional

import baukit
import torch

from src.models import ModelandTokenizer
from src.tokens import prepare_input
from src.utils.typing import TokenizerOutput

logger = logging.getLogger(__name__)


def untuple(object: Any):
    return object[0] if isinstance(object, tuple) else object


def any_is_nontrivial_prefix(predictions: list[str], target: str) -> bool:
    """Return true if any prediction is (case insensitive) prefix of the target."""
    return any(is_nontrivial_prefix(p, target) for p in predictions)


def is_nontrivial_prefix(prediction: str, target: str) -> bool:
    """Return true if prediction is (case insensitive) prefix of the target."""
    target = target.lower().strip()
    prediction = prediction.lower().strip()
    return len(prediction) > 0 and target.startswith(prediction)


@dataclass(frozen=False)
class PatchSpec:
    location: tuple[str, int]
    patch: torch.Tensor
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
    """In-place patch application."""
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
    unambiguous.

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
                _apply_patch_to_state(actual, p.location[1], p.patch, p.strategy)

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
    inputs: TokenizerOutput,
    max_new_tokens: int = 16,
    patches: Optional[list[PatchSpec]] = None,
    do_sample: bool = True,
    patch_interface: Literal["baukit"] | callable = "baukit",
    remove_prefix: bool = False,
    skip_special_tokens: bool = True,
    n_steer_tokens: Optional[int] = None,
    **kwargs,
) -> list[str]:
    """Generate text with optional activation patches.

    Args:
        mt:               Model + tokenizer.
        inputs:           Pre-tokenized TokenizerOutput.
        max_new_tokens:   Maximum tokens to generate.
        patches:          List of PatchSpec objects.  Each patch controls its own
                          ``strategy`` ("replace"|"add") and ``on_pass``
                          ("first"|"all").  Patches with ``on_pass="all"`` must
                          have ``location[1]`` in {0, -1}.
        do_sample:        Whether to sample (True) or greedy-decode (False).
        patch_interface:  "baukit" (default) patches via baukit forward hooks
                          (``_generate_baukit``); a callable is used as a custom
                          generation interface instead.
        remove_prefix:    Strip the input prompt tokens from the returned strings.
        skip_special_tokens: Passed to batch_decode.
        n_steer_tokens:   If set, patches with ``on_pass="all"`` are applied for
                          only the first ``n_steer_tokens`` decode steps; after
                          that the model generates freely.  Requires
                          ``patch_interface="baukit"``.
    """
    if "offset_mapping" in inputs:
        inputs.pop("offset_mapping")

    if patches is not None and isinstance(patches, PatchSpec):
        patches = [patches]

    if n_steer_tokens is not None and patch_interface != "baukit":
        raise ValueError(
            "n_steer_tokens requires patch_interface='baukit'."
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
        if patch_interface != "baukit":
            raise ValueError(f"patch_interface must be 'baukit' or a callable, got {patch_interface!r}")
        sequences = _generate_baukit(
            mt, inputs, patches or [], max_new_tokens, do_sample,
            n_steer_tokens=n_steer_tokens, **kwargs
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


def get_module_nnsight(model, layer_name):
    layer = model
    for name in layer_name.split("."):
        layer = layer[int(name)] if name.isdigit() else getattr(layer, name)
    return layer


def patch_with_baukit(
    mt: ModelandTokenizer,
    inputs: TokenizerOutput,
    patches: Optional[list[PatchSpec]] = None,
    model_kwargs: Optional[dict] = None,
):
    """Default ``patch_interface`` of ``src.attention.get_attention_matrices``: a plain forward
    (that function never passes patches)."""
    assert not patches, "patch_with_baukit is a plain forward; pass a custom patch_interface to patch"
    return mt._model(**inputs, **(model_kwargs or {}))


@torch.no_grad()
def get_hs(
    mt: ModelandTokenizer,
    input: str | TokenizerOutput,
    locations: tuple[str, int] | list[tuple[str, int]],
    return_dict: bool = False,
) -> torch.Tensor | dict[tuple[str, int], torch.Tensor]:
    """Hidden states (module outputs) at ``(module_name, token_index)`` locations."""
    if isinstance(input, TokenizerOutput):
        if "offset_mapping" in input:
            input.pop("offset_mapping")
    else:
        input = prepare_input(prompts=input, tokenizer=mt.tokenizer)

    if isinstance(locations, tuple):
        locations = [locations]

    unique_layer_names = []
    for layer_name, _ in locations:
        if layer_name not in unique_layer_names:
            unique_layer_names.append(layer_name)

    layer_states = {}
    with mt.trace(input):
        for layer_name in unique_layer_names:
            module = get_module_nnsight(mt, layer_name)
            layer_states[layer_name] = module.output.save()

    hs = {}
    for layer_name, index in locations:
        hs[(layer_name, index)] = (
            untuple(layer_states[layer_name])[:, index, :].squeeze().detach()
        )

    if len(hs) == 1 and not return_dict:
        return list(hs.values())[0]
    return hs


def free_gpu_cache():
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
