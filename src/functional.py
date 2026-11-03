import gc
import logging
import warnings
from dataclasses import dataclass
from typing import Any, Literal, Optional

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
    location: tuple[int | str, int]
    patch: torch.Tensor
    strategy: Literal["replace", "add", "add_norm"] = "replace"
    on_pass: Literal["first", "all"] = "all"
    """
    One edit of the residual stream leaving a decoder block, ``mt.layers[i].layer_output[:, index]``.

    ``location = (layer, index)``: ``layer`` is the block index ``i``, or, while unported callers
    still pass it, the native path ``mt.layer_name_format.format(i)``; ``index`` is the token
    position. ``patch`` is ``(d,)`` or ``(B, d)``, one row per batch item.

    ``strategy="add_norm"`` — norm-relative add: ``patch`` is a UNIT direction
    (scale it by any coefficient before passing it in) and the applied update is
    ``h + ||h|| * patch``, so the injection is a fixed fraction of the local
    residual norm instead of a fixed absolute magnitude.  Only the *direction*
    comes from the caller; the magnitude is read at application time, because
    ``||h||`` varies by layer, by decode step and by batch item.

    ``on_pass="first"`` — apply the patch only on the prompt call (generate step 0, the forward
    that consumes the full prompt and produces the first new token).

    ``on_pass="all"`` — apply on the prompt call and on the decode steps (all of them, or the first
    ``n_steer_tokens`` of them; see :func:`generate_with_patch`).  A decode step's stream is
    ``[B, 1, d]`` (KV cache), so only token positions 0 or -1 are valid for ``on_pass="all"``.
    """


def _apply_patch_to_state(
    current_state: torch.Tensor,
    index: int,
    patch: torch.Tensor,
    strategy: str,
) -> None:
    """In-place patch application on ``current_state[:, index, :]``."""
    if patch.dtype != current_state.dtype:
        patch = patch.to(current_state.dtype)
    if patch.device != current_state.device:
        patch = patch.to(current_state.device)
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


def _layer_index(mt: ModelandTokenizer, layer: int | str) -> int:
    """The block index of a ``PatchSpec`` location: an int as is, or the native layer path
    ``mt.layer_name_format.format(i)`` (``"model.layers.12"``) parsed back to ``i``."""
    if isinstance(layer, int):
        return layer
    prefix, _, suffix = mt.layer_name_format.partition("{}")
    middle = layer[len(prefix) : len(layer) - len(suffix)]
    if layer.startswith(prefix) and layer.endswith(suffix) and middle.isdigit():
        return int(middle)
    raise ValueError(
        f"PatchSpec location {layer!r} is not a decoder layer: pass the block index or "
        f"{mt.layer_name_format!r}. generate_with_patch edits mt.layers[i].layer_output only."
    )


# nnsight's warning when a ``tracer.iter[:n]`` loop asks for a step the run did not make. A
# patched generation whose every row hits EOS before step ``n`` ends that way; the edits for the
# steps that did run have landed, and nothing follows the loop in its invoke.
_LOOP_CUT_SHORT = r".*was never reached: the loop asked for a step the run did not make"


def generate_with_patch(
    mt: ModelandTokenizer,
    inputs: TokenizerOutput,
    max_new_tokens: int = 16,
    patches: Optional[list[PatchSpec]] = None,
    do_sample: bool = True,
    remove_prefix: bool = False,
    skip_special_tokens: bool = True,
    n_steer_tokens: Optional[int] = None,
    **kwargs,
) -> list[str]:
    """Generate text with optional residual-stream patches, on ``mt.generate`` + ``tracer.iter``.

    Each patch edits ``mt.layers[i].layer_output[:, index]`` in place (see :class:`PatchSpec`).
    Generate step 0 is the prompt call; step ``k >= 1`` is the decode step that produces new token
    ``k + 1``. ``on_pass="first"`` patches apply at step 0 only; ``on_pass="all"`` patches apply at
    step 0 and at decode steps ``1..n_steer_tokens`` (every step when ``n_steer_tokens`` is None).
    So ``n_steer_tokens`` counts DECODE steps, the prompt call not included: ``n_steer_tokens=k``
    patches ``k + 1`` forward passes. (``src.goals.goalscope.headpatched_generate`` counts the
    prompt call as the first of its ``n_steer_tokens``.) Within a step the edits run in layer
    order; patches at one layer apply in the order given.

    Args:
        mt:               Model + tokenizer.
        inputs:           Pre-tokenized TokenizerOutput (left-padded).
        max_new_tokens:   Maximum tokens to generate.
        patches:          List of PatchSpec objects.  Each patch controls its own
                          ``strategy`` ("replace"|"add"|"add_norm") and ``on_pass``
                          ("first"|"all").  Patches with ``on_pass="all"`` must
                          have ``location[1]`` in {0, -1}.
        do_sample:        Whether to sample (True) or greedy-decode (False).
        remove_prefix:    Strip the input prompt tokens from the returned strings.
        skip_special_tokens: Passed to batch_decode.
        n_steer_tokens:   If set, patches with ``on_pass="all"`` are applied on
                          the prompt call and the first ``n_steer_tokens`` decode
                          steps only; after that the model generates freely.
        **kwargs:         Forwarded to ``generate`` (decoding follows the checkpoint's
                          ``generation_config`` otherwise).
    """
    if "offset_mapping" in inputs:
        inputs.pop("offset_mapping")

    if patches is not None and isinstance(patches, PatchSpec):
        patches = [patches]
    patches = list(patches or [])

    # Group the patches by block, in forward order (patches at one block keep the given order).
    by_layer: dict[int, list[PatchSpec]] = {}
    for p in patches:
        if p.on_pass not in ("first", "all"):
            raise ValueError(f"on_pass must be 'first' or 'all', got {p.on_pass!r}")
        # A decode step's stream is [B, 1, d] (KV cache): only position 0 or -1 exists there.
        if p.on_pass == "all" and p.location[1] not in (0, -1):
            raise ValueError(
                f"PatchSpec with on_pass='all' requires location[1] to be "
                f"0 or -1 (got {p.location[1]}). With KV-cache, decode steps "
                f"have seq_len=1, so any other fixed position is out-of-bounds."
            )
        by_layer.setdefault(_layer_index(mt, p.location[0]), []).append(p)
    # Envoys bound once, outside the step loop.
    plan = [(mt.layers[i], by_layer[i]) for i in sorted(by_layer)]

    # Steps carrying an "all" patch: the prompt call + n_steer_tokens decode steps.
    n_all = max_new_tokens if n_steer_tokens is None else min(max_new_tokens, n_steer_tokens + 1)
    n_steps = n_all if any(p.on_pass == "all" for p in patches) else (1 if patches else 0)

    start = inputs["input_ids"].shape[1] if remove_prefix else 0

    with torch.no_grad(), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=_LOOP_CUT_SHORT, category=UserWarning)
        with mt.generate(max_new_tokens=max_new_tokens, do_sample=do_sample, **kwargs) as tracer:
            with tracer.invoke(dict(inputs)):
                if n_steps:
                    for step in tracer.iter[:n_steps]:
                        for layer, layer_patches in plan:
                            active = [
                                p for p in layer_patches
                                if (p.on_pass == "first" and step == 0)
                                or (p.on_pass == "all" and step < n_all)
                            ]
                            if active:
                                h = layer.layer_output
                                for p in active:
                                    _apply_patch_to_state(h, p.location[1], p.patch, p.strategy)
            # Its own invoke, so an early EOS that cuts the step loop short cannot drop it.
            with tracer.invoke():
                out = tracer.result.save()

    sequences = getattr(out, "sequences", out)
    return mt.tokenizer.batch_decode(
        sequences[:, start:], skip_special_tokens=skip_special_tokens
    )


def get_module_nnsight(model, layer_name):
    layer = model
    for name in layer_name.split("."):
        layer = layer[int(name)] if name.isdigit() else getattr(layer, name)
    return layer


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
