import logging

import torch

from src.models import ModelandTokenizer
from src.tokens import prepare_input

logger = logging.getLogger(__name__)


def apply_chat_template(
    mt: ModelandTokenizer,
    user_query: str,
    incomplete_code: str,
    system_prompt: str = "You are a helpful assistant proficient in programming. When completing code, ensure proper syntax and logical correctness. Make sure to have all codes inside the <code> </code> tags.",
) -> str:
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_query,
        },
        {
            "role": "assistant",
            "content": f"<code>\n{incomplete_code}",
        },
    ]

    prompt = mt.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=False,  # Don't add a new assistant header
        continue_final_message=True,  # Don't close the assistant turn with <|eot_id|>
        tokenize=False,
    )
    return prompt


def supports_chat_template(mt: ModelandTokenizer) -> bool:
    """
    Heuristic to determine if the model supports chat templates.
    """
    instruct_signatures = [
        "instruct",
        "it",
        "chat",
        "oss",
        "zai-org/glm",
        "next",
        "cwm",
        "distill",
    ]
    return any(
        inst_signature in mt.name.lower() for inst_signature in instruct_signatures
    )


@torch.inference_mode()
def get_last_token_latents_batch(
    mt: ModelandTokenizer,
    texts: list[str],
    layers: list[int] | None = None,
) -> dict[int, torch.Tensor]:
    """
    Extract residual stream hidden states at the last token position for a batch of inputs.

    Simplified and batched alternative to get_hs — no interventions, only grabs
    the last (non-padding) token's residual stream across specified layers.

    Each layer's output stays on its native GPU (no cross-device stacking).

    Args:
        mt: ModelandTokenizer instance
        texts: List of input strings
        layers: Layer indices to extract from. None = all layers.

    Returns:
        Dict mapping layer_idx -> Tensor of shape (batch, hidden_dim) on that layer's device.
    """
    if layers is None:
        layers = list(range(mt.n_layer))

    # Tokenize with left-padding so the last non-pad token is always at position -1
    inputs = prepare_input(prompts=texts, tokenizer=mt, padding_side="left")

    # Set up nnsight trace to capture layer outputs
    layer_saves = {}
    with mt.trace(inputs, scan=False):
        for layer_idx in layers:
            layer_name = mt.layer_name_format.format(layer_idx)
            module = mt
            for attr in layer_name.split("."):
                module = module[int(attr)] if attr.isdigit() else getattr(module, attr)
            output = module.output
            # Some architectures wrap output in a tuple
            if isinstance(output, tuple) or (
                hasattr(output, "__len__") and not hasattr(output, "shape")
            ):
                output = output[0]
            layer_saves[layer_idx] = output[:, -1, :].save()  # (batch, hidden_dim)

    # Return per-layer tensors, each staying on its native device
    return {l: layer_saves[l].float() for l in layers}
