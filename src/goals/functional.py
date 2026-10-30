import logging

import torch

from src.models import ModelandTokenizer
from src.models import supports_chat_template as _supports_chat_template
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
    if "gemma-2" in mt.name.lower():
        messages.pop(0)  # Gemma-2 doesn't support system prompts in chat templates
        messages[0]["content"] = f"{system_prompt}\n\n{user_query}" # prepend system prompt to user query instead

    prompt = mt.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=False,  # Don't add a new assistant header
        continue_final_message=True,  # Don't close the assistant turn with <|eot_id|>
        tokenize=False,
        chat_template_kwargs={"enable_thinking": False},  # only Qwen3 and Qwen3.5
    )
    return prompt


def supports_chat_template(mt: ModelandTokenizer) -> bool:
    """Does the model expose a chat template? Kept here for backward compatibility —
    delegates to ``src.models.supports_chat_template``, which asks the tokenizer instead of
    pattern-matching the model name (the name heuristic missed e.g. ``Qwen/Qwen3-32B``)."""
    return _supports_chat_template(mt)


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
