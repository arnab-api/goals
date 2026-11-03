import logging
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
from dataclasses_json import DataClassJsonMixin

from src.models import ModelandTokenizer
from src.tokens import prepare_input
from src.utils.typing import TokenizerOutput

logger = logging.getLogger(__name__)

def _safe_token(t: str) -> str:
    """Replace HTML angle-bracket characters with fullwidth Unicode look-alikes.

    circuitsvis passes tokens as JSON to a React component which renders them
    as text nodes.  Raw ``<`` / ``>`` characters inside the JSON get parsed as
    HTML tags by the browser before React runs, corrupting special tokens such
    as ``<|im_start|>``.  Fullwidth variants (U+FF1C / U+FF1E) are visually
    identical but are never treated as HTML markup.
    """
    return t.replace("<", "\uff1c").replace(">", "\uff1e")


@dataclass(frozen=False)
class AttentionInformation(DataClassJsonMixin):
    tokenized_prompt: list[str]
    attention_matrices: np.ndarray
    logits: torch.Tensor | None = None


@torch.inference_mode()
def get_attention_matrices(
    input: str | list[str] | TokenizerOutput,
    mt: ModelandTokenizer,
    value_weighted: bool = False,
    interventions: Optional[Callable[[ModelandTokenizer], None]] = None,
) -> "AttentionInformation | list[AttentionInformation]":
    """
    Every layer's attention pattern, ``mt.layers[l].self_attn.attention_probabilities``, read in one
    trace (needs ``attn_implementation="eager"``).

    Parameters:
        input: str, list[str], or TokenizerOutput.
            - str            => single prompt; returns one AttentionInformation
            - list[str]      => batched prompts; returns list[AttentionInformation]
            - TokenizerOutput=> already tokenized (batch dim required); returns list (or single
              AttentionInformation if batch size is 1).
        mt: ModelandTokenizer, model and tokenizer
        value_weighted: bool.
            - False => returns attention masks after softmax (the attention pattern produced inside the model)
            - True  => weights attention by ||v_k|| to reflect the actual contribution of source tokens
              (``attention_values``, expanded over the GQA groups with ``repeat_interleave``)
        interventions: optional ``interventions(mt)``, called inside the trace before the reads,
            in the invoke that carries the inputs; the reads run in a second, empty invoke. Same
            composition rule as :func:`src.goals.head_loc_attn.get_o_proj_inputs_batch`, so an
            edit of ``attention_scores`` (e.g. :func:`src.nrgs.functional.whitelist_head_attn`)
            shows in the pattern read here.
    Returns:
        AttentionInformation if a single str was passed, else list[AttentionInformation].
        Each AttentionInformation has attention_matrices of shape (layers, heads, tokens, tokens).
    """
    reason = mt.support(layer=0)["self_attn.attention_probabilities"]
    if reason is not None:
        raise ValueError(f"get_attention_matrices reads attention_probabilities: {reason}")

    single_input = False
    if isinstance(input, str):
        input = prepare_input(prompts=input, tokenizer=mt)
        single_input = True
    elif isinstance(input, list):
        input = prepare_input(prompts=input, tokenizer=mt)
    else:
        assert isinstance(input, TokenizerOutput), (
            "input must be a str, list[str], or TokenizerOutput object"
        )
        single_input = input.input_ids.shape[0] == 1
    if "offset_mapping" in input:
        input.pop("offset_mapping")

    attns = [mt.layers[l].self_attn for l in range(mt.num_layers)]
    patterns, values = [], []
    with mt.trace() as tracer:
        with tracer.invoke(dict(input)):
            if interventions is not None:
                interventions(mt)
        with tracer.invoke():
            for attn in attns:
                if value_weighted:
                    # (batch, kv_heads, seq, head_dim): enters the attention before the pattern
                    values.append(attn.attention_values.save())
                patterns.append(attn.attention_probabilities.save())  # (batch, heads, q, k)
            logits = mt.logits[:, -1].save()  # (batch, vocab)

    # Gathered on the first layer's device (layers can sit on several under device_map="auto").
    dev = patterns[0].device
    # (layers, batch, heads, q, k) -> (batch, layers, heads, q, k)
    attentions = torch.stack([a.to(dev) for a in patterns], dim=0).permute(1, 0, 2, 3, 4).contiguous()

    if value_weighted:
        # (layers, batch, kv_heads, seq, head_dim) -> (batch, layers, kv_heads, seq, head_dim)
        v = torch.stack([x.to(dev) for x in values], dim=0).permute(1, 0, 2, 3, 4)
        # ||v_k|| per kv head, expanded over the GQA groups: query head h reads kv head h // groups
        vnorm = v.norm(dim=-1).repeat_interleave(mt.num_heads // mt.num_kv_heads, dim=2)  # (b, L, H, s)
        # attentions: (b, L, H, q, s); value norms: (b, L, H, s) -> broadcast over q
        attentions = torch.einsum("blhqs,blhs->blhqs", attentions, vnorm)

    attentions_np = attentions.detach().cpu().to(torch.float32).numpy()
    logits_cpu = logits.detach().cpu()
    batch_size = input.input_ids.shape[0]

    infos = [
        AttentionInformation(
            tokenized_prompt=[
                _safe_token(mt.tokenizer.decode(tok)) for tok in input.input_ids[i]
            ],
            attention_matrices=attentions_np[i],
            logits=logits_cpu[i],
        )
        for i in range(batch_size)
    ]
    return infos[0] if single_input else infos
