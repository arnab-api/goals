import logging
from dataclasses import dataclass

import numpy as np
import torch
from dataclasses_json import DataClassJsonMixin

from src.functional import (
    patch_with_baukit,
    prepare_input,
)
from src.models import ModelandTokenizer
from src.utils.typing import TokenizerOutput

logger = logging.getLogger(__name__)

# Per-query-head value norms actually read by each head, for layers whose attention is rewired so that
# heads of one GQA group read DIFFERENT values (e.g. ``src.diff.functional.headwise_kv_patch``: routed
# heads read the donor span's v, the shared cache keeps the recipient v). Filled by such wrappers during
# the forward; ``get_attention_matrices(value_weighted=True)`` uses it instead of the cache's values for
# those layers. {layer: (batch, n_heads, seq) float32}
_EFFECTIVE_VALUE_NORMS: dict = {}


def record_effective_value_norms(layer_idx: int, norms: torch.Tensor) -> None:
    _EFFECTIVE_VALUE_NORMS[int(layer_idx)] = norms.detach()


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


@torch.inference_mode()
def get_attention_matrices(
    input: str | list[str] | TokenizerOutput,
    mt: ModelandTokenizer,
    value_weighted: bool = False,
    patch_interface: callable = patch_with_baukit,
) -> "AttentionInformation | list[AttentionInformation]":
    """
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
    Returns:
        AttentionInformation if a single str was passed, else list[AttentionInformation].
        Each AttentionInformation has attention_matrices of shape (layers, heads, tokens, tokens).
    """
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

    _EFFECTIVE_VALUE_NORMS.clear()
    output = patch_interface(
        mt=mt,
        inputs=input,
        patches=[],
        model_kwargs=dict(output_attentions=True, use_cache=True),
    )
    eff_vnorm = dict(_EFFECTIVE_VALUE_NORMS)
    _EFFECTIVE_VALUE_NORMS.clear()

    # logits: (batch, seq, vocab) -> last-position logits per sample
    logits = output.logits[:, -1, :]  # (batch, vocab)

    # attentions: list of (batch, heads, seq, seq) per layer
    output.attentions = [attn.cuda() for attn in output.attentions]
    # stack along new layer axis -> (layers, batch, heads, seq, seq) -> (batch, layers, heads, seq, seq)
    attentions = (
        torch.stack(output.attentions, dim=0).permute(1, 0, 2, 3, 4).contiguous()
    )

    if value_weighted:
        pkv = output.past_key_values
        # transformers ≥ 4.36 returns a Cache object; the internal structure
        # changed again in later versions.  Try the known access patterns:
        #   - new layered API  (DynamicCache.layers[i].values)
        #   - mid-era API      (.value_cache list)
        #   - legacy tuple     ([i][1])
        if hasattr(pkv, "layers") and hasattr(pkv.layers[0], "values"):
            value_list = [layer.values.cuda() for layer in pkv.layers]
        elif hasattr(pkv, "value_cache"):
            value_list = [v.cuda() for v in pkv.value_cache]
        else:
            value_list = [pkv[i][1].cuda() for i in range(mt.n_layer)]
        # each layer value: (batch, kv_heads, seq, head_dim)
        values = torch.stack(value_list, dim=0).permute(1, 0, 2, 3, 4).contiguous()
        # values: (batch, layers, kv_heads, seq, head_dim)
        n_rep = mt.n_heads // mt.n_kv_heads
        b, L, kvh, s, d = values.shape
        values = repeat_kv(values.reshape(b * L, kvh, s, d), n_rep=n_rep)
        values = values.reshape(b, L, mt.n_heads, s, d)
        vnorm = values.norm(dim=-1)  # (b, L, H, s)
        for layer, n in eff_vnorm.items():  # heads rewired to other values (headwise kv routing)
            vnorm[:, layer] = n.to(vnorm.device, vnorm.dtype)[..., -s:]
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
