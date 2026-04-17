"""Activation oracle utilities for GoalScope.

Thin wrappers around the Karvonen et al. (2025) activation oracle that plug
into the goals project's own interfaces (ModelandTokenizer, find_token_range,
render_patchscope_html).

The oracle takes a hidden-state vector extracted from a *target* model run,
injects it into an *oracle* model via a forward hook at layer 1, then asks a
natural-language question.  The oracle LoRA was trained to answer questions
about activations, so the generated text verbalizes what information is encoded
in that hidden state.

Paper: https://arxiv.org/pdf/2512.15674
Original code: https://github.com/adamkarvonen/activation_oracles
"""
from __future__ import annotations

import contextlib
from typing import Optional

import torch
import torch.nn.functional as F

from src.functional import get_module_nnsight
from src.models import ModelandTokenizer
from src.tokens import find_token_range


# ── Oracle configs ────────────────────────────────────────────────────────────

ORACLE_CONFIGS: dict[str, dict] = {
    "meta-llama/Llama-3.3-70B-Instruct": {
        "oracle_lora": "adamkarvonen/checkpoints_act_cls_latentqa_pretrain_mix_adding_Llama-3_3-70B-Instruct",
    },
    "Qwen/Qwen3-8B": {
        "oracle_lora": "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B",
    },
}

SPECIAL_TOKEN = " ?"          # activation placeholder token
INJECTION_LAYER = 1           # always layer 1 per Karvonen et al.
EXTRACT_LAYER_PCT = 40        # extract at 40% depth by default


# ── Internal helpers ──────────────────────────────────────────────────────────

@contextlib.contextmanager
def _hook(module, hook_fn):
    handle = module.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()


# ── Public API ────────────────────────────────────────────────────────────────

def collect_hidden_states(
    mt: ModelandTokenizer,
    prompt: str,
    layer: Optional[int] = None,
    layer_pct: int = EXTRACT_LAYER_PCT,
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """Run a forward pass and return hidden states at `layer`.

    Uses nnsight-native tracing so it is compatible with ModelandTokenizer's
    Envoy wrapper.  Call with oracle adapters disabled to get clean base-model
    activations.

    Args:
        mt: ModelandTokenizer.
        prompt: Already-formatted prompt string (no new tokens will be generated).
        layer: Absolute layer index.  If None, inferred from `layer_pct`.
        layer_pct: Percentage of total layers (used only when `layer` is None).

    Returns:
        Tuple of:
          - hidden [seq_len, d_model] float32 CPU tensor
          - offset_mapping list[tuple[int,int]] (token → char span in `prompt`)
    """
    if layer is None:
        layer = int(mt.n_layer * layer_pct / 100)

    enc = mt.tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offset_mapping = enc.pop("offset_mapping")[0].tolist()

    layer_name = mt.layer_name_format.format(layer)

    with torch.no_grad():
        with mt.trace(enc):
            module = get_module_nnsight(mt, layer_name)
            hidden = module.output[0].save()

    # nnsight with batch=1 saves output[0] as [seq, d_model] (batch already removed)
    return hidden.cpu(), offset_mapping  # [seq, d_model], list[(cs,ce)]


def query_oracle(
    peft_model,
    tokenizer,
    activation_vec: torch.Tensor,
    question: str,
    oracle_adapter_name: str,
    extract_layer: int,
    injection_layer: int = INJECTION_LAYER,
    max_new_tokens: int = 60,
    steering_coeff: float = 1.0,
) -> str:
    """Ask the oracle a question about a single activation vector.

    Args:
        peft_model: ``PeftModel`` returned by ``PeftModel.from_pretrained(mt._model, ...)``.
            Must have ``oracle_adapter_name`` loaded.  Call with adapters disabled
            before this function; it enables/disables the adapter internally.
        tokenizer: The model's tokenizer (``mt.tokenizer``).
        activation_vec: [d_model] float32 hidden state from the target model.
        question: Natural-language question for the oracle.
        oracle_adapter_name: Name of the loaded LoRA adapter.
        extract_layer: The layer the activation was extracted from (in oracle prompt).
        injection_layer: Where to inject into the oracle (default: layer 1).
        max_new_tokens: Max tokens for oracle to generate.
        steering_coeff: Scales the injected vector (1.0 = norm-matched).

    Returns:
        Oracle's generated response string.
    """
    # ── Build oracle prompt ───────────────────────────────────────────────────
    prefix = f"Layer: {extract_layer}\n{SPECIAL_TOKEN} \n"
    oracle_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prefix + question}],
        tokenize=False,
        add_generation_prompt=True,
    )
    enc = tokenizer(oracle_prompt, return_tensors="pt", add_special_tokens=False)
    device = next(peft_model.parameters()).device
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)

    # ── Find placeholder position ─────────────────────────────────────────────
    ph_id = tokenizer.encode(SPECIAL_TOKEN, add_special_tokens=False)
    ph_pos = next(
        (i for i, tid in enumerate(input_ids[0].tolist()) if tid == ph_id[0]),
        None,
    )
    if ph_pos is None:
        enc_off = tokenizer(
            oracle_prompt,
            return_tensors="pt",
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets = enc_off["offset_mapping"][0].tolist()
        char_pos = oracle_prompt.index(SPECIAL_TOKEN)
        ph_pos = next(
            (i for i, (cs, ce) in enumerate(offsets) if cs <= char_pos < ce),
            None,
        )

    # ── Injection hook ────────────────────────────────────────────────────────
    param_dtype = next(peft_model.parameters()).dtype
    vec = F.normalize(activation_vec.float(), dim=-1).to(dtype=param_dtype)

    def _inject_hook(_module, _inp, out):
        h = (out[0] if isinstance(out, tuple) else out).clone()
        if ph_pos is not None and ph_pos < h.shape[1]:
            norm = h[0, ph_pos].norm()
            h[0, ph_pos] = h[0, ph_pos] + steering_coeff * norm * vec.to(h.device)
        return (h,) + out[1:] if isinstance(out, tuple) else h

    # ── Switch to oracle adapter, generate, restore base ─────────────────────
    # PeftModel structure: peft_model.base_model.model.model.layers[i]
    # (LoraModel → LlamaForCausalLM → LlamaModel → ModuleList)
    inj_layer = peft_model.base_model.model.model.layers[injection_layer]

    # enable_adapter_layers() is required before set_adapter(): BaseTunerLayer.set_adapter()
    # only updates _active_adapter and requires_grad — it does NOT reset _disable_adapters.
    # Without this, every oracle query runs with the base model only (no LoRA active).
    peft_model.base_model.enable_adapter_layers()
    peft_model.set_adapter(oracle_adapter_name)

    with torch.no_grad(), _hook(inj_layer, _inject_hook):
        out_ids = peft_model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    peft_model.base_model.disable_adapter_layers()

    return tokenizer.decode(
        out_ids[0, input_ids.shape[1]:], skip_special_tokens=True
    ).strip()


def find_code_segment(
    prompt: str,
    offset_mapping: list[tuple[int, int]],
    token_ids: list[int],
) -> tuple[int, int]:
    """Return (code_start_tok, code_end_tok) for the assistant <code> block.

    Uses `find_token_range` from `src.tokens` with `occurrence=1` to skip the
    system-prompt example tags and find the assistant's actual code block.

    Args:
        prompt: The full formatted prompt string.
        offset_mapping: Token offset mapping (from tokenizer with return_offsets_mapping=True).
        token_ids: List of token ids corresponding to `offset_mapping`.

    Returns:
        (code_start, code_end) token indices — half-open, i.e. `token_ids[code_start:code_end]`.
    """
    import torch as _torch
    _off = _torch.tensor(offset_mapping)

    code_start, _ = find_token_range(
        string=prompt,
        substring="<code>",
        occurrence=1,
        offset_mapping=_off,
    )
    try:
        _, code_end = find_token_range(
            string=prompt,
            substring="</code>",
            occurrence=1,
            offset_mapping=_off,
        )
    except (ValueError, AssertionError):
        code_end = len(token_ids)

    return code_start, code_end
