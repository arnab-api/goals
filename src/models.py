import logging
import os
from typing import Any, Literal

import baukit
import torch
import transformers
from nnsight import LanguageModel
from transformers import AutoTokenizer

from src.utils.env_utils import DEFAULT_MODELS_DIR, HF_CACHE_DIR
from src.utils.typing import Model, Tokenizer

logger = logging.getLogger(__name__)


class ModelandTokenizer(LanguageModel):
    """nnsight ``LanguageModel`` + tokenizer with a unified config interface (``n_layer``,
    ``n_embd``, ``layer_name_format``, ``attn_module_name_format``, ...).

    Supported families (the reported models): Qwen2 (Qwen2.5, Qwen2.5-Coder), Qwen3 (Qwen3-32B), Llama (Llama-3.3),
    OLMo-3 (Olmo-3.1) and dense GLM-4 (GLM-4-32B-0414).
    """

    def __init__(
        self,
        model_key: str,
        device_map: dict | str = "auto",
        **kwargs,
    ) -> None:
        if HF_CACHE_DIR is not None:
            kwargs["cache_dir"] = HF_CACHE_DIR
        model_key = get_full_model_path(model_key)
        self.__dict__ = LanguageModel(
            model_key,
            device_map=device_map,
            dispatch=True,
            **kwargs,
        ).__dict__
        self.name = model_key

        self._model.eval()
        # Left-pad explicitly rather than relying on nnsight's default (or a raw HF tokenizer's,
        # which is "right" for e.g. Qwen). Every batched intervention here indexes with negative
        # positions (PATCH_POS=-2, [:, -1]); those only align with the real, right-aligned tokens
        # under LEFT padding. Set once, at construction, so no downstream call has to remember.
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token  # for generation
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if (
            hasattr(self.tokenizer, "bos_token") is False
            or self.tokenizer.bos_token is None
        ):
            self.tokenizer.bos_token = self.tokenizer.eos_token
            self.tokenizer.bos_token_id = self.tokenizer.eos_token_id
        self._model.generation_config.pad_token_id = self.tokenizer.eos_token_id
        self.parse_config()

        logger.info(
            f"loaded model <{model_key}> | size: {get_model_size(self._model)} | dtype: {determine_dtype(self._model)} | device: {self.device}"
        )

    @property
    def device(self):
        return determine_device(self._model)

    @property
    def n_heads(self) -> int:
        """Number of *query* heads (post repeat_kv width for GQA models)."""
        return determine_n_heads(self)

    @property
    def head_dim(self) -> int:
        """Per-head width of the attention inner space (read from the config when it declares
        ``head_dim``; not always ``n_embd // n_heads``)."""
        return determine_head_dim(self)

    @property
    def attn_inner_size(self) -> int:
        """Width of the o_proj INPUT (== n_heads * head_dim). Equals n_embd only when the model
        does not decouple head_dim from hidden_size."""
        return self.n_heads * self.head_dim

    @property
    def n_kv_heads(self) -> int:
        """Number of key/value heads (GQA). Equals ``n_heads`` for MHA models."""
        return determine_n_kv_heads(self)

    def parse_config(self) -> None:
        if not is_supported_variant(self):
            logger.error(
                f"Unknown model type: {type(unwrap_model(self)).__name__}. Parsing may fail."
            )

        fields = {
            "n_layer": len(determine_layers(self)),
            "n_embd": determine_hidden_size(self),
            "embedder_name": determine_embedding_layer_path(self),
            "final_layer_norm_name": determine_final_layer_norm_path(self),
            "lm_head_name": determine_lm_head_path(self),
            "layer_name_format": determine_layer_name_format(self),
            "mlp_module_name_format": "model.layers.{}.mlp",
            "attn_module_name_format": "model.layers.{}.self_attn",
        }
        fields["layer_names"] = [
            fields["layer_name_format"].format(i) for i in range(fields["n_layer"])
        ]
        for key, value in fields.items():
            setattr(self, key, value)

    def __call__(self, *args, **kwargs) -> Any:
        """Call the model."""
        return self._model(*args, **kwargs)


def get_model_size(
    model: torch.nn.Module, unit: Literal["B", "KB", "MB", "GB"] = "MB"
) -> float:
    size_all = model.get_memory_footprint()
    return bytes_to_human_readable(size_all, unit)


def bytes_to_human_readable(
    size: int, unit: Literal["B", "KB", "MB", "GB"] = "MB"
) -> str:
    denom = {"B": 1, "KB": 2**10, "MB": 2**20, "GB": 2**30}[unit]
    return f"{size / denom:.3f} {unit}"


def get_full_model_path(model_name: str) -> str:
    full_path = os.path.join(DEFAULT_MODELS_DIR, model_name)
    if os.path.exists(full_path):
        return full_path
    else:
        logger.warning(
            f"""{model_name} not found in {DEFAULT_MODELS_DIR}
If not found in cache, model will be downloaded from HuggingFace to cache directory: {HF_CACHE_DIR}"""
        )
        return model_name


def unwrap_model(
    net: ModelandTokenizer | LanguageModel | torch.nn.Module,
) -> torch.nn.Module:
    if isinstance(net, LanguageModel):
        return net._model
    if isinstance(net, torch.nn.Module):
        return net
    raise ValueError("mt must be a ModelandTokenizer or a torch.nn.Module")


def unwrap_tokenizer(mt: ModelandTokenizer | AutoTokenizer) -> AutoTokenizer:
    if isinstance(mt, ModelandTokenizer):
        return mt.tokenizer
    return mt


def _name_contains(mt, *substrings: str) -> bool:
    """Fallback family check on ``config._name_or_path``."""
    name = getattr(getattr(mt, "config", None), "_name_or_path", None)
    return name is not None and any(s in name.lower() for s in substrings)


def is_llama_variant(mt: Model | ModelandTokenizer) -> bool:
    """Determine if model is a Llama variant (Llama-3.3-70B-Instruct)."""
    if isinstance(mt, ModelandTokenizer) or isinstance(mt, LanguageModel):
        mt = unwrap_model(mt)
    return isinstance(mt, transformers.LlamaForCausalLM) or _name_contains(mt, "llama")


def is_qwen_variant(mt: Model | ModelandTokenizer) -> bool:
    """Determine if model is a Qwen2 / Qwen3 variant (Qwen2.5, Qwen2.5-Coder, Qwen3-32B)."""
    if isinstance(mt, ModelandTokenizer) or isinstance(mt, LanguageModel):
        mt = unwrap_model(mt)
    return (isinstance(mt, (transformers.Qwen2ForCausalLM, transformers.Qwen3ForCausalLM))
            or _name_contains(mt, "qwen"))


def is_olmo_variant(mt: Model | ModelandTokenizer) -> bool:
    """Determine if model is an OLMo-3 variant (Olmo-3.1-32B Instruct / SFT)."""
    if isinstance(mt, ModelandTokenizer) or isinstance(mt, LanguageModel):
        mt = unwrap_model(mt)
    return isinstance(mt, transformers.Olmo3ForCausalLM) or _name_contains(mt, "olmo")


def is_glm_variant(mt: Model | ModelandTokenizer) -> bool:
    """Determine if model is a dense GLM-4 variant (GLM-4-32B-0414)."""
    if isinstance(mt, ModelandTokenizer) or isinstance(mt, LanguageModel):
        mt = unwrap_model(mt)
    return isinstance(mt, transformers.Glm4ForCausalLM) or _name_contains(mt, "glm")


def is_supported_variant(mt: Model | ModelandTokenizer) -> bool:
    """All supported families share the ``model.layers.{i}`` / ``model.embed_tokens`` layout."""
    return (
        is_llama_variant(mt)
        or is_qwen_variant(mt)
        or is_olmo_variant(mt)
        or is_glm_variant(mt)
    )


# Model-name signatures of chat / instruct releases: only a fallback for when no tokenizer is reachable.
# Qwen3 (e.g. Qwen/Qwen3-32B) is instruct-tuned without an "-Instruct" suffix, hence the explicit entry.
INSTRUCT_NAME_SIGNATURES = ("instruct", "chat", "zai-org/glm", "qwen3")


def supports_chat_template(mt: Model | ModelandTokenizer | Tokenizer) -> bool:
    """Does this model ship a chat template (i.e. is it instruct / chat tuned)?

    Asks the tokenizer (base models ship no ``chat_template``); the name signatures above are only
    a fallback when no tokenizer is reachable."""
    tok = getattr(mt, "tokenizer", None)
    if tok is None and hasattr(mt, "apply_chat_template"):
        tok = mt  # a bare tokenizer
    if tok is not None:
        try:
            if getattr(tok, "chat_template", None):
                return True
            get_ct = getattr(tok, "get_chat_template", None)
            return bool(get_ct()) if callable(get_ct) else False
        except Exception as e:  # noqa: BLE001  (fall through to the name heuristic)
            logger.debug(f"chat_template lookup failed ({e}); falling back to the name signatures")
    name = getattr(mt, "name", None) or getattr(getattr(mt, "config", None), "_name_or_path", "")
    return any(sig in str(name).lower() for sig in INSTRUCT_NAME_SIGNATURES)


def any_parameter(model: ModelandTokenizer | Model) -> torch.nn.Parameter | None:
    """Get any example parameter for the model."""
    model = unwrap_model(model)
    return next(iter(model.parameters()), None)


def _check_supported(model) -> None:
    if not is_supported_variant(model):
        raise ValueError(f"unknown model type: {type(model).__name__}")


def determine_embedding_layer_path(model: ModelandTokenizer | Model) -> str:
    model = unwrap_model(model)
    _check_supported(model)
    return "model.embed_tokens"


def determine_final_layer_norm_path(model: ModelandTokenizer | Model) -> str:
    model = unwrap_model(model)
    _check_supported(model)
    return "model.norm"


def determine_lm_head_path(model: ModelandTokenizer | Model) -> str:
    model = unwrap_model(model)
    _check_supported(model)
    return "lm_head"


def determine_layers(model: ModelandTokenizer | Model) -> tuple[int, ...]:
    """Return all hidden layer indices for the given model."""
    model = unwrap_model(model)
    assert isinstance(model, Model)
    return (*range(model.config.num_hidden_layers),)


def determine_layer_name_format(
    model: ModelandTokenizer | Model,
) -> str | None:
    """Determine the format of layer names."""
    model = unwrap_model(model)
    if is_supported_variant(model):
        return "model.layers.{}"


def determine_hidden_size(model: ModelandTokenizer | Model) -> int:
    """Determine hidden rep size for the model."""
    model = unwrap_model(model)
    try:
        return model.config.hidden_size
    except AttributeError:
        embed = baukit.get_module(model, determine_embedding_layer_path(model))
        return embed.weight.shape[-1]


def determine_n_heads(model: ModelandTokenizer | Model) -> int:
    """Number of *query* attention heads."""
    model = unwrap_model(model)
    return model.config.num_attention_heads


def determine_n_kv_heads(model: ModelandTokenizer | Model) -> int:
    """Number of key/value heads (GQA). Falls back to n_heads (MHA) when unspecified."""
    model = unwrap_model(model)
    return getattr(model.config, "num_key_value_heads", None) or determine_n_heads(model)


def determine_head_dim(model: ModelandTokenizer | Model) -> int:
    """Per-head width of the attention inner space.

    Prefers the config's explicit ``head_dim``, which some architectures decouple from
    ``hidden_size``. Falls back to ``hidden_size // n_heads`` for models that don't declare it
    (Qwen2, Llama, GLM-4 dense, ... — where the two agree anyway)."""
    model = unwrap_model(model)
    hd = getattr(model.config, "head_dim", None)
    if hd:
        return int(hd)
    return determine_hidden_size(model) // determine_n_heads(model)


def determine_device(model: ModelandTokenizer | Model) -> torch.device | None:
    """Determine device model is running on."""
    parameter = any_parameter(model)
    return parameter.device if parameter is not None else None


def determine_dtype(model: ModelandTokenizer | Model) -> torch.dtype | None:
    """Determine dtype of model."""
    parameter = any_parameter(model)
    return parameter.dtype if parameter is not None else None
