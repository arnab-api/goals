import logging
import os
from typing import Literal

import torch
from nnterp import StandardizedTransformer
from transformers import AutoTokenizer

from src.utils.env_utils import DEFAULT_MODELS_DIR, HF_CACHE_DIR

logger = logging.getLogger(__name__)

# The reported model families (nnterp family = the config's ``model_type``). Their native layout is
# ``model.layers.{i}`` / ``self_attn`` / ``o_proj``, which the two path-format strings below spell out.
SUPPORTED_FAMILIES = ("qwen2", "llama", "olmo3", "glm4")


class ModelandTokenizer(StandardizedTransformer):
    """nnterp ``StandardizedTransformer`` (nnsight 0.8) + tokenizer, with the repo's older config
    interface kept as properties over nnterp's sizes.

    Reported families: Qwen2 (Qwen2.5, Qwen2.5-Coder), Llama (Llama-3.3), OLMo-3 (Olmo-3.1) and
    dense GLM-4 (GLM-4-32B-0414). ``self.family`` is the nnterp family module.

    Code uses nnterp's standard names and values (``mt.layers[i].layer_output``,
    ``mt.layers[i].self_attn.attention_scores``, ``mt.num_layers``, ...). What is left of the
    pre-nnterp interface:

    - ``_model``: the HF module (``self._module``). Deprecated, and read only by the head-mask
      training path of ``src/goals/head_loc_attn.py`` (until its port lands), the pre-port code the
      A/B tests in ``test/`` load from git, and two demo notebooks (``02.1_unspecified``,
      ``08_beyond_code``).
    - ``layer_name_format``, ``attn_module_name_format``: native module paths
      (``model.layers.{i}``, ``model.layers.{i}.self_attn``), read by the same pre-port code and by
      ``PatchSpec`` locations that name a layer by its path. They are the layout of the four
      reported families only, and are wrong on a family whose native names differ.
    - ``n_layer``, ``n_embd``, ``n_heads``, ``n_kv_heads``, ``attn_inner_size``: aliases of
      ``num_layers``, ``hidden_size``, ``num_heads``, ``num_kv_heads`` and
      ``num_heads * head_dim``. ``head_dim`` is nnterp's own (the config's ``head_dim`` when it
      declares one, else ``hidden_size // num_heads``).

    ``device`` (the first parameter's) and ``mt(...)`` outside a trace (a plain HF forward of the
    module) are nnsight's own.
    """

    def __init__(
        self,
        model_key: str,
        device_map: dict | str = "auto",
        **kwargs,
    ) -> None:
        model_key = get_full_model_path(model_key)
        if HF_CACHE_DIR is not None:
            kwargs["cache_dir"] = HF_CACHE_DIR
        super().__init__(
            model_key,
            device_map=device_map,
            dispatch=True,
            **kwargs,
        )
        self._model_key = model_key

        self._module.eval()
        # Left-pad explicitly rather than relying on the default (a raw HF tokenizer's is "right"
        # for e.g. Qwen). Every batched intervention here indexes with negative positions
        # (PATCH_POS=-2, [:, -1]); those only align with the real, right-aligned tokens under
        # LEFT padding. Set once, at construction, so no downstream call has to remember.
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token  # for generation
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if getattr(self.tokenizer, "bos_token", None) is None:
            self.tokenizer.bos_token = self.tokenizer.eos_token
            self.tokenizer.bos_token_id = self.tokenizer.eos_token_id
        self._module.generation_config.pad_token_id = self.tokenizer.eos_token_id

        if self.family_name not in SUPPORTED_FAMILIES:
            logger.warning(
                f"nnterp family {self.family_name!r} is not one of the reported families "
                f"{SUPPORTED_FAMILIES}; the native path strings (layer_name_format, "
                f"attn_module_name_format) may not match this model."
            )
        logger.info(
            f"loaded model <{model_key}> | family: {self.family_name} | "
            f"size: {get_model_size(self._module)} | dtype: {self._module.dtype} | "
            f"device: {self.device}"
        )

    # -- identity and the transition shim ----------------------------------------

    @property
    def name(self) -> str:
        """The model key (HF repo id or local path) the model was loaded from."""
        return self._model_key

    @property
    def family_name(self) -> str:
        """The nnterp family, i.e. the config's ``model_type`` (``"qwen2"``, ``"llama"``, ...)."""
        return self.family.__name__.rsplit(".", 1)[-1]

    @property
    def _model(self) -> torch.nn.Module:
        """Deprecated transition shim: the HF module (``self._module``), for the pre-port code listed
        in the class docstring; everything else goes through nnterp's envoys."""
        return self._module

    # -- legacy sizes (aliases of nnterp's) --------------------------------------

    @property
    def n_layer(self) -> int:
        return self.num_layers

    @property
    def n_embd(self) -> int:
        return self.hidden_size

    @property
    def n_heads(self) -> int:
        """Number of *query* heads (post repeat_kv width for GQA models)."""
        return self.num_heads

    @property
    def n_kv_heads(self) -> int:
        """Number of key/value heads (GQA). Equals ``n_heads`` for MHA models."""
        return self.num_kv_heads

    @property
    def attn_inner_size(self) -> int:
        """Width of the o_proj INPUT (== n_heads * head_dim). Equals n_embd only when the model
        does not decouple head_dim from hidden_size."""
        return self.num_heads * self.head_dim

    # -- legacy native module paths (reported families only) --------------------

    layer_name_format = "model.layers.{}"
    attn_module_name_format = "model.layers.{}.self_attn"


def get_model_size(
    model: torch.nn.Module, unit: Literal["B", "KB", "MB", "GB"] = "MB"
) -> str:
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


def unwrap_tokenizer(mt: ModelandTokenizer | AutoTokenizer) -> AutoTokenizer:
    if isinstance(mt, ModelandTokenizer):
        return mt.tokenizer
    return mt
