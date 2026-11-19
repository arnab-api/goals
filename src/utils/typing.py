"""Some useful type aliases relevant to this project."""

import pathlib

import numpy
import torch
import transformers
from nnsight import LanguageModel

ArrayLike = list | tuple | numpy.ndarray | torch.Tensor
PathLike = str | pathlib.Path

# Throughout this codebase, we use HuggingFace model implementations (the reported model families).
Model = (
    LanguageModel
    | transformers.LlamaForCausalLM
    | transformers.Qwen2ForCausalLM
    | transformers.Qwen3ForCausalLM
    | transformers.Olmo3ForCausalLM
    | transformers.Glm4ForCausalLM
)
# Qwen3.5 hybrid (e.g. Qwen/Qwen3.8-27B, loaded as the VLM wrapper); only in recent transformers releases
for _cls in ("Qwen3_5ForCausalLM", "Qwen3_5ForConditionalGeneration"):
    if hasattr(transformers, _cls):
        Model = Model | getattr(transformers, _cls)
Tokenizer = transformers.PreTrainedTokenizerFast
TokenizerOutput = transformers.tokenization_utils_base.BatchEncoding

# All strings are also Sequence[str], so we have to distinguish that we
# mean lists or tuples of strings, or sets of strings, not other strings.
StrSequence = list[str] | tuple[str, ...]
