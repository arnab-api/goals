"""Some useful type aliases relevant to this project."""

import pathlib

import numpy
import torch
from nnterp import StandardizedTransformer  # before any transformers.models.* import
import transformers

ArrayLike = list | tuple | numpy.ndarray | torch.Tensor
PathLike = str | pathlib.Path

# Throughout this codebase, we use HuggingFace model implementations (the reported model families).
Model = (
    StandardizedTransformer
    | transformers.LlamaForCausalLM
    | transformers.Qwen2ForCausalLM
    | transformers.Olmo3ForCausalLM
    | transformers.Glm4ForCausalLM
)
Tokenizer = transformers.PreTrainedTokenizerFast
TokenizerOutput = transformers.tokenization_utils_base.BatchEncoding

# All strings are also Sequence[str], so we have to distinguish that we
# mean lists or tuples of strings, or sets of strings, not other strings.
StrSequence = list[str] | tuple[str, ...]
