"""Some useful type aliases relevant to this project."""

import pathlib
from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import numpy
import torch
import transformers
import transformers.modeling_outputs
from dataclasses_json import DataClassJsonMixin
from nnsight import LanguageModel

ArrayLike = list | tuple | numpy.ndarray | torch.Tensor
PathLike = str | pathlib.Path
Device = str | torch.device

# Throughout this codebase, we use HuggingFace model implementations.
Model = (
    LanguageModel
    | transformers.GPT2LMHeadModel
    | transformers.GPTJForCausalLM
    | transformers.GPTNeoXForCausalLM
    | transformers.LlamaForCausalLM
    | transformers.Gemma3ForConditionalGeneration
    | transformers.Gemma2ForCausalLM
    | transformers.GemmaForCausalLM
    | transformers.Qwen2ForCausalLM
    | transformers.Olmo2ForCausalLM
    | transformers.OlmoForCausalLM
    | transformers.Qwen3ForCausalLM
    | transformers.Qwen3MoeForCausalLM
    | transformers.GptOssForCausalLM
    | transformers.Olmo3ForCausalLM
    | transformers.DeepseekV2ForCausalLM
    | transformers.Starcoder2ForCausalLM
    | transformers.Glm4MoeLiteForCausalLM
    | transformers.Glm4ForCausalLM
    | transformers.Qwen3NextForCausalLM
    | transformers.CwmForCausalLM
    | transformers.MistralForCausalLM
    | transformers.Qwen3_5ForCausalLM
    | transformers.Qwen3VLForConditionalGeneration  # VLM (Qwen/Qwen3-VL-*-Instruct)
)
Tokenizer = transformers.PreTrainedTokenizerFast
TokenizerOffsetMapping = Sequence[tuple[int, int]]
TokenizerOutput = transformers.tokenization_utils_base.BatchEncoding

ModelInput = transformers.BatchEncoding
ModelOutput = transformers.modeling_outputs.CausalLMOutput
ModelGenerateOutput = transformers.generation.utils.GenerateOutput | torch.LongTensor

Layer = int | Literal["emb"] | Literal["ln_f"]

# All strings are also Sequence[str], so we have to distinguish that we
# mean lists or tuples of strings, or sets of strings, not other strings.
StrSequence = list[str] | tuple[str, ...]


@dataclass(frozen=False)
class PredictedToken(DataClassJsonMixin):
    """A predicted token and its probability."""

    token: str
    prob: Optional[float] = None
    logit: Optional[float] = None
    token_id: Optional[int] = None
    metadata: Optional[dict] = None

    def __str__(self) -> str:
        rep = f"{self.token!r} [{self.token_id}]"
        # if self.prob is not None:
        #     rep += f" (p={self.prob:.3f})"
        # if self.logit is not None:
        #     rep += f" (logit={self.logit:.3f})"
        add = []
        if self.prob is not None:
            add.append(f"p={self.prob:.3f}")
        if self.logit is not None:
            add.append(f"logit={self.logit:.3f}")

        if self.metadata is not None:
            for k, v in self.metadata.items():
                if k not in ["token", "token_id"] and v is not None:
                    add.append(f"{k}={v}")

        if len(add) > 0:
            rep += " (" + ", ".join(add) + ")"
        return rep


# NOTE: the SVD dataclass moved to src.analysis.algo
