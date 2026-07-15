#!/usr/bin/env python
"""Shared FV (Function Vector) localization helpers — data, prompts, mean activations.

Faithful port of the pieces of Todd et al. 2024 ("Function Vectors in Large Language
Models", ICLR 2024; ericwtodd/function_vectors) that both localization scripts need:

  * abstractive task loading + train/valid split  (`load_fv_task`)
  * the Q:/A: ICL prompt template with optional label shuffling for the *corrupted*
    (uninformative) prompt used in causal mediation  (`build_icl_prompt`)
  * the *contextualized first answer token* id — Todd et al. track the indirect effect
    on ONLY the first token of the target word  (`first_answer_token_id`)
  * per-head mean activations `ā_ℓj^t` = mean o_proj input at the last token over N clean
    ICL prompts  (`compute_mean_head_activations`), which equals the mean head OV output
    the paper transplants (o_proj is linear).

The paper's reference implementation edits the o_proj INPUT (reshaped to
(…, n_heads, head_dim), overwrites head_n's slice, re-projects). In this codebase that is
exactly what `get_o_proj_inputs_batch` captures and what `forward_with_src_replacement`
patches — so both localization scripts reuse those primitives.

Template / hyperparameter defaults mirror the FV repo:
  prefixes  = {"input": "Q:", "output": "A:", "instructions": ""}
  separators= {"input": "\\n", "output": "\\n\\n", "instructions": ""}
  n_shots=10, prepend_space=True.
"""
import json
import os
import random
from dataclasses import dataclass

import numpy as np
import torch

from src.goals.head_loc_attn import get_o_proj_inputs_batch
from src.tokens import prepare_input

# ── template defaults (verbatim from the FV repo) ─────────────────────────────
PREFIXES = {"input": "Q:", "output": "A:", "instructions": ""}
SEPARATORS = {"input": "\n", "output": "\n\n", "instructions": ""}

# Tracked copy of the FV dataset_files (from ericwtodd/function_vectors); we only use abstractive.
DEFAULT_ABSTRACTIVE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "FV_dataset", "abstractive",
)

# the 6 tasks the paper foregrounds; a sensible small default for quick runs
REPRESENTATIVE_TASKS = [
    "antonym", "capitalize", "country-capital",
    "english-french", "present-past", "singular-plural",
]


# ── data ──────────────────────────────────────────────────────────────────────
@dataclass
class FVTask:
    """One abstractive task split into train (draw demos) and valid (draw queries)."""

    name: str
    train: list[dict]  # [{"input": x, "output": y}, ...]
    valid: list[dict]


def load_fv_task(name: str, root_dir: str = DEFAULT_ABSTRACTIVE_DIR,
                 test_size: float = 0.3, seed: int = 42) -> FVTask:
    """Load one abstractive task JSON and split into train/valid.

    Mirrors the FV repo's `load_dataset`: a fixed-seed shuffle then a train/valid cut.
    Demos are drawn from `train`, queries from `valid` (no leakage)."""
    path = os.path.join(root_dir, f"{name}.json")
    with open(path) as f:
        pairs = json.load(f)
    pairs = [{"input": str(p["input"]), "output": str(p["output"])} for p in pairs]
    rng = random.Random(seed)
    rng.shuffle(pairs)
    n_valid = max(1, int(round(len(pairs) * test_size)))
    valid, train = pairs[:n_valid], pairs[n_valid:]
    return FVTask(name=name, train=train, valid=valid)


def list_abstractive_tasks(root_dir: str = DEFAULT_ABSTRACTIVE_DIR) -> list[str]:
    return sorted(f[:-5] for f in os.listdir(root_dir) if f.endswith(".json"))


# ── prompt construction ─────────────────────────────────────────────────────────
def build_icl_prompt(demos: list[dict], query_input: str,
                     shuffle_labels: bool = False, prepend_space: bool = True,
                     rng: random.Random | None = None) -> str:
    """Build one N-shot ICL prompt string in the Q:/A: template.

    `shuffle_labels=True` produces the *corrupted* (uninformative) prompt `p̃` used in
    causal mediation: demo outputs are permuted so no systematic x→y relation remains
    (the query input is unchanged and its true label is NOT part of the prompt).

    Ends with the query's "A:" separator so the next token is the model's answer.
    `prepend_space=True` puts a leading space on every input/output value (FV default)."""
    sp = " " if prepend_space else ""
    outs = [d["output"] for d in demos]
    if shuffle_labels:
        rng = rng or random
        outs = outs[:]
        rng.shuffle(outs)
    lines = []
    for d, y in zip(demos, outs):
        lines.append(f"{PREFIXES['input']}{sp}{d['input']}{SEPARATORS['input']}"
                     f"{PREFIXES['output']}{sp}{y}")
    body = SEPARATORS["output"].join(lines)
    # query: input line + separator + bare "A:" (model completes the answer)
    prompt = (body + SEPARATORS["output"]
              + f"{PREFIXES['input']}{sp}{query_input}{SEPARATORS['input']}"
              + f"{PREFIXES['output']}")
    return prompt


def first_answer_token_id(mt, prompt: str, answer: str, prepend_space: bool = True) -> int:
    """Contextualized id of the FIRST answer token (Todd et al. `get_answer_id`[:1]).

    Tokenize `prompt` vs `prompt + <answer>` and take the first differing token, so the
    id reflects how the answer actually tokenizes right after "A:" (e.g. leading-space
    merge). This is the single token whose probability the indirect effect is read on."""
    ans = (" " if prepend_space else "") + answer
    src = mt.tokenizer(prompt, add_special_tokens=False)["input_ids"]
    tgt = mt.tokenizer(prompt + ans, add_special_tokens=False)["input_ids"]
    assert len(tgt) > len(src), f"answer added no tokens: {answer!r}"
    return int(tgt[len(src)])


# ── mean head activations ā_ℓj^t ──────────────────────────────────────────────
@torch.inference_mode()
def compute_mean_head_activations(mt, task: FVTask, layers: list[int],
                                  n_shots: int = 10, n_trials: int = 100,
                                  batch_size: int = 16, seed: int = 42) -> torch.Tensor:
    """Mean o_proj input (per layer) at the last token over `n_trials` CLEAN ICL prompts.

    Returns `(len(layers), n_embd)` float32 CPU — stacked across `layers`. Each layer's
    row, reshaped to (n_heads, head_dim), gives the per-head mean OV output ā_ℓj^t that
    the intervention transplants. Clean = correct (un-shuffled) labels."""
    rng = np.random.RandomState(seed)
    prompts = []
    for _ in range(n_trials):
        demo_idx = rng.choice(len(task.train), n_shots, replace=False)
        q_idx = int(rng.choice(len(task.valid)))
        demos = [task.train[i] for i in demo_idx]
        prompts.append(build_icl_prompt(demos, task.valid[q_idx]["input"],
                                        shuffle_labels=False))
    acc = {ly: [] for ly in layers}
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        inputs = prepare_input(prompts=batch, tokenizer=mt, padding_side="left")
        o = get_o_proj_inputs_batch(mt, inputs, layers)  # {ly: (b, n_embd)} last token
        for ly in layers:
            acc[ly].append(o[ly])
    return torch.stack([torch.cat(acc[ly], 0).mean(0) for ly in layers])  # (L, n_embd)
