"""Subspace Probe v2 — causal-transfer objective via KL self-distillation.

Design: agent_reports/claude/design/subspace_probe_v2_causal_design.md.

Differences from v1 (``src/goals/subspace_probe.py``):
  * the ONLY trainable parameter is ``W`` (one subspace at a single read layer); **no soft prompt**.
  * read-out ``o = Wᵀ W h`` (the in-subspace component of the source frontier residual) is used as an
    ``additive_norm`` steering vector injected into a C++ causal stub — byte-for-byte the eval causality
    path (``head_additive_generate`` / OVReader).
  * objective = ``KL( teacher ‖ student )`` over the FULL vocab, where
      - teacher = the model told the goal explicitly (no steering), and
      - student = the model steered with ``o`` (goal hidden), teacher-forced on the teacher trajectory.

The teacher distributions are recomputed on the fly each step (frozen model), so nothing but the token
trajectory ``y`` is cached — the KL is exact over the full vocabulary.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Optional

import baukit
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.goals.goalscope import GOALSCOPE_SYS
from src.models import ModelandTokenizer
from src.tokens import prepare_input

logger = logging.getLogger(__name__)

CAUSAL_PREFIX_TMPL = "bool {func}(auto {arg}){{"

# Destination-prompt augmentation pools (§6 of the design doc) — keep the "C++ body, no comments" intent.
FUNC_NAMES = ["check", "func", "f", "g", "predicate", "test", "cond"]
ARG_NAMES = ["value", "val", "x", "v", "n", "arg", "input"]
SYS_PARAPHRASES = [
    GOALSCOPE_SYS,
    "You are an expert C++ programmer.",
    "You write clean, correct, idiomatic C++.",
    "You are a coding assistant specialised in C++.",
]
USER_PARAPHRASES = [
    "You are implementing a simple C++ function that checks for a condition. Just write the "
    "function body, without any comments or explanations. Assume all the necessary headers are "
    "already included.",
    "Implement the body of this C++ predicate function. Output only code — no comments, no prose. "
    "All headers are already included.",
    "Complete this C++ function that returns a boolean. Write just the body, no comments or "
    "explanation; assume every needed header is present.",
    "Fill in the body of the following C++ boolean function. Code only, no comments; headers are "
    "already included.",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Config + module
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class SubspaceV2Config:
    d_model: int
    r: int = 256
    read_layer: int = 50  # h taken after this layer of the SOURCE (structured) prompt
    steer_layers: list[int] = field(default_factory=lambda: list(range(35, 46)))
    alpha: float = 0.1
    lambda_ortho: float = 0.1


class SubspaceProbeV2(nn.Module):
    """A single learnable subspace ``W ∈ R^{r×d}`` (orthonormal rows). Read-out ``o = Wᵀ W h``."""

    def __init__(self, config: SubspaceV2Config, init_W: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.config = config
        W = init_W if init_W is not None else _rand_orthonormal_rows(config.r, config.d_model)
        self.W = nn.Parameter(W.float().clone())

    def readout_vec(self, h: torch.Tensor) -> torch.Tensor:
        """``o = Wᵀ W h`` — projection of ``h`` onto span(W). Inputs/outputs: (..., d)."""
        return h @ self.W.T @ self.W

    def orthogonality_loss(self) -> torch.Tensor:
        I_r = torch.eye(self.config.r, device=self.W.device, dtype=self.W.dtype)
        return (self.W @ self.W.T - I_r).pow(2).sum()

    @torch.no_grad()
    def reorthogonalize(self) -> None:
        """Project W onto the Stiefel manifold (orthonormal rows) via QR — call after each opt step."""
        Q, _ = torch.linalg.qr(self.W.T)  # (d, r), orthonormal cols
        self.W.copy_(Q.T)


def _rand_orthonormal_rows(r: int, d: int) -> torch.Tensor:
    Q, _ = torch.linalg.qr(torch.randn(d, r))
    return Q.T


# ═══════════════════════════════════════════════════════════════════════════════
# Destination-prompt (causal C++ stub) construction + augmentation
# ═══════════════════════════════════════════════════════════════════════════════
def sample_stub(rng: random.Random) -> dict:
    """One augmented C++ stub: func name, arg name, system + user paraphrase, assistant prefill."""
    func = rng.choice(FUNC_NAMES)
    arg = rng.choice(ARG_NAMES)
    return dict(
        func=func,
        arg=arg,
        sys=rng.choice(SYS_PARAPHRASES),
        user=rng.choice(USER_PARAPHRASES),
        prefill=CAUSAL_PREFIX_TMPL.format(func=func, arg=arg),
    )


def build_causal_chat(mt: ModelandTokenizer, stub: dict, goal_desc: Optional[str] = None) -> str:
    """Chat string ending in the assistant prefill. ``goal_desc`` != None → TEACHER prompt (goal
    stated explicitly); None → STUDENT prompt (goal hidden, supplied only via steering)."""
    user = stub["user"]
    if goal_desc is not None:
        user = user + f" Specifically, the function should check {goal_desc}."
    msgs = [
        {"role": "system", "content": stub["sys"]},
        {"role": "user", "content": user},
        {"role": "assistant", "content": stub["prefill"]},
    ]
    kw = dict(tokenize=False, add_generation_prompt=False, continue_final_message=True)
    try:
        return mt.tokenizer.apply_chat_template(
            msgs, chat_template_kwargs={"enable_thinking": False}, **kw
        )
    except TypeError:
        return mt.tokenizer.apply_chat_template(msgs, **kw)


def _ids(mt: ModelandTokenizer, text: str) -> torch.Tensor:
    return prepare_input(prompts=text, tokenizer=mt)["input_ids"][0]


# ═══════════════════════════════════════════════════════════════════════════════
# Teacher trajectory generation (offline; cache tokens only)
# ═══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def generate_teacher_trajectory(
    mt: ModelandTokenizer,
    teacher_prompt: str,
    max_new_tokens: int = 32,
    stop_str: str = "</code>",
) -> list[int]:
    """Greedy continuation of the goal-informed teacher prompt. Stops at ``max_new_tokens``, EOS,
    ``stop_str``, or when the C++ body's brace depth returns to 0 (the closing ``}``). Returns the
    generated token ids (the stop token is included when it is ``}`` / EOS)."""
    inp = prepare_input(prompts=teacher_prompt, tokenizer=mt)
    input_ids = inp["input_ids"].to(mt.device)
    attn = inp["attention_mask"].to(mt.device)
    eos = mt.tokenizer.eos_token_id
    gen: list[int] = []
    depth = 1  # the prefill ends with the opening '{'
    for _ in range(max_new_tokens):
        out = mt._model(input_ids=input_ids, attention_mask=attn)
        nxt = int(out.logits[0, -1].argmax())
        gen.append(nxt)
        if nxt == eos:
            break
        piece = mt.tokenizer.decode([nxt])
        depth += piece.count("{") - piece.count("}")
        text_so_far = mt.tokenizer.decode(gen)
        if depth <= 0 or stop_str in text_so_far:
            break
        input_ids = torch.cat([input_ids, torch.tensor([[nxt]], device=mt.device)], dim=1)
        attn = torch.cat([attn, torch.ones((1, 1), dtype=attn.dtype, device=mt.device)], dim=1)
    return gen


# ═══════════════════════════════════════════════════════════════════════════════
# Teacher / student forwards
# ═══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def teacher_logits(
    mt: ModelandTokenizer, teacher_prefix_ids: torch.Tensor, y_ids: torch.Tensor
) -> torch.Tensor:
    """Full-vocab logits of the goal-informed teacher at the T positions predicting ``y``.
    No steering, no grad. Returns (T, vocab) float32."""
    seq = torch.cat([teacher_prefix_ids.to(mt.device), y_ids.to(mt.device)]).unsqueeze(0)
    out = mt._model(input_ids=seq)
    P = teacher_prefix_ids.shape[0]
    T = y_ids.shape[0]
    return out.logits[0, P - 1 : P - 1 + T, :].float()


def steered_student_logits(
    mt: ModelandTokenizer,
    probe: SubspaceProbeV2,
    student_prefix_ids: torch.Tensor,
    y_ids: torch.Tensor,
    h_src: torch.Tensor,
) -> torch.Tensor:
    """Full-vocab logits of the STUDENT (goal hidden) teacher-forced on ``y`` while steered with the
    subspace read-out ``o = Wᵀ W h_src`` (additive_norm over the steer band). Differentiable in ``W``.
    Returns (T, vocab) float32.

    The injection mirrors ``head_additive_generate(strategy="additive_norm")`` at ``patch_input_positions
    =[-1]`` / ``n_steer_tokens=None``: every position from the last prefix token onward (the ones that
    predict ``y``) gets ``h += alpha * ||h|| * unit(o)`` at each steer layer."""
    cfg = probe.config
    o = probe.readout_vec(h_src.to(probe.W.device).float())  # (d,), grad via W
    unit_o = o / o.norm().clamp_min(1e-6)

    seq = torch.cat([student_prefix_ids.to(mt.device), y_ids.to(mt.device)]).unsqueeze(0)
    P = student_prefix_ids.shape[0]
    T = y_ids.shape[0]
    patch_start = P - 1  # first position whose residual predicts y_1

    handles = []
    for l in cfg.steer_layers:
        module = baukit.get_module(mt._model, mt.layer_name_format.format(l))

        def hook(mod, inp, output):
            hidden = output[0] if isinstance(output, tuple) else output
            rest = output[1:] if isinstance(output, tuple) else ()
            region = hidden[:, patch_start:, :]  # (1, n, d)
            norms = region.norm(dim=-1, keepdim=True)  # (1, n, 1)
            # device_map="auto" can place steer layers on a different GPU than probe.W — match hidden.
            add = cfg.alpha * norms * unit_o.to(device=hidden.device, dtype=hidden.dtype)  # (d,)
            new_hidden = torch.cat([hidden[:, :patch_start, :], region + add], dim=1)
            return (new_hidden,) + rest if rest else new_hidden

        handles.append(module.register_forward_hook(hook))
    try:
        out = mt._model(input_ids=seq)
    finally:
        for h in handles:
            h.remove()
    return out.logits[0, P - 1 : P - 1 + T, :].float()


# ═══════════════════════════════════════════════════════════════════════════════
# Batched forwards (train loop) — left-pad, corrected position ids, per-row steer mask
# ═══════════════════════════════════════════════════════════════════════════════
def _left_pad(mt: ModelandTokenizer, seqs: list[list[int]]):
    """Left-pad token-id lists to a common length. Returns (input_ids, attention_mask, position_ids,
    L) on the model device. position_ids are recomputed from the mask so RoPE ignores the pad."""
    pad = mt.tokenizer.pad_token_id
    if pad is None:
        pad = mt.tokenizer.eos_token_id
    B = len(seqs)
    L = max(len(s) for s in seqs)
    ids = torch.full((B, L), pad, dtype=torch.long)
    mask = torch.zeros((B, L), dtype=torch.long)
    for b, s in enumerate(seqs):
        ids[b, L - len(s):] = torch.as_tensor(s, dtype=torch.long)
        mask[b, L - len(s):] = 1
    pos = (mask.cumsum(-1) - 1).clamp_min(0)
    dev = mt.device
    return ids.to(dev), mask.to(dev), pos.to(dev), L


def _y_predict_slices(logits: torch.Tensor, starts: list[int], T_list: list[int]) -> torch.Tensor:
    """Concatenate, over rows, the logits at the positions predicting ``y`` → (ΣT, V)."""
    return torch.cat([logits[b, starts[b]:starts[b] + T_list[b], :] for b in range(len(starts))], dim=0)


@torch.no_grad()
def batched_teacher_logits(
    mt: ModelandTokenizer, prefix_list: list[torch.Tensor], y_list: list[torch.Tensor]
) -> torch.Tensor:
    """Full-vocab teacher logits at the y-predicting positions for a batch of rows, concatenated to
    (ΣT, V). No steering, no grad. Row order must match the student call for the KL to align."""
    seqs = [list(p.tolist()) + list(y.tolist()) for p, y in zip(prefix_list, y_list)]
    ids, mask, pos, L = _left_pad(mt, seqs)
    out = mt._model(input_ids=ids, attention_mask=mask, position_ids=pos)
    T_list = [int(y.shape[0]) for y in y_list]
    starts = [L - T - 1 for T in T_list]  # left-padded: position predicting y_1
    return _y_predict_slices(out.logits.float(), starts, T_list)


def batched_steered_student_logits(
    mt: ModelandTokenizer,
    probe: SubspaceProbeV2,
    prefix_list: list[torch.Tensor],
    y_list: list[torch.Tensor],
    h_list: list[torch.Tensor],
) -> torch.Tensor:
    """Full-vocab student logits at the y-predicting positions for a batch of rows, concatenated to
    (ΣT, V). Each row is steered with its own ``o_b = Wᵀ W h_b`` (additive_norm over the steer band,
    injected from the last prefix token onward). Differentiable in ``W``."""
    cfg = probe.config
    H = torch.stack([h.to(probe.W.device).float() for h in h_list])  # (B, d)
    O = probe.readout_vec(H)  # (B, d), grad via W
    unit_O = O / O.norm(dim=-1, keepdim=True).clamp_min(1e-6)  # (B, d)

    seqs = [list(p.tolist()) + list(y.tolist()) for p, y in zip(prefix_list, y_list)]
    ids, mask, pos, L = _left_pad(mt, seqs)
    B = len(seqs)
    T_list = [int(y.shape[0]) for y in y_list]
    starts = [L - T - 1 for T in T_list]  # patch/inject start per row (last prefix token)
    inj = torch.zeros((B, L), dtype=torch.bool)
    for b in range(B):
        inj[b, starts[b]:] = True  # inject at the last prefix token + every y position

    handles = []
    for l in cfg.steer_layers:
        module = baukit.get_module(mt._model, mt.layer_name_format.format(l))

        def hook(mod, inp, output):
            hidden = output[0] if isinstance(output, tuple) else output
            rest = output[1:] if isinstance(output, tuple) else ()
            dev, dt = hidden.device, hidden.dtype
            norms = hidden.norm(dim=-1, keepdim=True)  # (B, L, 1)
            add = cfg.alpha * norms * unit_O.to(device=dev, dtype=dt)[:, None, :]  # (B, L, d)
            m = inj.to(dev)[:, :, None]  # (B, L, 1)
            return (hidden + m * add,) + rest if rest else hidden + m * add

        handles.append(module.register_forward_hook(hook))
    try:
        out = mt._model(input_ids=ids, attention_mask=mask, position_ids=pos)
    finally:
        for h in handles:
            h.remove()
    return _y_predict_slices(out.logits.float(), starts, T_list)


def distill_kl(teacher_logits_t: torch.Tensor, student_logits_t: torch.Tensor) -> torch.Tensor:
    """Forward KL( teacher ‖ student ) averaged over positions (T). Full vocab, T=1 (no softening)."""
    log_p = F.log_softmax(teacher_logits_t, dim=-1)  # teacher (target)
    log_q = F.log_softmax(student_logits_t, dim=-1)  # student (model output)
    # F.kl_div(input=log_q, target=log_p, log_target) = Σ p·(log p − log q) = KL(p‖q); batchmean → /T
    return F.kl_div(log_q, log_p, log_target=True, reduction="batchmean")
