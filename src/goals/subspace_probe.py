"""Subspace probe for intent/goal extraction from LM residual streams.

Learns jointly:
  W_l ∈ R^{r × d}  — per-layer projection to intent subspace
  e_1..e_k ∈ R^d   — soft prompt token embeddings for destination context

Patching formula (equivalent to projection-replace when rows are orthonormal):
  r_patched = r_dest + W_l^T W_l (h_src - r_dest)

The same patch is applied at ALL positions from the last prefix token onward
(mirrors on_pass="all" in the current GoalScope patchscope).

Training objective:
  L = L_CE  +  λ_ortho * ||W_l W_l^T - I_r||_F^2  +  weight_decay
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import baukit
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models import ModelandTokenizer
from src.tokens import prepare_input

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config + cache dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SubspaceProbeConfig:
    d_model: int
    r: int = 256
    layers: list[int] = field(default_factory=list)
    n_soft_tokens: int = 4
    lambda_ortho: float = 0.1

    @property
    def n_layers(self) -> int:
        return len(self.layers)


@dataclass
class SubspaceCache:
    """Pre-computed source residuals for one training instance."""

    task: str
    block_kind: str
    label: str  # ground-truth label text (target for CE loss)
    src_residuals: dict[int, torch.Tensor]  # layer → (d,) float32 CPU
    token_pos: int  # token index in the source sequence (for reference)
    is_preceding: bool  # True = preceding token; False = first token of block


# ---------------------------------------------------------------------------
# SubspaceProbe module
# ---------------------------------------------------------------------------


class SubspaceProbe(nn.Module):
    """Learnable subspace projection matrices + soft prompt token embeddings.

    Parameters
    ----------
    config : SubspaceProbeConfig
    init_soft_tokens : optional (n_soft_tokens, d_model) tensor to initialise
        the soft prompt embeddings (e.g. from real token embeddings).
    """

    def __init__(
        self,
        config: SubspaceProbeConfig,
        init_soft_tokens: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.config = config

        # W : (n_layers, r, d) — initialised with orthonormal rows
        W_init = torch.stack(
            [_rand_orthonormal_rows(config.r, config.d_model) for _ in range(config.n_layers)]
        )
        self.W = nn.Parameter(W_init)

        # Soft prompt embeddings : (n_soft_tokens, d_model)
        if init_soft_tokens is not None:
            assert init_soft_tokens.shape == (config.n_soft_tokens, config.d_model)
            self.soft_tokens = nn.Parameter(init_soft_tokens.float().clone())
        else:
            self.soft_tokens = nn.Parameter(
                torch.randn(config.n_soft_tokens, config.d_model) * 0.02
            )

    # ── Core operations ──────────────────────────────────────────────────────

    def patch_delta(
        self,
        h_src: torch.Tensor,
        h_dest: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """W_l^T W_l (h_src - h_dest).  Inputs/outputs: (..., d)."""
        W = self.W[layer_idx]  # (r, d)
        return (h_src - h_dest) @ W.T @ W  # (..., d)

    def patched_residual(
        self,
        h_src: torch.Tensor,
        h_dest: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """r_dest + W_l^T W_l (h_src - r_dest).  Inputs/outputs: (..., d)."""
        return h_dest + self.patch_delta(h_src, h_dest, layer_idx)

    def orthogonality_loss(self) -> torch.Tensor:
        """sum_l ||W_l W_l^T - I_r||_F^2  (correct form for orthonormal rows)."""
        I_r = torch.eye(self.config.r, device=self.W.device, dtype=self.W.dtype)
        loss = self.W.new_zeros(())
        for i in range(self.config.n_layers):
            W = self.W[i]  # (r, d)
            loss = loss + (W @ W.T - I_r).pow(2).sum()
        return loss

    @torch.no_grad()
    def reorthogonalize(self) -> None:
        """Project each W_l onto the Stiefel manifold via QR (hard constraint).

        After this call every W_l has orthonormal rows: W_l W_l^T = I_r.
        Call once after each optimizer.step(), analogous to HeadMask.clamp_().
        """
        for i in range(self.config.n_layers):
            Q, _ = torch.linalg.qr(self.W[i].T)  # (d, r) Q has orthonormal cols
            self.W[i].copy_(Q.T)  # (r, d) orthonormal rows


# ---------------------------------------------------------------------------
# Destination template helpers
# ---------------------------------------------------------------------------

_DEST_SYSTEM = "You are a helpful assistant proficient in programming."
_SOFT_SENTINEL = "SOFTTOKEN_PLACEHOLDER_XYZ"


def build_dest_template_parts(
    mt: ModelandTokenizer,
    system_content: str = _DEST_SYSTEM,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the destination chat template at the soft-token insertion point.

    Returns
    -------
    before_soft_ids : (n_before,) int64 — token IDs up to (but not including)
        the soft token positions.
    after_soft_ids  : (n_after,)  int64 — token IDs from after the soft tokens
        to the end of the prefix (the open assistant turn).
    """
    tok = mt.tokenizer
    msgs = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": _SOFT_SENTINEL},
        {"role": "assistant", "content": ""},
    ]
    template_str = tok.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=True,
    )
    before_str, after_str = template_str.split(_SOFT_SENTINEL)
    before_ids = torch.tensor(tok.encode(before_str, add_special_tokens=False), dtype=torch.long)
    after_ids = torch.tensor(tok.encode(after_str, add_special_tokens=False), dtype=torch.long)
    return before_ids, after_ids


# ---------------------------------------------------------------------------
# Source residual caching
# ---------------------------------------------------------------------------


@torch.inference_mode()
def cache_source_residuals(
    mt: ModelandTokenizer,
    source_text: str,
    token_positions: list[int],
    layers: list[int],
) -> dict[int, dict[int, torch.Tensor]]:
    """Extract layer residuals at specified token positions in one forward pass.

    Returns
    -------
    {token_pos: {layer: (d_model,) float32 CPU tensor}}
    """
    inputs = prepare_input(source_text, return_offsets_mapping=False, tokenizer=mt)
    layer_names = [mt.layer_name_format.format(l) for l in layers]

    with baukit.TraceDict(mt._model, layer_names) as td:
        mt._model(
            input_ids=inputs["input_ids"].to(mt.device),
            attention_mask=inputs["attention_mask"].to(mt.device),
        )

    results: dict[int, dict[int, torch.Tensor]] = {pos: {} for pos in token_positions}
    for l, lname in zip(layers, layer_names):
        hidden = td[lname].output
        if isinstance(hidden, tuple):
            hidden = hidden[0]  # (1, seq_len, d)
        hidden = hidden.float().cpu()
        for pos in token_positions:
            results[pos][l] = hidden[0, pos, :]

    return results


# ---------------------------------------------------------------------------
# Pivot position finding
# ---------------------------------------------------------------------------


def find_block_pivot_positions(
    block_tree,  # IntentBlock root
    source_prompt: str,
    code_str: str,
    offset_mapping: torch.Tensor,  # (seq_len, 2) from prepare_input
    skip_kinds: Optional[set[str]] = None,
) -> list[tuple[object, int, int]]:
    """For each labeled block in the tree return (block, first_tok, preceding_tok).

    Parameters
    ----------
    block_tree    : IntentBlock root (from load_labeled_tree).
    source_prompt : The full source prompt string (prompt + generation).
    code_str      : The code string used to build the block tree.
    offset_mapping: (seq_len, 2) tensor from tokenizer with offset_mapping=True.
    skip_kinds    : Block kinds to skip (default: {"module"}).

    Returns
    -------
    List of (block, first_token_idx, preceding_token_idx).
    Only blocks with a non-None label and first_token_idx > 0 are included.
    """
    if skip_kinds is None:
        skip_kinds = {"module"}

    # Locate where the code starts in the source prompt (char level)
    code_char_offset = source_prompt.rfind(code_str)
    if code_char_offset < 0:
        # fallback: find via <code> tag
        tag = "<code>"
        idx = source_prompt.rfind(tag)
        code_char_offset = idx + len(tag) if idx >= 0 else 0
        logger.warning(
            "code_str not found in source_prompt; falling back to <code> tag offset=%d",
            code_char_offset,
        )

    offsets = offset_mapping.tolist()  # list of [start, end]
    results = []

    def _char_to_token(char_pos: int) -> Optional[int]:
        for i, (start, end) in enumerate(offsets):
            if start == end:
                continue  # special token
            if start <= char_pos < end:
                return i
            if start > char_pos:
                return i
        return None

    def visit(block) -> None:
        if block.label is not None and block.kind not in skip_kinds:
            abs_char = code_char_offset + block.char_start
            tok_idx = _char_to_token(abs_char)
            if tok_idx is not None and tok_idx > 0:
                results.append((block, tok_idx, tok_idx - 1))
        for child in block.children:
            visit(child)

    visit(block_tree)
    return results


# ---------------------------------------------------------------------------
# Differentiable patched forward pass (for training)
# ---------------------------------------------------------------------------


def patched_lm_forward(
    mt: ModelandTokenizer,
    probe: SubspaceProbe,
    before_soft_ids: torch.Tensor,  # (n_before,) int64 CPU
    after_soft_ids: torch.Tensor,  # (n_after,)  int64 CPU
    label_ids: torch.Tensor,  # (n_label,)  int64 CPU
    src_residuals: dict[int, torch.Tensor],  # layer → (d,) float32 CPU
) -> torch.Tensor:
    """One teacher-forced forward pass with subspace patching.

    The destination prefix is:
        [before_soft_tokens | soft_token_embeddings | after_soft_tokens]
    followed by label_ids for teacher forcing.

    The patch  r_patched = r_dest + W_l^T W_l (h_src - r_dest)  is applied at
    every position from the last prefix token onward (mirrors on_pass="all").

    Returns
    -------
    logits : (n_label, vocab_size) float32 — differentiable w.r.t. probe params.
    """
    dev = next(probe.parameters()).device
    dtype = next(mt._model.parameters()).dtype

    # ── Build prefix embeddings (inject soft tokens) ─────────────────────────
    embed_fn = baukit.get_module(mt._model, mt.embedder_name)

    with torch.no_grad():
        before_emb = embed_fn(before_soft_ids.to(dev))  # (n_before, d)
        after_emb = embed_fn(after_soft_ids.to(dev))  # (n_after,  d)
        label_emb = embed_fn(label_ids.to(dev))  # (n_label,  d)

    soft_emb = probe.soft_tokens.to(dtype)  # (n_soft, d) — has grad
    before_emb = before_emb.to(dtype)
    after_emb = after_emb.to(dtype)
    label_emb = label_emb.to(dtype)

    prefix_emb = torch.cat([before_emb, soft_emb, after_emb], dim=0)  # (prefix_len, d)
    all_emb = torch.cat([prefix_emb, label_emb], dim=0).unsqueeze(0)  # (1, total, d)

    prefix_len = prefix_emb.shape[0]
    total_len = all_emb.shape[1]
    patch_start = prefix_len - 1  # last prefix token; patch from here onward

    # ── Register patching hooks (one per layer) ───────────────────────────────
    layer_to_idx = {l: i for i, l in enumerate(probe.config.layers)}
    handles = []

    for l in probe.config.layers:
        li = layer_to_idx[l]
        lname = mt.layer_name_format.format(l)
        module = baukit.get_module(mt._model, lname)
        src_h = src_residuals[l].to(dev).to(dtype)  # (d,) — no grad

        def _make_hook(li_: int, src_h_: torch.Tensor) -> object:
            def hook(mod, inp, output):
                hidden = output[0] if isinstance(output, tuple) else output
                rest = output[1:] if isinstance(output, tuple) else ()

                region = hidden[:, patch_start:, :]  # (1, n_patch, d)
                src_exp = src_h_.view(1, 1, -1).expand_as(region)  # broadcast
                patched_region = region + (src_exp - region) @ probe.W[li_].T @ probe.W[li_]

                new_hidden = torch.cat(
                    [hidden[:, :patch_start, :], patched_region], dim=1
                )
                return (new_hidden,) + rest if rest else new_hidden

            return hook

        handles.append(module.register_forward_hook(_make_hook(li, src_h)))

    try:
        out = mt._model(inputs_embeds=all_emb)
    finally:
        for h in handles:
            h.remove()

    # Logits for the label positions: position patch_start predicts label[0], etc.
    logits = out.logits[0, patch_start : patch_start + label_ids.shape[0], :].float()
    return logits  # (n_label, vocab)


def compute_loss(
    logits: torch.Tensor,  # (n_label, vocab)
    label_ids: torch.Tensor,  # (n_label,)
    probe: SubspaceProbe,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CE loss + orthogonality loss.  Returns (total, ce, ortho)."""
    ce = F.cross_entropy(logits, label_ids.to(logits.device))
    ortho = probe.orthogonality_loss()
    total = ce + probe.config.lambda_ortho * ortho
    return total, ce, ortho


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _rand_orthonormal_rows(r: int, d: int) -> torch.Tensor:
    """(r, d) tensor with orthonormal rows."""
    Q, _ = torch.linalg.qr(torch.randn(d, r))  # Q: (d, r) orthonormal cols
    return Q.T  # (r, d) orthonormal rows
