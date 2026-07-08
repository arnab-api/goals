"""Order-ID gaze experiment (Track-2, clock/03.1).

Question: at a condition's ``=`` pivot (e.g. ``b =``), do attention heads in the *early*
layers (``< layer_boundary``) carry the decision of *which instruction to implement next*?

Objective — the raw **Q·K gaze affinity** from the pivot query to each instruction span,
measured on the locked late-layer "gaze" heads (selected by ``score_condition_heads`` on the
value-weighted look-back ratio).  We extract those heads' ``q_proj``/``k_proj`` and compute the
affinity *ourselves* from the residual latent that feeds them — no softmax, no value-weighting,
no ``output_attentions``.  Because the span key tokens precede the pivot and we patch only the
pivot position, the keys are constant (causality); the entire causal effect is on the query.

Intervention — transplant the source pivot's per-head ``o_proj`` output into the destination
pivot (same spec, different condition), and ask whether the gaze redirects from the destination's
condition (``orig``) toward the source's condition (``target``).

Used by ``scripts/clock/screen_oid_gaze.py`` (per-head screen) and
``scripts/clock/train_oid_gaze_mask.py`` (joint sparse mask).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

import numpy as np
import torch

from src.functional import get_module_nnsight
from src.goals.head_loc_attn import get_o_proj_inputs_batch
from src.goals.multistep_label import LabeledGeneration
from src.tokens import prepare_input
from src.utils import env_utils

# single-letter result variable per condition index (a, b, c, …) — the pivot is `<var> =`
PIVOT_VARS = {i: chr(ord("a") + i) for i in range(8)}


def cache_root(model: str, version: str) -> str:
    return os.path.join(
        env_utils.DEFAULT_RESULTS_DIR, "multistep_clock", version, model.split("/")[-1]
    )


# ──────────────────────────────────────────────────────────────────────────────
# Gaze-head selection (top-N look-back heads, restricted to the late layers).
# ──────────────────────────────────────────────────────────────────────────────


def load_gaze_heads(
    model: str,
    version: str,
    q_span: int,
    compare_with: str,
    top_n: int,
    layer_boundary: int,
    which: str = "combined",
) -> list[tuple[int, int]]:
    """Top-`top_n` look-back heads on the `which` score map, kept only for layers > layer_boundary."""
    from scripts.clock.score_condition_heads import load_head_scores  # lazy (script dep)

    hs = load_head_scores(model=model, version=version, q_span=q_span, compare_with=compare_with)
    score = hs[which]
    flat = np.argsort(score, axis=None)[::-1][:top_n]
    gaze_all = (np.unravel_index(i, score.shape) for i in flat)
    return [(int(l), int(h)) for l, h in gaze_all if int(l) > layer_boundary]


# ──────────────────────────────────────────────────────────────────────────────
# Same-spec oid pairs.
# ──────────────────────────────────────────────────────────────────────────────


def _load_labeled(croot: str, modularity: str, sample: str) -> LabeledGeneration:
    with open(os.path.join(croot, modularity, sample, "labels.json")) as f:
        return LabeledGeneration.from_dict(json.load(f))


def cut_after_assignment(generation: str, var: str) -> int:
    """Char offset just after the `<var> =` assignment in `generation`."""
    m = re.search(rf"(?m)^\s*{re.escape(var)}\s*=", generation)
    if m is None:
        raise ValueError(f"no '{var} =' assignment line found")
    return m.end()


def build_prefill(croot: str, modularity: str, sample: str, var: str) -> str:
    """Destination/source prefill: full prompt + generation truncated right after `<var> =`."""
    lab = _load_labeled(croot, modularity, sample)
    return lab.full_prompt + lab.generation[: cut_after_assignment(lab.generation, var)]


def _key_idx(step_ranges: list[list[int]], seq: int) -> np.ndarray:
    idx: set[int] = set()
    for r in step_ranges:
        if r:
            idx.update(t for t in range(int(r[0]), int(r[1])) if 0 <= t < seq)
    return np.array(sorted(idx), dtype=np.int64)


@dataclass
class OidPair:
    """One same-spec oid pair, cut at the orig/target pivots.

    ``dest_inputs`` is the B=1 tokenized destination (cut at ``orig`` pivot).
    ``src_hv`` / ``dest_hv`` are the source/destination pivot o_proj inputs for the
    ``< boundary`` layers, shape ``(n_layers_below, n_embd)``.
    """

    sample: str
    dest_inputs: dict
    src_hv: torch.Tensor
    dest_hv: torch.Tensor
    spans: list[np.ndarray]
    orig_idx: int
    target_idx: int
    step_texts: list[str]

    @property
    def span_orig(self) -> np.ndarray:
        return self.spans[self.orig_idx]

    @property
    def span_target(self) -> np.ndarray:
        return self.spans[self.target_idx]


def build_oid_pairs(
    mt,
    model: str,
    version: str,
    modularity: str,
    layers_below: list[int],
    n_pairs: int,
    orig_idx: int | None = None,
    target_idx: int | None = None,
    samples: list[str] | None = None,
) -> list[OidPair]:
    """Build up to `n_pairs` same-spec oid pairs.

    For each sample we form pairs over condition indices ``(orig, target)`` with ``orig < target``,
    both having a findable ``=`` pivot and a non-empty key span.  ``orig_idx`` / ``target_idx``, when
    given, **pin** that side; when ``None`` (default) that side **varies** over every valid index.
    The source is cut at ``target``, the destination at ``orig``; the swap-delta
    ``T(target_span) - T(orig_span)`` is read at the destination pivot.  The per-condition ``=``-pivot
    o_proj vector is identical whether the condition plays source or destination, so we extract them
    once per sample (one batched forward).
    """
    croot = cache_root(model, version)
    mdir = os.path.join(croot, modularity)
    if samples is None:
        samples = sorted(os.listdir(mdir))

    pairs: list[OidPair] = []
    for s in samples:
        kr_path = os.path.join(mdir, s, "key_ranges.json")
        if not (os.path.exists(kr_path) and os.path.exists(os.path.join(mdir, s, "labels.json"))):
            continue
        kr = json.load(open(kr_path))["step_key_token_ranges"]

        # condition indices with a non-empty key span and a findable `=` pivot
        valid: dict[int, str] = {}
        for k in range(len(kr)):
            if not (kr[k] and kr[k][0]):
                continue
            try:
                valid[k] = build_prefill(croot, modularity, s, PIVOT_VARS[k])
            except Exception:
                continue

        orig_cand = [orig_idx] if orig_idx is not None else sorted(valid)
        tgt_cand = [target_idx] if target_idx is not None else sorted(valid)
        combos = [
            (o, t)
            for o in orig_cand
            for t in tgt_cand
            if o is not None and t is not None and o in valid and t in valid and o < t
        ]
        if not combos:
            continue

        # one batched forward for the o_proj pivots of every condition index we need
        needed = sorted({i for c in combos for i in c})
        hv_batch = get_o_proj_inputs_batch(mt, [valid[i] for i in needed], layers_below)
        hv = {i: torch.stack([hv_batch[ly][bi] for ly in layers_below]) for bi, i in enumerate(needed)}

        lab = _load_labeled(croot, modularity, s)
        step_texts = [sl.step_text for sl in lab.labels]

        di_cache: dict[int, dict | None] = {}
        span_cache: dict[int, list[np.ndarray]] = {}
        for (oi, ti) in combos:
            if oi not in di_cache:  # destination tokenization + spans, cached per origin
                di = prepare_input(prompts=[valid[oi]], tokenizer=mt)
                seq = di["input_ids"].shape[1]
                if int(di["attention_mask"].sum()) != seq:  # B=1 must not pad
                    di_cache[oi] = None
                else:
                    di_cache[oi] = di
                    span_cache[oi] = [_key_idx(kr[k], seq) for k in range(len(kr))]
            di = di_cache[oi]
            if di is None:
                continue
            spans = span_cache[oi]
            if spans[oi].size == 0 or spans[ti].size == 0:
                continue
            pairs.append(
                OidPair(
                    sample=s,
                    dest_inputs=di,
                    src_hv=hv[ti],
                    dest_hv=hv[oi],
                    spans=spans,
                    orig_idx=oi,
                    target_idx=ti,
                    step_texts=step_texts,
                )
            )
            if len(pairs) >= n_pairs:
                return pairs
    return pairs

def replicate_inputs(inputs: dict, batch: int) -> dict:
    """Tile a B=1 tokenized input to batch size `batch` (no re-tokenization)."""
    return {
        k: (v.repeat(batch, *([1] * (v.dim() - 1))) if torch.is_tensor(v) else v)
        for k, v in inputs.items()
    }


# ──────────────────────────────────────────────────────────────────────────────
# Q·K gaze scorer.
# ──────────────────────────────────────────────────────────────────────────────


def _norm_spec(mod):
    """Snapshot a q_norm/k_norm module as ``(detached_weight, eps)`` for manual RMSNorm, or
    ``None`` if the model has no QK-norm (e.g. Qwen2.5). Detached because the model is frozen
    during mask training (mirrors the detached q_proj/k_proj weights)."""
    if mod is None:
        return None
    eps = getattr(mod, "variance_epsilon", getattr(mod, "eps", 1e-6))
    return (mod.weight.detach(), float(eps))


def _rmsnorm(y, spec):
    """Apply RMSNorm ``(y / rms(y)) * weight`` over the last dim, in float32 (matching HF's
    Olmo/Gemma/Qwen3 RMSNorm). ``spec`` = ``(weight, eps)`` or ``None`` (identity)."""
    if spec is None:
        return y
    w, eps = spec
    dt = y.dtype
    yf = y.float()
    yf = yf * torch.rsqrt(yf.pow(2).mean(-1, keepdim=True) + eps)
    return (w.to(yf.device) * yf).to(dt)


class GazeScorer:
    """Computes the raw Q·K gaze affinity on the locked gaze heads, ourselves.

    ``normalize=False`` → raw dot product (no ``/sqrt(d)``); ``True`` → cosine similarity.
    ``include_rope`` applies the model's rotary to q/k before the dot (device-safe, manual).
    """

    def __init__(self, mt, gaze_heads, normalize=False, include_rope=False, device="cuda:0"):
        self.mt = mt
        self.gaze_heads = list(gaze_heads)
        self.normalize = normalize
        self.include_rope = include_rope
        self.device = device

        cfg = mt.model.config
        self.n_q = cfg.num_attention_heads
        self.n_kv = getattr(cfg, "num_key_value_heads", self.n_q)
        self.head_dim = getattr(cfg, "head_dim", mt.n_embd // self.n_q)
        self.gqa = self.n_q // self.n_kv

        self.gaze_layers = sorted({l for l, _ in self.gaze_heads})
        self.heads_by_layer: dict[int, list[int]] = {}
        for (l, h) in self.gaze_heads:
            self.heads_by_layer.setdefault(l, []).append(h)

        # frozen q_proj/k_proj weights+biases for the gaze layers, extracted once
        self.W = {}
        # frozen QK-norm specs (weight, eps) per gaze layer — Olmo2/3, Gemma2, Qwen3, … apply
        # RMSNorm to q/k *after* projection and *before* reshape+RoPE. Absent on Qwen2.5 (→ None,
        # no-op). We replicate it with the *detached* weight (model is frozen during mask training,
        # matching the detached q_proj/k_proj weights above) so its manual Q·K matches real attention.
        self.qk_norm = {}
        for l in self.gaze_layers:
            base = mt.attn_module_name_format.format(l)
            qp = get_module_nnsight(mt._model, base + ".q_proj")
            kp = get_module_nnsight(mt._model, base + ".k_proj")
            self.W[l] = (
                qp.weight.detach(),
                None if qp.bias is None else qp.bias.detach(),
                kp.weight.detach(),
                None if kp.bias is None else kp.bias.detach(),
            )
            attn = get_module_nnsight(mt._model, base)
            self.qk_norm[l] = (
                _norm_spec(getattr(attn, "q_norm", None)),
                _norm_spec(getattr(attn, "k_norm", None)),
            )

    def _proj(self, x, W, b, n_heads, norm_spec=None):
        y = torch.matmul(x, W.t().to(x.dtype))
        if b is not None:
            y = y + b.to(x.dtype)
        y = _rmsnorm(y, norm_spec)   # QK-norm (Olmo2/3 etc.); no-op when norm_spec is None
        return y.view(*x.shape[:-1], n_heads, self.head_dim)

    def _rope(self, vec, pos):
        """vec (B, P, n_heads, D); rotate at integer positions `pos`. `pos` may be 1-D (P,) —
        shared across the batch (no padding) — or 2-D (B, P) — per-row, e.g. position_ids derived
        from a left-padded attention mask. Manual + device-safe."""
        if not self.include_rope:
            return vec
        dev, dt = vec.device, vec.dtype
        rotary = self.mt._model.model.rotary_emb
        inv_freq = rotary.inv_freq.to(dev).float()
        scaling = float(getattr(rotary, "attention_scaling", 1.0))
        pos = pos.to(dev).float()
        if pos.dim() == 1:
            freqs = torch.outer(pos, inv_freq)                  # (P, d/2)
            emb = torch.cat([freqs, freqs], dim=-1)             # (P, D)
            cos = (emb.cos() * scaling).to(dt)[None, :, None, :]
            sin = (emb.sin() * scaling).to(dt)[None, :, None, :]
        else:  # (B, P) → per-row
            freqs = pos.unsqueeze(-1) * inv_freq                # (B, P, d/2)
            emb = torch.cat([freqs, freqs], dim=-1)             # (B, P, D)
            cos = (emb.cos() * scaling).to(dt).unsqueeze(2)     # (B, P, 1, D)
            sin = (emb.sin() * scaling).to(dt).unsqueeze(2)
        d = vec.shape[-1]
        rot = torch.cat([-vec[..., d // 2:], vec[..., : d // 2]], dim=-1)  # rotate_half (HF)
        return vec * cos + rot * sin

    def _latents(self, dest_inputs, pre_hooks=()):
        """Forward with `pre_hooks` (o_proj patches); capture each gaze layer's q_proj INPUT latent."""
        handles, saved = [], {}
        try:
            for mod, fn in pre_hooks:
                handles.append(mod.register_forward_pre_hook(fn))
            for l in self.gaze_layers:
                qp = get_module_nnsight(self.mt._model, self.mt.attn_module_name_format.format(l) + ".q_proj")

                def mk(l_):
                    def cap(module, args):
                        saved[l_] = args[0]
                        return None
                    return cap

                handles.append(qp.register_forward_pre_hook(mk(l)))
            self.mt._model(**dict(dest_inputs), use_cache=False)
        finally:
            for h in handles:
                h.remove()
        return saved

    def compute_T(self, dest_inputs, spans, pre_hooks=()):
        """T (B, n_span): mean over gaze heads of the pivot query's raw Q·K affinity to each span's
        key tokens.  Differentiable through any mask living in `pre_hooks`."""
        lat = self._latents(dest_inputs, pre_hooks)
        B = lat[self.gaze_layers[0]].shape[0]
        seq = lat[self.gaze_layers[0]].shape[1]
        pos = torch.arange(seq)
        T = torch.zeros(B, len(spans), device=self.device)
        n_used = 0
        for l, hs in self.heads_by_layer.items():
            x = lat[l]
            Wq, bq, Wk, bk = self.W[l]
            qn, kn = self.qk_norm[l]
            Q = self._rope(self._proj(x[:, -1:, :], Wq, bq, self.n_q, qn), pos[-1:])[:, 0]  # (B, n_q, D)
            K = self._rope(self._proj(x, Wk, bk, self.n_kv, kn), pos)                       # (B, seq, n_kv, D)
            for h in hs:
                qh = Q[:, h, :]                          # (B, D)
                kh = K[:, :, h // self.gqa, :]           # (B, seq, D) — GQA mapping
                if self.normalize:
                    qh = qh / (qh.norm(dim=-1, keepdim=True) + 1e-9)
                    kh = kh / (kh.norm(dim=-1, keepdim=True) + 1e-9)
                for c, span in enumerate(spans):
                    idx = torch.as_tensor(span, dtype=torch.long, device=kh.device)
                    if idx.numel() == 0:
                        continue
                    aff = (qh.unsqueeze(1) * kh[:, idx, :]).sum(-1).mean(dim=1)  # (B,)
                    T[:, c] = T[:, c] + aff.to(self.device)
                n_used += 1
        return T / max(n_used, 1)

    def compute_T_masked(self, dest_inputs, key_masks, pre_hooks=()):
        """Like ``compute_T`` but with PER-ROW key masks, for a left-padded batch where each row's
        span sits at different absolute key positions.

        ``key_masks`` (n_span, B, seq) float — 1.0 on the key tokens of each span per row, else 0.
        Query is the LAST position (-1, the decision token). Returns ``T`` (B, n_span): mean over
        gaze heads of the query's raw Q·K affinity to each span (per-row masked mean over keys).
        Differentiable through any mask living in ``pre_hooks``."""
        lat = self._latents(dest_inputs, pre_hooks)
        L0 = self.gaze_layers[0]
        B, seq = lat[L0].shape[0], lat[L0].shape[1]
        if self.include_rope:  # position_ids from left-padded mask (real tokens start at 0)
            am = dest_inputs["attention_mask"].to(self.device)
            pos_all = (am.long().cumsum(-1) - 1).clamp_min(0)   # (B, seq)
            pos_q = pos_all[:, -1:]                             # (B, 1)
        else:
            pos_all = torch.arange(seq)
            pos_q = pos_all[-1:]
        key_masks = key_masks.to(self.device).float()           # (n_span, B, seq)
        n_span = key_masks.shape[0]
        denom = key_masks.sum(-1).clamp_min(1.0)                # (n_span, B)
        T = torch.zeros(n_span, B, device=self.device)
        n_used = 0
        for l, hs in self.heads_by_layer.items():
            x = lat[l]
            Wq, bq, Wk, bk = self.W[l]
            qn, kn = self.qk_norm[l]
            Q = self._rope(self._proj(x[:, -1:, :], Wq, bq, self.n_q, qn), pos_q)[:, 0]  # (B, n_q, D)
            K = self._rope(self._proj(x, Wk, bk, self.n_kv, kn), pos_all)               # (B, seq, n_kv, D)
            for h in hs:
                qh = Q[:, h, :]                          # (B, D)
                kh = K[:, :, h // self.gqa, :]           # (B, seq, D) — GQA mapping
                if self.normalize:
                    qh = qh / (qh.norm(dim=-1, keepdim=True) + 1e-9)
                    kh = kh / (kh.norm(dim=-1, keepdim=True) + 1e-9)
                aff = (qh.unsqueeze(1) * kh).sum(-1).to(self.device)  # (B, seq)
                for c in range(n_span):
                    T[c] = T[c] + (aff * key_masks[c]).sum(-1) / denom[c]
                n_used += 1
        return (T / max(n_used, 1)).t()  # (B, n_span)


# ──────────────────────────────────────────────────────────────────────────────
# o_proj patch hooks (shared by screen & trainer).
# ──────────────────────────────────────────────────────────────────────────────


def add_patch_hooks(mt, layer, add):
    """Additive o_proj-input patch at the pivot (last pos). `add`: (B, n_embd)."""
    o_proj = get_module_nnsight(mt._model, mt.attn_module_name_format.format(layer) + ".o_proj")

    def pre(module, args):
        x = args[0].clone()
        x[:, -1, :] = x[:, -1, :] + add.to(x.device, x.dtype)
        return (x,) + tuple(args[1:])

    return [(o_proj, pre)]


def mask_patch_hooks(mt, layers, src_hv, mask, n_q, head_dim):
    """Mask-weighted per-head o_proj-input transplant at the pivot: m*src + (1-m)*cascade.
    `src_hv`: (len(layers), n_embd). Differentiable through `mask.values`."""
    hooks, src_b = [], src_hv.unsqueeze(0)
    n_embd = n_q * head_dim
    for li, layer in enumerate(layers):
        o_proj = get_module_nnsight(mt._model, mt.attn_module_name_format.format(layer) + ".o_proj")

        def mk(li_):
            def pre(module, args):
                x = args[0]
                dev, dt, B_ = x.device, x.dtype, x.shape[0]
                src3 = src_b[:, li_].to(dev).view(B_, n_q, head_dim)
                m = mask.values[li_].to(dev).view(1, n_q, 1)
                cas3 = x[:, -1, :].float().view(B_, n_q, head_dim)
                new_last = (m * src3 + (1 - m) * cas3).view(B_, n_embd).to(dt)
                return (torch.cat([x[:, :-1, :], new_last.unsqueeze(1)], dim=1),) + tuple(args[1:])
            return pre

        hooks.append((o_proj, mk(li)))
    return hooks
