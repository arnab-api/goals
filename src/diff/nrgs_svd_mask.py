"""Sub-head (SVD-component) query mask for the NRGS common-prefix setup — the filter paper's App. I recipe
(Overleaf-draft/REF/papers/filter.pdf, "Decomposing W_QK into smaller functional units with SVD"),
applied to the pre-RoPE query transplant of :mod:`src.diff.nrgs_mask`.

Per query head h (GQA: its key group g(h)), ignoring RoPE the score is a bilinear form in the normed residual::

    a_qk = h~_q W_Q W_K^T h~_k^T = h~_q W_QK h~_k^T,     W_QK = U S V^T = sum_i s_i u_i v_i^T   (rank d_head)

The paper patches the LEFT projection u = h~ U component-wise, u_patch = u_dst + (u_src - u_dst) * m, and
recomputes a_qk = u_patch S V^T h~_k^T without RoPE. Because U lies in W_Q's column space, u is an invertible
linear map of the head's query, u = q M with M = W_K^T V S^-1 (d_head x d_head), so the same patch is::

    q_patch = q_dst + (q_src - q_dst) M diag(m) M^-1

— exact for RoPE = I, and applied here to the pre-RoPE q_proj output, so attention (with RoPE) runs as usual.
The q_proj bias cancels in the difference. m = 1 everywhere reproduces the head-level transplant
(``nrgs_mask.q_blend_hooks`` with m = 1), so fixed head sets are evaluated in the same code path.

The basis is computed from small factors (no d_embd x d_embd matrix): W_Q = Q1 R1, W_K = Q2 R2 (QR), then
A = R1 R2^T = Ua S Va^T gives U = Q1 Ua, V = Q2 Va, and M = R1^-1 Ua, M^-1 = Ua^T R1.
Component order is by singular value (i = 0 is the largest).

Training reuses the head-level objective and caches (:class:`src.diff.nrgs_mask.NRGSCache`): KL to DST* +
directional d-over-b + optional SRC' term + lambda * mean(m), projected GD with m clamped to [0, 1]. ``allowed``
restricts which heads may carry components (e.g. only the goal heads, as in the paper's "only filter heads").
"""
from __future__ import annotations

import logging
import random
import time
from contextlib import contextmanager

import baukit
import torch
import torch.nn.functional as F

import src.diff.nrgs_mask as NM
from src.tokens import prepare_input

logger = logging.getLogger(__name__)


def _proj(mt, layer, which):
    return mt.attn_module_name_format.format(layer) + f".{which}_proj"


@torch.no_grad()
def svd_bases(mt, layers: list[int]) -> dict[int, dict[str, torch.Tensor]]:
    """{layer: {M, Minv: (n_heads, d, d), S: (n_heads, d)}} float32 on that layer's q_proj device.
    ``q @ M`` gives the U-basis coordinates of a (pre-bias-difference) query; ``u @ Minv`` maps back."""
    H, d = mt.n_heads, mt.head_dim
    group = H // mt.n_kv_heads
    out = {}
    for l in layers:
        wq = baukit.get_module(mt._model, _proj(mt, l, "q")).weight        # (H*d, d_embd)
        wk = baukit.get_module(mt._model, _proj(mt, l, "k")).weight        # (n_kv*d, d_embd)
        dev = wq.device
        Wq = wq.float().view(H, d, -1).transpose(1, 2)                      # (H, d_embd, d)
        Wk = wk.float().to(dev).view(mt.n_kv_heads, d, -1).transpose(1, 2)  # (n_kv, d_embd, d)
        _, R1 = torch.linalg.qr(Wq)                                         # (H, d, d)
        _, R2 = torch.linalg.qr(Wk)                                         # (n_kv, d, d)
        R2 = R2.repeat_interleave(group, dim=0)                             # query head h -> its key group
        Ua, S, _ = torch.linalg.svd(R1 @ R2.transpose(1, 2))                # A = R1 R2^T = Ua S Va^T
        M = torch.linalg.solve_triangular(R1, Ua, upper=True)               # R1^-1 Ua
        out[l] = dict(M=M.contiguous(), Minv=(Ua.transpose(1, 2) @ R1).contiguous(), S=S)
    return out


def svd_blend_hooks(mt, comp_mask: torch.Tensor, layers: list[int], bases, q_src: torch.Tensor):
    """Hooks on each candidate q_proj: LAST position <- q_dst + (q_src - q_dst) M diag(m) M^-1 per head.
    ``comp_mask`` (len(layers), n_heads, d) — trainable or fixed; ``q_src`` (B, len(layers), width)."""
    H, d = mt.n_heads, mt.head_dim
    handles = []
    for li, layer in enumerate(layers):
        mod = baukit.get_module(mt._model, _proj(mt, layer, "q"))
        b = bases[layer]

        def make(li, b):
            def hook(module, inp, out):
                o = out[0] if isinstance(out, tuple) else out                    # (B, T, width)
                last = o[:, -1]
                delta = (q_src[:, li].to(o.device) - last.float()).view(-1, H, d)     # (B, H, d) float32
                u = torch.einsum("bhd,hde->bhe", delta, b["M"].to(o.device))
                u = u * comp_mask[li].to(o.device, torch.float32)
                dq = torch.einsum("bhe,hed->bhd", u, b["Minv"].to(o.device)).reshape(last.shape)
                new_last = (last.float() + dq).to(o.dtype)
                o_new = torch.cat([o[:, :-1], new_last.unsqueeze(1)], dim=1)
                return (o_new,) + tuple(out[1:]) if isinstance(out, tuple) else o_new
            return hook

        handles.append(mod.register_forward_hook(make(li, b)))
    return handles


@contextmanager
def svd_blend(mt, comp_mask, layers, bases, caches: list[NM.NRGSCache], use_ctrl: bool = False):
    """Hooks live for the whole block (forward and backward, as :func:`nrgs_mask.q_blend`)."""
    qs = torch.stack([(c.q_ctrl if use_ctrl else c.q_src)[layers] for c in caches])   # (B, L, width)
    handles = svd_blend_hooks(mt, comp_mask, layers, bases, qs)
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def patched_logprobs(mt, comp_mask, layers, bases, caches, use_ctrl=False):
    with svd_blend(mt, comp_mask, layers, bases, caches, use_ctrl):
        return NM.dst_logprobs(mt, caches)


@torch.no_grad()
def evaluate(mt, comp_mask, layers, bases, caches, batch_size=8, use_ctrl=False) -> dict:
    """Same scores as :func:`nrgs_mask.evaluate` (flip / margin_gain / kl, pooled + by_config)."""
    if use_ctrl:
        caches = [c for c in caches if c.q_ctrl is not None]
    rows = []
    for s in range(0, len(caches), batch_size):
        cs = caches[s: s + batch_size]
        lp = patched_logprobs(mt, comp_mask, layers, bases, cs, use_ctrl=use_ctrl)
        for i, c in enumerate(cs):
            rows.append((c.config, int(lp[i].argmax()) == c.d_id,
                         float(lp[i][c.d_id] - lp[i][c.b_id]) - c.clean_margin,
                         float(F.kl_div(c.log_p_target.to(lp.device), lp[i], reduction="sum", log_target=True))))

    def agg(rs):
        n = max(1, len(rs))
        return dict(n=len(rs), flip=sum(r[1] for r in rs) / n, margin_gain=sum(r[2] for r in rs) / n,
                    kl=sum(r[3] for r in rs) / n)
    out = agg(rows)
    out["by_config"] = {cf: agg([r for r in rows if r[0] == cf]) for cf in sorted({r[0] for r in rows})}
    return out


class ComponentMask(torch.nn.Module):
    """(len(layers), n_heads, d_head) mask in [0, 1], init all-ones on ``allowed`` heads (zeros elsewhere,
    never trained). ``values`` is what the hooks read; ``allowed`` is a fixed (L, H, 1) 0/1 buffer."""

    def __init__(self, n_layers, n_heads, d_head, allowed=None):
        super().__init__()
        allowed = torch.ones(n_layers, n_heads) if allowed is None else allowed.float()
        self.register_buffer("allowed", allowed.unsqueeze(-1))
        self.raw = torch.nn.Parameter(torch.ones(n_layers, n_heads, d_head))

    @property
    def values(self) -> torch.Tensor:
        return self.raw * self.allowed

    @torch.no_grad()
    def clamp_(self):
        self.raw.clamp_(0.0, 1.0)

    def sparsity(self) -> torch.Tensor:
        """Mean over ALL candidate components (not just the allowed ones), so the per-component sparsity
        pull is the same whether or not the heads are restricted (restricting to 250 of 2560 heads would
        otherwise make it ~10x stronger)."""
        return self.values.sum() / self.values.numel()

    @torch.no_grad()
    def n_active(self, threshold=0.9) -> int:
        return int((self.values > threshold).sum())

    @torch.no_grad()
    def n_heads_active(self, threshold=0.9) -> int:
        return int(((self.values > threshold).sum(-1) > 0).sum())


def comp_mask_from_heads(heads, layers, n_heads, d_head, device=None) -> torch.Tensor:
    """All components of ``heads`` on (= the head-level transplant of a fixed head set)."""
    m = torch.zeros(len(layers), n_heads, d_head, device=device)
    row = {l: i for i, l in enumerate(layers)}
    for l, h in heads:
        if l in row:
            m[row[l], h] = 1.0
    return m


def allowed_from_heads(heads, layers, n_heads) -> torch.Tensor:
    a = torch.zeros(len(layers), n_heads)
    row = {l: i for i, l in enumerate(layers)}
    for l, h in heads:
        if l in row:
            a[row[l], h] = 1.0
    return a


def train_svd_mask(mt, train, val, layers, bases, *, allowed=None, n_epochs=50, lr=1e-2, lambda_sparse=1.0,
                   w_kl=1.0, w_dir=1.0, w_ctrl=0.0, dir_loss="logistic", batch_size=64, micro_batch=8, seed=0,
                   threshold=0.9, on_epoch=None):
    """Projected-GD training of a :class:`ComponentMask` with the :func:`nrgs_mask.train_nrgs_mask` objective
    (sparsity = lambda * mean over all candidate components). Gradient accumulation over ``micro_batch``
    forwards; OOM halves ``micro_batch`` and redoes the step."""
    if w_ctrl > 0 and not any(c.q_ctrl is not None for c in train):
        raise ValueError("w_ctrl > 0 needs SRC' queries; no training instance's config defines one")
    for p in mt._model.parameters():
        p.requires_grad_(False)
    dev = next(mt._model.parameters()).device
    mask = ComponentMask(len(layers), mt.n_heads, mt.head_dim, allowed).to(dev)
    opt = torch.optim.AdamW(mask.parameters(), lr=lr)
    rng, hist = random.Random(seed), NM.History()

    def _accumulate(batch, micro):
        opt.zero_grad()
        loss_sparse = lambda_sparse * mask.sparsity()
        loss_sparse.backward()
        step = dict(kl=0.0, dir=0.0, ctrl=0.0, sparse=float(loss_sparse.detach()))
        n_ctrl = sum(c.q_ctrl is not None for c in batch)
        for m0 in range(0, len(batch), micro):
            cs = batch[m0: m0 + micro]
            w = len(cs) / len(batch)
            with svd_blend(mt, mask.values, layers, bases, cs):
                lp = NM.dst_logprobs(mt, cs)
                tgt = torch.stack([c.log_p_target for c in cs]).to(lp.device)
                loss_kl = F.kl_div(tgt, lp, reduction="batchmean", log_target=True)
                ar = torch.arange(len(cs), device=lp.device)
                b = torch.tensor([c.b_id for c in cs], device=lp.device)
                d = torch.tensor([c.d_id for c in cs], device=lp.device)
                loss_dir = NM._dir_term(lp[ar, d] - lp[ar, b], dir_loss).mean()
                (w * (w_kl * loss_kl + w_dir * loss_dir)).backward()
            step["kl"] += w * float(loss_kl.detach())
            step["dir"] += w * float(loss_dir.detach())
            cc = [c for c in cs if c.q_ctrl is not None]
            if w_ctrl > 0 and cc:
                wc = len(cc) / n_ctrl
                with svd_blend(mt, mask.values, layers, bases, cc, use_ctrl=True):
                    lpc = NM.dst_logprobs(mt, cc)
                    arc = torch.arange(len(cc), device=lpc.device)
                    bc = torch.tensor([c.b_id for c in cc], device=lpc.device)
                    dc = torch.tensor([c.d_id for c in cc], device=lpc.device)
                    clean = torch.tensor([c.clean_margin for c in cc], device=lpc.device)
                    loss_ctrl = F.relu(lpc[arc, dc] - lpc[arc, bc] - clean).mean()
                    (wc * w_ctrl * loss_ctrl).backward()
                step["ctrl"] += wc * float(loss_ctrl.detach())
        return step

    steps_per_epoch = -(-len(train) // batch_size)
    n_allowed = int(mask.allowed.sum())
    logger.info(f"train_svd_mask: {len(train)} train / {len(val)} val · {len(layers)} layers x {mt.n_heads} heads "
                f"({n_allowed} allowed) x {mt.head_dim} components · {n_epochs} epochs x {steps_per_epoch} steps "
                f"(batch {batch_size}, micro {micro_batch}) · lr={lr} lambda={lambda_sparse} w_kl={w_kl} "
                f"w_dir={w_dir} w_ctrl={w_ctrl} dir={dir_loss}")
    t_start = time.time()
    for epoch in range(n_epochs):
        t_ep = time.time()
        NM._reset_peak_mem()
        order = list(range(len(train)))
        rng.shuffle(order)
        agg = dict(loss=0.0, kl=0.0, dir=0.0, ctrl=0.0, sparse=0.0)
        n_steps = 0
        for s in range(0, len(order), batch_size):
            idx = order[s: s + batch_size]
            while True:
                try:
                    step = _accumulate([train[i] for i in idx], micro_batch)
                    break
                except torch.cuda.OutOfMemoryError:
                    if micro_batch == 1:
                        raise
                    micro_batch = max(1, micro_batch // 2)
                    opt.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    logger.warning(f"CUDA OOM -> micro_batch={micro_batch}; redoing e{epoch + 1} step {n_steps + 1}")
            opt.step()
            mask.clamp_()
            step["loss"] = w_kl * step["kl"] + w_dir * step["dir"] + w_ctrl * step["ctrl"] + step["sparse"]
            for k in agg:
                agg[k] += step[k]
            n_steps += 1
            logger.info(f"  e{epoch + 1:03d} step {n_steps}/{steps_per_epoch}  loss={step['loss']:.3f}  "
                        f"kl={step['kl']:.3f}  dir={step['dir']:.3f}  ctrl={step['ctrl']:.3f}  "
                        f"mask_mean={float(mask.sparsity().detach()):.3f}  comps>{threshold}={mask.n_active(threshold)}  "
                        f"heads>{threshold}={mask.n_heads_active(threshold)}  ({time.time() - t_ep:.0f}s into epoch)")
        for k in agg:
            getattr(hist, k).append(agg[k] / max(1, n_steps))
        hist.n_active.append(mask.n_active(threshold))
        t_tr = time.time() - t_ep
        v = evaluate(mt, mask.values.detach(), layers, bases, val, micro_batch) if val else {}
        hist.val.append(v)
        el = time.time() - t_start
        logger.info(f"e{epoch + 1:03d}/{n_epochs}  loss={hist.loss[-1]:.3f}  kl={hist.kl[-1]:.3f}  dir={hist.dir[-1]:.3f}  "
                    f"comps>{threshold}={hist.n_active[-1]}  heads>{threshold}={mask.n_heads_active(threshold)}  "
                    f"val_flip={v.get('flip', float('nan')):.2f}  val_gain={v.get('margin_gain', float('nan')):+.2f}  "
                    f"val_kl={v.get('kl', float('nan')):.2f} | train {t_tr:.0f}s + val {time.time() - t_ep - t_tr:.0f}s · "
                    f"peak mem {NM._peak_mem()} · elapsed {el / 60:.1f}m · ETA {el / (epoch + 1) * (n_epochs - epoch - 1) / 60:.1f}m")
        if on_epoch is not None:
            on_epoch(epoch, mask, hist)
    return mask, hist


@torch.no_grad()
def component_projectors(mt, mask_path, threshold=None, eps=1e-3):
    """Load a trained component mask (``train_nrgs_svd_mask.py`` -> mask.pt) as per-head projectors for the
    generation evals: ``{layer: {head: P}}`` with ``P = M diag(m) M^-1`` (d x d, float32, CPU), plus the
    sorted list of (layer, head) that carry any component. ``threshold=None`` uses the SOFT trained values
    (the intervention as trained); a float binarizes ``m > threshold``. Heads with max(m) <= ``eps`` are
    dropped (left unpatched)."""
    ck = torch.load(mask_path, map_location="cpu", weights_only=False)
    mv = ck["mask_values"].float()                                  # (n_layer, n_heads, d), full depth
    if threshold is not None:
        mv = (mv > threshold).float()
    layers = sorted({int(l) for l in torch.nonzero(mv.amax(dim=(1, 2)) > eps).flatten()})
    bases = svd_bases(mt, layers)
    proj, heads = {}, []
    for l in layers:
        b = bases[l]
        for h in torch.nonzero(mv[l].amax(dim=1) > eps).flatten().tolist():
            m = mv[l, h].to(b["M"].device)
            proj.setdefault(l, {})[h] = (b["M"][h] * m.unsqueeze(0)) @ b["Minv"][h]   # M diag(m) M^-1
            proj[l][h] = proj[l][h].cpu()
            heads.append((l, h))
    return proj, heads
