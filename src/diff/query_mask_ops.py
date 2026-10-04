"""Shared differentiable pre-RoPE query kernels; no model-wrapper dependencies.

Factored unchanged from nrgs_mask/nrgs_svd_mask, except module lookup uses
PyTorch get_submodule. The adapter supplies the original mt shim fields.
"""
import torch


def _q_proj(mt, layer):
    return mt.attn_module_name_format.format(layer) + ".q_proj"


def _proj(mt, layer, which):
    return mt.attn_module_name_format.format(layer) + f".{which}_proj"


def q_blend_hooks(mt, mask_values: torch.Tensor, layers: list[int], q_src: torch.Tensor):
    """Forward hooks on each candidate q_proj blending the LAST position (differentiable, out-of-place).
    ``mask_values`` (len(layers), n_heads) — a trainable parameter or a fixed 0/1 tensor;
    ``q_src`` (B, len(layers), width)."""
    hd = mt.head_dim
    handles = []
    for li, layer in enumerate(layers):
        mod = mt._model.get_submodule(_q_proj(mt, layer))

        def make(li):
            def hook(module, inp, out):
                o = out[0] if isinstance(out, tuple) else out                   # (B, T, width)
                m = mask_values[li].to(o.device, o.dtype).repeat_interleave(hd)  # (width,)
                last = m * q_src[:, li].to(o.device, o.dtype) + (1 - m) * o[:, -1]
                o_new = torch.cat([o[:, :-1], last.unsqueeze(1)], dim=1)
                return (o_new,) + tuple(out[1:]) if isinstance(out, tuple) else o_new
            return hook

        handles.append(mod.register_forward_hook(make(li)))
    return handles


@torch.no_grad()
def svd_bases(mt, layers: list[int]) -> dict[int, dict[str, torch.Tensor]]:
    """{layer: {M, Minv: (n_heads, d, d), S: (n_heads, d)}} float32 on that layer's q_proj device.
    ``q @ M`` gives the U-basis coordinates of a (pre-bias-difference) query; ``u @ Minv`` maps back."""
    H, d = mt.n_heads, mt.head_dim
    group = H // mt.n_kv_heads
    out = {}
    for l in layers:
        wq = mt._model.get_submodule(_proj(mt, l, "q")).weight        # (H*d, d_embd)
        wk = mt._model.get_submodule(_proj(mt, l, "k")).weight        # (n_kv*d, d_embd)
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
        mod = mt._model.get_submodule(_proj(mt, layer, "q"))
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
