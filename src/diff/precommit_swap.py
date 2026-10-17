"""Pre-commitment swap on the multi-predicate setup (notebook ``notebooks/precommitment/replace_context.ipynb``; batch
driver ``scripts/diff/precommit_swap_eval.py``).

    p_src: a'b'c -> c     the residual donor
    p_dst: a'b'd -> d     the patched run

Both prefill ``a`` and ``b`` and stop at the frontier's ``{var} =``; everything from the user turn's ``<|im_end|>`` up to
``{var}`` is token-identical. Over that window every decoder layer's OUTPUT residual of DST is replaced with SRC's on the
prompt pass (the pre-commitment-swap residual transfer, I2, with ``p_src`` itself as the donor). The DST task list and the
eq pivot `` =`` are never overwritten; the pivot reads the swapped k/v from layer 1 on (:func:`pivot_sanity`).

* patched implements **c** -> the pending operation was already committed in the progress latents (pre-commitment effect);
* patched implements **d** -> the pre-commitment is redundant, the goal is re-derived from the task span (no-op).

``include_src_front=True`` also transfers the frontier description (SRC's ``c`` text -> DST's ``d`` text; the shorter
span is ``\\n``-padded right before it, :func:`src.diff.functional.align_kv_spans` ``mode="pad"``, and the padded prompts
are used for every run). Then DST's task list itself carries ``c`` — a positive control for the intervention logic."""

from contextlib import contextmanager

import torch

import src.diff.data as DATA
import src.diff.functional as FUNC
import src.diff.path_patch as PP
from src.tokens import prepare_input


def swap_geometry(inst, *, include_src_front=False, ctx):
    """Token sequences + the transferred windows (SRC / DST positions pair up 1:1 in order). The pivot is never in them."""
    mt = ctx.mt
    m = DATA.materialize(inst, ctx=ctx)
    c_desc, d_desc = inst.src_frontier.description, inst.dst_frontier.description
    assert m["src_prompt"].count(c_desc) == 1 and m["dst_prompt"].count(d_desc) == 1, "frontier text not unique"
    fs0, fd0 = FUNC.token_span(m["src_prompt"], c_desc, ctx=ctx), FUNC.token_span(m["dst_prompt"], d_desc, ctx=ctx)
    if include_src_front:
        src_tok, dst_tok, _, _, src_fspan, dst_fspan = FUNC.align_kv_spans(
            mt, m["src_prompt"], m["dst_prompt"], fs0, fd0, mode="pad", ctx=ctx)
    else:
        src_tok = prepare_input(prompts=m["src_prompt"], tokenizer=mt)
        dst_tok = prepare_input(prompts=m["dst_prompt"], tokenizer=mt)
        src_fspan, dst_fspan = tuple(fs0), tuple(fd0)
    s_ids, d_ids = PP._ids(src_tok), PP._ids(dst_tok)
    L_src, L_dst = len(s_ids), len(d_ids)
    n_pad = (L_src + L_dst - len(PP._ids(prepare_input(prompts=m["src_prompt"], tokenizer=mt)))
             - len(PP._ids(prepare_input(prompts=m["dst_prompt"], tokenizer=mt))))   # "\n" pads (src front only)

    # shared segment = last <|im_end|> (end of the user turn) .. pivot, aligned from the END of each prompt
    im_end = mt.tokenizer.convert_tokens_to_ids("<|im_end|>")
    seg_start = max(i for i, t in enumerate(d_ids) if t == im_end)
    n_seg = L_dst - seg_start
    assert s_ids[L_src - n_seg:] == d_ids[seg_start:], "SRC/DST code prefix is not token-identical"
    n_pat = n_seg - 1                                   # <|im_end|> .. {var}; the pivot ` =` stays clean
    seg_src = (L_src - n_seg, L_src - n_seg + n_pat)
    seg_dst = (seg_start, seg_start + n_pat)

    src_win = ([tuple(src_fspan)] if include_src_front else []) + [seg_src]
    dst_win = ([tuple(dst_fspan)] if include_src_front else []) + [seg_dst]
    src_pos = [q for lo, hi in src_win for q in range(lo, hi)]
    dst_pos = [q for lo, hi in dst_win for q in range(lo, hi)]
    assert len(src_pos) == len(dst_pos), (len(src_pos), len(dst_pos))
    return dict(include_src_front=include_src_front, src_tok=src_tok, dst_tok=dst_tok,
                src_prompt=m["src_prompt"], dst_prompt=m["dst_prompt"], c_desc=c_desc, d_desc=d_desc,
                fs0=tuple(fs0), fd0=tuple(fd0), src_fspan=tuple(src_fspan), dst_fspan=tuple(dst_fspan),
                src_win=src_win, dst_win=dst_win, src_pos=src_pos, dst_pos=dst_pos,
                piv_src=L_src - 1, piv_dst=L_dst - 1, pad=n_pad)


def geometry_info(g) -> dict:
    """JSON-native summary of a geometry (for the saved record)."""
    return dict(include_src_front=g["include_src_front"], src_win=[list(w) for w in g["src_win"]],
                dst_win=[list(w) for w in g["dst_win"]], n_pos=len(g["dst_pos"]), pad_tokens=g["pad"],
                L_src=g["piv_src"] + 1, L_dst=g["piv_dst"] + 1)


def swap_cache(g, layers, *, ctx):
    """SRC's layer outputs over the window — what overwrites DST's."""
    return PP.layer_outputs(ctx.mt, g["src_tok"], g["src_pos"], layers)


def swap_patch(g, cache, layers, *, ctx):
    """Context manager: DST's window residuals <- ``cache`` (prompt pass only; generated tokens read the cache)."""
    return PP.residual_patch(ctx.mt, cache, g["dst_pos"], layers)


@torch.no_grad()
def _next_top(tok, mt, k=5):
    lg = mt._model(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"]).logits[0, -1].float()
    v, ix = lg.softmax(-1).topk(k)
    return [(mt.tokenizer.decode([int(t)]), round(float(x), 4)) for x, t in zip(v, ix)]


def pivot_sanity(g, cache, layers, *, ctx, k=5):
    """Does the swap reach the (clean) eq pivot? Controls: every position before the first transferred token is
    bit-identical to clean DST (``pre_max_abs`` == 0) and the window equals SRC's residuals (``win_max_abs``). At the pivot:
    per-layer ‖h_patched − h_dst‖ / ‖h_dst‖ (expected 0 at layer 0, > 0 from layer 1 on) and the top-k next tokens."""
    mt = ctx.mt
    every = list(range(mt.n_layer))
    pre = list(range(min(g["dst_pos"])))
    probe = pre + [g["piv_dst"]]
    h_dst = PP.layer_outputs(mt, g["dst_tok"], probe, every)
    with swap_patch(g, cache, layers, ctx=ctx):
        h_pat = PP.layer_outputs(mt, g["dst_tok"], probe, every)
        h_win = PP.layer_outputs(mt, g["dst_tok"], g["dst_pos"], layers)
    pre_max = max(float((h_pat[l][:-1] - h_dst[l][:-1]).abs().max()) for l in every) if pre else 0.0
    win_max = max(float((h_win[l] - cache[l]).abs().max()) for l in layers)
    rel = [float((h_pat[l][-1] - h_dst[l][-1]).norm() / h_dst[l][-1].norm()) for l in every]
    moved = [l for l in every if rel[l] > 1e-3]
    with swap_patch(g, cache, layers, ctx=ctx):
        top_pat = _next_top(g["dst_tok"], mt, k)
    top_src, top_dst = _next_top(g["src_tok"], mt, k), _next_top(g["dst_tok"], mt, k)
    return dict(pre_max_abs=pre_max, win_max_abs=win_max, pivot_rel_delta=[round(x, 4) for x in rel],
                pivot_affected=bool(moved), first_layer_moved=(moved[0] if moved else None),
                pivot_l0_delta=rel[0], pivot_max_rel_delta=max(rel),
                top_src=top_src, top_dst=top_dst, top_patched=top_pat,
                top1_patched_is_src=top_pat[0][0] == top_src[0][0], top1_patched_is_dst=top_pat[0][0] == top_dst[0][0])


def runs(g, cache, layers, *, readout_heads, max_new=64, ctx, clean=None):
    """Clean SRC / clean DST / patched DST greedy generations + GoalScope readouts at `` =`` (``PP.judge``'s ``raw``).
    ``clean`` = an earlier result's ``gen_src / gen_dst / ro_src / ro_dst`` on the same token sequences (reused)."""
    out = dict(clean) if clean is not None else PP.clean_runs(
        dict(src_tok=g["src_tok"], dst_tok=g["dst_tok"]), readout_heads=readout_heads, max_new=max_new, ctx=ctx)
    with swap_patch(g, cache, layers, ctx=ctx):
        out["gen_patched"] = FUNC.greedy(g["dst_tok"], max_new=max_new, ctx=ctx)

    def iface(mt_, inputs, patches=None, model_kwargs=None):
        with swap_patch(g, cache, layers, ctx=ctx):
            return mt_._model(**dict(inputs), **dict(model_kwargs or {}))

    out["ro_patched"] = FUNC.goalscope_readout("", g["dst_tok"], g["piv_dst"], heads=readout_heads, iface=iface,
                                               ctx=ctx)[0]["patched_gen"]
    return out
