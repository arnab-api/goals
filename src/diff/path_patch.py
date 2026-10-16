"""Path patching on the multi-predicate kv-transfer setup (notebook ``notebooks/diff/05_kv_transfer.ipynb``,
sections "Path patching" / "Path-patching variants"; batch driver ``scripts/diff/path_patch_eval.py``).

Three prompts sharing the prefilled progress (a, b) — DST's and p_res's progress prefill is token-identical:

    p_src: a'b'c -> c     k/v donor (I1, I3)
    p_dst: a'b'd -> d     the patched run
    p_res: a'b'e -> e     residual donor for I2 (frontier e not in {c, d}; rendered in DST's language)

Interventions on the patched DST run (each optional):

* **I1** — ``task_heads`` read SRC's window k/v at query rule ``i1_query`` (default "all": the queries inside and after
  the window, incl. every generated token).
* **I2** — over ``<|im_end|> .. {var}`` (the pivot ` =` excluded) every decoder layer's OUTPUT residual is overwritten
  by the donor's (``p_res`` by default, or the clean DST run) on the prompt pass; generated tokens see it via the cache.
* **I3** — ``goal_heads`` read SRC's window k/v from the pivot on (rule "last"), optionally blind to their own key.

The transplanted window is the task span (``window="span"``) or the task span + the progress up to ` =`
(``window="span+progress"``; ``include_pivot`` adds ` =` itself). Both are aligned with
:func:`src.diff.functional.align_kv_spans` (``mode="pad"``): the shorter SRC / DST window gets ``\\n`` tokens right before
it, so the two windows have equal length and the transplant stays position-aligned across languages ("A1"). All runs of a
geometry (clean SRC, clean DST, patched) use these padded token sequences."""

import dataclasses
import hashlib
import random
from contextlib import ExitStack, contextmanager

import baukit
import torch

import src.diff.data as DATA
import src.diff.functional as FUNC
from src.tokens import prepare_input


# ══════════════════════════════════════════════════════════════════════════════
# residual tracing / overwriting (every decoder layer's OUTPUT)
# ══════════════════════════════════════════════════════════════════════════════
def layer_outputs(mt, tok, positions, layers):
    """{layer: (len(positions), d)} — each decoder layer's OUTPUT residual at ``positions`` (CPU float32)."""
    names = {l: mt.layer_name_format.format(l) for l in layers}
    with torch.no_grad(), baukit.TraceDict(mt._model, list(names.values()), retain_output=True) as tr:
        mt._model(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"])
    out = lambda o: o[0] if isinstance(o, tuple) else o
    return {l: out(tr[n].output)[0, positions].detach().float().cpu() for l, n in names.items()}


@contextmanager
def residual_patch(mt, cache, positions, layers):
    """Overwrite each decoder layer's OUTPUT residual at absolute ``positions`` with ``cache[layer]`` (rows in
    ``positions`` order). Forwards that do not contain the positions (cached decode steps) are left alone."""
    handles = []

    def make(src):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if h.shape[1] <= max(positions):
                return out
            h = h.clone()
            h[:, positions] = src.to(h.device, h.dtype)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

        return hook

    try:
        for l in layers:
            mod = baukit.get_module(mt._model, mt.layer_name_format.format(l))
            handles.append(mod.register_forward_hook(make(cache[l])))
        yield
    finally:
        for hd in handles:
            hd.remove()


# ══════════════════════════════════════════════════════════════════════════════
# geometry: token sequences + the transplant window + the I2 window + p_res
# ══════════════════════════════════════════════════════════════════════════════
def _ids(tok):
    return tok["input_ids"][0].tolist()


def inst_key(inst) -> str:
    return (f"multi|{inst.datatype}|shared={'-'.join(p.id for p in inst.shared)}"
            f"|src={inst.src_frontier.id}|dst={inst.dst_frontier.id}")


def res_candidates(inst):
    """Predicates allowed as p_res's frontier e: not a shared one, not c, not d, and compatible with all of them."""
    dt = inst.datatype
    used = [p.id for p in inst.shared] + [inst.src_frontier.id, inst.dst_frontier.id]
    return [p for p in DATA.QTASKS[dt].predicates
            if p.id not in used and all(DATA.qpred_compatible(dt, p.id, u) and DATA.qpred_compatible(dt, u, p.id)
                                        for u in used)]


def pick_res_frontier(inst, res_frontier=None, seed=0):
    """e: ``res_frontier`` (an id) if given, else a pick seeded by the instance key (stable across configs)."""
    cands = res_candidates(inst)
    assert cands, f"no allowed p_res frontier for {inst_key(inst)}"
    if res_frontier is not None:
        e = next((p for p in cands if p.id == res_frontier), None)
        assert e is not None, f"{res_frontier!r} not allowed; choose from {[p.id for p in cands]}"
        return e
    h = int(hashlib.md5(f"{seed}|{inst_key(inst)}".encode()).hexdigest()[:8], 16)
    return random.Random(h).choice(cands)


def build_geometry(inst, *, window="span", span="list", include_pivot=True, i2_from="im_end", res_frontier=None,
                   res_seed=0, ctx):
    """Token sequences and positions for one (instance, window) pair. See the module docstring."""
    mt = ctx.mt
    m = DATA.materialize(inst, ctx=ctx)
    c_desc, d_desc = inst.src_frontier.description, inst.dst_frontier.description
    txt_s, txt_d = (m["src_task_text"], m["dst_task_text"]) if span == "list" else (c_desc, d_desc)
    assert m["src_prompt"].count(txt_s) == 1 and m["dst_prompt"].count(txt_d) == 1, "task span text not unique"
    s0, d0 = FUNC.token_span(m["src_prompt"], txt_s, ctx=ctx), FUNC.token_span(m["dst_prompt"], txt_d, ctx=ctx)
    Ls = len(_ids(prepare_input(prompts=m["src_prompt"], tokenizer=mt)))
    Ld = len(_ids(prepare_input(prompts=m["dst_prompt"], tokenizer=mt)))
    if window == "span":
        s_win, d_win = s0, d0
    elif window == "span+progress":   # ONE contiguous window: task-span start .. ` =` (or just before it)
        cut = 0 if include_pivot else 1
        s_win, d_win = (s0[0], Ls - cut), (d0[0], Ld - cut)
    else:
        raise ValueError(f"window must be 'span' | 'span+progress', got {window!r}")
    src_tok, dst_tok, src_w, dst_w, _, _ = FUNC.align_kv_spans(
        mt, m["src_prompt"], m["dst_prompt"], s_win, d_win, mode="pad", ctx=ctx)
    assert len(src_w) == len(dst_w)
    d_ids = _ids(dst_tok)
    L = len(d_ids)
    pad_s, pad_d = len(_ids(src_tok)) - Ls, L - Ld

    # I2 window: <|im_end|> (or the end of DST's task span) .. {var}; the pivot ` =` (last token) excluded
    im_end = mt.tokenizer.convert_tokens_to_ids("<|im_end|>")
    lo = max(i for i, t in enumerate(d_ids) if t == im_end) if i2_from == "im_end" else d0[1] + pad_d
    seg = list(range(lo, L - 1))

    # p_res: a, b, e in DST's language -> its <|im_end|> .. {var} is token-identical to DST's (aligned from the END)
    e = pick_res_frontier(inst, res_frontier, seed=res_seed)
    inst_res = dataclasses.replace(inst, src_frontier=e, src_lang=inst.dst_lang)
    mres = DATA.materialize(inst_res, ctx=ctx)
    res_tok = prepare_input(prompts=mres["src_prompt"], tokenizer=mt)
    r_ids = _ids(res_tok)
    res_seg = [len(r_ids) - L + q for q in seg]
    assert [r_ids[q] for q in res_seg] == [d_ids[q] for q in seg], "p_res / p_dst progress tokens differ"
    return dict(
        window=window, src_tok=src_tok, dst_tok=dst_tok, src_w=list(src_w), dst_w=list(dst_w), seg=seg,
        res_tok=res_tok, res_seg=res_seg, e=e.id, c_desc=c_desc, d_desc=d_desc, e_desc=e.description,
        pad_src=pad_s, pad_dst=pad_d, src_prompt=m["src_prompt"], dst_prompt=m["dst_prompt"], res_prompt=mres["src_prompt"],
    )


def geometry_info(g) -> dict:
    """JSON-native summary of a geometry (for the saved record)."""
    return dict(window=g["window"], n_window=len(g["dst_w"]), dst_window=[g["dst_w"][0], g["dst_w"][-1]],
                src_window=[g["src_w"][0], g["src_w"][-1]], i2_window=[g["seg"][0], g["seg"][-1]],
                pad_src=g["pad_src"], pad_dst=g["pad_dst"], L_src=g["src_tok"]["input_ids"].shape[1],
                L_dst=g["dst_tok"]["input_ids"].shape[1], res_frontier=g["e"], res_frontier_desc=g["e_desc"])


# ══════════════════════════════════════════════════════════════════════════════
# runs
# ══════════════════════════════════════════════════════════════════════════════
def clean_runs(g, *, readout_heads, max_new=64, ctx):
    """Clean SRC / DST greedy generations + GoalScope readouts at the pivot, on the geometry's token sequences."""
    out = {}
    for k in ("src", "dst"):
        tok = g[f"{k}_tok"]
        out[f"gen_{k}"] = FUNC.greedy(tok, max_new=max_new, ctx=ctx)
        out[f"ro_{k}"] = FUNC.goalscope_readout("", tok, tok["input_ids"].shape[1] - 1, heads=readout_heads,
                                                ctx=ctx)[0]["patched_gen"]
    return out


def run_patched(g, *, task_heads, goal_heads, readout_heads, i1=False, i2=True, i3=True, i1_query="all",
                i3_query="last", blind_self=False, alpha=1.0, donor="res", i2_layers=None, max_new=64, ctx):
    """One patched DST run (prompt ending with ` =`): greedy generation + GoalScope readout at ` =` under the enabled
    interventions. Returns ``gen_patched`` / ``ro_patched``."""
    mt = ctx.mt
    i2_layers = list(range(mt.n_layer)) if i2_layers is None else list(i2_layers)
    routes = ([(task_heads, i1_query)] if i1 else []) + ([(goal_heads, i3_query, {"blind_self": blind_self})] if i3 else [])
    if routes:   # one SRC k/v cache over the kv-heads of every enabled route
        hs = [h for r in routes for h in r[0]]
        layers = sorted(FUNC.heads_per_layer(hs, "k", layers={l for l, _ in hs}, ctx=ctx))
        k_c = FUNC.cache_src_proj(g["src_tok"], g["src_w"], layers, "k", ctx=ctx)
        v_c = FUNC.cache_src_proj(g["src_tok"], g["src_w"], layers, "v", ctx=ctx)
    restore = None
    if i2:
        restore = (layer_outputs(mt, g["res_tok"], g["res_seg"], i2_layers) if donor == "res"
                   else layer_outputs(mt, g["dst_tok"], g["seg"], i2_layers) if donor == "dst" else None)
        assert restore is not None, f"donor must be 'res' | 'dst', got {donor!r}"

    def stack():
        s = ExitStack()
        if routes:
            s.enter_context(FUNC.headwise_kv_patch_routes(mt, routes, k_c, v_c, g["dst_w"]))
        if i2:
            s.enter_context(residual_patch(mt, restore, g["seg"], i2_layers))
        if i3 and alpha != 1.0:
            s.enter_context(FUNC.boost_head_ov(mt, goal_heads, alpha, positions=(-1,)))
        return s

    def iface(mt_, inputs, patches=None, model_kwargs=None):
        with stack():
            return mt_._model(**dict(inputs), **dict(model_kwargs or {}))

    with stack():
        gen = FUNC.greedy(g["dst_tok"], max_new=max_new, ctx=ctx)
    L = g["dst_tok"]["input_ids"].shape[1]
    ro = FUNC.goalscope_readout("", g["dst_tok"], L - 1, heads=readout_heads, iface=iface, ctx=ctx)[0]["patched_gen"]
    return dict(gen_patched=gen, ro_patched=ro)
