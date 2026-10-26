"""Path patching on the multi-predicate kv-transfer setup (notebook ``notebooks/diff/05_kv_transfer.ipynb``,
sections "Path patching" / "Path-patching variants"; batch driver ``scripts/diff/path_patch_eval.py``).

Three prompts sharing the prefilled progress (a, b) — DST's and p_res's progress prefill is token-identical:

    p_src: a'b'c -> c     k/v donor (I1, I3)
    p_dst: a'b'd -> d     the patched run
    p_res: a'b'e -> e     residual donor for I2 (frontier e not in {c, d}; rendered in DST's language)

Interventions on the patched DST run (each optional):

* **I1, task-onwards KV transfer** — ``task_heads`` read SRC's window k/v at query rule ``i1_query`` (default "all": the queries inside and after
  the window, incl. every generated token).
* **I2, pre-commitment-swap residual transfer** — over ``<|im_end|> .. {var}`` (the pivot ` =` excluded) every decoder layer's OUTPUT residual is overwritten
  by the donor's (``p_res`` by default, or the clean DST run) on the prompt pass; generated tokens see it via the cache.
  If the next pending operation is already decided in the progress latents, this swaps DST's pre-commitment for p_res's.
* **I3, pivot-onwards KV transfer** — ``goal_heads`` read SRC's window k/v from the pivot on (rule "last"), optionally blind to their own key.

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
import src.diff.eval as E
import src.diff.functional as FUNC
import src.diff.hier_data as H
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


def _safe(dt: str) -> str:
    return dt.replace("[", "_").replace("]", "").replace(" ", "")


def _md5_int(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


def sample_multi_instances(datatype, src_lang, dst_lang, n_samples=32, seed=1123):
    """The batch eval's instances (``scripts/diff/path_patch_eval.py``): ``multi_predicate_setups`` (k=3, letters, var_eq,
    listed) with a per-(datatype, direction) seed. Returns ``(instances, sample_seed)``; instance i here is instance i
    there (same seed -> same draw)."""
    s = (seed + _md5_int(f"{_safe(datatype)}__{src_lang}-{dst_lang}")) % (2**32)
    insts = DATA.multi_predicate_setups(datatype, n_conditions=3, n_samples=n_samples, seed=s, presentation="listed",
                                        naming="letters", letter_stub="var_eq", src_lang=src_lang, dst_lang=dst_lang)
    return insts, s


def sample_hier_instances(datatype, connective, order, src_lang, dst_lang, n_samples=32, seed=1123):
    """The multi-hierarchy kv eval's instances (``scripts/diff/hier_kv_eval.py``, via
    :func:`src.diff.hier_data.seeded_setups`). Returns ``(instances, sample_seed)``."""
    return H.seeded_setups(datatype, connective, order, src_lang, dst_lang, n_samples=n_samples, seed=seed)


def inst_key(inst) -> str:
    if isinstance(inst, H.HierKVInstance):
        return H.inst_key(inst)
    return (f"multi|{inst.datatype}|shared={'-'.join(p.id for p in inst.shared)}"
            f"|src={inst.src_frontier.id}|dst={inst.dst_frontier.id}")


def _progress_text(prompt: str) -> str:
    """The prefill shared by DST and p_res: everything from the user turn's ``<|im_end|>`` on."""
    return prompt[prompt.rindex("<|im_end|>"):]


def res_instance(inst, e):
    """p_res for the candidate ``e``, rendered on its SRC side in DST's language.
    multi-predicate: frontier e (a q-bank predicate) -> a'b'e. multi-hierarchy: second condition e (a filter condition)
    -> a'(x <conn> e), spec order x-first."""
    if isinstance(inst, H.HierKVInstance):
        ft = DATA.FILTER_TASKS[inst.datatype]
        return dataclasses.replace(
            inst, src_frontier=H._frontier(ft, inst.var_name, inst.x, e, inst.connective), src_cond=e, order="match",
            src_lang=inst.dst_lang, meta={**inst.meta, "z": e.id, "order": "match", "src_lang": inst.dst_lang})
    return dataclasses.replace(inst, src_frontier=e, src_lang=inst.dst_lang)


def res_candidates(inst, ctx=None):
    """Allowed p_res contrasts e.
    multi-predicate: q-bank predicates other than the shared ones / c / d, compatible with all of them.
    multi-hierarchy: filter conditions other than x / y / z, compatible with x / y / z, not overlapping a, whose
    ``x <conn> e`` result differs from both ``x <conn> y`` and ``x <conn> z`` (an observable third sub-sub-goal), and
    whose rendered progress prefill is string-identical to DST's (C++ emptiness guards depend on all conditions)."""
    dt = inst.datatype
    if isinstance(inst, H.HierKVInstance):
        conds = {c.id: c for c in DATA.FILTER_TASKS[dt].conditions}
        x, y, z, conn = inst.x.id, inst.dst_cond.id, inst.src_cond.id, inst.connective
        cpp = "cpp" in (inst.src_lang, inst.dst_lang)
        ref = {tuple(H._filter((x, c), conn, dt, conds)) for c in (y, z)}
        overlap = H.A_OVERLAP[dt].get(inst.shared[0].id, set())
        dst_prog = _progress_text(DATA.materialize(inst, ctx=ctx)["dst_prompt"])
        out = []
        for w, cw in conds.items():
            if w in (x, y, z) or w in overlap or (cpp and w in H.CPP_EXCLUDE):
                continue
            if not all(DATA.filter_compatible(w, u) and DATA.filter_compatible(u, w) for u in (x, y, z)):
                continue
            rw = H._filter((x, w), conn, dt, conds)
            if not rw or tuple(rw) in ref:
                continue
            if _progress_text(DATA.materialize(res_instance(inst, cw), ctx=ctx)["src_prompt"]) != dst_prog:
                continue
            out.append(cw)
        return out
    used = [p.id for p in inst.shared] + [inst.src_frontier.id, inst.dst_frontier.id]
    return [p for p in DATA.QTASKS[dt].predicates
            if p.id not in used and all(DATA.qpred_compatible(dt, p.id, u) and DATA.qpred_compatible(dt, u, p.id)
                                        for u in used)]


def pick_res_frontier(inst, res_frontier=None, seed=0, ctx=None):
    """e: ``res_frontier`` (an id) if given, else a pick seeded by the instance key (stable across configs)."""
    cands = res_candidates(inst, ctx=ctx)
    assert cands, f"no allowed p_res frontier for {inst_key(inst)}"
    if res_frontier is not None:
        e = next((p for p in cands if p.id == res_frontier), None)
        assert e is not None, f"{res_frontier!r} not allowed; choose from {[p.id for p in cands]}"
        return e
    return random.Random(_md5_int(f"{seed}|{inst_key(inst)}")).choice(cands)


def build_geometry(inst, *, window="span", span="list", include_pivot=True, i2_from="im_end", res_frontier=None,
                   res_seed=0, ctx):
    """Token sequences and positions for one (instance, window) pair. See the module docstring."""
    mt = ctx.mt
    m = DATA.materialize(inst, ctx=ctx)
    c_desc, d_desc = inst.expected_next_desc, inst.dst_next_desc          # what the oracle judges against
    txt_s, txt_d = ((m["src_task_text"], m["dst_task_text"]) if span in ("list", "list+gap")
                    else (inst.src_frontier.description, inst.dst_frontier.description))
    assert m["src_prompt"].count(txt_s) == 1 and m["dst_prompt"].count(txt_d) == 1, "task span text not unique"
    s0, d0 = FUNC.token_span(m["src_prompt"], txt_s, ctx=ctx), FUNC.token_span(m["dst_prompt"], txt_d, ctx=ctx)
    ids_s = _ids(prepare_input(prompts=m["src_prompt"], tokenizer=mt))
    ids_d = _ids(prepare_input(prompts=m["dst_prompt"], tokenizer=mt))
    Ls, Ld = len(ids_s), len(ids_d)
    im_end = mt.tokenizer.convert_tokens_to_ids("<|im_end|>")
    n_gap = 0
    if span == "list+gap":   # task span = `(1)` .. the user turn's <|im_end|> (excluded): the list + everything after it
        e_s = max(i for i, t in enumerate(ids_s) if t == im_end)
        e_d = max(i for i, t in enumerate(ids_d) if t == im_end)
        n_gap = e_d - d0[1]
        s0, d0 = (s0[0], e_s), (d0[0], e_d)
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

    # I2 window: <|im_end|> (or the end of DST's task span) .. the token before the pivot (` =`; ` and` / ` &&` ...)
    lo = max(i for i, t in enumerate(d_ids) if t == im_end) if i2_from == "im_end" else d0[1] + pad_d
    seg = list(range(lo, L - 1))

    # p_res in DST's language -> its <|im_end|> .. pivot-1 is token-identical to DST's (aligned from the END)
    e = pick_res_frontier(inst, res_frontier, seed=res_seed, ctx=ctx)
    inst_res = res_instance(inst, e)
    mres = DATA.materialize(inst_res, ctx=ctx)
    res_tok = prepare_input(prompts=mres["src_prompt"], tokenizer=mt)
    r_ids = _ids(res_tok)
    res_seg = [len(r_ids) - L + q for q in seg]
    assert [r_ids[q] for q in res_seg] == [d_ids[q] for q in seg], "p_res / p_dst progress tokens differ"
    return dict(
        window=window, span=span, n_gap=n_gap, src_tok=src_tok, dst_tok=dst_tok, src_w=list(src_w), dst_w=list(dst_w), seg=seg,
        res_tok=res_tok, res_seg=res_seg, e=e.id, c_desc=c_desc, d_desc=d_desc, e_desc=inst_res.expected_next_desc,
        pad_src=pad_s, pad_dst=pad_d, src_prompt=m["src_prompt"], dst_prompt=m["dst_prompt"], res_prompt=mres["src_prompt"],
    )


def geometry_info(g) -> dict:
    """JSON-native summary of a geometry (for the saved record)."""
    return dict(window=g["window"], span=g.get("span"), n_gap=g.get("n_gap", 0), n_window=len(g["dst_w"]), dst_window=[g["dst_w"][0], g["dst_w"][-1]],
                src_window=[g["src_w"][0], g["src_w"][-1]], i2_window=[g["seg"][0], g["seg"][-1]],
                pad_src=g["pad_src"], pad_dst=g["pad_dst"], L_src=g["src_tok"]["input_ids"].shape[1],
                L_dst=g["dst_tok"]["input_ids"].shape[1], res_frontier=g["e"], res_frontier_desc=g["e_desc"])


# ══════════════════════════════════════════════════════════════════════════════
# runs
# ══════════════════════════════════════════════════════════════════════════════
def clean_runs(g, *, readout_heads, max_new=64, readout=True, ctx):
    """Clean SRC / DST greedy generations + GoalScope readouts at the pivot, on the geometry's token sequences
    (``readout=False``: generations only, ``ro_*`` = None)."""
    out = {}
    for k in ("src", "dst"):
        tok = g[f"{k}_tok"]
        out[f"gen_{k}"] = FUNC.greedy(tok, max_new=max_new, ctx=ctx)
        out[f"ro_{k}"] = (FUNC.goalscope_readout("", tok, tok["input_ids"].shape[1] - 1, heads=readout_heads,
                                                 ctx=ctx)[0]["patched_gen"] if readout else None)
    return out


def res_readout(g, *, readout_heads, ctx):
    """Clean p_res GoalScope readout at its pivot (validity baseline for the 3-way readout judge)."""
    tok = g["res_tok"]
    return FUNC.goalscope_readout("", tok, tok["input_ids"].shape[1] - 1, heads=readout_heads, ctx=ctx)[0]["patched_gen"]


def intervention_stack(g, *, task_heads, goal_heads, i1=False, i2=True, i3=True, i1_query="all", i3_query="last",
                       blind_self=False, v_scale=1.0, alpha=1.0, donor="res", i2_layers=None, ctx):
    """``() -> ExitStack`` with the enabled interventions (single-use: call it once per forward / generation).
    I1 (task-onwards) / I3 (pivot-onwards) KV transfer = one SRC window k/v transplant read by two head routes;
    I2 (pre-commitment-swap residual transfer) = residual overwrite on ``g["seg"]`` from the
    donor; ``alpha`` = OV boost of the I3 heads at -1. Used by :func:`run_patched` and by the notebooks' sanity checks."""
    mt = ctx.mt
    i2_layers = list(range(mt.n_layer)) if i2_layers is None else list(i2_layers)
    routes = ([(task_heads, i1_query)] if i1 else []) + ([(goal_heads, i3_query, {"blind_self": blind_self, "v_scale": v_scale})] if i3 else [])
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

    return stack


def run_patched(g, *, task_heads, goal_heads, readout_heads, i1=False, i2=True, i3=True, i1_query="all",
                i3_query="last", blind_self=False, v_scale=1.0, alpha=1.0, donor="res", i2_layers=None, max_new=64,
                readout=True, ctx):
    """One patched DST run (prompt ending at the pivot): greedy generation + GoalScope readout at the pivot under the
    enabled interventions (:func:`intervention_stack`). ``v_scale`` multiplies the transplanted SRC window VALUES read
    by the I3 heads (keys / attention pattern unchanged; 1.0 = off). Returns ``gen_patched`` / ``ro_patched``."""
    stack = intervention_stack(g, task_heads=task_heads, goal_heads=goal_heads, i1=i1, i2=i2, i3=i3, i1_query=i1_query,
                               i3_query=i3_query, blind_self=blind_self, v_scale=v_scale, alpha=alpha, donor=donor,
                               i2_layers=i2_layers, ctx=ctx)

    def iface(mt_, inputs, patches=None, model_kwargs=None):
        with stack():
            return mt_._model(**dict(inputs), **dict(model_kwargs or {}))

    with stack():
        gen = FUNC.greedy(g["dst_tok"], max_new=max_new, ctx=ctx)
    L = g["dst_tok"]["input_ids"].shape[1]
    ro = (FUNC.goalscope_readout("", g["dst_tok"], L - 1, heads=readout_heads, iface=iface, ctx=ctx)[0]["patched_gen"]
          if readout else None)
    return dict(gen_patched=gen, ro_patched=ro)


# ══════════════════════════════════════════════════════════════════════════════
# oracle (the batch eval's scoring)
# ══════════════════════════════════════════════════════════════════════════════
_JOINT2 = {"native": "dst", "src_target": "src", "diff_target": "res", "neither": "neither", "parse_error": "parse_error"}


def eval_readouts_res(inst, e_desc, patched_readout, res_readout_text, model_name="claude-sonnet-4-6"):
    """3-way (mutually exclusive) verdicts over {d = DST's next, c = SRC's, e = p_res's} for the patched DST readout and
    the clean p_res readout. ``eval_readouts`` judges each readout c-vs-d only, so a readout that expresses e would come
    out "neither" there. ``res_valid`` = the clean p_res readout expresses e; ``shifted_to_res`` = the patched one does."""
    kind = "a short natural-language phrase the model 'read out' internally"

    def j(text):
        r = E.classify_joint(text, native_desc=inst.dst_next_desc, src_desc=inst.expected_next_desc,
                             diff_desc=e_desc, task_desc=inst.task_desc, kind=kind, model_name=model_name)
        return dict(r, label=_JOINT2[r["label"]])

    pat, res = j(patched_readout), j(res_readout_text)
    return dict(patched=pat["label"], res_clean=res["label"], res_valid=res["label"] == "res",
                shifted_to_res=pat["label"] == "res", details=dict(patched=pat, res_clean=res))


def judge(inst, raw, e_desc=None, model_name="claude-sonnet-4-6"):
    """The batch eval's oracle: ``(causal, readout, readout_res)``. causal = ``judge_causal`` on the patched generation
    (first computation; BOTH is a miss); readout = ``eval_readouts`` (c-vs-d, BOTH is a miss); readout_res = the 3-way
    {d, c, e} verdicts, only when ``raw`` carries the clean p_res readout ``ro_res`` (I2 with the p_res donor)."""
    causal = E.judge_causal(inst, raw["gen_patched"], model_name=model_name)
    readout = (E.eval_readouts(inst, raw["ro_src"], raw["ro_dst"], raw["ro_patched"], model_name=model_name)
               if raw.get("ro_patched") is not None else None)       # None when run without readouts
    readout_res = (eval_readouts_res(inst, e_desc, raw["ro_patched"], raw["ro_res"], model_name)
                   if raw.get("ro_res") is not None and e_desc else None)
    return causal, readout, readout_res
