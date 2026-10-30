"""Task vs progress: which span does the pivot read the goal from? (batch driver
``scripts/diff/task_vs_progress_eval.py``).

    p_src: a'b'c -> c     source (k/v or residual donor)
    p_dst: a'b'd -> d     the patched run (prompt ends at the eq pivot `` =``)

Two spans of DST can be read from SRC, giving a 2 x 2 grid (none / task / progress / task+progress):

* **task** — the task list ``(1) .. (3)`` + its closing ``\\n`` (``PP.build_geometry(span="task")``: up to the user
  turn's ``<|im_end|>``, excluded; ``\\n``-padded at the front to equal length);
* **progress** — ``<|im_end|> .. {var}`` (``g["seg"]``; token-identical in SRC and DST, aligned from the end).

Methods (:func:`make_stack`):

* ``kv`` — I3 only: the chosen heads (goal-250 | all) read SRC's span k/v from the pivot on (values x ``v_scale``);
  every pre-pivot latent stays clean DST.
* ``residual`` — the span's input embeddings and every layer's output <- SRC; every other position from the span start to
  pivot-1 is pinned to clean DST (so "task, no progress" keeps DST's progress latents exactly). Equals ``kv`` on all
  heads at V x 1.

The pivot is never overwritten: it computes freely under the intervention (hence the readout can shift)."""

import hashlib
import json
import os
from contextlib import ExitStack, contextmanager

import torch

import src.diff.functional as FUNC
import src.diff.path_patch as PP

# (method, heads, span, V) — "none" = the clean DST run (no-op corner of the grid)
CONFIGS = (
    [("none", None, "none", 1.0)]
    + [("kv", h, s, v) for h in ("goal250", "all") for s in ("task", "progress", "task+progress") for v in (1.0, 2.0, 4.0)]
    + [("residual", None, s, 1.0) for s in ("task", "progress", "task+progress")]
)


def cfg_name(m, h, s, v):
    return "clean DST (no-op)" if m == "none" else f"{m}{'·' + h if h else ''} · {s} · V×{v:g}"


def geometry(inst, *, ctx):
    """SRC / DST tokens and the span windows ``W[span] = (src_positions, dst_positions)`` (paired 1:1 in order)."""
    g = PP.build_geometry(inst, span="task", ctx=ctx)
    s_ids, d_ids = PP._ids(g["src_tok"]), PP._ids(g["dst_tok"])
    L_s, L_d = len(s_ids), len(d_ids)
    prog_d = list(g["seg"])                                   # <|im_end|> .. pivot-1 on DST
    prog_s = [q - L_d + L_s for q in prog_d]
    assert [s_ids[q] for q in prog_s] == [d_ids[q] for q in prog_d], "SRC/DST progress tokens differ"
    task_s, task_d = list(g["src_w"]), list(g["dst_w"])
    assert len(task_s) == len(task_d) and max(task_d) + 1 == min(prog_d), "task span must end right before the progress"
    W = {"none": ([], []), "task": (task_s, task_d), "progress": (prog_s, prog_d),
         "task+progress": (task_s + prog_s, task_d + prog_d)}
    return dict(inst=inst, src_tok=g["src_tok"], dst_tok=g["dst_tok"], W=W, piv_d=L_d - 1)


@contextmanager
def embedding_patch(mt, src_ids, dst_pos):
    """Replace the input-embedding output at `dst_pos` with the embeddings of `src_ids` (prompt pass only)."""
    emb = mt._model.get_input_embeddings()
    with torch.no_grad():
        e_src = emb(src_ids.to(emb.weight.device))

    def hook(module, inp, out):
        if out.shape[1] <= max(dst_pos):
            return out
        out = out.clone()
        out[:, dst_pos] = e_src.to(out.dtype)
        return out

    h = emb.register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


def make_stack(G, method, span, heads=None, v_scale=1.0, *, headsets, ctx):
    """``() -> ExitStack`` (single use per forward / generation) for one config. ``headsets`` maps a heads name to
    ``[(layer, head), ...]``; the residual method patches every layer."""
    mt = ctx.mt
    layers = list(range(mt.n_layer))
    src_pos, dst_pos = G["W"][span]
    if method == "none" or span == "none":
        return ExitStack
    if method == "kv":    # I3 only: the chosen heads read SRC's span k/v from the pivot on (values x v_scale)
        gk = dict(src_tok=G["src_tok"], dst_tok=G["dst_tok"], src_w=src_pos, dst_w=dst_pos)
        return PP.intervention_stack(gk, goal_heads=headsets[heads], v_scale=v_scale, ctx=ctx)
    if method == "residual":   # span <- SRC; every other position between the span and the pivot <- clean DST
        assert v_scale == 1.0, "the residual method has no value boost (use kv with heads='all')"
        span_set = set(dst_pos)
        pin = [q for q in range(min(dst_pos), G["piv_d"]) if q not in span_set]
        c_src = PP.layer_outputs(mt, G["src_tok"], src_pos, layers)
        c_dst = PP.layer_outputs(mt, G["dst_tok"], pin, layers) if pin else None
        pos = list(dst_pos) + pin
        cache = {l: torch.cat([c_src[l], c_dst[l]]) if pin else c_src[l] for l in layers}
        # layer 0 reads k/v from the token EMBEDDINGS (not a layer output), so the span's embeddings are swapped too:
        # without this the pivot's layer-0 attention would still see DST's task tokens (progress tokens are identical)
        src_ids = G["src_tok"]["input_ids"][0, src_pos]

        def stack():
            s = ExitStack()
            s.enter_context(embedding_patch(mt, src_ids, dst_pos))
            s.enter_context(PP.residual_patch(mt, cache, pos, layers))
            return s
        return stack
    raise ValueError(method)


def run_patched(G, stack, *, max_new=64, ctx):
    """Greedy patched DST generation."""
    with stack():
        return FUNC.greedy(G["dst_tok"], max_new=max_new, ctx=ctx)


def judge(G, clean, gen, model_name="claude-sonnet-4-6"):
    causal, _ = PP.judge(G["inst"], dict(**clean, gen_patched=gen, ro_patched=None), model_name=model_name)
    return causal


# ══════════════════════════════════════════════════════════════════════════════
# resumable sweep (shared by the notebook and the batch driver)
# ══════════════════════════════════════════════════════════════════════════════
def cache_dir(results_dir, model_short, datatype, src_lang, dst_lang, *, seed, max_new):
    # `ro0` (no GoalScope readout) and `task+gap` (the task span) are kept for cached-result compatibility
    return os.path.join(results_dir, "notebook_caches", "task_vs_progress", model_short,
                        f"{PP._safe(datatype)}__{src_lang}-{dst_lang}__seed{seed}__mx{max_new}__ro0__task+gap")


def cache_path(cdir, inst):
    ik = PP.inst_key(inst)
    return os.path.join(cdir, f"{inst.src_frontier.id}__{inst.dst_frontier.id}__{hashlib.md5(ik.encode()).hexdigest()[:10]}.json")


def cache_write(fp, entry):
    tmp = fp + ".tmp"
    with open(tmp, "w") as f:
        json.dump(entry, f, indent=1)
    os.replace(tmp, fp)


def sweep_instance(inst, idx, configs, cdir, *, headsets, max_new=64, oracle_model="claude-sonnet-4-6", resume=True,
                   ctx):
    """Every config on one instance, cached per config in ``cdir`` (resumable). Returns ``({name: row}, n_run, n_hit)``."""
    fp = cache_path(cdir, inst)
    entry = json.load(open(fp)) if (resume and os.path.exists(fp)) else {}
    if entry.get("key") not in (None, PP.inst_key(inst)):
        entry = {}                                             # hash collision / stale file: recompute
    entry.update(key=PP.inst_key(inst), idx=idx)
    entry.setdefault("configs", {})
    G = clean = None
    rows, n_run, n_hit = {}, 0, 0
    for m, h, s, v in configs:
        name = cfg_name(m, h, s, v)
        row = entry["configs"].get(name)
        if row is None:                                        # not cached: run it (geometry + clean runs only once)
            if G is None:
                G = geometry(inst, ctx=ctx)
                clean = entry.get("clean")
                if clean is None:
                    clean = PP.clean_runs(dict(src_tok=G["src_tok"], dst_tok=G["dst_tok"]), readout_heads=None,
                                          max_new=max_new, readout=False, ctx=ctx)
                    entry["clean"] = clean
            gen = (clean["gen_dst"] if m == "none" else
                   run_patched(G, make_stack(G, m, s, h, v, headsets=headsets, ctx=ctx), max_new=max_new, ctx=ctx))
            causal = judge(G, clean, gen, model_name=oracle_model)
            # readout / ro_shifted / ro (no readout: None) keep the cached rows' schema
            row = dict(idx=idx, method=m, heads=h, span=s, v=v, causal=causal["label"], shifted=causal["label"] == "src",
                       readout=None, ro_shifted=None, gen=gen, ro=None, c=inst.src_frontier.id, d=inst.dst_frontier.id)
            entry["configs"][name] = row
            cache_write(fp, entry)                             # after every config, so an interrupt loses at most one
            n_run += 1
        else:
            n_hit += 1
        rows[name] = dict(row, idx=idx)
    return rows, n_run, n_hit
