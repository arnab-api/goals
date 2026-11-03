"""A/B of the NRGS port: the hook implementations (forward swap, q/k/v and o_proj hooks, mask pre-hook, manual
decode loops), loaded from git, against the nnterp interventions, in one process on Qwen/Qwen2.5-0.5B-Instruct
(CPU, float32, eager attention; 14 query heads over 2 kv heads, so GQA groups of 7).

Run from the repo root (a few minutes on CPU):

    JE_ARROW_MALLOC_CONF=background_thread:false HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="" \\
        .venv/bin/python test/test_nrgs_port.py

The old implementations are ``git show <OLD_REF>:<path>`` of the files in ``OLD_FILES``, written to a temp directory
(under ``$TMPDIR``) with their imports of each other pointed at the old copies; PR 1's ``_model`` shim makes them run
on the nnterp ``ModelandTokenizer``. ``OLD_REF`` defaults to the ``nnterp`` branch before the GoalScope and NRGS
ports (``NRGS_OLD_REF`` overrides it). ``src.nrgs.data`` / ``eval`` / ``hier_data`` are shared: the port does not
touch them. Every comparison feeds both sides identical inputs and passes ``ctx`` explicitly.

The functions are also pytest tests (``pytest test/test_nrgs_port.py``, if pytest is installed).
"""

import contextlib
import importlib
import os
import re
import subprocess
import sys
import tempfile
import time
import types

sys.path.append(".")

import nnterp  # noqa: E402  (before any transformers.models.* import)
import nnsight  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402

import src.nrgs.data as DATA  # noqa: E402
import src.nrgs.functional as FUNC  # noqa: E402
import src.nrgs.kv_transfer as PP  # noqa: E402
import src.nrgs.task_vs_progress as TP  # noqa: E402
from src.attention import get_attention_matrices  # noqa: E402
from src.goals.head_loc_attn import get_o_proj_inputs_batch  # noqa: E402
from src.models import ModelandTokenizer  # noqa: E402
from src.tokens import prepare_input  # noqa: E402

MODEL_KEY = "Qwen/Qwen2.5-0.5B-Instruct"
OLD_REF = os.environ.get("NRGS_OLD_REF", "e1d9701f2d9e0bfc72a7875dc8445d33927930d9")
OLD_FILES = {
    "old_functional": "src/functional.py",
    "old_head_loc_attn": "src/goals/head_loc_attn.py",
    "old_goalscope": "src/goals/goalscope.py",
    "old_attention": "src/attention.py",
    "old_nrgs_functional": "src/nrgs/functional.py",
    "old_kv_transfer": "src/nrgs/kv_transfer.py",
    "old_task_vs_progress": "src/nrgs/task_vs_progress.py",
}
# `from src.X import ...` and `import src.X as Y` forms, both pointed at the old copies
_OLD_IMPORTS = {
    "src.functional": "old_functional",
    "src.goals.head_loc_attn": "old_head_loc_attn",
    "src.goals.goalscope": "old_goalscope",
    "src.attention": "old_attention",
    "src.nrgs.functional": "old_nrgs_functional",
    "src.nrgs.kv_transfer": "old_kv_transfer",
}

MAX_NEW = 24
SPARSE = [(6, 2), (6, 9), (10, 0), (12, 5), (12, 6), (15, 13), (17, 1)]  # 7 heads, kv groups shared and not
DENSE = [(l, h) for l in range(6, 19) for h in range(14)]  # every head of layers 6-18 (182 heads)

_MT = None
_OLD = None
_GEOM = {}
RESULTS: list[tuple[str, str]] = []  # (comparison, outcome) rows for the summary table


def model() -> ModelandTokenizer:
    global _MT
    if _MT is None:
        _MT = ModelandTokenizer(MODEL_KEY, dtype=torch.float32, attn_implementation="eager")
    return _MT


def ctx() -> FUNC.Ctx:
    return FUNC.Ctx(mt=model())


def old() -> types.SimpleNamespace:
    """The old modules, imported from ``OLD_REF`` (cached for the process)."""
    global _OLD
    if _OLD is None:
        root = tempfile.mkdtemp(prefix="nrgs_old_")
        names = "|".join(map(re.escape, _OLD_IMPORTS))
        from_re = re.compile(r"\bfrom (" + names + r") import")
        import_re = re.compile(r"\bimport (" + names + r") as")
        for name, path in OLD_FILES.items():
            src = subprocess.run(["git", "show", f"{OLD_REF}:{path}"], capture_output=True,
                                 text=True, check=True).stdout
            src = from_re.sub(lambda m: f"from {_OLD_IMPORTS[m.group(1)]} import", src)
            src = import_re.sub(lambda m: f"import {_OLD_IMPORTS[m.group(1)]} as", src)
            with open(os.path.join(root, f"{name}.py"), "w") as f:
                f.write(src)
        sys.path.insert(0, root)
        _OLD = types.SimpleNamespace(
            **{name: importlib.import_module(name) for name in OLD_FILES}, root=root
        )
    return _OLD


@contextlib.contextmanager
def _cuda_is_noop():
    """The old ``get_attention_matrices`` moves every tensor with ``.cuda()``; on a CPU run that
    call becomes the identity (the arithmetic after it is unchanged)."""
    orig = torch.Tensor.cuda
    torch.Tensor.cuda = lambda self, *a, **k: self
    try:
        yield
    finally:
        torch.Tensor.cuda = orig


def _record(name: str, outcome: str) -> None:
    RESULTS.append((name, outcome))
    print(f"    {name}: {outcome}")


def _maxdiff(a, b) -> float:
    return (torch.as_tensor(a).float() - torch.as_tensor(b).float()).abs().max().item()


def _old_runner(stack):
    """The old ``patch_interface`` form of a hook context factory: run the raw forward under it."""
    def iface(mt_, inputs, patches=None, model_kwargs=None):
        with stack():
            return mt_._model(**dict(inputs), **dict(model_kwargs or {}))

    return iface


def geometry(lt: str) -> dict:
    """``build_geometry`` of instance 0 of the int multi-setup draw (seed 1123) for one language pair."""
    if lt not in _GEOM:
        old_note = DATA.CPP_NOTE_IN_SYSTEM
        DATA.CPP_NOTE_IN_SYSTEM = True  # as kv_transfer_eval.main sets it
        try:
            insts, _ = PP.sample_multi_instances("int", *PP.LANG_TRANSFERS[lt], n_samples=32, seed=1123)
            _GEOM[lt] = PP.build_geometry(insts[0], span="task", ctx=ctx())
        finally:
            DATA.CPP_NOTE_IN_SYSTEM = old_note
    return _GEOM[lt]


def _old_stack(g, heads, v):
    O, c = old(), ctx()
    return O.old_kv_transfer.intervention_stack(g, goal_heads=heads, v_scale=v, ctx=c)


# ---------------------------------------------------------------------------------------------
# KV transfer
# ---------------------------------------------------------------------------------------------


def test_kv_transfer():
    """``run_patched`` (readout off) and ``clean_runs`` on instance 0 of int py-py and int py-cpp (the cross-language
    pair puts the SRC window at other positions, so the source keys are RoPE'd at the DST positions), V x1 and x4, a
    sparse and a dense head set, 24 tokens: token-identical; the patched layers' o_proj inputs at the pivot
    (prompt pass) allclose 1e-5, the patched heads' slices and every other head."""
    mt, O, c = model(), old(), ctx()
    hd = mt.head_dim
    rows, worst_ho, n_moved = 0, 0.0, 0
    for lt in ("py-py", "py-cpp"):
        g = geometry(lt)
        a = O.old_kv_transfer.clean_runs(g, readout_heads=None, max_new=MAX_NEW, readout=False, ctx=c)
        b = PP.clean_runs(g, readout_heads=None, max_new=MAX_NEW, readout=False, ctx=c)
        assert a == b, (lt, a, b)
        for name, heads in (("sparse", SPARSE), ("dense", DENSE)):
            layers = sorted({l for l, _ in heads})
            for v in (1.0, 4.0):
                kw = dict(goal_heads=heads, readout_heads=heads, v_scale=v, max_new=MAX_NEW, readout=False, ctx=c)
                pa = O.old_kv_transfer.run_patched(g, **kw)
                pb = PP.run_patched(g, **kw)
                assert pa == pb, (lt, name, v, pa, pb)
                rows += 1
                n_moved += pb["gen_patched"] != b["gen_dst"]
                # the patched layers' o_proj inputs at the pivot, prompt pass
                oa = O.old_head_loc_attn.get_o_proj_inputs_batch(
                    mt, g["dst_tok"], layers, patch_interface=_old_runner(_old_stack(g, heads, v)))
                iv = PP.intervention_stack(g, goal_heads=heads, v_scale=v, ctx=c)
                ob = get_o_proj_inputs_batch(mt, g["dst_tok"], layers, interventions=iv)
                for l in layers:
                    torch.testing.assert_close(ob[l], oa[l], rtol=1e-5, atol=1e-5)
                    worst_ho = max(worst_ho, _maxdiff(oa[l], ob[l]))
                print(f"      {lt} {name:6s} V{v:g}: {pb['gen_patched'][:50]!r}")
    _record("KV transfer: `run_patched` + `clean_runs`, int py-py and py-cpp, sparse 7 / dense 182 heads, V x1 / x4, "
            f"{MAX_NEW} tok",
            f"token-identical {rows}/{rows} (+ clean SRC/DST 2/2 per pair; {n_moved}/{rows} differ from clean DST); "
            f"patched layers' o_proj inputs at the pivot: max |old - new| = {worst_ho:.1e}")


def test_gqa_bleed():
    """At the lowest patched layer, a NON-patched head in a patched kv group reads clean K/V: its output at the pivot
    equals the clean run's bit for bit, while the patched head moves. A per-kv-head K/V write (assigning
    ``attention_keys`` / ``attention_values``) would move the whole group."""
    mt, c = model(), ctx()
    g = geometry("py-py")
    group = mt.num_heads // mt.num_kv_heads
    L, h = 6, 2  # patched; head 3 shares kv head 0 and is not patched; head 9 is patched, kv head 1
    other = 3
    assert other // group == h // group and (L, other) not in SPARSE
    iv = PP.intervention_stack(g, goal_heads=SPARSE, v_scale=4.0, ctx=c)
    clean = get_o_proj_inputs_batch(mt, g["dst_tok"], [L])[L].unflatten(-1, (mt.num_heads, mt.head_dim))
    kv = get_o_proj_inputs_batch(mt, g["dst_tok"], [L], interventions=iv)[L].unflatten(-1, (mt.num_heads, mt.head_dim))
    assert torch.equal(kv[:, other], clean[:, other])
    moved = _maxdiff(kv[:, h], clean[:, h])
    assert moved > 1e-3
    # the bleed a kv-head-wide write causes: the source keys / values over the span for kv head h // group
    layers = sorted({l for l, _ in SPARSE})
    k_c = FUNC.cache_src_proj(g["src_tok"], g["src_w"], layers, "k", ctx=c)
    v_c = FUNC.cache_src_proj(g["src_tok"], g["src_w"], layers, "v", ctx=c)
    src = FUNC._rope_source_span(mt, k_c, v_c, g["dst_w"], {L: [h // group]})
    span = torch.tensor(g["dst_w"])
    attn = mt.layers[L].self_attn

    def kv_wide(m):
        k, v = attn.attention_keys.clone(), attn.attention_values.clone()
        k[:, h // group, span] = src[L][0][0]
        v[:, h // group, span] = src[L][1][0] * 4.0
        attn.attention_keys = k
        attn.attention_values = v

    bled = get_o_proj_inputs_batch(mt, g["dst_tok"], [L], interventions=kv_wide)[L].unflatten(-1, (mt.num_heads, mt.head_dim))
    torch.testing.assert_close(bled[:, h], kv[:, h], rtol=1e-5, atol=1e-5)  # same patched head
    bleed = _maxdiff(bled[:, other], clean[:, other])
    assert bleed > 1e-3
    _record("GQA bleed (layer 6, patched head 2, unpatched head 3 of the same kv group, V x4)",
            f"head 3 bit-identical to clean; head 2 moves by {moved:.3f}; a kv-head-wide K/V write would move "
            f"head 3 by {bleed:.3f}")


def test_kv_readout():
    """``goalscope_readout`` at the pivot under the KV stack (old: ``iface`` hook context through
    ``patch_interface``; new: the intervention through ``interventions``), additive_norm and replace, 8 tokens:
    identical strings."""
    mt, O, c = model(), old(), ctx()
    out = []
    for lt in ("py-py", "py-cpp"):
        g = geometry(lt)
        L = g["dst_tok"]["input_ids"].shape[1]
        ro_heads = [(13, 1), (14, 4), (16, 8), (17, 1), (18, 11)]
        for heads in (SPARSE, DENSE):
            for strategy in ("additive_norm", "replace"):
                kw = dict(heads=ro_heads, strategy=strategy, alpha=0.1 if strategy == "additive_norm" else 1.0,
                          layers=(8, 12), maxnew=8, ctx=c)
                a = O.old_nrgs_functional.goalscope_readout(
                    "", g["dst_tok"], L - 1, iface=_old_runner(_old_stack(g, heads, 4.0)), **kw)
                b = FUNC.goalscope_readout(
                    "", g["dst_tok"], L - 1, interventions=PP.intervention_stack(g, goal_heads=heads, v_scale=4.0, ctx=c),
                    **kw)
                assert a == b, (lt, strategy, a, b)
                out.append(b[0]["patched_gen"])
    plain = FUNC.goalscope_readout("", geometry("py-py")["dst_tok"], geometry("py-py")["dst_tok"]["input_ids"].shape[1] - 1,
                                   heads=[(13, 1), (14, 4), (16, 8), (17, 1), (18, 11)], layers=(8, 12), maxnew=8,
                                   ctx=c)[0]["patched_gen"]
    _record("`goalscope_readout` under the KV stack (py-py, py-cpp; sparse, dense; additive_norm, replace)",
            f"identical {len(out)}/{len(out)}; {sum(o != plain for o in out[:4])}/4 py-py readouts differ from the "
            f"unpatched one ({plain!r})")


# ---------------------------------------------------------------------------------------------
# Q transfer (gaze-select-restrict), OV boost
# ---------------------------------------------------------------------------------------------


def _q_instance():
    """Instance 0 of the order_diff int py-py cell, the cell seed computed as ``q_transfer_eval`` does."""
    import hashlib

    tag = "int__python-python"
    seed = int(hashlib.md5(f"order_diff/{tag}".encode()).hexdigest()[:8], 16) % (2**32)
    return DATA.diff_trigger_setups("int", variant="order_diff", n_samples=4, seed=seed, src_lang="python",
                                    dst_lang="python")


def test_q_transfer():
    """``run_q_transfer_intervention`` (compute_readout off), sparse and dense head sets, two instances: every raw
    field identical (generations, selection, restrict set) or allclose 1e-5 (gaze scores, pivot attention);
    ``on_gaze`` sees the three passes and reproduces the selection scores."""
    mt, O, c = model(), old(), ctx()
    n, worst, n_moved = 0, 0.0, 0
    for inst in _q_instance()[:2]:
        for heads in (SPARSE, DENSE):
            kw = dict(readout_heads=heads, max_new=MAX_NEW, alpha=2.0, compute_readout=False, gsr_max_new=MAX_NEW, ctx=c)
            with _cuda_is_noop():
                a = O.old_nrgs_functional.run_q_transfer_intervention(mt, inst, heads, **kw)
            seen = {}
            b = FUNC.run_q_transfer_intervention(mt, inst, heads, on_gaze=lambda k, i: seen.setdefault(k, i), **kw)
            assert set(a) == set(b)
            for k in a:
                if k in ("cl_select_scores",):
                    assert a[k].keys() == b[k].keys()
                    worst = max(worst, max(abs(a[k][x] - b[k][x]) for x in a[k]))
                    assert all(abs(a[k][x] - b[k][x]) < 1e-5 for x in a[k]), (k, a[k], b[k])
                elif k.startswith("attn_"):
                    worst = max(worst, _maxdiff(a[k], b[k]))
                    torch.testing.assert_close(torch.tensor(b[k]), torch.tensor(a[k]), rtol=1e-5, atol=1e-5)
                else:
                    assert a[k] == b[k], (k, a[k], b[k])
            assert list(seen) == ["select", "clean", "patched"]
            span = sorted({q for p in inst.task.predicates
                           for q in range(*FUNC.token_span(b["dst_prompt"], p.description, ctx=c))})
            at = dict(zip(span, FUNC.pivot_attn(seen["select"], heads, span, b["piv_dst"])))
            for p in inst.task.predicates:
                ps = list(range(*FUNC.token_span(b["dst_prompt"], p.description, ctx=c)))
                assert abs(sum(at[q] for q in ps) / len(ps) - b["cl_select_scores"][p.id]) < 1e-6
            n += 1
            n_moved += b["gen_patched"] != b["gen_dst"]
            print(f"      selected {b['cl_selected']} {b['cl_select_scores']}: {b['gen_patched'][:50]!r}")
    _record("Q transfer + GSR (`run_q_transfer_intervention`, 2 instances x sparse / dense)",
            f"identical {n}/{n} (generations, selection, restrict set; {n_moved}/{n} GSR generations differ from clean "
            f"DST); gaze scores and pivot attention max |old - new| = {worst:.1e}; `on_gaze` reproduces the scores")


def test_q_patched_readout():
    """The patched GoalScope readout of ``run_q_transfer_intervention``: the SRC pivot query at the DST pivot + an OV
    boost x2 of the patch heads (old: ``proj_patch_hooks`` + ``boost_head_ov`` hooks; new: ``proj_patch(...) +
    boost_head_ov(...)``), and the query patch alone: identical strings."""
    mt, O, c = model(), old(), ctx()
    inst = _q_instance()[0]
    m = DATA.materialize(inst, ctx=c)
    src_tok, dst_tok = (prepare_input(m[k], tokenizer=mt) for k in ("src_prompt", "dst_prompt"))
    piv_src, piv_dst = src_tok["input_ids"].shape[1] - 1, dst_tok["input_ids"].shape[1] - 1
    out = []
    for heads in (SPARSE, DENSE):
        q_layers = sorted({l for l, _ in heads})
        q_src = FUNC.cache_src_proj(m["src_prompt"], [piv_src], q_layers, "q", ctx=c)
        torch.testing.assert_close(  # the ported source cache is the old one
            torch.stack(list(q_src.values())),
            torch.stack(list(O.old_nrgs_functional.cache_src_proj(m["src_prompt"], [piv_src], q_layers, "q",
                                                                    ctx=c).values())), rtol=0, atol=0)
        pf = [dict(which="q", cache=q_src, heads=FUNC.heads_per_layer(heads), positions=[piv_dst])]
        boost = sorted(set(heads))

        @contextlib.contextmanager
        def old_hooks():
            with O.old_nrgs_functional.proj_patch_hooks(mt, pf), O.old_nrgs_functional.boost_head_ov(mt, boost, 2.0):
                yield

        kw = dict(heads=[(13, 1), (14, 4), (16, 8), (17, 1), (18, 11)], layers=(8, 12), maxnew=8, ctx=c)
        for name, old_ctx, new_iv in (
                ("q", lambda: O.old_nrgs_functional.proj_patch_hooks(mt, pf), FUNC.proj_patch(mt, pf)),
                ("q+boost", old_hooks, FUNC.proj_patch(mt, pf) + FUNC.boost_head_ov(mt, boost, 2.0))):
            a = O.old_nrgs_functional.goalscope_readout("", m["dst_prompt"], piv_dst, iface=_old_runner(old_ctx), **kw)
            b = FUNC.goalscope_readout("", m["dst_prompt"], piv_dst, interventions=new_iv, **kw)
            assert a == b, (name, a, b)
            out.append(f"{name} {b[0]['patched_gen']!r}")
    _record("q-transfer patched readout (query patch; query patch + OV boost x2), sparse / dense",
            "identical: " + "; ".join(out))


def test_ov_boost_generation():
    """OV boost x2 at the last token of every forward around a cached greedy decode (old: ``boost_head_ov`` hooks
    around ``greedy``; new: ``greedy(interventions=boost_head_ov(...))``), sparse and dense: token-identical."""
    mt, O, c = model(), old(), ctx()
    g = geometry("py-py")
    base = FUNC.greedy(g["dst_tok"], max_new=MAX_NEW, ctx=c)
    out = []
    for heads in (SPARSE, DENSE):
        with O.old_nrgs_functional.boost_head_ov(mt, heads, 2.0):
            a = O.old_nrgs_functional.greedy(g["dst_tok"], max_new=MAX_NEW, ctx=c)
        b = FUNC.greedy(g["dst_tok"], max_new=MAX_NEW, ctx=c, interventions=FUNC.boost_head_ov(mt, heads, 2.0))
        assert a == b, (a, b)
        out.append(b)
    assert out[1] != base
    _record("OV boost x2 around `greedy` (sparse, dense)",
            f"token-identical 2/2; dense differs from unboosted ({out[1][:40]!r} vs {base[:40]!r})")


# ---------------------------------------------------------------------------------------------
# task vs progress: residual + embedding transplants, KV configs
# ---------------------------------------------------------------------------------------------


def test_task_vs_progress():
    """``task_vs_progress.make_stack`` + ``run_patched`` on instance 0 of int py-py: the residual method (embedding +
    every layer's output transplanted on the prompt pass) on the three spans, and two KV configs (the sparse heads,
    progress, V x2; every head of every layer, task, V x1): token-identical."""
    mt, O, c = model(), old(), ctx()
    insts, _ = PP.sample_multi_instances("int", "python", "python", n_samples=32, seed=1123)
    G = TP.geometry(insts[0], ctx=c)
    Go = O.old_task_vs_progress.geometry(insts[0], ctx=c)
    assert {k: (list(a), list(b)) for k, (a, b) in G["W"].items()} == {k: (list(a), list(b)) for k, (a, b) in Go["W"].items()}
    headsets = {"sparse": SPARSE, "all": [(l, h) for l in range(mt.num_layers) for h in range(mt.num_heads)]}
    configs = ([("residual", None, s, 1.0) for s in ("task", "progress", "task+progress")]
               + [("kv", "sparse", "progress", 2.0), ("kv", "all", "task", 1.0), ("none", None, "none", 1.0)])
    clean = FUNC.greedy(G["dst_tok"], max_new=MAX_NEW, ctx=c)
    rows = []
    for m, h, s, v in configs:
        a = O.old_task_vs_progress.run_patched(
            Go, O.old_task_vs_progress.make_stack(Go, m, s, h, v, headsets=headsets, ctx=c), max_new=MAX_NEW, ctx=c)
        b = TP.run_patched(G, TP.make_stack(G, m, s, h, v, headsets=headsets, ctx=c), max_new=MAX_NEW, ctx=c)
        assert a == b, (m, h, s, v, a, b)
        rows.append((TP.cfg_name(m, h, s, v), b != clean))
        print(f"      {TP.cfg_name(m, h, s, v)}: {b[:50]!r}")
    _record("task_vs_progress `run_patched`: residual (+ embeddings) on task / progress / task+progress, KV sparse "
            "progress V x2, KV all heads task V x1, none",
            f"token-identical {len(rows)}/{len(rows)}; differ from clean DST: "
            + ", ".join(n for n, moved in rows if moved))


# ---------------------------------------------------------------------------------------------
# composition and timing
# ---------------------------------------------------------------------------------------------


def test_scores_edit_composes_with_attention_matrices():
    """An edit of ``attention_scores`` (``whitelist_head_attn``) passed to ``get_attention_matrices`` as
    ``interventions`` runs in the first invoke while the pattern is read in the second: the read sees the edit (the
    nnterp docs say two invokes cannot both touch the scores / pattern; on nnsight 0.8.2 they can)."""
    mt = model()
    g = geometry("py-py")
    L = g["dst_tok"]["input_ids"].shape[1]
    keep = list(range(g["dst_w"][0], g["dst_w"][0] + 5))
    iv = FUNC.whitelist_head_attn(mt, SPARSE, keep, q_from=L - 1)
    info = get_attention_matrices(g["dst_tok"], mt, interventions=iv)
    A = torch.from_numpy(info.attention_matrices)  # (layers, heads, q, k)
    for l, h in SPARSE:
        row = A[l, h, L - 1]
        assert row[keep].sum() > 0.999 and row.sum() > 0.999, (l, h, row[keep].sum())
    clean = torch.from_numpy(get_attention_matrices(g["dst_tok"], mt).attention_matrices)
    assert torch.equal(A[:, :, : L - 1], clean[:, :, : L - 1])  # rows before q_from untouched
    _record("`whitelist_head_attn` (a scores edit) composed with `get_attention_matrices` (two invokes)",
            "the pattern read in the second invoke carries the edit: each restricted head's pivot row puts all its "
            "mass on the 5 kept keys; earlier rows equal the clean pattern")


def test_kv_timing():
    """Old vs new KV-transfer generation, 24 tokens, the dense head set, V x4 (mean of 3 after a warm-up). CPU timing
    is only indicative; the 7B GPU timing is in the PR."""
    mt, O, c = model(), old(), ctx()
    g = geometry("py-py")
    kw = dict(goal_heads=DENSE, readout_heads=DENSE, v_scale=4.0, max_new=MAX_NEW, readout=False, ctx=c)
    times = {}
    for name, fn in (("old", O.old_kv_transfer.run_patched), ("new", PP.run_patched)):
        fn(g, **kw)
        t = time.perf_counter()
        for _ in range(3):
            res = fn(g, **kw)
        times[name] = (time.perf_counter() - t) / 3
        times[name + "_res"] = res
    assert times["old_res"] == times["new_res"]
    _record(f"KV-transfer `run_patched` timing ({MAX_NEW} tok, 182 heads, CPU)",
            f"old {times['old']:.2f} s, new {times['new']:.2f} s per call (mean of 3)")


if __name__ == "__main__":
    print(f"nnsight {nnsight.__version__} | nnterp {nnterp.__version__} | "
          f"transformers {transformers.__version__} | torch {torch.__version__} | old ref {OLD_REF[:10]}")
    t0 = time.time()
    model()
    old()
    print(f"loaded {MODEL_KEY} and the old modules in {time.time() - t0:.1f}s")
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        t = time.time()
        try:
            fn()
            print(f"PASS {fn.__name__} ({time.time() - t:.1f}s)")
        except Exception as e:
            failed += 1
            print(f"FAIL {fn.__name__} ({time.time() - t:.1f}s): {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed in {time.time() - t0:.1f}s\n")
    print("| comparison | result |\n|---|---|")
    for name, outcome in RESULTS:
        print(f"| {name} | {outcome} |")
    sys.exit(1 if failed else 0)
