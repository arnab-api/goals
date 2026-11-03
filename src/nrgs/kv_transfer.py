"""Pivot-onwards KV transfer on the multi-predicate setup (batch driver ``scripts/nrgs/kv_transfer_eval.py``).

Two prompts sharing the prefilled progress (a, b):

    p_src: a'b'c -> c     k/v donor
    p_dst: a'b'd -> d     the patched run

(multi-hierarchy, one level down: p_src a'(x'z) -> z, p_dst a'(x'y) -> y, pivot = the connective;
:mod:`src.nrgs.hier_data`.)

The intervention is the **pivot-onwards KV transfer**: ``goal_heads`` read SRC's window k/v from the pivot on (the last
query of each forward), optionally with the transplanted values scaled by ``v_scale``. The transplanted window is the
task span (``span="task"``: ``(1)`` .. the user turn's end-of-turn token, excluded) or the task span + the progress
through the pivot ` =` (``span="task+progress"``). Both are aligned with :func:`src.nrgs.functional.align_kv_spans`: the shorter
SRC / DST window gets ``\\n`` tokens right before it, so the two windows have equal length and the transplant stays
position-aligned across languages ("A1"). All runs of a geometry (clean SRC, clean DST, patched) use these padded token
sequences."""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

import src.nrgs.data as DATA
import src.nrgs.eval as E
import src.nrgs.functional as FUNC
import src.nrgs.hier_data as H
from src.tokens import prepare_input


# ══════════════════════════════════════════════════════════════════════════════
# residual tracing / overwriting (every decoder layer's OUTPUT; the task-vs-progress residual method)
# ══════════════════════════════════════════════════════════════════════════════
def layer_outputs(mt, tok, positions, layers):
    """{layer: (len(positions), d)} — each decoder layer's OUTPUT residual at ``positions`` (CPU float32)."""
    layers = sorted(layers)
    saves = {}
    with torch.no_grad(), mt.trace(dict(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"])):
        for l in layers:
            saves[l] = mt.layers[l].layer_output[0, positions].save()
    return {l: saves[l].detach().float().cpu() for l in layers}


def residual_patch(mt, cache, positions, layers) -> FUNC.Intervention:
    """Overwrite each decoder layer's OUTPUT residual (``layer_output``) at absolute ``positions`` with
    ``cache[layer]`` (rows in ``positions`` order), on the prompt pass only (the decode steps do not contain them)."""
    positions = list(positions)

    def make(block, src):
        def edit(mt_, step):
            block.layer_output[:, positions] = src

        return edit

    edits = []
    for l in layers:
        w = mt.layers[l].self_attn.o_proj.weight  # the block's device and dtype
        edits.append(FUNC.Edit(l, "layer_output", make(mt.layers[l], cache[l].to(w.device, w.dtype)), first_only=True))
    return FUNC.Intervention(edits, mt)


# ══════════════════════════════════════════════════════════════════════════════
# geometry: token sequences + the transplant window + the progress span
# ══════════════════════════════════════════════════════════════════════════════
def _ids(tok):
    return tok["input_ids"][0].tolist()


def _safe(dt: str) -> str:
    return dt.replace("[", "_").replace("]", "").replace(" ", "")


def _md5_int(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


def sample_multi_instances(datatype, src_lang, dst_lang, n_samples=32, seed=1123):
    """The batch eval's instances (``scripts/nrgs/kv_transfer_eval.py``): ``multi_predicate_setups`` (k=3, letters, var_eq,
    listed) with a per-(datatype, direction) seed. Returns ``(instances, sample_seed)``; instance i here is instance i
    there (same seed -> same draw)."""
    s = (seed + _md5_int(f"{_safe(datatype)}__{src_lang}-{dst_lang}")) % (2**32)
    insts = DATA.multi_predicate_setups(datatype, n_conditions=3, n_samples=n_samples, seed=s, presentation="listed",
                                        naming="letters", letter_stub="var_eq", src_lang=src_lang, dst_lang=dst_lang)
    return insts, s


def sample_hier_instances(datatype, connective, order, src_lang, dst_lang, n_samples=32, seed=1123):
    """The multi-hierarchy instances (:func:`src.nrgs.hier_data.seeded_setups`). Returns ``(instances, sample_seed)``."""
    return H.seeded_setups(datatype, connective, order, src_lang, dst_lang, n_samples=n_samples, seed=seed)


def inst_key(inst) -> str:
    if isinstance(inst, H.HierKVInstance):
        return H.inst_key(inst)
    return (f"multi|{inst.datatype}|shared={'-'.join(p.id for p in inst.shared)}"
            f"|src={inst.src_frontier.id}|dst={inst.dst_frontier.id}")


def build_geometry(inst, *, span="task", ctx):
    """Token sequences and positions for one (instance, span) pair. ``span`` = the transplanted SRC window:
    ``"task"`` = ``(1)`` .. the user turn's end-of-turn token (excluded), i.e. the task list + everything after it;
    ``"task+progress"`` = one contiguous window from the task start through the pivot ` =`. ``g["seg"]`` = the
    progress, end-of-turn token .. ``{var}`` (pivot excluded) on DST, used by :mod:`src.nrgs.task_vs_progress`.

    The end-of-turn token is the first special token after the task text: ``<|im_end|>`` on Qwen and OLMo-3,
    ``<|eot_id|>`` on Llama-3, ``<end_of_turn>`` on Gemma, ``<|end|>`` on Phi-3, ``<|assistant|>`` on GLM-4."""
    mt = ctx.mt
    m = DATA.materialize(inst, ctx=ctx)
    c_desc, d_desc = inst.expected_next_desc, inst.dst_next_desc          # what the oracle judges against
    txt_s, txt_d = m["src_task_text"], m["dst_task_text"]
    assert m["src_prompt"].count(txt_s) == 1 and m["dst_prompt"].count(txt_d) == 1, "task span text not unique"
    s0, d0 = FUNC.token_span(m["src_prompt"], txt_s, ctx=ctx), FUNC.token_span(m["dst_prompt"], txt_d, ctx=ctx)
    ids_s = _ids(prepare_input(prompts=m["src_prompt"], tokenizer=mt))
    ids_d = _ids(prepare_input(prompts=m["dst_prompt"], tokenizer=mt))
    Ls, Ld = len(ids_s), len(ids_d)
    special = {i for i, t in mt.tokenizer.added_tokens_decoder.items() if t.special}
    e_s = next(i for i in range(s0[1], Ls) if ids_s[i] in special)  # the user turn's end-of-turn token
    e_d = next(i for i in range(d0[1], Ld) if ids_d[i] in special)
    n_gap = e_d - d0[1]
    s0, d0 = (s0[0], e_s), (d0[0], e_d)
    if span == "task":
        s_win, d_win = s0, d0
    elif span == "task+progress":   # ONE contiguous window: task start .. ` =` (included)
        s_win, d_win = (s0[0], Ls), (d0[0], Ld)
    else:
        raise ValueError(f"span must be 'task' | 'task+progress', got {span!r}")
    src_tok, dst_tok, src_w, dst_w = FUNC.align_kv_spans(mt, m["src_prompt"], m["dst_prompt"], s_win, d_win, ctx=ctx)
    assert len(src_w) == len(dst_w)
    d_ids = _ids(dst_tok)
    L = len(d_ids)
    pad_s, pad_d = len(_ids(src_tok)) - Ls, L - Ld

    # progress: the end-of-turn token .. the token before the pivot (` =`; ` and` / ` &&` ...); the alignment's
    # newlines went in before the window, so every position from the window on moved by pad_d
    seg = list(range(e_d + pad_d, L - 1))
    return dict(
        span=span, n_gap=n_gap, src_tok=src_tok, dst_tok=dst_tok, src_w=list(src_w), dst_w=list(dst_w), seg=seg,
        c_desc=c_desc, d_desc=d_desc, pad_src=pad_s, pad_dst=pad_d, src_prompt=m["src_prompt"], dst_prompt=m["dst_prompt"],
    )


def geometry_info(g) -> dict:
    """JSON-native summary of a geometry (for the saved record)."""
    return dict(span=g["span"], n_gap=g["n_gap"], n_window=len(g["dst_w"]), dst_window=[g["dst_w"][0], g["dst_w"][-1]],
                src_window=[g["src_w"][0], g["src_w"][-1]], pad_src=g["pad_src"], pad_dst=g["pad_dst"],
                L_src=g["src_tok"]["input_ids"].shape[1], L_dst=g["dst_tok"]["input_ids"].shape[1])


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


def intervention_stack(g, *, goal_heads, v_scale=1.0, ctx):
    """The pivot-onwards KV transfer as an :class:`src.nrgs.functional.Intervention` (``iv(mt, step=0)``):
    ``goal_heads`` read SRC's window k/v (values x ``v_scale``) from the pivot on. Reusable: pass it to
    ``FUNC.greedy(..., interventions=)`` for a generation and to ``goalscope_readout(..., interventions=)`` for the
    readout's source trace. Used by :func:`run_patched` and by :func:`src.nrgs.task_vs_progress.make_stack`."""
    mt = ctx.mt
    layers = sorted({l for l, _ in goal_heads})
    k_c = FUNC.cache_src_qkv(g["src_tok"], g["src_w"], layers, "k", ctx=ctx)
    v_c = FUNC.cache_src_qkv(g["src_tok"], g["src_w"], layers, "v", ctx=ctx)
    return FUNC.headwise_kv_patch(mt, goal_heads, k_c, v_c, g["dst_w"], v_scale=v_scale)


def run_patched(g, *, goal_heads, readout_heads, v_scale=1.0, max_new=64, readout=True, ctx):
    """One patched DST run (prompt ending at the pivot): greedy generation + GoalScope readout at the pivot under
    :func:`intervention_stack`. ``v_scale`` multiplies the transplanted SRC window VALUES read by the goal heads
    (keys / attention pattern unchanged; 1.0 = off). Returns ``gen_patched`` / ``ro_patched``."""
    iv = intervention_stack(g, goal_heads=goal_heads, v_scale=v_scale, ctx=ctx)
    gen = FUNC.greedy(g["dst_tok"], max_new=max_new, ctx=ctx, interventions=iv)
    L = g["dst_tok"]["input_ids"].shape[1]
    ro = (FUNC.goalscope_readout("", g["dst_tok"], L - 1, heads=readout_heads, interventions=iv, ctx=ctx)[0]["patched_gen"]
          if readout else None)
    return dict(gen_patched=gen, ro_patched=ro)


# ══════════════════════════════════════════════════════════════════════════════
# oracle (the batch eval's scoring)
# ══════════════════════════════════════════════════════════════════════════════
def judge(inst, raw, model_name="claude-sonnet-4-6"):
    """The batch eval's oracle: ``(causal, readout)``. causal = ``judge_causal`` on the patched generation
    (first computation; BOTH is a miss); readout = ``eval_readouts`` (c-vs-d, BOTH is a miss)."""
    causal = E.judge_causal(inst, raw["gen_patched"], model_name=model_name)
    readout = (E.eval_readouts(inst, raw["ro_src"], raw["ro_dst"], raw["ro_patched"], model_name=model_name)
               if raw.get("ro_patched") is not None else None)       # None when run without readouts
    return causal, readout


# ══════════════════════════════════════════════════════════════════════════════
# batch-eval records (``scripts/nrgs/kv_transfer_eval.py``, ``scripts/nrgs/task_vs_progress_eval.py``)
# ══════════════════════════════════════════════════════════════════════════════
LANG_TRANSFERS = {
    "py-py": ("python", "python"),
    "py-cpp": ("python", "cpp"),
    "cpp-py": ("cpp", "python"),
    "cpp-cpp": ("cpp", "cpp"),
}


@dataclass
class InstanceRecord:
    """Everything about one contrastive instance: identity + RAW artifacts + eval verdicts."""

    key: str
    setup: str  # "multi_pred" | "multi_hier"
    datatype: str
    group: str  # multi: the shared prefix (T1,T2) ids ; hier: "<x>|<connective>"
    naming: str
    presentation: str
    src_lang: str
    dst_lang: str
    var_name: str
    expected_next: str  # SRC target (transplant target)
    expected_next_desc: str
    dst_next: str  # clean DST target
    dst_next_desc: str
    letter_stub: str = "na"  # frontier-pivot shape ("var_eq")
    raw: dict = field(default_factory=dict)  # raw intervention artifacts (prompts, generations, readouts)
    causal: Optional[dict] = None  # eval verdicts
    readout: Optional[dict] = None


def patch_heads_tag(spec, n) -> str:
    """Run-tag label for a JSON head set: ``<label>_<n>`` (label = filename stem before '__'), e.g. ``goal_250``."""
    return f"{Path(spec).stem.split('__')[0] or 'heads'}_{n}"


def summarize(records: list[InstanceRecord]) -> dict:
    def rate(pred, over=None):
        pool = [r for r in records if (over is None or over(r))]
        vals = [pred(r) for r in pool if pred(r) is not None]
        return (sum(vals) / len(vals)) if vals else None

    causal = rate(lambda r: r.causal["triggered"] if r.causal else None)
    r_valid = rate(lambda r: r.readout["valid"] if r.readout else None)
    r_shift = rate(lambda r: r.readout["shifted"] if r.readout else None)
    r_shift_valid = rate(
        lambda r: (r.readout["shifted"] if r.readout else None),
        over=lambda r: r.readout and r.readout["valid"],
    )

    def _has_parse_error(r) -> bool:
        if r.causal and r.causal.get("label") == "parse_error":
            return True
        if r.readout:
            return any(
                r.readout.get(k) == "parse_error" for k in ("source", "dest", "patched")
            )
        return False

    n_parse_error = sum(1 for r in records if _has_parse_error(r))
    return dict(
        causal_trigger_rate=causal,
        readout_valid_rate=r_valid,
        readout_shift_rate=r_shift,
        readout_shift_rate_valid=r_shift_valid,
        n_parse_error=n_parse_error,
    )
