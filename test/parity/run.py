#!/usr/bin/env python
"""Run the parity case set (``cases.py``) through the repo's public entry points and write one
``<case>.json`` (strings, scalars, selections) + one ``<case>.pt`` (tensors) per case, plus ``env.json``.

    python test/parity/run.py --model Qwen/Qwen2.5-Coder-7B-Instruct --out results_parity/run1
    python test/parity/compare.py test/parity/reference/Qwen2.5-Coder-7B-Instruct results_parity/run1

No LM judge and no network: ``ask_claude`` is replaced by a function that raises before any repo module
imports it, and ``HF_HUB_OFFLINE=1`` is set unless the caller set it. Download the model beforehand.
"""

from __future__ import annotations

import argparse
import datetime
import importlib.metadata as md
import json
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TQDM_DISABLE", "1")

ROOT = Path(__file__).resolve().parents[2]
_CWD = Path.cwd()
os.chdir(ROOT)  # the eval scripts resolve data/ and head_sets/ relative to the repo root
sys.path.insert(0, str(ROOT))

# ── no judge: swap ask_claude out BEFORE any module does `from src.utils.oracle_llms import ask_claude` ──
import src.utils.oracle_llms as _oracle  # noqa: E402


def _no_judge(*args, **kwargs):
    raise RuntimeError("parity harness: an LM-judge (ask_claude) call was attempted")


_oracle.ask_claude = _no_judge

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import scripts.evaluation.obstacle_persistence_ov_boost_eval as OVB  # noqa: E402
import src.nrgs.data as DATA  # noqa: E402
import src.nrgs.functional as FUNC  # noqa: E402
import src.nrgs.kv_transfer as PP  # noqa: E402
from scripts.evaluation.ablation_common import (  # noqa: E402
    complement_heads,
    frontier_block_positions,
    generate_blocking_span_cached,
    random_heads_like,
    sink_generate_cached,
)
from scripts.evaluation.eval_goalscope import (  # noqa: E402
    ANCHOR_ORDER,
    build_prompt,
    frontier_impl_span,
    generate_completion,
    load_json_heads,
    pivot_positions,
)
from scripts.evaluation.obstacle_persistence_eval import build_obstacle_prompt  # noqa: E402
from src.attention import get_attention_matrices  # noqa: E402
from src.functional import get_hs  # noqa: E402
from src.goals import temporal_persistence as TP  # noqa: E402
from src.goals.goalscope import (  # noqa: E402
    OVReader,
    _build_causality_prompt,
    _build_goalscope_prompt,
    apply_goalscope,
    greedy_continue,
    head_additive_generate,
)
from src.goals.head_loc_attn import (  # noqa: E402
    HeadMask,
    collect_head_steering_vecs,
    forward_with_src_replacement,
    get_o_proj_inputs_batch,
)
from src.goals.head_loc_pivot import get_next_token_probs, get_next_token_probs_batch  # noqa: E402
from src.tokens import prepare_input  # noqa: E402
from src.utils.experiment_utils import set_seed  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cases as C  # noqa: E402

TOPK = 50


# ══════════════════════════════════════════════════════════════════════════════
# record: JSON payload + tensors with a manifest (kind -> tolerance class in compare.py)
# ══════════════════════════════════════════════════════════════════════════════
class Rec:
    """``data`` -> ``<case>.json``; ``tensors`` -> ``<case>.pt``; ``manifest`` (in the json) says what each tensor is.

    Tensor kinds: ``act`` (activations read from the bf16 model), ``vec`` (fp32 vectors the code derives from
    activations), ``logits``, ``probs``, ``attn`` (attention probabilities), ``attn_vw`` (value-weighted attention),
    ``grad``, ``topk_ids`` / ``topk_vals`` (top-50 of a logits / probs row)."""

    def __init__(self, padded: bool = False):
        self.data: dict = {}
        self.tensors: dict[str, torch.Tensor] = {}
        self.manifest: dict[str, dict] = {}
        self.padded = padded

    def t(self, key: str, x, kind: str) -> None:
        x = torch.as_tensor(x).detach().cpu().contiguous().clone()
        stored = x
        # activations / attention read from a bf16 model and upcast to fp32 by the code: keep them in bf16
        # when that is lossless (halves the reference size; compare.py upcasts both sides before comparing)
        if kind in ("act", "attn", "attn_vw") and x.dtype == torch.float32:
            xb = x.to(torch.bfloat16)
            if torch.equal(xb.float(), x):
                stored = xb
        self.tensors[key] = stored
        self.manifest[key] = dict(kind=kind, shape=list(x.shape), dtype=str(x.dtype).replace("torch.", ""),
                                  stored_dtype=str(stored.dtype).replace("torch.", ""))

    def topk(self, key: str, x: torch.Tensor, of: str = "logits") -> None:
        x = x.detach().float().cpu()
        v, i = x.topk(TOPK, dim=-1)
        self.t(f"{key}/top_ids", i, "topk_ids")
        self.t(f"{key}/top_vals", v, "topk_vals")
        self.manifest[f"{key}/top_vals"]["of"] = of


def _ids(mt, text: str) -> list[int]:
    """Re-tokenization of a generated string, stored next to it for inspection (compare.py skips these lists in
    tolerance mode: the string itself is compared, with a token-level summary of the first divergence)."""
    return mt.tokenizer(text, add_special_tokens=False)["input_ids"]


def _layer(mt, L: int) -> str:
    return f"model.layers.{L}"  # get_hs takes native module paths


class first_call_logits:
    """Observe (never modify) the logits: keeps ``mt.logits[:, -1]`` of the FIRST forward of the first run inside
    the block, through a persistent nnsight edit (``mt.edit(inplace=True)``, cleared on exit) that reads them at
    their first occurrence in each run. Used around a ``max_new_tokens=1`` generation to read the step-0 logits of
    a steered probe."""

    def __init__(self, mt):
        self.mt, self.logits, self._seen = mt, None, []

    def __enter__(self):
        mt, seen = self.mt, self._seen
        with mt.edit(inplace=True):
            seen.append(mt.logits[:, -1].save())
        return self

    def __exit__(self, *exc):
        self.mt.clear_edits()
        if self._seen:
            self.logits = self._seen[0].detach().float().cpu()


def _readouts(mt, res) -> dict:
    return {str(x["pos"]): dict(token=x["token"], patched_gen=x["patched_gen"], ids=_ids(mt, x["patched_gen"]))
            for x in res}


# ══════════════════════════════════════════════════════════════════════════════
# cases
# ══════════════════════════════════════════════════════════════════════════════
def _rollout(E, sid):
    """``eval_goalscope.main``'s per-sample prefix: prompt -> greedy completion -> pivots."""
    mt = E.mt
    inst = E.specs[sid]
    prompt, var = build_prompt(mt, inst)
    completion = generate_completion(mt, prompt, C.GEN_TOKENS)
    full_text = prompt + completion
    pivs = pivot_positions(mt, prompt, full_text, ANCHOR_ORDER)
    return dict(prompt=prompt, var=var, completion=completion, completion_ids=_ids(mt, completion),
                full_text=full_text, pivots=pivs, impl_span=frontier_impl_span(prompt, completion, var),
                frontier=inst.expected_next)


def _public(ro: dict) -> dict:
    return {k: v for k, v in ro.items() if k != "full_text"}


def case_hs_probs(E) -> Rec:
    """``get_hs`` at the embedding, layer and final-norm outputs + ``get_next_token_probs`` (full distribution).
    Locations are listed in module execution order: the nnsight 0.5-0.7 trace in ``get_hs`` raises
    ``MissedProviderError`` when an earlier module is requested after a later one."""
    mt, r = E.mt, Rec()
    prompt, _ = build_prompt(mt, E.specs[C.HS_SPEC])
    locs = [("model.embed_tokens", -1)] + [(_layer(mt, L), p) for L in C.HS_LAYERS for p in C.HS_POSITIONS]
    locs += [("model.norm", -1)]
    hs = get_hs(mt, prompt, locations=locs, return_dict=True)
    for (name, p), v in hs.items():
        r.t(f"hs/{name}@{p}", v, "act")
    probs = get_next_token_probs(mt, prompt)
    r.t("next_token_probs", probs, "probs")
    r.topk("next_token_probs", probs, of="probs")
    r.data = dict(spec=C.HS_SPEC, prompt=prompt, locations=[[n, p] for n, p in locs],
                  top1=mt.tokenizer.decode([int(probs.argmax())]))
    return r


def case_attention(E) -> Rec:
    """``get_attention_matrices`` (all layers) plain and value-weighted; 2 layers saved in full, the last
    query row of every layer, and the last-position logits."""
    mt, r = E.mt, Rec()
    prompt, _ = build_prompt(mt, E.specs[C.ATTN_SPEC])
    for vw in (False, True):
        info = get_attention_matrices(prompt, mt, value_weighted=vw)
        A = torch.from_numpy(info.attention_matrices)  # (L, H, T, T) float32
        tag, kind = ("value_weighted", "attn_vw") if vw else ("plain", "attn")
        for L in C.ATTN_SAVE_LAYERS:
            r.t(f"{tag}/layer{L}", A[L], kind)
        r.t(f"{tag}/last_query_all_layers", A[:, :, -1, :], kind)
        r.topk(f"{tag}/logits", info.logits, of="logits")
        r.data[tag] = dict(shape=list(A.shape), tokenized_prompt=info.tokenized_prompt)
    r.data.update(spec=C.ATTN_SPEC, prompt=prompt, saved_layers=C.ATTN_SAVE_LAYERS)
    return r


def _headvec_record(E, r, prompts, tag):
    mt, heads = E.mt, E.heads
    layers = sorted({l for l, _ in heads})
    tok = prepare_input(prompts=prompts, tokenizer=mt)  # left-padded when lengths differ
    T = tok["input_ids"].shape[1]
    pos = list(range(T - C.HEADVEC_N_LAST, T))
    sv = collect_head_steering_vecs(mt, tok, heads, token_positions=pos)  # (B, n_pos, n_embd) fp32
    r.t(f"{tag}/steering_vecs", sv, "vec")
    o = get_o_proj_inputs_batch(mt, tok, layers, token_positions=pos)  # {layer: (B, n_pos, n_embd)}
    r.t(f"{tag}/o_proj_inputs", torch.stack([o[l] for l in layers]), "act")  # (n_layers, B, n_pos, n_embd)
    o_last = get_o_proj_inputs_batch(mt, list(prompts), layers)  # list[str] path, last token: (B, n_embd)
    r.t(f"{tag}/o_proj_inputs_last", torch.stack([o_last[l] for l in layers]), "act")
    r.data[tag] = dict(prompts=list(prompts), seq_len=T, positions=pos, layers=layers,
                       n_pad=[int((tok["attention_mask"][i] == 0).sum()) for i in range(len(prompts))])
    return sv


def case_head_vecs_single(E) -> Rec:
    """``collect_head_steering_vecs`` + ``get_o_proj_inputs_batch`` on 1 prompt (no padding)."""
    r = Rec()
    prompt = build_prompt(E.mt, E.specs[C.HEADVEC_SPECS[1]])[0]
    _headvec_record(E, r, [prompt], "single")
    r.data["spec"] = C.HEADVEC_SPECS[1]
    return r


def case_head_vecs_padded(E) -> Rec:
    """The same extraction on 2 prompts of DIFFERENT lengths in one left-padded batch."""
    r = Rec(padded=True)
    prompts = [build_prompt(E.mt, E.specs[s])[0] for s in C.HEADVEC_SPECS]
    sv2 = _headvec_record(E, r, prompts, "batch2")
    # diagnostic (not compared): the padded row vs the same prompt run alone (= head_vecs_single). A plain forward
    # gives left-padded rows position_ids = arange(T), so the pads shift that row's RoPE positions.
    i_pad = next(i for i, n in enumerate(r.data["batch2"]["n_pad"]) if n > 0)
    tok1 = prepare_input(prompts=prompts[i_pad], tokenizer=E.mt)
    T1 = tok1["input_ids"].shape[1]
    sv1 = collect_head_steering_vecs(E.mt, tok1, E.heads, token_positions=list(range(T1 - C.HEADVEC_N_LAST, T1)))
    r.data["_meta_padded_row_vs_alone_rel_err"] = float((sv2[i_pad] - sv1[0]).norm() / sv1[0].norm())
    r.data["specs"] = C.HEADVEC_SPECS
    return r


def case_gs_additive(E) -> Rec:
    """``apply_goalscope`` additive_norm (alpha 0.1, steer layers 15-20) at the 5 pivots of 2 specs (faithfulness
    probe) and the causality (C++ stub) probe on the first spec, for both head sets; the steering vectors; and the
    probe's step-0 logits (``head_additive_generate`` with 1 new token, lm_head observed)."""
    mt, r = E.mt, Rec()
    rolls = {sid: _rollout(E, sid) for sid in C.GS_ADDITIVE_SPECS}
    dest = prepare_input(prompts=_build_goalscope_prompt(mt), tokenizer=mt)
    steer_layers = list(range(*C.STEER_LAYER_RANGE))
    r.data["rollouts"] = {sid: _public(ro) for sid, ro in rolls.items()}
    for arm, heads in E.head_sets.items():
        gs = dict(selected_heads=heads, strategy="additive_norm", alpha=C.GS_ADDITIVE_ALPHA,
                  steer_layer_range=C.STEER_LAYER_RANGE, n_steer_tokens="all", max_new_tokens=C.READOUT_TOKENS)
        out = {}
        for sid, ro in rolls.items():
            apos = sorted(set(ro["pivots"].values()))
            res = apply_goalscope(ro["full_text"], mt=mt, include_noncode=False, token_positions=apos, **gs)
            out[sid] = dict(readouts=_readouts(mt, res))
            if sid == C.GS_CAUSALITY_SPEC:
                res_c = apply_goalscope(ro["full_text"], mt=mt, include_noncode=False, token_positions=apos,
                                        goalscope_prompt=_build_causality_prompt(mt), **gs)
                out[sid]["readouts_causality"] = _readouts(mt, res_c)
            sv = collect_head_steering_vecs(mt=mt, src_tokenized=prepare_input(prompts=ro["full_text"], tokenizer=mt),
                                            selected_heads=heads, token_positions=apos)[0]
            r.t(f"{arm}/{sid}/steering_vecs", sv, "vec")
            with first_call_logits(mt) as obs:
                head_additive_generate(mt, dest, sv, steer_layers, max_new_tokens=1, alpha=C.GS_ADDITIVE_ALPHA)
            r.topk(f"{arm}/{sid}/step0_logits", obs.logits)
        r.data[arm] = out
    # step counting: n_steer_tokens=2 (counts DECODE steps here: the prefill + 2 decode steps are patched)
    sid = C.GS_ADDITIVE_SPECS[0]
    ro = rolls[sid]
    apos = sorted(set(ro["pivots"].values()))
    res = apply_goalscope(ro["full_text"], mt=mt, include_noncode=False, token_positions=apos,
                          selected_heads=E.head_sets["band448"], strategy="additive_norm", alpha=C.GS_ADDITIVE_ALPHA,
                          steer_layer_range=C.STEER_LAYER_RANGE, n_steer_tokens=C.GS_NSTEER,
                          max_new_tokens=C.READOUT_TOKENS)
    r.data["band448_nsteer2"] = {sid: dict(readouts=_readouts(mt, res))}
    r.data["config"] = dict(alpha=C.GS_ADDITIVE_ALPHA, steer_layer_range=list(C.STEER_LAYER_RANGE),
                            max_new_tokens=C.READOUT_TOKENS, arms={a: len(h) for a, h in E.head_sets.items()})
    return r


def case_gs_replace(E) -> Rec:
    """``apply_goalscope`` strategy="replace" (alpha 1) -> ``headpatched_generate``, both head sets; plus its step-0
    logits from the same public pieces (HeadMask + get_o_proj_inputs_batch + forward_with_src_replacement)."""
    mt, r = E.mt, Rec()
    sid = C.GS_REPLACE_SPEC
    ro = _rollout(E, sid)
    apos = sorted(set(ro["pivots"].values()))
    src_tok = prepare_input(prompts=ro["full_text"], tokenizer=mt)
    dest = prepare_input(prompts=_build_goalscope_prompt(mt), tokenizer=mt)
    r.data["rollout"] = _public(ro)
    for arm, heads in E.head_sets.items():
        res = apply_goalscope(ro["full_text"], mt=mt, selected_heads=heads, strategy="replace",
                              alpha=C.GS_REPLACE_ALPHA, steer_layer_range=C.STEER_LAYER_RANGE, n_steer_tokens="all",
                              max_new_tokens=C.READOUT_TOKENS, include_noncode=False, token_positions=apos)
        r.data[arm] = dict(readouts=_readouts(mt, res))
        # step 0 of headpatched_generate for the (single) chunk, as apply_goalscope builds it
        layers = sorted({l for (l, _) in heads})
        mask = HeadMask(n_layers_tested=len(layers), n_heads=mt.n_heads)
        with torch.no_grad():
            mask.values.zero_()
            for l, h in heads:
                mask.values[layers.index(l), h] = 1.0
        src_o = get_o_proj_inputs_batch(mt, src_tok, layers, token_positions=apos)
        src_all = torch.stack([src_o[l][0] for l in layers], dim=1)  # (n_pos, n_layers, n_embd)
        if arm == "heads60":
            r.t("src_o_proj_inputs", src_all, "act")
        B, Lp = len(apos), dest["input_ids"].shape[1]
        inputs = dict(input_ids=dest["input_ids"].expand(B, -1).clone(),
                      attention_mask=dest["attention_mask"].expand(B, -1).clone())
        with torch.no_grad():
            logits = forward_with_src_replacement(mt, inputs, src_all, mask, layers, alpha=C.GS_REPLACE_ALPHA,
                                                  patch_positions=[Lp - 1])
        r.topk(f"{arm}/step0_logits", logits, of="logits")
    # step counting: n_steer_tokens=2 (counts forward passes, the prefill included)
    res = apply_goalscope(ro["full_text"], mt=mt, selected_heads=E.head_sets["band448"], strategy="replace",
                          alpha=C.GS_REPLACE_ALPHA, steer_layer_range=C.STEER_LAYER_RANGE,
                          n_steer_tokens=C.GS_NSTEER, max_new_tokens=C.READOUT_TOKENS, include_noncode=False,
                          token_positions=apos)
    r.data["band448_nsteer2"] = dict(readouts=_readouts(mt, res))
    return r


def case_ov_reader(E) -> Rec:
    """``OVReader.apply_probe_on_prompt`` at the `eq` pivot, 2 probe layers (alpha 0.15, steer layers 15-20), both
    head sets; the residual latents, the OV-write vectors, and the probe's step-0 logits."""
    mt, r = E.mt, Rec()
    ro = _rollout(E, C.OV_SPEC)
    pos = ro["pivots"]["eq"]
    steer_layers = list(range(*C.STEER_LAYER_RANGE))
    hs = get_hs(mt, prepare_input(prompts=ro["full_text"], tokenizer=mt),
                locations=[(_layer(mt, L), pos) for L in C.OV_PROBE_LAYERS], return_dict=True)
    lat = torch.stack([hs[(_layer(mt, L), pos)] for L in C.OV_PROBE_LAYERS])
    r.t("latents", lat, "act")
    r.data = dict(spec=C.OV_SPEC, pivot_pos=pos, probe_layers=C.OV_PROBE_LAYERS, rollout=_public(ro))
    for arm, heads in E.head_sets.items():
        ov = OVReader(mt, heads)
        by_layer = ov.apply_probe_on_prompt(ro["full_text"], pos, C.OV_PROBE_LAYERS, mode="faithfulness",
                                            steer_layers=steer_layers, max_new_tokens=C.READOUT_TOKENS,
                                            alpha=C.OV_ALPHA)
        vecs = torch.stack([ov.resid_ov_vec(h) for h in lat])
        r.t(f"{arm}/ov_vecs", vecs, "vec")
        with first_call_logits(mt) as obs:
            head_additive_generate(mt, ov.probe_inputs("faithfulness"), vecs, steer_layers, max_new_tokens=1,
                                   alpha=C.OV_ALPHA)
        r.topk(f"{arm}/step0_logits", obs.logits)
        r.data[arm] = {str(L): dict(text=t, ids=_ids(mt, t)) for L, t in by_layer.items()}
    return r


def case_temporal(E) -> Rec:
    """``temporal_persistence.goal_encoding`` at the `=` pivot + the implementation tokens of one frontier-prefill
    probe, both head sets. The oracle segmentation is replaced by "the first generated line" (no judge)."""
    mt, r = E.mt, Rec()
    inst = C.tp_instance()
    preds = inst._preds_for(inst.src_frontier)
    prompt, tv = TP.build_probe_prompt(mt, inst, preds, C.TP_TARGET)
    full_text = TP.rollout_probe(mt, prompt, max_new=C.GEN_TOKENS)
    gen = full_text[len(prompt):]
    seg = gen.split("\n", 1)[0]
    offs = TP._offsets(mt, full_text)
    P = TP.prompt_boundary(mt, prompt, full_text)
    eq_tok, impl = TP.locate_value(offs, full_text, len(prompt), P, seg)
    r.data = dict(ctx_key=TP.context_key(inst), target=preds[C.TP_TARGET].id, var=tv, prompt=prompt, gen=gen,
                  gen_ids=_ids(mt, gen), P=P, eq_tok=eq_tok, impl=impl)
    for arm, heads in E.head_sets.items():
        enc = TP.goal_encoding(mt, full_text, [eq_tok] + impl, heads)
        r.t(f"{arm}/encoding", enc, "vec")
        u = enc / enc.norm(dim=-1, keepdim=True)
        r.data[f"{arm}_cos_pivot_traj"] = [float(x) for x in (u[1:] @ u[0])]
    return r


def case_ablation_blind(E) -> Rec:
    """``generate_blocking_span_cached`` (sufficiency: blind all non-spared heads in layers >= 9 to the frontier
    span) for baseline / ablated (spare the 60 heads) / control (spare a random set), as the script builds them."""
    mt, heads, r = E.mt, E.heads, Rec()
    inst = E.specs[C.ABL_BLIND_SPEC]
    prompt, var = build_prompt(mt, inst)
    prompt_ids = prepare_input(prompts=prompt, tokenizer=mt)["input_ids"][0].tolist()
    q_start = len(prompt_ids) - 1
    block_pos = frontier_block_positions(mt, prompt, inst.expected_next_desc)
    ablated = complement_heads(heads, C.ABLATION_FROM_LAYER, mt.n_layer, mt.n_heads)
    gh = set(map(tuple, heads))
    rng = random.Random(C.ABL_CONTROL_SEED)
    pool = [(l, h) for l in range(mt.n_layer // 2, mt.n_layer) for h in range(mt.n_heads) if (l, h) not in gh]
    control = complement_heads(rng.sample(pool, min(len(heads), len(pool))), C.ABLATION_FROM_LAYER, mt.n_layer,
                               mt.n_heads)
    gens = {}
    for cname, hs in {"baseline": [], "ablated": ablated, "control": control}.items():
        g = generate_blocking_span_cached(mt, prompt_ids, hs, block_pos, q_start, max_new=C.GEN_TOKENS)
        gens[cname] = dict(gen=g, code=g.split("\n", 1)[0].strip(), ids=_ids(mt, g), n_heads=len(hs))
    r.data = dict(spec=C.ABL_BLIND_SPEC, prompt=prompt, q_start=q_start, block_positions=block_pos,
                  from_layer=C.ABLATION_FROM_LAYER, condition=inst.expected_next_desc, gens=gens)
    return r


def case_ablation_sink(E) -> Rec:
    """``sink_generate_cached`` (necessity: the heads attend only to key 0 from the pivot on) for baseline / ablated
    (60 heads; and the 448-head stress set) / control (a random 60 from the later half, seed 123)."""
    mt, heads, r = E.mt, E.heads, Rec()
    inst = E.specs[C.ABL_SINK_SPEC]
    prompt, var = build_prompt(mt, inst)
    prompt_ids = prepare_input(prompts=prompt, tokenizer=mt)["input_ids"][0].tolist()
    q_start = len(prompt_ids) - 1
    control = random_heads_like(heads, mt.n_layer // 2, mt.n_layer, mt.n_heads, heads,
                                random.Random(C.ABL_CONTROL_SEED))
    gens = {}
    conds = {"baseline": [], "ablated": heads, "control": control, "ablated_band448": E.head_sets["band448"]}
    for cname, hs in conds.items():
        g = sink_generate_cached(mt, prompt_ids, hs, q_start, sink_pos=0, max_new=C.GEN_TOKENS)
        gens[cname] = dict(gen=g, code=g.split("\n", 1)[0].strip(), ids=_ids(mt, g), n_heads=len(hs))
    r.data = dict(spec=C.ABL_SINK_SPEC, prompt=prompt, q_start=q_start, condition=inst.expected_next_desc,
                  control_heads=[list(h) for h in control], gens=gens)
    return r


def _traced_logits(mt, input_ids, attention_mask, interventions):
    """Last-position logits of one traced forward under ``interventions(mt)`` (an NRGS intervention)."""
    with torch.no_grad(), mt.trace(dict(input_ids=input_ids, attention_mask=attention_mask)):
        interventions(mt)
        logits = mt.logits[0, -1].save()
    return logits


def case_ov_boost(E) -> Rec:
    """OV boost x2 at the last token of every forward of the no-cache greedy loop:
    ``greedy_continue(..., interventions=boost_head_ov(...))`` (both head sets) and the script's own ``run_obstacle``
    (60 heads), which runs the same; plus step-0 logits of each path."""
    mt, r = E.mt, Rec()
    dt, pid = OVB.candidates()[C.OVB_CANDIDATE]
    prompt, p = build_obstacle_prompt(mt, dt, pid, OVB.NUMERIC_OBSTACLE)
    inp = prepare_input(prompts=prompt, tokenizer=mt)
    gc = {}
    with torch.no_grad():
        base = greedy_continue(mt, **inp, max_new_tokens=C.GEN_TOKENS)
        gc["base"] = dict(gen=base, ids=_ids(mt, base))
        r.topk("step0_logits/clean", mt(**inp).logits[0, -1], of="logits")  # a plain HF forward
        for arm, heads in E.head_sets.items():
            boost = FUNC.boost_head_ov(mt, heads, C.OVB_ALPHA)
            g = greedy_continue(mt, **inp, max_new_tokens=C.GEN_TOKENS, interventions=boost)
            r.topk(f"step0_logits/boost_head_ov/{arm}",
                   _traced_logits(mt, inp["input_ids"], inp["attention_mask"], boost), of="logits")
            gc[f"boost_{arm}"] = dict(gen=g, ids=_ids(mt, g))
        # the reference's key for the script's own boost (a hook context then); run_obstacle now boosts with the same
        # intervention
        r.topk("step0_logits/boosted_heads/heads60",
               _traced_logits(mt, inp["input_ids"], inp["attention_mask"],
                              FUNC.boost_head_ov(mt, E.heads, C.OVB_ALPHA)), of="logits")
    old = OVB.MAX_NEW
    OVB.MAX_NEW = C.GEN_TOKENS  # run_obstacle reads the module constant (48) at call time
    try:
        s_base = OVB.run_obstacle(mt, E.heads, dt, pid, boost_alpha=None)
        s_boost = OVB.run_obstacle(mt, E.heads, dt, pid, boost_alpha=C.OVB_ALPHA)
    finally:
        OVB.MAX_NEW = old
    r.data = dict(datatype=dt, predicate=pid, frontier=p.description, prompt=prompt, greedy_continue=gc,
                  run_obstacle=dict(base_rhs=s_base[0], base_gen=s_base[2], boost_rhs=s_boost[0],
                                    boost_gen=s_boost[2]),
                  _meta_boost_paths_agree=(gc["boost_heads60"]["gen"] == s_boost[2]))
    return r


def case_kv_transfer(E) -> Rec:
    """NRGS pivot-onwards KV transfer (``kv_transfer.build_geometry`` / ``clean_runs`` / ``run_patched``), 1 multi-setup
    instance (int, py-py), V x1 and x4, ``readout=False``, both head sets. Extra: step-0 logits under
    ``intervention_stack``; the cached source k/v; and GoalScope read-outs at the pivot through ``goalscope_readout``
    with the 7B steer layers and 32 tokens (its defaults, layers 35-46 and 128 tokens, do not fit a 28-layer model).
    The patched read-out passes ``intervention_stack`` to ``apply_goalscope``'s source trace (``interventions``)."""
    mt, ctx, r = E.mt, E.ctx, Rec()
    old = DATA.CPP_NOTE_IN_SYSTEM
    DATA.CPP_NOTE_IN_SYSTEM = True  # kv_transfer_eval.main sets this module global
    try:
        inst, seed = C.kv_instance()
        set_seed(seed)
        g = PP.build_geometry(inst, span="task", ctx=ctx)
        dst = g["dst_tok"]
        L = dst["input_ids"].shape[1]
        clean = PP.clean_runs(g, readout_heads=E.heads, max_new=C.GEN_TOKENS, readout=False, ctx=ctx)
        with torch.no_grad():
            r.topk("step0_logits/clean_dst", mt(input_ids=dst["input_ids"],
                                                attention_mask=dst["attention_mask"]).logits[0, -1])
        r.data = dict(key=PP.inst_key(inst), sample_seed=seed, geometry=PP.geometry_info(g),
                      src_prompt=g["src_prompt"], dst_prompt=g["dst_prompt"],
                      src_ids=PP._ids(g["src_tok"]), dst_ids=PP._ids(dst), src_w=g["src_w"], dst_w=g["dst_w"],
                      clean=dict(gen_src=clean["gen_src"], gen_dst=clean["gen_dst"], ro_src_default=clean["ro_src"],
                                 gen_src_ids=_ids(mt, clean["gen_src"]), gen_dst_ids=_ids(mt, clean["gen_dst"])))
        for arm, heads in E.head_sets.items():
            ro_kw = dict(heads=heads, layers=C.STEER_LAYER_RANGE, maxnew=C.READOUT_TOKENS, ctx=ctx)
            out = {k: FUNC.goalscope_readout("", g[f"{k}_tok"], g[f"{k}_tok"]["input_ids"].shape[1] - 1,
                                             **ro_kw)[0]["patched_gen"] for k in ("src", "dst")}
            out = dict(ro_src_7b=out["src"], ro_dst_7b=out["dst"])
            for v in C.KV_V_SCALES:
                pat = PP.run_patched(g, goal_heads=heads, readout_heads=heads, v_scale=v, max_new=C.GEN_TOKENS,
                                     readout=False, ctx=ctx)
                stack = PP.intervention_stack(g, goal_heads=heads, v_scale=v, ctx=ctx)
                lg = _traced_logits(mt, dst["input_ids"], dst["attention_mask"], stack)
                r.topk(f"{arm}/step0_logits/v{v:g}", lg)

                # the readout's source trace runs under the stack, as `run_patched` passes it
                ro = FUNC.goalscope_readout("", dst, L - 1, interventions=stack, **ro_kw)[0]["patched_gen"]
                out[f"v{v:g}"] = dict(gen_patched=pat["gen_patched"], gen_ids=_ids(mt, pat["gen_patched"]),
                                      ro_patched_7b=ro, ro_ids=_ids(mt, ro), ro_patched_default=pat["ro_patched"])
            r.data[arm] = out
        layers = sorted({l for l, _ in E.heads})[: C.KV_SAVE_LAYERS]
        for which in ("k", "v"):
            c = FUNC.cache_src_proj(g["src_tok"], g["src_w"], layers, which, ctx=ctx)
            r.t(f"src_cache_{which}", torch.stack([c[l] for l in layers]), "act")
        r.data["cache_layers"] = layers
    finally:
        DATA.CPP_NOTE_IN_SYSTEM = old
    return r


def case_q_transfer(E) -> Rec:
    """``run_q_transfer_intervention`` (gaze-select-restrict) on 1 order-diff int py-py instance, both head sets,
    ``compute_readout=False`` (its read-outs use ``goalscope_readout``'s 32B defaults; the judges live in the script)."""
    mt, r = E.mt, Rec()
    inst, cfg_seed = C.q_instance()
    r.data = dict(cfg_seed=cfg_seed, src_order=[p.id for p in inst.src_order], dst_order=[p.id for p in inst.dst_order],
                  src_progress=list(inst.src_progress), dst_progress=list(inst.dst_progress))
    for arm, heads in E.head_sets.items():
        set_seed(cfg_seed)
        raw = FUNC.run_q_transfer_intervention(mt, inst, heads, readout_heads=heads, max_new=C.GEN_TOKENS,
                                               alpha=C.Q_ALPHA, compute_readout=False, gsr_max_new=C.GEN_TOKENS)
        for k in ("gen_src", "gen_dst", "gen_patched"):
            raw[f"{k}_ids"] = _ids(mt, raw[k])
        raw["cl_select_scores"] = {k: float(v) for k, v in raw["cl_select_scores"].items()}
        r.data[arm] = raw
    q_layers = sorted({l for l, _ in E.heads})[:2]
    qc = FUNC.cache_src_proj(r.data["heads60"]["src_prompt"], [r.data["heads60"]["piv_src"]], q_layers, "q")
    r.t("src_q_cache", torch.stack([qc[l] for l in q_layers]), "act")
    r.data["q_cache_layers"] = q_layers
    return r


def _mask_grad(E, which: str) -> Rec:
    """One ``train_head_mask`` step (lazy / augment path) with a fixed HeadMask = 0.5 everywhere: target log-probs
    (``get_next_token_probs_batch``), source o_proj inputs (``get_o_proj_inputs_batch``), ``forward_with_src_replacement``
    on the destination batch, KL(P_patched || P_target) batchmean + lambda*mean(mask), ``loss.backward()``."""
    mt = E.mt
    r = Rec(padded=(which == "padded"))
    sel = E.mask_sel
    idx = sel[which]
    rows = [sel["rows"][i] for i in idx]
    src_ps, dest_ps, targ_ps = ([row[k] for row in rows] for k in ("src", "dst", "targ"))
    layers = list(range(mt.n_layer))
    req = [p.requires_grad for p in mt.parameters()]
    for p in mt.parameters():  # train_head_mask freezes the model
        p.requires_grad_(False)
    try:
        with torch.inference_mode():  # cache_pivot_data
            p_t = get_next_token_probs_batch(mt, targ_ps)
            log_p_target = torch.log(p_t.clamp(min=1e-10))
            src_o = get_o_proj_inputs_batch(mt, src_ps, layers)
        B = len(src_ps)
        batch_src_hv = torch.stack([torch.stack([src_o[ly][i_] for ly in layers]) for i_ in range(B)])
        batch_log_p = torch.stack([log_p_target[i_] for i_ in range(B)])
        batch_inputs = prepare_input(dest_ps, tokenizer=mt)
        mask = HeadMask(len(layers), mt.n_heads)
        with torch.no_grad():
            mask.values.fill_(C.HEADMASK_INIT)
        with torch.enable_grad():
            logits = forward_with_src_replacement(mt, batch_inputs, batch_src_hv, mask, layers)
            log_p_patched = F.log_softmax(logits, dim=-1)
            loss_kl = F.kl_div(batch_log_p.to(logits.device), log_p_patched, reduction="batchmean", log_target=True)
            loss_sparse = C.HEADMASK_LAMBDA * mask.values.mean()
            loss = loss_kl + loss_sparse
            loss.backward()
    finally:
        for p, q in zip(mt.parameters(), req):
            p.requires_grad_(q)
    r.t("mask_grad", mask.values.grad, "grad")
    r.t("src_head_vals_last", batch_src_hv, "act")
    if which == "equal":
        r.t("patched_logits", logits.detach(), "logits")
    r.topk("patched_logits", logits.detach(), of="logits")
    r.topk("target_probs", p_t.clone(), of="probs")
    r.data = dict(instances=idx, pairs=[row["pair"] for row in rows], pivot_prefixes=[row["prefix"] for row in rows],
                  lens_src_dst_targ=[list(row["lens"]) for row in rows], src_prompts=src_ps, dest_prompts=dest_ps,
                  targ_prompts=targ_ps, loss_kl=float(loss_kl), loss_sparse=float(loss_sparse),
                  mask_init=C.HEADMASK_INIT, lambda_sparse=C.HEADMASK_LAMBDA,
                  n_pad_dest=[int((batch_inputs["attention_mask"][i] == 0).sum()) for i in range(B)])
    return r


def case_mask_grad_equal(E) -> Rec:
    return _mask_grad(E, "equal")


def case_mask_grad_padded(E) -> Rec:
    return _mask_grad(E, "padded")


CASES = [
    ("hs_probs", case_hs_probs),
    ("attention", case_attention),
    ("head_vecs_single", case_head_vecs_single),
    ("head_vecs_padded", case_head_vecs_padded),
    ("gs_additive", case_gs_additive),
    ("gs_replace", case_gs_replace),
    ("ov_reader", case_ov_reader),
    ("temporal", case_temporal),
    ("ablation_blind", case_ablation_blind),
    ("ablation_sink", case_ablation_sink),
    ("ov_boost", case_ov_boost),
    ("kv_transfer", case_kv_transfer),
    ("q_transfer", case_q_transfer),
    ("mask_grad_equal", case_mask_grad_equal),
    ("mask_grad_padded", case_mask_grad_padded),
]


# ══════════════════════════════════════════════════════════════════════════════
# environment record
# ══════════════════════════════════════════════════════════════════════════════
def _sh(*cmd) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=ROOT).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"<unavailable: {e}>"


def env_record(model_id: str, mt) -> dict:
    import transformers

    dists = sorted({f"{d.metadata['Name'].lower()}=={d.version}" for d in md.distributions()})
    snapshot = None
    try:
        from huggingface_hub import try_to_load_from_cache

        snapshot = Path(try_to_load_from_cache(model_id, "config.json")).parent.name
    except Exception:  # noqa: BLE001
        pass
    import nnsight

    return dict(
        date=datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        git_sha=_sh("git", "rev-parse", "HEAD"),
        git_branch=_sh("git", "rev-parse", "--abbrev-ref", "HEAD"),
        git_tracked_changes=_sh("git", "status", "--porcelain", "--untracked-files=no"),
        python=platform.python_version(),
        torch=torch.__version__, torch_cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        transformers=transformers.__version__, nnsight=nnsight.__version__,
        gpu=torch.cuda.get_device_name(0), gpu_capability=list(torch.cuda.get_device_capability(0)),
        driver=_sh("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader").splitlines()[0:1],
        model=model_id, model_snapshot=snapshot, dtype="bfloat16", attn_implementation="eager",
        device_map={"": 0}, matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        packages=dists,
    )


# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=C.MODEL_ID)
    ap.add_argument("--out", required=True, help="output directory (created; existing case files are overwritten)")
    ap.add_argument("--heads", default=str(C.HEADS_JSON), help="[[layer, head], ...] json (default: the 7B set)")
    ap.add_argument("--cases", nargs="+", default=None, help="run only these cases (default: all)")
    args = ap.parse_args()
    out = Path(args.out) if Path(args.out).is_absolute() else _CWD / args.out
    heads_path = Path(args.heads) if Path(args.heads).is_absolute() else _CWD / args.heads
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    ctx = FUNC.setup(args.model, device_map={"": 0})  # binds src.nrgs.functional.mt (run_q_transfer_intervention needs it)
    ctx.readout_heads, ctx.gsp = None, None
    mt = ctx.mt

    class E:  # run environment shared by the cases
        pass

    E.mt, E.ctx = mt, ctx
    E.heads = load_json_heads(str(heads_path), mt.n_layer, mt.n_heads)
    E.head_sets = {"heads60": E.heads, "band448": C.stress_head_set()}
    ctx.readout_heads = E.heads
    E.specs = C.spec_index()
    E.mask_sel = C.mask_instances(mt)
    t_load = time.time() - t0

    env = env_record(args.model, mt)
    env["heads_json"] = os.path.relpath(heads_path, ROOT)
    env["n_heads_in_set"] = len(E.heads)
    env["model_load_seconds"] = round(t_load, 1)

    todo = [(n, f) for n, f in CASES if args.cases is None or n in args.cases]
    timings, failed = {}, []
    for name, fn in todo:
        set_seed(0)
        t1 = time.time()
        try:
            rec = fn(E)
            err = None
        except Exception:  # noqa: BLE001  record and keep going: the report lists cases the old code cannot run
            rec, err = Rec(), traceback.format_exc()
            failed.append(name)
            print(f"[parity] {name}: ERROR\n{err}", file=sys.stderr, flush=True)
        dt = time.time() - t1
        timings[name] = round(dt, 2)
        payload = dict(case=name, padded=rec.padded, data=rec.data, manifest=rec.manifest,
                       _meta=dict(seconds=round(dt, 2)))
        if err:
            payload["error"] = err
        (out / f"{name}.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
        pt = out / f"{name}.pt"
        if rec.tensors:
            torch.save(rec.tensors, pt)
        elif pt.exists():
            pt.unlink()
        print(f"[parity] {name}: {'ERROR' if err else 'ok'} ({dt:.1f}s)", flush=True)

    env["case_seconds"] = timings
    env["total_seconds"] = round(time.time() - t0, 1)
    env["cases"] = [n for n, _ in todo]
    env["failed_cases"] = failed
    (out / "env.json").write_text(json.dumps(env, indent=1) + "\n")
    print(f"[parity] wrote {len(todo)} cases to {out} in {env['total_seconds']}s; failed: {failed or 'none'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
