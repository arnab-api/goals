#!/usr/bin/env python
"""Cross-family check: run every experiment primitive of the repo once on one model, through the repo's public
functions, and report which run.

    python scripts/cross_family_check.py --model meta-llama/Llama-3.1-8B-Instruct \\
        --out test/cross_family/Llama-3.1-8B-Instruct.json

The model is loaded with ``src.nrgs.functional.setup`` (bf16, eager attention) on one GPU (``device_map={"": 0}``).
Every primitive runs on a tiny budget (``--n-tokens`` new tokens per generation and read-out) with a head set drawn
from the model's own shape: ``--heads random:N`` (default 60) samples N (layer, head) pairs from the middle third of
the layers with ``random.Random(--seed)``; ``--heads <file>.json`` reads a ``[[layer, head], ...]`` head set instead.
The paper's 32B layer constants are scaled by depth: steer layers 35-45 of 64, the OVReader probe layer 48 of 64, the
sufficiency blind from layer 20 of 64.

Each primitive is caught on its own, so one failure does not hide the others. The table printed at the end, and the
``--out`` json, hold the status and a one-line reason per primitive (shapes, sizes, the exception); no generated text.
A primitive passes when it runs, its outputs have the expected shapes and are finite, and its own check holds:
attention rows sum to 1; ``get_next_token_probs`` is the softmax of the logits the model returns; a sink-ablated head
puts all its pivot-row mass on key 0 and a blinded head none on the blocked keys; the KV transfer of a window onto
itself leaves the logits unchanged (which holds only if the transplanted keys are the keys the model reads and the
recomputed attention is the model's own). Whether a readout MEANS the same on another family is not tested (see
CLAUDE.md, "Running on another model"). No primitive calls the LM judge: ``ask_claude`` is replaced by a function that
raises.

There is no model-specific code in this script: anything a primitive needs from the model it reads through nnterp
(``mt.num_layers``, ``mt.num_heads``, ``mt.support()``) or through the repo's functions.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TQDM_DISABLE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nnterp  # noqa: E402  (before any transformers.models.* import)
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

N_BOOST = 2.0  # OV boost factor (obstacle_persistence_ov_boost_eval.BOOST_ALPHA)
V_SCALE = 4.0  # KV transfer value scale (the paper's V x 4)
Q_ALPHA = 2.0  # q-transfer readout OV boost (q_transfer_eval --alphas default; the readout is off here)
# A KV self-transplant must leave the logits bit-identical: the recompute runs the forward's own GEMM shapes
# (NNTERP_GAPS.md, NRGS item 1), so any change means it is not the model's attention.
KV_IDENTITY_TOL = 0.0
# get_next_token_probs vs the softmax of mt.logits: 0.0 where lm_head's output is the logits the model returns; a
# transform after the head (Gemma-2's final logit softcap) makes them differ
PROBS_TOL = 1e-6

# The rows of the report, in order.
PRIMITIVES = [
    "load + support()",
    "GoalScope additive",
    "GoalScope replace",
    "OVReader probe",
    "attention matrices",
    "head vectors, padded batch",
    "KV transfer",
    "Q transfer (GSR)",
    "sink ablation decode",
    "blind-span ablation decode",
    "OV boost (greedy_continue)",
    "temporal goal_encoding",
    "get_hs + next-token probs",
    "head-mask gradient step",
]


# ══════════════════════════════════════════════════════════════════════════════
# the run context: the model, the head set, the scaled constants, shared inputs
# ══════════════════════════════════════════════════════════════════════════════
class Run:
    def __init__(self, args):
        import src.nrgs.functional as FUNC

        self.args = args
        self.n = args.n_tokens
        self.ctx = FUNC.setup(args.model, device_map={"": 0})  # bf16, eager; binds src.nrgs.functional.mt
        self.mt = mt = self.ctx.mt
        L = mt.num_layers
        self.heads = self._heads(args.heads, args.seed)
        self.ctx.readout_heads = self.heads
        self.steer = (round(L * 35 / 64), round(L * 46 / 64))  # half-open, the 32B's (35, 46)
        self.probe_layer = round(L * 48 / 64)
        self.from_layer = round(L * 20 / 64)
        self._rollout = None

    def _heads(self, spec, seed):
        mt = self.mt
        if spec.startswith("random:"):
            n = int(spec.split(":", 1)[1])
            L = mt.num_layers
            pool = [(l, h) for l in range(L // 3, 2 * L // 3) for h in range(mt.num_heads)]
            return sorted(random.Random(seed).sample(pool, min(n, len(pool))))
        from src.goals.head_sets import load_json_heads

        return load_json_heads(spec, mt.num_layers, mt.num_heads)

    def rollout(self):
        """The GoalScope eval's per-spec prefix on ``int_python_000``: prompt -> greedy completion -> pivots."""
        if self._rollout is None:
            from scripts.evaluation.eval_goalscope import (
                ALL_DATATYPES, ANCHOR_ORDER, DEFAULT_DATASET, build_prompt, generate_completion,
                load_locked_datapoints, pivot_positions,
            )

            specs = dict(load_locked_datapoints(DEFAULT_DATASET, ALL_DATATYPES)[0])
            inst = specs["int_python_000"]
            prompt, var = build_prompt(self.mt, inst)
            full_text = prompt + generate_completion(self.mt, prompt, self.n)
            pivs = pivot_positions(self.mt, prompt, full_text, ANCHOR_ORDER)
            self._rollout = dict(inst=inst, specs=specs, prompt=prompt, full_text=full_text, pivots=pivs)
        return self._rollout


def _finite(x) -> bool:
    return bool(torch.isfinite(torch.as_tensor(x).float()).all())


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _gens(*texts) -> str:
    return f"{len(texts)} generation{'s' * (len(texts) > 1)}, {sum(len(t) for t in texts)} chars"


# ══════════════════════════════════════════════════════════════════════════════
# the primitives: each returns a one-line note, or raises
# ══════════════════════════════════════════════════════════════════════════════
INTERIOR = ["attention_probabilities", "attention_scores", "attention_queries", "attention_keys",
            "attention_values", "attention_head_outputs", "attention_premix"]


def p_load(R):
    mt = R.mt
    sup = mt.support()
    missing = [k for k in INTERIOR if sup.get(f"self_attn.{k}", "absent") is not None]
    norm = sup.get("attention_norm_output", "absent")
    out = (f"{mt.family_name}: {mt.num_layers} layers x {mt.num_heads} heads ({mt.num_kv_heads} kv, head_dim "
           f"{mt.head_dim}), hidden {mt.hidden_size}; attention interior "
           + ("available" if not missing else f"UNAVAILABLE: {missing}")
           + ("" if norm is None else f"; attention_norm_output unavailable ({str(norm)[:60]})"))
    _check(not missing, out)
    return out


def p_goalscope_additive(R):
    from src.goals.goalscope import apply_goalscope

    ro = R.rollout()
    pos = sorted(set(ro["pivots"].values()))[-2:]
    res = apply_goalscope(ro["full_text"], mt=R.mt, selected_heads=R.heads, strategy="additive_norm", alpha=0.1,
                          steer_layer_range=R.steer, max_new_tokens=R.n, token_positions=pos)
    _check([r["pos"] for r in res] == pos and all(isinstance(r["patched_gen"], str) for r in res), "bad rows")
    return f"{len(res)} readouts at pivots {pos}, steer layers {R.steer[0]}-{R.steer[1] - 1}"


def p_goalscope_replace(R):
    from src.goals.goalscope import apply_goalscope

    ro = R.rollout()
    pos = sorted(set(ro["pivots"].values()))[-2:]
    res = apply_goalscope(ro["full_text"], mt=R.mt, selected_heads=R.heads, strategy="replace", alpha=1.0,
                          max_new_tokens=R.n, token_positions=pos)
    _check([r["pos"] for r in res] == pos, "bad rows")
    return f"{len(res)} readouts, {len({l for l, _ in R.heads})} patched layers"


def p_ovreader(R):
    from src.goals.goalscope import OVReader

    ro = R.rollout()
    ov = OVReader(R.mt, R.heads)
    out = ov.apply_probe_on_prompt(ro["full_text"], ro["pivots"]["eq"], [R.probe_layer], mode="faithfulness",
                                   steer_layers=list(range(*R.steer)), max_new_tokens=R.n, alpha=0.15)
    vec = ov.resid_ov_vec(torch.ones(R.mt.hidden_size))
    _check(set(out) == {R.probe_layer} and _finite(vec), "bad output")
    return f"readout at probe layer {R.probe_layer}; OV vector finite"


def p_attention(R):
    from src.attention import get_attention_matrices

    mt = R.mt
    prompt = R.rollout()["prompt"]
    plain = get_attention_matrices(prompt, mt, value_weighted=False)
    vw = get_attention_matrices(prompt, mt, value_weighted=True)
    A = torch.from_numpy(plain.attention_matrices)
    L, H, T, T2 = A.shape
    _check((L, H) == (mt.num_layers, mt.num_heads) and T == T2 == len(plain.tokenized_prompt), f"shape {tuple(A.shape)}")
    dev = float((A.sum(-1) - 1).abs().max())
    _check(dev < 2e-2, f"pattern rows sum to 1 within {dev:.1e} only")
    _check(vw.attention_matrices.shape == A.shape and _finite(vw.attention_matrices), "value-weighted: bad output")
    return f"plain + value-weighted {tuple(A.shape)}; rows sum to 1 within {dev:.0e}"


def p_head_vecs_padded(R):
    from scripts.evaluation.eval_goalscope import build_prompt
    from src.goals.head_loc_attn import collect_head_steering_vecs
    from src.tokens import prepare_input

    mt, specs = R.mt, R.rollout()["specs"]
    prompts = [build_prompt(mt, specs[s])[0] for s in ("int_python_001", "list[int]_python_001")]
    tok = prepare_input(prompts=prompts, tokenizer=mt)
    T = tok["input_ids"].shape[1]
    n_pad = [int((tok["attention_mask"][i] == 0).sum()) for i in range(2)]
    sv = collect_head_steering_vecs(mt, tok, R.heads, token_positions=list(range(T - 3, T)))
    _check(tuple(sv.shape) == (2, 3, mt.hidden_size) and _finite(sv), f"shape {tuple(sv.shape)}")
    return f"(2, 3, {mt.hidden_size}), pads per row {n_pad}"


def _last_logits(mt, tok, interventions=None):
    """float32 last-position logits of one traced forward, under ``interventions(mt)`` when given."""
    with torch.no_grad(), mt.trace(dict(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"])):
        if interventions:
            interventions(mt)
        logits = mt.logits[0, -1].save()
    return logits.float().cpu()


def p_kv_transfer(R):
    """The KV transfer on one int py-py instance (clean SRC / DST, patched at V x4), then an identity check: the DST
    window transplanted onto itself (V x1) must leave the forward unchanged. It does only when the transplanted keys
    are the keys the model reads and the recomputed attention is the model's own (score function, mask)."""
    import src.nrgs.data as DATA
    import src.nrgs.kv_transfer as PP

    old = DATA.CPP_NOTE_IN_SYSTEM
    DATA.CPP_NOTE_IN_SYSTEM = True  # kv_transfer_eval.main sets this module global
    try:
        insts, _ = PP.sample_multi_instances("int", "python", "python", n_samples=32, seed=1123)
        g = PP.build_geometry(insts[0], span="task", ctx=R.ctx)
        clean = PP.clean_runs(g, readout_heads=R.heads, max_new=R.n, readout=False, ctx=R.ctx)
        pat = PP.run_patched(g, goal_heads=R.heads, readout_heads=R.heads, v_scale=V_SCALE, max_new=R.n,
                             readout=False, ctx=R.ctx)
        g_self = dict(g, src_tok=g["dst_tok"], src_w=g["dst_w"])
        iv = PP.intervention_stack(g_self, goal_heads=R.heads, v_scale=1.0, ctx=R.ctx)
        gap = float((_last_logits(R.mt, g["dst_tok"], iv) - _last_logits(R.mt, g["dst_tok"])).abs().max())
    finally:
        DATA.CPP_NOTE_IN_SYSTEM = old
    _check(gap <= KV_IDENTITY_TOL, f"a self-transplant moves the last logits by {gap:.2g} (max abs; it must leave "
                                   f"them bit-identical): the recomputed attention is not the model's")
    return (f"window {len(g['dst_w'])} tokens, V x{V_SCALE:g}; " + _gens(clean["gen_src"], clean["gen_dst"],
                                                                        pat["gen_patched"])
            + f"; self-transplant moves logits by {gap:.2g}")


def p_q_transfer(R):
    import hashlib

    import src.nrgs.data as DATA
    import src.nrgs.functional as FUNC

    seed = int(hashlib.md5(b"order_diff/int__python-python").hexdigest()[:8], 16) % (2**32)  # q_transfer_eval's
    inst = DATA.diff_trigger_setups("int", variant="order_diff", n_samples=32, seed=seed, src_lang="python",
                                    dst_lang="python")[0]
    raw = FUNC.run_q_transfer_intervention(R.mt, inst, R.heads, readout_heads=R.heads, max_new=R.n, alpha=Q_ALPHA,
                                           compute_readout=False, gsr_max_new=R.n)
    _check(raw["cl_selected"] in raw["cl_select_scores"] and len(raw["attn_clean"]) == len(raw["task_positions"]),
           "bad gaze fields")
    return f"selected 1 of {len(raw['cl_select_scores'])} tasks; " + _gens(raw["gen_src"], raw["gen_dst"],
                                                                           raw["gen_patched"])


def _ablation_inputs(R):
    from scripts.evaluation.eval_goalscope import build_prompt
    from src.tokens import prepare_input

    inst = R.rollout()["specs"]["str_python_001"]
    prompt, _ = build_prompt(R.mt, inst)
    ids = prepare_input(prompts=prompt, tokenizer=R.mt)["input_ids"][0].tolist()
    return inst, prompt, ids, len(ids) - 1


def _pattern_under(R, prompt, iv):
    """(layers, heads, T, T) attention pattern of ``prompt`` with the scores edit ``iv`` applied."""
    from src.attention import get_attention_matrices
    from src.tokens import prepare_input

    return get_attention_matrices(prepare_input(prompts=prompt, tokenizer=R.mt), R.mt, interventions=iv).attention_matrices


def p_sink(R):
    """Sink-ablation decode (baseline and the head set), then the edit itself: under ``whitelist_head_attn`` the sunk
    heads' pattern at the pivot row must put all its mass on key 0."""
    from scripts.evaluation.ablation_common import sink_generate_cached
    from src.nrgs.functional import whitelist_head_attn

    _, prompt, ids, q_start = _ablation_inputs(R)
    base = sink_generate_cached(R.mt, ids, [], q_start, sink_pos=0, max_new=R.n)
    abl = sink_generate_cached(R.mt, ids, R.heads, q_start, sink_pos=0, max_new=R.n)
    A = _pattern_under(R, prompt, whitelist_head_attn(R.mt, R.heads, [0], q_from=q_start))
    mass = min(float(A[l, h, q_start, 0]) for l, h in R.heads)
    _check(mass > 0.99, f"a sunk head keeps only {mass:.3f} of its pivot-row mass on key 0")
    return f"{len(R.heads)} heads sunk to key 0 (pivot-row mass on it >= {mass:.3f}); " + _gens(base, abl)


def p_blind(R):
    from scripts.evaluation.ablation_common import (
        complement_heads, frontier_block_positions, generate_blocking_span_cached,
    )
    from src.nrgs.functional import whitelist_head_attn

    inst, prompt, ids, q_start = _ablation_inputs(R)
    block = frontier_block_positions(R.mt, prompt, inst.expected_next_desc)
    _check(block, "frontier description not located in the prompt")
    blind = complement_heads(R.heads, R.from_layer, R.mt.num_layers, R.mt.num_heads)
    g = generate_blocking_span_cached(R.mt, ids, blind, block, q_start, max_new=R.n)
    keep = [k for k in range(len(ids)) if k not in set(block)]
    A = _pattern_under(R, prompt, whitelist_head_attn(R.mt, blind, keep, q_from=q_start))
    leak = max(float(A[l, h, q_start, block].sum()) for l, h in blind)
    _check(leak < 1e-3, f"a blinded head still puts {leak:.3f} of its pivot-row mass on the blocked keys")
    return (f"{len(blind)} heads (layers >= {R.from_layer}) blind to {len(block)} keys (mass left on them "
            f"{leak:.0e}); " + _gens(g))


def p_ov_boost(R):
    import scripts.evaluation.obstacle_persistence_ov_boost_eval as OVB
    from scripts.evaluation.obstacle_persistence_eval import build_obstacle_prompt
    from src.goals.goalscope import greedy_continue
    from src.nrgs.functional import boost_head_ov
    from src.tokens import prepare_input

    dt, pid = OVB.candidates()[0]
    prompt, _ = build_obstacle_prompt(R.mt, dt, pid, OVB.NUMERIC_OBSTACLE)
    inp = prepare_input(prompts=prompt, tokenizer=R.mt)
    g = greedy_continue(R.mt, **inp, max_new_tokens=R.n, interventions=boost_head_ov(R.mt, R.heads, N_BOOST))
    return f"x{N_BOOST:g} on {len(R.heads)} heads, every forward; " + _gens(g)


def p_temporal(R):
    import src.nrgs.data as D
    from src.goals import temporal_persistence as TP

    inst = D.multi_predicate_setups("int", n_conditions=list(TP.N_COND), n_samples=256, seed=1123,
                                    naming="letters", src_lang="python", dst_lang="python")[0]
    preds = inst._preds_for(inst.src_frontier)
    prompt, _ = TP.build_probe_prompt(R.mt, inst, preds, 1)
    full_text = TP.rollout_probe(R.mt, prompt, max_new=R.n)
    P = TP.prompt_boundary(R.mt, prompt, full_text)
    n_tok = len(TP._offsets(R.mt, full_text))
    positions = list(range(P - 1, min(P + 4, n_tok)))
    enc = TP.goal_encoding(R.mt, full_text, positions, R.heads)
    _check(tuple(enc.shape) == (len(positions), R.mt.hidden_size) and _finite(enc), f"shape {tuple(enc.shape)}")
    u = enc / enc.norm(dim=-1, keepdim=True)
    return f"encodings at {len(positions)} positions; cos(pivot, next) = {float(u[0] @ u[1]):.2f}"


def p_hs_probs(R):
    from src.functional import get_hs
    from src.goals.head_loc_pivot import get_next_token_probs

    mt, prompt = R.mt, R.rollout()["prompt"]
    mid = mt.num_layers // 2
    locs = [("embed_tokens", -1), (f"layers.{mid}", -1), (f"layers.{mt.num_layers - 1}", -1), ("norm", -1)]
    hs = get_hs(mt, prompt, locs, return_dict=True)
    _check(all(tuple(v.shape) == (mt.hidden_size,) and _finite(v) for v in hs.values()), "bad hidden states")
    probs = get_next_token_probs(mt, prompt)
    with torch.no_grad(), mt.trace(prompt):  # the logits the model returns (after any softcap)
        ref = mt.logits[0, -1].float().softmax(-1).save()
    gap = float((probs - ref.cpu()).abs().max())
    _check(abs(float(probs.sum()) - 1) < 1e-3, "probs do not sum to 1")
    _check(gap <= PROBS_TOL, f"get_next_token_probs differs from softmax(mt.logits) by {gap:.1e} (max abs)")
    return f"{len(hs)} locations (standard names); probs = softmax(mt.logits) within {gap:.0e}"


def p_head_mask_step(R):
    from src.goals.head_loc_attn import HeadMask, forward_with_src_replacement, get_o_proj_inputs_batch
    from src.goals.head_loc_data import HeadLocDataset
    from src.goals.head_loc_pivot import build_cpp_prompt, build_python_prompt, get_next_token_probs_batch
    from src.tokens import prepare_input

    mt = R.mt
    ds = HeadLocDataset.load(model_name="Qwen2.5-Coder-32B-Instruct")  # train_head_mask.py's fallback pool
    rows = []
    for pair, pi in ds.pivot_instances[:2]:
        a, b = ds.get_sample(pair.source_predicate), ds.get_sample(pair.destination_predicate)
        rows.append((build_cpp_prompt(mt, a, include_docstring=True),
                     build_python_prompt(mt, b, pivot_prefix=pi.pivot_prefix, include_docstring=True),
                     build_python_prompt(mt, a, pivot_prefix=pi.pivot_prefix, include_docstring=True)))
    src, dst, targ = (list(x) for x in zip(*rows))
    layers = list(range(mt.num_layers))
    req = [p.requires_grad for p in mt.parameters()]
    for p in mt.parameters():  # train_head_mask freezes the model
        p.requires_grad_(False)
    try:
        with torch.inference_mode():
            log_p_target = torch.log(get_next_token_probs_batch(mt, targ).clamp(min=1e-10))
            src_o = get_o_proj_inputs_batch(mt, src, layers)
        src_hv = torch.stack([torch.stack([src_o[ly][i] for ly in layers]) for i in range(len(src))])
        mask = HeadMask(len(layers), mt.num_heads)
        with torch.no_grad():
            mask.values.fill_(0.5)
        with torch.enable_grad():
            logits = forward_with_src_replacement(mt, prepare_input(dst, tokenizer=mt), src_hv, mask, layers)
            loss = F.kl_div(log_p_target.to(logits.device), F.log_softmax(logits, dim=-1), reduction="batchmean",
                            log_target=True) + 20.0 * mask.values.mean()
            loss.backward()
    finally:
        for p, q in zip(mt.parameters(), req):
            p.requires_grad_(q)
    grad = mask.values.grad
    _check(grad is not None and _finite(grad) and float(grad.abs().sum()) > 0, "no finite, nonzero mask gradient")
    return f"mask {tuple(grad.shape)}, loss {float(loss.detach()):.2f}, grad norm {float(grad.norm()):.2e}"


FUNCS = [p_load, p_goalscope_additive, p_goalscope_replace, p_ovreader, p_attention, p_head_vecs_padded,
         p_kv_transfer, p_q_transfer, p_sink, p_blind, p_ov_boost, p_temporal, p_hs_probs, p_head_mask_step]


# ══════════════════════════════════════════════════════════════════════════════
def _reason(exc: BaseException) -> str:
    """``Type: first line of the message (at file:line in function)``, the location being the innermost frame in
    this repo's ``src/`` or ``scripts/``."""
    where = None
    for fr in traceback.extract_tb(exc.__traceback__):
        rel = os.path.relpath(fr.filename, ROOT)
        if rel.startswith(("src/", "scripts/")) and not rel.endswith("cross_family_check.py"):
            where = f"{rel}:{fr.lineno} in {fr.name}"
    msg = (str(exc).strip().splitlines() or [""])[0][:220]
    return f"{type(exc).__name__}: {msg}" + (f" (at {where})" if where else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF id (or local path) of the checkpoint")
    ap.add_argument("--heads", default="random:60", help="random:N (middle third of the layers) or a json head set")
    ap.add_argument("--seed", type=int, default=0, help="seed of the random head set")
    ap.add_argument("--n-tokens", type=int, default=16, help="new tokens per generation and read-out")
    ap.add_argument("--out", default=None, help="write the summary json here")
    args = ap.parse_args()
    os.chdir(ROOT)  # data/ and head_sets/ are read relative to the repo root

    # No judge: swap ask_claude out before any repo module binds it (`from src.utils.oracle_llms import ask_claude`);
    # every repo import of this script happens after this line.
    import src.utils.oracle_llms as oracle

    def _no_judge(*a, **k):
        raise RuntimeError("cross_family_check: an LM-judge (ask_claude) call was attempted")

    oracle.ask_claude = _no_judge

    import nnsight
    import transformers

    from src.utils.experiment_utils import set_seed

    t0 = time.time()
    rows, R = [], None
    try:
        R = Run(args)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        rows.append(dict(primitive=PRIMITIVES[0], status="FAIL", reason=_reason(e), seconds=round(time.time() - t0, 1)))
        rows += [dict(primitive=p, status="FAIL", reason="not run: the model did not load", seconds=0.0)
                 for p in PRIMITIVES[1:]]
    if R is not None:
        for name, fn in zip(PRIMITIVES, FUNCS):
            set_seed(args.seed)
            t1 = time.time()
            try:
                note, status = fn(R), "OK"
            except Exception as e:  # noqa: BLE001  one row per primitive, keep going
                print(f"[cross_family] {name}: FAIL", file=sys.stderr)
                traceback.print_exc()
                note, status = _reason(e), "FAIL"
            rows.append(dict(primitive=name, status=status, reason=note, seconds=round(time.time() - t1, 1)))
            print(f"[cross_family] {name}: {status} ({time.time() - t1:.1f}s)", flush=True)

    w = max(len(p) for p in PRIMITIVES)
    print(f"\n{args.model}" + (f" ({R.mt.family_name})" if R else ""))
    print(f"{'PRIMITIVE':<{w}} | {'STATUS':<6} | REASON")
    print(f"{'-' * w}-+--------+-------")
    for r in rows:
        print(f"{r['primitive']:<{w}} | {r['status']:<6} | {r['reason']}")
    n_ok = sum(r["status"] == "OK" for r in rows)
    print(f"{n_ok}/{len(rows)} OK in {time.time() - t0:.0f}s")

    if args.out:
        mt = R.mt if R else None
        summary = dict(
            model=args.model,
            family=mt.family_name if mt else None,
            sizes=(dict(num_layers=mt.num_layers, num_heads=mt.num_heads, num_kv_heads=mt.num_kv_heads,
                        head_dim=mt.head_dim, hidden_size=mt.hidden_size) if mt else None),
            heads=dict(spec=args.heads, seed=args.seed, n=len(R.heads) if R else None,
                       layers=[min(l for l, _ in R.heads), max(l for l, _ in R.heads)] if R else None),
            constants=(dict(n_tokens=args.n_tokens, steer_layers=list(R.steer), ov_probe_layer=R.probe_layer,
                            blind_from_layer=R.from_layer) if R else dict(n_tokens=args.n_tokens)),
            versions=dict(nnterp=nnterp.__version__, nnsight=nnsight.__version__,
                          transformers=transformers.__version__, torch=torch.__version__),
            dtype="bfloat16", attn_implementation="eager",
            n_ok=n_ok, n_primitives=len(rows),
            results=[{k: r[k] for k in ("primitive", "status", "reason")} for r in rows],
        )
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
        print(f"wrote {out}")
    sys.exit(0 if n_ok == len(rows) else 1)


if __name__ == "__main__":
    main()
