#!/usr/bin/env python
"""Depth-collapse layerwise eval — by which layer L is the next condition (a) faithfully readable and
(b) the target of the goal heads' gaze, if ALL goal heads collectively read the residual h^L?

Same prefill-to-frontier corpus / prompt / oracle as ``eval_ov_reader.py`` (QTASKS letters-naming,
prompt ends at the frontier's ``{var} =`` pivot). The intervention is the notebook 10_disent
depth-collapse: overwrite every goal-head layer's BLOCK INPUT with the cached residual h^L (so the
model applies its own RMSNorm + RoPE), then, at the frontier pivot (the ``=`` token), measure:

  * TOOL 2 (faithful read-out): the goal heads' collective OV write, verbalized through GoalScope
    (``apply_goalscope`` additive_norm) and graded by the SAME oracle as eval_goalscope/eval_ov_reader
    (``hard_eval`` faithfulness) against the frontier's actual code. -> per-layer hard_yes rate.
  * TOOL 1 (gaze): value-weighted pivot->task-list attention (all heads reading h^L). Alignment with the
    next condition = did the pivot look MOST at the frontier predicate's description span (argmax over the
    T predicate spans), and how much of the on-task mass landed there. -> per-layer argmax-hit / mass-frac.

Distinct from ``eval_ov_reader.py`` (which reads each residual through the OV circuit assuming full
attention to the pivot position): here the heads run their REAL attention over the whole context, all
collapsed onto h^L.

Usage:
    conda run -n goals python scripts/evaluation/eval_collapse_layerwise.py --cuda 2,3,6,7
    # smaller/faster:
    ... --n_samples 16 --datatypes list[str] --layers 24 28 32 36 40 44 48 --max_new_tokens 32
"""
import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from contextlib import contextmanager

sys.path.append(".")

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--cuda", default=None)
_cuda = _pre.parse_known_args()[0].cuda
if _cuda is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_cuda)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import baukit  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import src.diff.data as DATA  # noqa: E402
from scripts.evaluation.eval_goalscope import hard_eval, load_json_heads  # noqa: E402
from scripts.evaluation.eval_ov_reader import (  # noqa: E402
    build_prompt,
    frontier_impl_span,
    sample_datapoints,
)
from src.attention import get_attention_matrices  # noqa: E402
from src.goals.goalscope import apply_goalscope, greedy_continue  # noqa: E402
from src.models import ModelandTokenizer  # noqa: E402
from src.tokens import prepare_input  # noqa: E402
from src.utils import env_utils, logging_utils  # noqa: E402

logger = logging.getLogger(__name__)

STEER_LAYER_RANGE = (35, 46)  # GoalScope steer band (same as eval_ov_reader skyline)
DEFAULT_HEADS_JSON = "head_sets/{slug}/goal__lam20_thr0.9_band25-56.json"  # the 122 goal heads


# ══════════════════════════════════════════════════════════════════════════════
# depth-collapse mechanism (verbatim logic from notebooks/diff/10_disent.ipynb, mt passed explicitly)
# ══════════════════════════════════════════════════════════════════════════════
def resid_all(mt, tok, layer):
    """Residual after ``layer`` at EVERY position -> (seq, n_embd) CPU f32 (one clean forward)."""
    ln = mt.layer_name_format.format(layer)
    unt = lambda x: x[0] if isinstance(x, tuple) else x
    with torch.no_grad(), baukit.TraceDict(mt._model, [ln], retain_output=True) as tr:
        mt._model(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"])
    return unt(tr[ln].output)[0].detach().float().cpu()


@contextmanager
def read_from_layer(mt, h_L, heads):
    """Force each layer in ``heads`` to read residual h_L (seq, n_embd) as its block input this forward.
    Seq-length-guarded, so it is a no-op on forwards of a different length (e.g. the GoalScope probe gen)."""
    ref = next(mt._model.parameters())
    hL = h_L.to(ref.device, ref.dtype)
    handles = []
    try:
        for L in sorted({l for l, _ in heads}):
            blk = baukit.get_module(mt._model, mt.layer_name_format.format(L))

            def pre_hook(module, args, kwargs):
                hs = args[0] if args else kwargs.get("hidden_states")
                if hs is None or hs.shape[1] != hL.shape[0]:
                    return None
                new = hL.unsqueeze(0).expand(hs.shape[0], -1, -1).to(hs.dtype).to(hs.device)
                if args:
                    return (new,) + tuple(args[1:]), kwargs
                kwargs = dict(kwargs)
                kwargs["hidden_states"] = new
                return args, kwargs

            handles.append(blk.register_forward_pre_hook(pre_hook, with_kwargs=True))
        yield
    finally:
        for h in handles:
            h.remove()


def read_from_layer_iface(mt, h_L, heads):
    # NB: get_attention_matrices calls this as patch_interface(mt=mt, inputs=..., ...), so the first
    # parameter MUST be named `mt` (it shadows the closed-over mt with the same object).
    def iface(mt, inputs, patches=None, model_kwargs=None):
        with read_from_layer(mt, h_L, heads):
            return mt._model(**dict(inputs), **dict(model_kwargs or {}))

    return iface


# ══════════════════════════════════════════════════════════════════════════════
# the two per-datapoint measurements
# ══════════════════════════════════════════════════════════════════════════════
def collapsed_readout_str(mt, full_text, pivot_pos, heads, h_L, alpha, max_new, goalscope_prompt):
    """TOOL 2: verbalize the goal heads' collective OV write when all read h_L (GoalScope additive_norm)."""
    iface = read_from_layer_iface(mt, h_L, heads)
    res = apply_goalscope(
        full_text, mt=mt, selected_heads=heads, strategy="additive_norm",
        steer_layer_range=STEER_LAYER_RANGE, alpha=alpha, n_steer_tokens="all",
        max_new_tokens=max_new, show=False, include_noncode=False,
        token_positions=[pivot_pos], goalscope_prompt=goalscope_prompt, patch_interface=iface,
    )
    return res[0]["patched_gen"] if res else ""


def collapsed_gaze_alignment(mt, tok, pivot_pos, heads, h_L, pred_spans, frontier_idx):
    """TOOL 1: value-weighted pivot->task attention (all heads read h_L); does it look MOST at the
    frontier predicate span? Returns (argmax_hit, frontier_mass_frac, per_pred_mass)."""
    iface = read_from_layer_iface(mt, h_L, heads)
    info = get_attention_matrices(tok, mt, value_weighted=True, patch_interface=iface)
    A = info.attention_matrices  # (L, H, q, s)
    # mean value-weighted attention from the pivot onto each key position, averaged over goal heads
    per_pos = np.stack([A[l, h, pivot_pos, :] for (l, h) in heads]).mean(0)  # (s,)
    # per-predicate: mean attention per token in that description span (length-normalized)
    pred_mean = [per_pos[lo:hi].mean() if hi > lo else 0.0 for (lo, hi) in pred_spans]
    pred_sum = [float(per_pos[lo:hi].sum()) for (lo, hi) in pred_spans]
    total = sum(pred_sum) or 1e-9
    argmax_hit = int(int(np.argmax(pred_mean)) == frontier_idx)
    frontier_mass_frac = pred_sum[frontier_idx] / total
    return argmax_hit, frontier_mass_frac, pred_sum


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    ap.add_argument("--cuda", default=None, help="CUDA_VISIBLE_DEVICES (applied before torch import)")
    ap.add_argument("--datatypes", nargs="+", default=["list[str]"], choices=list(DATA.DATATYPES))
    ap.add_argument("--lang", default="python", choices=["python", "cpp"])
    ap.add_argument("--n_conditions", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--n_samples", type=int, default=16, help="datapoints per datatype")
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="collapse layers L to sweep (default: 20..n_layer step 2)")
    ap.add_argument("--heads", default=None, help="json head set (default: 122-head goal band 25-56)")
    ap.add_argument("--alpha", type=float, default=0.1, help="GoalScope read-out alpha (additive_norm)")
    ap.add_argument("--max_new_tokens", type=int, default=32, help="read-out length")
    ap.add_argument("--gen_max_new_tokens", type=int, default=64, help="frontier-body greedy budget")
    ap.add_argument("--faith_thr", type=float, default=0.5, help="hard_yes threshold for 'earliest layer'")
    ap.add_argument("--gaze_thr", type=float, default=0.5, help="argmax-hit threshold for 'earliest layer'")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", default=None, help="output dir (default: <results>/evaluation/collapse_layerwise/<model>)")
    logging_utils.add_logging_args(ap)
    return ap.parse_args()


def main():
    args = parse_args()
    logging_utils.configure(args)
    mt = ModelandTokenizer(model_key=args.model, dtype=torch.bfloat16,
                           device_map="auto", attn_implementation="eager")
    torch.set_grad_enabled(False)
    slug = args.model.split("/")[-1]
    heads = load_json_heads(args.heads or DEFAULT_HEADS_JSON.format(slug=slug), mt.n_layer, mt.n_heads)
    heads = [tuple(h) for h in heads]
    layers = args.layers or list(range(20, mt.n_layer, 2))
    logger.info(f"{slug} n_layer={mt.n_layer} | {len(heads)} goal heads "
                f"(L{min(l for l,_ in heads)}-{max(l for l,_ in heads)}) | sweep {layers}")

    # corpus: exactly eval_ov_reader's sampler, pooled over datatypes to n=sum
    datapoints = []
    for dt in args.datatypes:
        for i, inst in enumerate(sample_datapoints(dt, args.n_conditions, args.n_samples, args.lang, args.seed)):
            datapoints.append((f"{dt}_{args.lang}_{i:03d}", inst))
    logger.info(f"n = {len(datapoints)} datapoints")

    # accumulators
    faith_hits = defaultdict(list)   # layer -> [0/1 hard_yes]
    gaze_hits = defaultdict(list)    # layer -> [0/1 argmax hit]
    gaze_mass = defaultdict(list)    # layer -> [frontier mass frac]
    records = []
    t0 = time.time()

    for n, (sid, inst) in enumerate(datapoints, 1):
        prompt, var = build_prompt(mt, inst, args.lang)
        completion = greedy_continue(mt, **prepare_input(prompts=prompt, tokenizer=mt),
                                     max_new_tokens=args.gen_max_new_tokens)
        full_text = prompt + completion
        impl_span = frontier_impl_span(prompt, completion, var)
        frontier_desc = inst.expected_next_desc

        # per-predicate description spans + pivot, in the PROMPT tokenization (pivot = the `=` token)
        ptok = prepare_input(prompts=prompt, tokenizer=mt)
        pivot_prompt = ptok["input_ids"].shape[1] - 1
        try:
            from src.tokens import find_token_range
            pred_spans = [find_token_range(prompt, p.description, mt.tokenizer) for p in inst.task.predicates]
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[{sid}] span-locate failed ({e}); gaze skipped for this datapoint")
            pred_spans = None
        frontier_idx = [p.id for p in inst.task.predicates].index(inst.expected_next)

        # pivot in the FULL-text tokenization (what apply_goalscope sees) — last prompt token
        pivot_full = pivot_prompt  # prompt is a prefix of full_text; `=` index is stable

        for L in layers:
            hL_prompt = resid_all(mt, ptok, L)   # for gaze (prompt-length residual)
            # read-out (Tool 2) — needs residual at full_text length
            hL_full = resid_all(mt, prepare_input(prompts=full_text, tokenizer=mt), L)
            ro = collapsed_readout_str(mt, full_text, pivot_full, heads, hL_full,
                                       args.alpha, args.max_new_tokens, None)
            hv = hard_eval(ro, impl_span, "faithfulness")
            is_yes = int(hv["verdict"] == "YES")
            faith_hits[L].append(is_yes)

            rec = dict(spec=sid, layer=L, readout=ro, impl_span=impl_span,
                       condition=frontier_desc, hard=hv["verdict"], faith_yes=is_yes)
            if pred_spans is not None:
                ah, fm, _ = collapsed_gaze_alignment(mt, ptok, pivot_prompt, heads, hL_prompt,
                                                     pred_spans, frontier_idx)
                gaze_hits[L].append(ah)
                gaze_mass[L].append(fm)
                rec.update(gaze_argmax_hit=ah, gaze_frontier_massfrac=round(fm, 4))
            records.append(rec)
        logger.info(f"[{n}/{len(datapoints)}] {sid} done ({time.time()-t0:.0f}s)")

    # ── summarize ──
    def rate(d, L):
        return float(np.mean(d[L])) if d[L] else float("nan")

    summary = {"model": slug, "n": len(datapoints), "layers": layers,
               "faith_thr": args.faith_thr, "gaze_thr": args.gaze_thr, "by_layer": {}}
    for L in layers:
        summary["by_layer"][L] = dict(
            faithfulness=round(rate(faith_hits, L), 4),
            gaze_argmax_hit=round(rate(gaze_hits, L), 4),
            gaze_frontier_massfrac=round(rate(gaze_mass, L), 4),
        )
    faith_cross = next((L for L in layers if rate(faith_hits, L) >= args.faith_thr), None)
    gaze_cross = next((L for L in layers if rate(gaze_hits, L) >= args.gaze_thr), None)
    summary["earliest_layer"] = dict(faithfulness=faith_cross, gaze_argmax_hit=gaze_cross)

    out_dir = args.out or os.path.join(
        env_utils.DEFAULT_RESULTS_DIR, "evaluation", "collapse_layerwise", slug)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "records.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ── print ──
    print("\n" + "=" * 68)
    print(f"depth-collapse layerwise · {slug} · n={len(datapoints)} · heads={len(heads)}")
    print(f"{'layer L':>8} | {'faithful':>9} | {'gaze argmax':>11} | {'gaze mass':>9}")
    print("-" * 48)
    for L in layers:
        s = summary["by_layer"][L]
        print(f"{L:>8} | {s['faithfulness']:>9.3f} | {s['gaze_argmax_hit']:>11.3f} | {s['gaze_frontier_massfrac']:>9.3f}")
    print("-" * 48)
    print(f"earliest layer  faithful>= {args.faith_thr}: {faith_cross}   "
          f"gaze argmax>= {args.gaze_thr}: {gaze_cross}")
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
