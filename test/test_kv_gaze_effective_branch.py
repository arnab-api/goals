"""Smoke test: under headwise kv routing, the attention weights and value norms that gaze uses must describe
the branch each query head actually reads (routed heads: donor k2/v2; others, incl. GQA siblings: recipient).

Checks, on real kv-design instances (p_kv = A'BD, p_dst = A'XY, goal-250 heads):
  1. reconstruction: for routed heads and for non-routed siblings, the head's o_proj input at the pivot
     equals sum_s w[h, piv, s] * v_eff[h, s], with w = the returned attention and v_eff = the branch's values;
  2. the returned weights of routed heads DIFFER from the recipient-branch weights (the old behaviour),
     so the check above can actually detect the bug;
  3. value norms used by get_attention_matrices(value_weighted=True) equal ||v2|| (routed) / ||v|| (others);
  4. logits are identical with and without output_attentions (diagnostics do not change the computation).

Usage:  CUDA_VISIBLE_DEVICES=5 conda run -n goals python test/test_kv_gaze_effective_branch.py
"""
import os
import sys

sys.path.append(".")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "5")

import torch

import scripts.diff.diff_trigger_eval as DTE
import src.attention as ATT
import src.diff.data as DATA
import src.diff.functional as FUNC
from src.functional import prepare_input

HEADS = "head_sets/Qwen2.5-Coder-32B-Instruct/goal_topk/goal__lam20_top250.json"


def kv_setup(ctx, dt, k):
    setups = DATA.diff_trigger_setups(dt, variant="order_diff", n_samples=8, seed=1234, naming="letters",
                                      letter_stub="var_eq", src_lang="python", dst_lang="python",
                                      distinct_families=True, nontrivial_frontier=True)
    for inst in setups[k:]:
        roles = DTE._kv_roles(dt, inst, 1234, distinct=True)
        if roles is None:
            continue
        kv_prompt = DATA.materialize(inst)["dst_prompt"]
        import dataclasses
        run_inst = dataclasses.replace(inst, dst_order=(roles["A"], roles["X"], roles["Y"]), dst_native=roles["X"].id)
        dst_prompt = DATA.materialize(run_inst)["dst_prompt"]

        def span(p, preds):
            sp = [FUNC.token_span(p, q.description, ctx=ctx) for q in preds]
            return min(a for a, _ in sp), max(b for _, b in sp)

        kv_tok, dst_tok, kv_pos, dst_pos, _, _ = FUNC.align_kv_spans(
            ctx.mt, kv_prompt, dst_prompt, span(kv_prompt, inst.dst_order), span(dst_prompt, run_inst.dst_order),
            mode="pad", ctx=ctx)
        return kv_tok, dst_tok, kv_pos, dst_pos
    raise RuntimeError("no kv instance")


def main():
    ctx = FUNC.setup("Qwen/Qwen2.5-Coder-32B-Instruct")
    mt = ctx.mt
    heads = DTE._resolve_heads(ctx, HEADS)
    hd, rep = mt.head_dim, mt.n_heads // mt.config.num_key_value_heads
    fails = []

    for dt, k in (("int", 0), ("list[str]", 0)):
        kv_tok, dst_tok, kv_pos, dst_pos = kv_setup(ctx, dt, k)
        kvh = FUNC.heads_per_layer(heads, "k", layers=set(FUNC._layers_of(heads, ctx=ctx)), ctx=ctx)
        K = FUNC.cache_src_proj(kv_tok, kv_pos, sorted(kvh), "k", ctx=ctx)
        V = FUNC.cache_src_proj(kv_tok, kv_pos, sorted(kvh), "v", ctx=ctx)
        piv = dst_tok["input_ids"].shape[1] - 1
        by_layer = {}
        for l, h in heads:
            by_layer.setdefault(l, set()).add(h)
        # test layers: a few routed layers that also contain non-routed siblings of a routed head
        test_layers = [l for l in sorted(by_layer) if any((h // rep) * rep + i not in by_layer[l]
                                                          for h in by_layer[l] for i in range(rep))][:4]
        cap = {}
        hooks = [FUNC._attn_module(mt, l).o_proj.register_forward_pre_hook(
            (lambda l: lambda m, a: cap.__setitem__(l, a[0][0, piv].float()))(l)) for l in test_layers]
        ATT._EFFECTIVE_VALUE_NORMS.clear()
        with torch.no_grad(), FUNC.headwise_kv_patch(mt, heads, K, V, dst_pos) as state:
            out = mt._model(**dst_tok, output_attentions=True, use_cache=True)
            eff = dict(ATT._EFFECTIVE_VALUE_NORMS)
            out_noattn = mt._model(**dst_tok, output_attentions=False, use_cache=False)
        for hk in hooks:
            hk.remove()
        ld = (out.logits[0, -1].float() - out_noattn.logits[0, -1].float()).abs().max().item()
        print(f"[{dt}] logits |Δ| with vs without output_attentions: {ld:.2e}")
        if ld > 0:
            fails.append(f"{dt}: logits changed by diagnostics ({ld})")

        # recipient-branch weights for comparison (no kv)
        with torch.no_grad():
            out_clean = mt._model(**dst_tok, output_attentions=True, use_cache=False)

        span = state["span"]
        for l in test_layers:
            w = out.attentions[l][0, :, piv].float()                 # (H, S) returned weights
            v = out.past_key_values.layers[l].values[0].float() if hasattr(out.past_key_values, "layers") \
                else out.past_key_values[l][1][0].float()            # (kvH, S, hd) recipient values (cache)
            v2 = v.clone()
            for g, sv in state["src_v"][l].items():
                v2[g, span] = sv.float().to(v2.device)
            routed = sorted(by_layer[l])
            sib = sorted({(h // rep) * rep + i for h in routed for i in range(rep)} - set(routed))
            for h in routed + sib:
                vv = (v2 if h in routed else v)[h // rep]           # (S, hd)
                recon = w[h] @ vv
                real = cap[l][h * hd:(h + 1) * hd]
                err = (recon - real).abs().max().item()
                tol = 0.03125 + 0.015625 * real.abs().max().item()
                if err > tol:
                    fails.append(f"{dt} L{l}h{h} ({'routed' if h in routed else 'sibling'}) recon err {err:.3g} > {tol:.3g}")
                vn_expect = vv.norm(dim=-1)
                vn_err = (eff[l][0, h].float() - vn_expect).abs().max().item()
                if vn_err > 1e-2 * vn_expect.max().item():
                    fails.append(f"{dt} L{l}h{h} value-norm err {vn_err:.3g}")
            wd = max((w[h] - out_clean.attentions[l][0, h, piv].float().to(w.device)).abs()[span].max().item()
                     for h in routed)
            sd = max((w[h] - out_clean.attentions[l][0, h, piv].float().to(w.device)).abs().max().item()
                     for h in sib) if sib else 0.0
            print(f"[{dt}] L{l}: {len(routed)} routed + {len(sib)} siblings checked · routed w vs recipient-branch "
                  f"max|Δ| on span = {wd:.3g} (should be > 0) · sibling w vs clean max|Δ| = {sd:.3g}")
            if wd < 1e-4:
                fails.append(f"{dt} L{l}: routed weights equal recipient weights — test cannot detect the bug")

    print("\nFAIL:\n  " + "\n  ".join(fails) if fails else "\nALL CHECKS PASSED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
