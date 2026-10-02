"""Equivalence check for the SVD-component query patch in run_q_transfer_intervention:
P = I -> identical to the full head replacement (alpha=2);  P = 0 at alpha=1 -> identical to clean DST
(at alpha != 1 the OV boost of the patch heads stays on, so P = 0 is not the clean run)."""
import sys
sys.path.append(".")
import torch

import src.diff.data as D
import src.diff.functional as FUNC
import src.diff.nrgs_prefix as N

ctx = N.load_ctx("Qwen/Qwen2.5-Coder-14B-Instruct")
mt, d = ctx.mt, ctx.mt.head_dim
import json
heads = [tuple(x) for x in json.load(open("head_sets/Qwen2.5-Coder-14B-Instruct/goal__lam20_thr0.9_band0-48.json"))]
eye, zero = {}, {}
for l, h in heads:
    eye.setdefault(l, {})[h] = torch.eye(d); zero.setdefault(l, {})[h] = torch.zeros(d, d)
for inst in D.diff_trigger_setups("int", variant="order_diff", n_samples=4, seed=0, letter_stub="var_eq"):
    kw = dict(n_steer=8, max_new=40, n_pivot=1, compute_gaze=False, compute_readout=False, ctx=ctx)
    full = FUNC.run_q_transfer_intervention(mt, inst, heads, alpha=2.0, **kw)            # P = I vs full, with OV boost
    ident = FUNC.run_q_transfer_intervention(mt, inst, heads, component_proj=eye, alpha=2.0, **kw)
    zer = FUNC.run_q_transfer_intervention(mt, inst, heads, component_proj=zero, alpha=1.0, **kw)  # P = 0, no boost
    print("P=I == full replacement:", ident["gen_patched"] == full["gen_patched"],
          "| P=0 == clean DST:", zer["gen_patched"] == full["gen_dst"],
          "| full differs from clean:", full["gen_patched"] != full["gen_dst"])
    assert ident["gen_patched"] == full["gen_patched"] and zer["gen_patched"] == full["gen_dst"]
print("OK")
