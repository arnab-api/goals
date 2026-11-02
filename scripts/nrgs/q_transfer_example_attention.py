"""Per-token pivot attention over the task span for ONE gsr instance (for the gsr schematic).

Re-runs ``run_q_transfer_intervention`` (gaze_select_restrict) exactly as
``q_transfer_eval.py`` does for the cell, and records, for the chosen instance:
  - value-weighted attention from the DST pivot to every task-description token, mean over the
    goal heads, with the SRC query transplanted (the selection pass) and on the clean DST run;
  - the per-task gaze scores (must match ``cl_select_scores`` in the stored results.json).

Usage (one GPU is enough for Coder-32B; the paper's example):
  python scripts/nrgs/q_transfer_example_attention.py --cuda 5 --datatype int --lang py-py \
      --variant order_diff --key-contains "dst=is_prime-is_positive-last_digit"
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda", default=None)
    ap.add_argument("--datatype", default="int")
    ap.add_argument("--lang", default="py-py")
    ap.add_argument("--variant", default="order_diff")
    ap.add_argument("--key-contains", default="dst=is_prime-is_positive-last_digit")
    a = ap.parse_args()
    if a.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = a.cuda

    ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    spec = importlib.util.spec_from_file_location("dte", ROOT / "scripts/nrgs/q_transfer_eval.py")
    dte = importlib.util.module_from_spec(spec)
    sys.argv = ["q_transfer_eval.py", "--datatypes", a.datatype, "--variants", a.variant, "--lang-transfers", a.lang]
    spec.loader.exec_module(dte)
    args = dte.parse_args()   # the diff_trigger defaults = the reported cell (seed, n_samples, alpha, heads)

    import numpy as np  # noqa: E402
    import src.attention as ATT  # noqa: E402

    DATA, FUNC = dte.DATA, dte.FUNC

    # capture every attention-matrix call (patched selection pass vs clean) and the pivot_attn lookups
    CAP = {"infos": [], "pivot_calls": []}
    _gam = ATT.get_attention_matrices
    def _gam_wrap(*args_, **kw):
        info = _gam(*args_, **kw)
        CAP["infos"].append(dict(patched=kw.get("legacy_hooks") is not None, info=info[0] if isinstance(info, list) else info))
        return info
    ATT.get_attention_matrices = _gam_wrap
    _pa = FUNC.pivot_attn
    def _pa_wrap(info, heads, positions, pivot):
        CAP["pivot_calls"].append(dict(positions=list(positions), pivot=int(pivot)))
        return _pa(info, heads, positions, pivot)
    FUNC.pivot_attn = _pa_wrap

    ctx = FUNC.setup(dte.MODEL)
    mt = ctx.mt
    heads = dte._resolve_heads(args.patch_heads)

    sl, dl = dte.LANG_TRANSFERS[a.lang]
    dt = a.datatype
    tag = f"{dt.replace('[', '(').replace(']', ')')}__{sl}-{dl}"   # q_transfer_eval's cell tag
    cfg_seed = (args.seed + int(hashlib.md5(f"{a.variant}/{tag}".encode()).hexdigest()[:8], 16)) % (2**32)
    dte.set_seed(cfg_seed)
    setups = DATA.diff_trigger_setups(dt, variant=a.variant, n_samples=args.n_samples, seed=cfg_seed, src_lang=sl, dst_lang=dl)
    inst = next(i for i in setups if a.key_contains in dte._inst_key(i))
    key = dte._inst_key(inst)
    print("instance:", key)

    raw = FUNC.run_q_transfer_intervention(
        mt, inst, heads, readout_heads=heads, max_new=dte.MAX_NEW, alpha=args.alphas[0], compute_readout=True,
        gsr_max_new=dte.GSR_MAX_NEW)

    # the selection pass is the patched get_attention_matrices call whose pivot_attn lookup follows it
    sel_call = CAP["pivot_calls"][0]
    positions, piv = sel_call["positions"], sel_call["pivot"]
    patched = next(c["info"] for c in CAP["infos"] if c["patched"])
    clean = next(c["info"] for c in CAP["infos"] if not c["patched"])   # the clean gaze pass of run_q
    a_p = _pa(patched, heads, positions, piv)
    a_c = _pa(clean, heads, positions, piv)
    toks = [patched.tokenized_prompt[q] for q in positions]

    mat = DATA.materialize(inst, ctx=ctx)
    dst_prompt = mat["dst_prompt"]
    sel_items = {p.id: list(range(*FUNC.token_span(dst_prompt, p.description, ctx=ctx))) for p in inst.task.predicates}
    scores = {pid: float(np.mean([a_p[positions.index(q)] for q in ps])) for pid, ps in sel_items.items()}
    print("recomputed scores:", scores)
    print("stored-style raw scores:", raw.get("cl_select_scores"), "selected:", raw.get("cl_selected"))

    out = dict(
        key=key, datatype=dt, lang=a.lang, variant=a.variant, pivot=piv, positions=positions, tokens=toks,
        attn_patched=[float(x) for x in a_p], attn_clean=[float(x) for x in a_c],
        task_positions={pid: ps for pid, ps in sel_items.items()},
        task_desc={p.id: p.description for p in inst.task.predicates},
        scores_patched=scores,
        scores_clean={pid: float(np.mean([a_c[positions.index(q)] for q in ps])) for pid, ps in sel_items.items()},
        cl_select_scores=raw.get("cl_select_scores"), cl_selected=raw.get("cl_selected"),
        src_prompt=mat.get("src_prompt"), dst_prompt=dst_prompt,
        gen_src=raw["gen_src"], gen_dst=raw["gen_dst"], gen_patched=raw["gen_patched"],
        ro_src=raw.get("ro_src"), ro_dst=raw.get("ro_dst"), ro_patched=raw.get("ro_patched"),
        heads=args.patch_heads,
    )
    # legacy results dir names (diff/, gsr_example/), kept for cache compatibility
    fp = (ROOT / "results/diff/Qwen2.5-Coder-32B-Instruct/gsr_example"
          / f"{a.variant}__{tag}__{hashlib.md5(key.encode()).hexdigest()[:8]}.json")
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps(out, indent=1))
    print("saved", fp)


if __name__ == "__main__":
    main()
