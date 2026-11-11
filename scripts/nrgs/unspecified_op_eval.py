#!/usr/bin/env python
"""Broader-goal KV transfer (``tab:kv-goal``): the prompt names only an algorithm, the next operation is never written
(``src.nrgs.unspecified_data``).

    p_src: "Implement A in <lang>." + shared prefix -> A     (k/v donor)
    p_dst: "Implement B in <lang>." + shared prefix -> B     (patched)    pivot = the indentation token

From the pivot on, ``--patch-heads`` read SRC's k/v over the algorithm NAME (values x ``--v-scale``); every other head
and position is untouched (:func:`src.nrgs.kv_transfer.intervention_stack`). Greedy ``--max-new`` tokens per run (the
whole function); the oracle (``classify_concept``) judges which algorithm the WHOLE function implements, and which one
the GoalScope readout at each pivot names (``--readout-heads``, 64 tokens). causal-shift = the patched run implements
SRC's algorithm (BOTH is a miss). Defaults = the paper's configuration: the 250 goal heads, V x 4, 512 tokens, all 12
pairs x 2 directions x 4 language pairs (n = 96).

Results: ``results/kv_transfer/<model>/broader_goal/<run_tag>/{results.json, summary.json}`` (resumable per instance).

    conda run -n goals python scripts/nrgs/unspecified_op_eval.py --cuda 6
    conda run -n goals python scripts/nrgs/unspecified_op_eval.py --cuda 6 --pairs merge_quick --langs py-cpp --results-dir /tmp/x   # smoke
"""

import argparse
import collections
import json
import logging
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _preparse_cuda(argv):
    for i, a in enumerate(argv):
        if a == "--cuda" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--cuda="):
            return a.split("=", 1)[1]
    return None


_cuda = _preparse_cuda(sys.argv[1:])   # CUDA_VISIBLE_DEVICES must be set before torch is imported
if _cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = _cuda
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402

import src.nrgs.functional as FUNC  # noqa: E402
import src.nrgs.kv_transfer as PP  # noqa: E402
import src.nrgs.unspecified_data as U  # noqa: E402
from src.utils import env_utils, logging_utils  # noqa: E402

logger = logging.getLogger("unspecified_op_eval")

MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"
GOAL250 = "head_sets/Qwen2.5-Coder-32B-Instruct/goal_topk/goal__lam20_top250.json"
RO_MAX_NEW = 64
ORACLE_MODEL = "claude-sonnet-4-6"


def _load_heads(path):
    return [tuple(h) for h in json.load(open(path))]


def _heads_tag(path, n):
    stem = Path(path).stem
    name = "goal" if stem.startswith("goal__") else stem.split("__")[0]
    return f"{name}_{n}"


def summarize(records) -> dict:
    rate = lambda rs, f: (sum(f(r) for r in rs) / len(rs)) if rs else None
    out = dict(n=len(records), causal_shift=rate(records, lambda r: r["shifted"]),
               valid=rate(records, lambda r: r["valid"]),
               patched=dict(collections.Counter(r["v_patched"] for r in records)))
    ro = [r for r in records if r.get("r_patched") is not None]
    if ro:
        out["readout_shift"] = rate(ro, lambda r: r["ro_shifted"])
    out["by_langs"] = {lt: dict(n=len(rs), causal_shift=rate(rs, lambda r: r["shifted"]))
                       for lt, (sl, dl) in U.LANGS.items()
                       if (rs := [r for r in records if (r["src_lang"], r["dst_lang"]) == (sl, dl)])}
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cuda", default=None, help="CUDA_VISIBLE_DEVICES (applied before torch import)")
    p.add_argument("--patch-heads", default=GOAL250, help="head set json whose k/v read is transplanted")
    p.add_argument("--readout-heads", default=None, help="GoalScope readout heads (default: --patch-heads)")
    p.add_argument("--no-readout", action="store_true", help="skip the GoalScope readouts")
    p.add_argument("--v-scale", type=float, default=4.0, help="x the transplanted values (1.0 = off)")
    p.add_argument("--max-new", type=int, default=512, help="greedy tokens per generation")
    p.add_argument("--pairs", nargs="+", default=list(U.PAIRS), choices=list(U.PAIRS))
    p.add_argument("--langs", nargs="+", default=list(U.LANGS), choices=list(U.LANGS))
    p.add_argument("--force", action="store_true", help="recompute instances already in results.json")
    p.add_argument("--results-dir", default=env_utils.DEFAULT_RESULTS_DIR, help="results root (smoke runs: a scratch dir)")
    logging_utils.add_logging_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    logging_utils.configure(args)
    torch.set_grad_enabled(False)
    patch = _load_heads(args.patch_heads)
    readout = _load_heads(args.readout_heads or args.patch_heads)
    tag = f"ph-{_heads_tag(args.patch_heads, len(patch))}__span-name__V{args.v_scale:g}__mx{args.max_new}"
    if args.no_readout:
        tag += "__ro-off"
    out_dir = Path(args.results_dir) / "kv_transfer" / MODEL.split("/")[-1] / "broader_goal" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / "results.json"
    records = {} if args.force or not fp.exists() else {r["key"]: r for r in json.load(open(fp))}

    ctx = FUNC.setup(MODEL, device_map="auto")
    ctx.gsp = None
    todo = [i for i in U.instances() if i[0] in args.pairs and (i[2], i[3]) in [U.LANGS[l] for l in args.langs]]
    logger.info(f"{tag}: {len(todo)} instances, {sum(U.inst_key(*i) in records for i in todo)} done -> {out_dir}")

    def ro(tok, iface=None):
        return FUNC.goalscope_readout("", tok, tok["input_ids"].shape[1] - 1, heads=readout, iface=iface,
                                      maxnew=RO_MAX_NEW, ctx=ctx)[0]["patched_gen"]

    for n, (key, flip, sl, dl) in enumerate(todo, 1):
        k = U.inst_key(key, flip, sl, dl)
        if k in records:
            continue
        t0 = time.time()
        g = U.build_geometry(key, flip, sl, dl, ctx=ctx)
        stack = PP.intervention_stack(g, goal_heads=patch, v_scale=args.v_scale, ctx=ctx)

        def iface(mt_, inputs, patches=None, model_kwargs=None):
            with stack():
                return mt_._model(**dict(inputs), **dict(model_kwargs or {}))

        gens = dict(src=FUNC.greedy(g["src_tok"], max_new=args.max_new, ctx=ctx),
                    dst=FUNC.greedy(g["dst_tok"], max_new=args.max_new, ctx=ctx))
        with stack():
            gens["patched"] = FUNC.greedy(g["dst_tok"], max_new=args.max_new, ctx=ctx)
        lang = dict(src=sl, dst=dl, patched=dl)
        v = {r: U.judge_code(key, flip, gen, lang[r], model_name=ORACLE_MODEL) for r, gen in gens.items()}
        src, dst = U.sides(key, flip)
        rec = dict(key=k, pair=key, flip=flip, src=src["id"], dst=dst["id"], src_lang=sl, dst_lang=dl,
                   n_window=len(g["dst_w"]), gens=gens, oracle={r: dict(verdict=x["verdict"], why=x["why"]) for r, x in v.items()},
                   v_src=v["src"]["verdict"], v_dst=v["dst"]["verdict"], v_patched=v["patched"]["verdict"],
                   valid=v["src"]["verdict"] == "SRC" and v["dst"]["verdict"] == "DST",
                   shifted=v["patched"]["verdict"] == "SRC")
        if not args.no_readout:
            ros = dict(src=ro(g["src_tok"]), dst=ro(g["dst_tok"]), patched=ro(g["dst_tok"], iface=iface))
            vr = {r: U.judge_readout(key, flip, text, model_name=ORACLE_MODEL)["verdict"] for r, text in ros.items()}
            rec.update(readouts=ros, r_src=vr["src"], r_dst=vr["dst"], r_patched=vr["patched"],
                       ro_valid=vr["src"] == "SRC" and vr["dst"] == "DST", ro_shifted=vr["patched"] == "SRC")
        records[k] = rec
        fp.write_text(json.dumps(list(records.values()), indent=1, ensure_ascii=False))
        logger.info(f"({n}/{len(todo)}) {k}: clean {rec['v_src']}/{rec['v_dst']} -> patched {rec['v_patched']}"
                    + (f" · readout {rec['r_src']}/{rec['r_dst']} -> {rec['r_patched']}" if "r_patched" in rec else "")
                    + f" ({time.time() - t0:.0f}s)")

    summary = dict(model=MODEL, patch_heads=args.patch_heads, n_patch_heads=len(patch),
                   readout_heads=None if args.no_readout else (args.readout_heads or args.patch_heads),
                   v_scale=args.v_scale, max_new=args.max_new, span="name", oracle_model=ORACLE_MODEL,
                   **summarize(list(records.values())))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"SUMMARY {tag}: causal_shift={summary['causal_shift']} n={summary['n']} by_langs="
                + " ".join(f"{lt}:{d['causal_shift']:.2f}" for lt, d in summary["by_langs"].items()))


if __name__ == "__main__":
    main()
