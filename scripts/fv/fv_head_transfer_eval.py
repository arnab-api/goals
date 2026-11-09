"""FV recovery through different head sets on the abstractive FV tasks (Todd et al. 2024).

Protocol = master `notebooks/clock/10_goal_vs_FV_heads.ipynb` §4, scaled to every abstractive task and several
head sets. Per task:
  1. ā = mean o_proj input at the last token over `--mean-trials` clean 10-shot ICL prompts (all 64 layers).
  2. For each query x (`--n-query` from the task's valid split): zero-shot prompt "\\n\\nQ: x\\nA:"; replace the o_proj
     input of every head in the set, at that head's own layer, with ā (binary HeadMask, alpha 1, last token only).
  3. Score top-1 == contextualized first answer token, and dP(correct) vs the unpatched zero-shot run.
     The 10-shot ICL accuracy on the same queries is recorded as the ceiling.
ā and the queries are shared across head sets, so the arms differ only in the head mask.

    python scripts/fv/fv_head_transfer_eval.py --cuda 4 --tasks antonym capitalize --out <dir>
Writes <out>/<task>.json (resumable per task) and never touches results/.
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.append(".")


def _preparse_cuda(argv):
    for i, a in enumerate(argv):
        if a == "--cuda" and i + 1 < len(argv):
            return argv[i + 1]
    return None


if _preparse_cuda(sys.argv[1:]):
    os.environ["CUDA_VISIBLE_DEVICES"] = _preparse_cuda(sys.argv[1:])

import numpy as np  # noqa: E402
import torch  # noqa: E402

from scripts.fv.fv_common import (build_icl_prompt, compute_mean_head_activations,  # noqa: E402
                                  first_answer_token_id, list_abstractive_tasks, load_fv_task)
from src.goals.head_loc_attn import HeadMask, forward_with_src_replacement  # noqa: E402
from src.models import ModelandTokenizer  # noqa: E402
from src.tokens import prepare_input  # noqa: E402

HS = "head_sets/Qwen2.5-Coder-32B-Instruct"
HEAD_SETS = {
    "fv122": f"{HS}/fv__cma_top122.json",
    "fv150": f"{HS}/fv__cma_top150.json",
    "fv250": f"{HS}/fv__cma_top250.json",
    "goal122": f"{HS}/goal__lam20_thr0.9_band25-56.json",
    "goal150": f"{HS}/goal_topk/goal__lam20_top150.json",
    "goal250": f"{HS}/goal_topk/goal__lam20_top250.json",
}


def binmask(heads, n_layer, n_heads):
    m = HeadMask(n_layer, n_heads)
    with torch.no_grad():
        m.values.zero_()
        for (l, h) in heads:
            m.values[l, h] = 1.0
    return m


@torch.inference_mode()
def last_logits(mt, prompts):
    inp = prepare_input(prompts=prompts, tokenizer=mt, padding_side="left")
    return mt._model(**dict(inp)).logits[:, -1, :].float(), inp


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cuda", default=None)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    ap.add_argument("--tasks", nargs="+", default=None, help="default: all abstractive tasks")
    ap.add_argument("--head-sets", nargs="+", default=list(HEAD_SETS), choices=list(HEAD_SETS))
    ap.add_argument("--n-query", type=int, default=100)
    ap.add_argument("--mean-trials", type=int, default=100)
    ap.add_argument("--n-shots", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert not os.path.abspath(args.out).startswith(os.path.abspath("results") + os.sep), "never write into results/"
    os.makedirs(args.out, exist_ok=True)
    tasks = args.tasks or list_abstractive_tasks()

    mt = ModelandTokenizer(model_key=args.model, dtype=torch.bfloat16, device_map="auto",
                           attn_implementation="eager")
    L, H = mt.n_layer, mt.model.config.num_attention_heads
    layers = list(range(L))
    sets = {k: [tuple(h) for h in json.load(open(HEAD_SETS[k]))] for k in args.head_sets}
    masks = {k: binmask(v, L, H) for k, v in sets.items()}
    json.dump(dict(vars(args), head_set_files={k: HEAD_SETS[k] for k in sets},
                   n_heads={k: len(v) for k, v in sets.items()}),
              open(os.path.join(args.out, f"config_{os.getpid()}.json"), "w"), indent=1)

    for t in tasks:
        fp = os.path.join(args.out, f"{t}.json")
        if os.path.exists(fp):
            print(f"[{t}] exists, skip")
            continue
        t0 = time.time()
        task = load_fv_task(t, seed=args.seed)
        a_bar = compute_mean_head_activations(mt, task, layers, n_shots=args.n_shots, n_trials=args.mean_trials,
                                              batch_size=8, seed=args.seed)  # (L, n_embd)
        qs = task.valid[:args.n_query]
        rng = np.random.RandomState(args.seed + 1)
        recs = []
        for i in range(0, len(qs), args.batch_size):
            qb = qs[i:i + args.batch_size]
            zs = [build_icl_prompt([], q["input"]) for q in qb]
            icl = []
            for q in qb:
                idx = rng.choice(len(task.train), args.n_shots, replace=False)
                icl.append(build_icl_prompt([task.train[j] for j in idx], q["input"]))
            tids = torch.tensor([first_answer_token_id(mt, p, q["output"]) for p, q in zip(zs, qb)])
            icl_tids = torch.tensor([first_answer_token_id(mt, p, q["output"]) for p, q in zip(icl, qb)])
            cl, inp = last_logits(mt, zs)
            cp = torch.softmax(cl, -1).cpu()
            icl_l, _ = last_logits(mt, icl)
            icl_top = icl_l.argmax(-1).cpu()
            src = a_bar.unsqueeze(0).expand(len(qb), -1, -1)  # 3D keeps batched logits; hooks move it to device
            patched = {}
            with torch.inference_mode():
                for k, m in masks.items():
                    pl = forward_with_src_replacement(mt, inp, src, m, mask_layers=layers, alpha=1.0,
                                                      patch_positions=[-1]).float()
                    patched[k] = torch.softmax(pl, -1).cpu()
            for b, q in enumerate(qb):
                tid = int(tids[b])
                r = dict(input=q["input"], output=q["output"], tid=tid,
                         clean_top=mt.tokenizer.decode([int(cp[b].argmax())]),
                         clean_hit=int(cp[b].argmax()) == tid, clean_p=float(cp[b, tid]),
                         icl_top=mt.tokenizer.decode([int(icl_top[b])]),
                         icl_hit=int(icl_top[b]) == int(icl_tids[b]))
                for k, pp in patched.items():
                    r[f"{k}_top"] = mt.tokenizer.decode([int(pp[b].argmax())])
                    r[f"{k}_hit"] = int(pp[b].argmax()) == tid
                    r[f"{k}_dP"] = float(pp[b, tid] - cp[b, tid])
                recs.append(r)
        n = len(recs)
        summ = dict(task=t, n=n, clean=sum(r["clean_hit"] for r in recs) / n,
                    icl=sum(r["icl_hit"] for r in recs) / n,
                    **{k: sum(r[f"{k}_hit"] for r in recs) / n for k in masks},
                    **{f"{k}_dP": float(np.mean([r[f"{k}_dP"] for r in recs])) for k in masks})
        json.dump(dict(summary=summ, records=recs), open(fp, "w"), indent=1)
        print(f"[{t}] n={n} clean {summ['clean']:.2f} icl {summ['icl']:.2f} " +
              " ".join(f"{k} {summ[k]:.2f}" for k in masks) + f" ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
