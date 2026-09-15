#!/usr/bin/env python
"""Generate the teacher-target corpus for the Subspace Probe v2 (KL self-distillation).

For each training example we need: (i) a structured SOURCE prompt (prefill-to-frontier, exactly the
OVReader corpus) whose frontier predicate is the goal ``g_k``; (ii) an augmented C++ causal STUB; and
(iii) the goal-informed TEACHER trajectory ``y`` (greedy C++ body when the goal is stated explicitly).
We cache ONLY the token trajectory ``y`` + the recipe to rebuild both prompts — teacher distributions
are recomputed on the fly during training so the KL is exact over the full vocab.

Train/eval split is by PREDICATE (held-out predicates never appear as a training frontier), so the eval
(``eval_subspace_probe_v2.py`` on the locked corpus) tests goal generalization, not spec memorization.

Output: data/subspace_v2/teacher_targets/<slug>/<split>.json  (+ a predicate-split sidecar)

Usage (14B smoke on cuda 0,1):
    conda run -n goals python scripts/evaluation/gen_teacher_targets_v2.py \
        --model Qwen/Qwen2.5-Coder-14B-Instruct --cuda 0,1 --split train \
        --n_per_datatype 8 --datatypes int str --show 4
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.append(".")

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--cuda", default=None)
_c = _pre.parse_known_args()[0].cuda
if _c is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_c)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402

import src.diff.data as DATA  # noqa: E402
from scripts.evaluation.eval_ov_reader import (  # noqa: E402
    build_prompt,
    frontier_impl_span,
    pivot_positions,
    sample_datapoints,
)
from src.goals.goalscope import greedy_continue  # noqa: E402
from src.goals.subspace_probe_v2 import (  # noqa: E402
    build_causal_chat,
    generate_teacher_trajectory,
    sample_stub,
)
from src.models import ModelandTokenizer  # noqa: E402
from src.tokens import prepare_input  # noqa: E402


def predicate_split(datatype, holdout_frac, seed):
    """Deterministic per-datatype split of predicate ids into (train, eval). Trivial frontiers are
    excluded from BOTH frontier pools (they can't be measured), but may still appear as context."""
    bank = DATA.QTASKS[datatype]
    trivial = DATA.TRIVIAL_FRONTIERS.get(datatype, frozenset())
    ids = [p.id for p in bank.predicates if p.id not in trivial]
    rng = random.Random(hash((datatype, seed)) & 0xFFFFFFFF)
    rng.shuffle(ids)
    n_eval = max(1, round(holdout_frac * len(ids)))
    eval_ids = set(ids[:n_eval])
    train_ids = set(ids[n_eval:])
    return train_ids, eval_ids


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-14B-Instruct")
    ap.add_argument("--cuda", default=None)
    ap.add_argument("--split", default="train", choices=["train", "eval"])
    ap.add_argument("--datatypes", nargs="+", default=list(DATA.DATATYPES))
    ap.add_argument("--lang", default="python", choices=["python", "cpp"])
    ap.add_argument("--n_conditions", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--n_per_datatype", type=int, default=32, help="source instances kept per datatype")
    ap.add_argument("--pivots", nargs="+", default=["eq", "first_impl"])
    ap.add_argument("--holdout_frac", type=float, default=0.3, help="fraction of predicates held out for eval")
    ap.add_argument("--gen_max_new_tokens", type=int, default=64, help="source frontier-body budget (for first_impl)")
    ap.add_argument("--teacher_max_new_tokens", type=int, default=32, help="teacher trajectory length")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--split_seed", type=int, default=7, help="seed for the predicate split (share across train/eval!)")
    ap.add_argument("--show", type=int, default=4, help="print this many teacher trajectories for review")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    slug = args.model.split("/")[-1]
    out = Path(args.out) if args.out else Path("data/subspace_v2/teacher_targets") / slug / f"{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    mt = ModelandTokenizer(model_key=args.model, dtype=torch.bfloat16,
                           device_map="auto", attn_implementation="eager")
    torch.set_grad_enabled(False)
    rng = random.Random(args.seed)

    # persist the predicate split so train + eval share it
    split_path = out.parent / f"predicate_split__seed{args.split_seed}__ho{args.holdout_frac}.json"
    splits = {}
    for dt in args.datatypes:
        tr, ev = predicate_split(dt, args.holdout_frac, args.split_seed)
        splits[dt] = {"train": sorted(tr), "eval": sorted(ev)}
    split_path.write_text(json.dumps(splits, indent=2))
    print(f"predicate split -> {split_path}")
    for dt in args.datatypes:
        print(f"  [{dt}] train={splits[dt]['train']}  eval={splits[dt]['eval']}")

    examples = []
    shown = 0
    for dt in args.datatypes:
        allowed = set(splits[dt][args.split])
        # oversample source instances, keep those whose FRONTIER predicate is in this split
        pool = sample_datapoints(dt, args.n_conditions, args.n_per_datatype * 8, args.lang, args.seed)
        kept = 0
        for inst in pool:
            if kept >= args.n_per_datatype:
                break
            if inst.expected_next not in allowed:
                continue
            prompt, var = build_prompt(mt, inst, args.lang)
            completion = greedy_continue(
                mt, **prepare_input(prompts=prompt, tokenizer=mt),
                max_new_tokens=args.gen_max_new_tokens,
            )
            full_text = prompt + completion
            pivs = pivot_positions(mt, prompt, full_text, args.pivots)
            if not pivs:
                continue
            goal_desc = inst.expected_next_desc
            stub = sample_stub(rng)
            teacher_prompt = build_causal_chat(mt, stub, goal_desc=goal_desc)
            y_ids = generate_teacher_trajectory(
                mt, teacher_prompt, max_new_tokens=args.teacher_max_new_tokens
            )
            if len(y_ids) < 2:
                continue
            y_text = mt.tokenizer.decode(y_ids)
            examples.append(dict(
                datatype=dt, frontier_id=inst.expected_next, goal_desc=goal_desc,
                lang=args.lang, var=var, T=inst.meta["T"], j=inst.meta["j"],
                source_prompt=prompt, source_completion=completion,
                impl_span=frontier_impl_span(prompt, completion, var),
                pivots=pivs, stub=stub, y_ids=y_ids, y_text=y_text,
            ))
            kept += 1
            if shown < args.show:
                shown += 1
                print(f"\n── [{dt}] frontier={inst.expected_next} :: {goal_desc}")
                print(f"   stub: bool {stub['func']}(auto {stub['arg']}){{")
                print(f"   teacher y: {y_text!r}")
        print(f"[{dt}] kept {kept} source instances (frontier ∈ {sorted(allowed)})")

    payload = dict(
        model=args.model, split=args.split, lang=args.lang, seed=args.seed,
        split_seed=args.split_seed, holdout_frac=args.holdout_frac,
        pivots=args.pivots, n_examples=len(examples), examples=examples,
    )
    out.write_text(json.dumps(payload))
    print(f"\nwrote {len(examples)} examples -> {out}")


if __name__ == "__main__":
    main()
