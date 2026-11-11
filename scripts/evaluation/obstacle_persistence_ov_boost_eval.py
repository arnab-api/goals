#!/usr/bin/env python
"""Obstacle OV-boost: does boosting the goal heads' OV write get the LM back on track?

At a numeric frontier ``b =`` (list[str] / list[int] predicates whose easy obstacle is ``" 0 + "``,
prompt built exactly as in ``obstacle_persistence_eval.py``) the forced ``0 +`` prefix sometimes derails
Qwen2.5-Coder-32B-Instruct into a complex-number placeholder (``0j``). For each derailed case:

  1. baseline  — greedy-continue after ``b = 0 +``;
  2. readout   — GoalScope read-out (250 goal heads, additive_norm alpha 0.1, steer layers 35-46) at
                 the ``+`` token: is the frontier still represented despite the derailment?
  3. OV-boost  — re-generate while multiplying the 250 goal heads' slice of the ``o_proj`` input by
                 ``BOOST_ALPHA = 2.0`` at the current token of every forward pass.

Baseline and boosted continuations are judged with ``obstacle_persistence_judge.judge_adapts``. Results are cached
under ``<results>/obstacle_persistence/<model>/ov_boost/`` (``base_<h>.json`` per candidate scan,
``cmp_<h>.json`` per derailed case, ``results.json`` = all derailed cases), from which
``notebooks/fig_n_table_generators/obstacle_persistence.ipynb`` renders the OV-boost table.

Usage:
    conda run -n goals python scripts/evaluation/obstacle_persistence_ov_boost_eval.py --cuda 0
    # subset (writes results_subset.json):  --datatypes 'list[str]' --predicates distinct_count
"""
import argparse
import hashlib
import json
import logging
import os
import re
import sys
from collections import defaultdict
from contextlib import contextmanager

sys.path.append(".")

from scripts.evaluation.obstacle_persistence_eval import _preparse_cuda  # noqa: E402

_cuda = _preparse_cuda(sys.argv[1:])
if _cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = _cuda  # must precede torch init

import torch  # noqa: E402
from baukit import get_module  # noqa: E402

from scripts.evaluation.eval_goalscope import greedy_continue, load_json_heads  # noqa: E402
from scripts.evaluation.obstacle_persistence_judge import judge_adapts  # noqa: E402
from scripts.evaluation.obstacle_persistence_prefills import OBSTACLE_PREFILLS  # noqa: E402
from scripts.evaluation.obstacle_persistence_eval import build_obstacle_prompt  # noqa: E402
from src.goals.goalscope import apply_goalscope  # noqa: E402
from src.tokens import prepare_input  # noqa: E402
from src.utils import env_utils, logging_utils  # noqa: E402

logger = logging.getLogger("obstacle_ov_boost")

MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"
HEADS_JSON = "head_sets/Qwen2.5-Coder-32B-Instruct/goal_topk/goal__lam20_top250.json"  # 250 goal heads
NUMERIC_OBSTACLE = " 0 + "
BOOST_ALPHA = 2.0
MAX_NEW = 48
SCAN_DTS = ["list[str]", "list[int]"]
DERAIL_RE = re.compile(r"0j|1j|complex|\.imag", re.I)  # complex-placeholder derailment
GS = dict(strategy="additive_norm", alpha=0.1, steer_layer_range=(35, 46), n_steer_tokens="all",
          max_new_tokens=48)


def cache_dir():
    return os.path.join(env_utils.DEFAULT_RESULTS_DIR, "obstacle_persistence", MODEL.split("/")[-1], "ov_boost")


def cache_path(dt, pid, tag):
    return os.path.join(cache_dir(), f"{tag}_{hashlib.md5(f'{dt}|{pid}'.encode()).hexdigest()[:12]}.json")


def candidates():
    """Numeric (dt, pid) whose easy obstacle is ``" 0 + "``."""
    return [(dt, pid) for dt in SCAN_DTS for pid, spec in OBSTACLE_PREFILLS[dt].items()
            if spec["easy"][0] == NUMERIC_OBSTACLE]


@contextmanager
def boosted_heads(mt, heads, alpha):
    """Pre-hook on each layer's o_proj: scale the given heads' slice of its INPUT (the concatenated
    head outputs) by ``alpha`` at the current (last) token."""
    H = mt.head_dim

    def prehook(head_idxs):
        def hook(module, args):
            x = args[0].clone()  # (batch, seq, n_heads*H)
            for h in head_idxs:
                x[:, -1, h * H:(h + 1) * H] *= alpha
            return (x,) + tuple(args[1:])
        return hook

    by_layer = defaultdict(list)
    for (l, h) in heads:
        by_layer[l].append(h)
    handles = [get_module(mt._model, mt.attn_module_name_format.format(l) + ".o_proj")
               .register_forward_pre_hook(prehook(hs)) for l, hs in by_layer.items()]
    try:
        yield
    finally:
        for hh in handles:
            hh.remove()


@torch.no_grad()
def run_obstacle(mt, heads, dt, pid, boost_alpha=None):
    """Greedy-continue after ``b =`` + the obstacle. ``boost_alpha=None`` -> clean."""
    prompt, p = build_obstacle_prompt(mt, dt, pid, NUMERIC_OBSTACLE)
    inp = prepare_input(prompts=prompt, tokenizer=mt)
    if boost_alpha is None:
        gen = greedy_continue(mt, **inp, max_new_tokens=MAX_NEW)
    else:
        with boosted_heads(mt, heads, boost_alpha):
            gen = greedy_continue(mt, **inp, max_new_tokens=MAX_NEW)
    rhs = (NUMERIC_OBSTACLE + gen).split("\n")[0].strip()
    return rhs, prompt, gen, p


@torch.no_grad()
def readout_at_plus(mt, heads, prompt):
    """GoalScope read-out of the goal encoding AT the last ``+`` token (not judged)."""
    inp = prepare_input(prompts=prompt, tokenizer=mt)
    ids = inp.input_ids[0].tolist()
    pos = next((i for i in range(len(ids) - 1, -1, -1) if "+" in mt.tokenizer.decode([ids[i]])),
               len(ids) - 1)
    r = apply_goalscope(inp, mt=mt, selected_heads=heads, token_positions=[pos], **GS)
    return (r[0].get("patched_gen") or "").strip() if r else ""


def _dump(obj, path):
    json.dump(obj, open(path + ".tmp", "w"), indent=2)
    os.replace(path + ".tmp", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cuda", default=None, help="CUDA_VISIBLE_DEVICES (applied before torch import)")
    ap.add_argument("--datatypes", nargs="+", default=None, help="subset: only these datatypes (default: all)")
    ap.add_argument("--predicates", nargs="+", default=None, help="subset: only these predicate ids (default: all)")
    logging_utils.add_logging_args(ap)
    args = ap.parse_args()
    logging_utils.configure(args)
    torch.set_grad_enabled(False)

    from src.models import ModelandTokenizer
    mt = ModelandTokenizer(model_key=MODEL, dtype=torch.bfloat16, device_map="auto",
                           attn_implementation="eager")
    heads = load_json_heads(HEADS_JSON, mt.n_layer, mt.n_heads)
    os.makedirs(cache_dir(), exist_ok=True)
    cands = [(dt, pid) for dt, pid in candidates() if (args.datatypes is None or dt in args.datatypes)
             and (args.predicates is None or pid in args.predicates)]
    assert cands, "no candidates match --datatypes / --predicates"
    subset = args.datatypes is not None or args.predicates is not None
    logger.info(f"{len(heads)} goal heads | {len(cands)} numeric candidates with the {NUMERIC_OBSTACLE!r} obstacle")

    # §1 baseline scan: flag the 0j / complex derailments
    derailed = []
    for dt, pid in cands:
        cp = cache_path(dt, pid, "base")
        if os.path.exists(cp):
            row = json.load(open(cp))
        else:
            rhs, _, _, p = run_obstacle(mt, heads, dt, pid)
            row = dict(dt=dt, pid=pid, desc=p.description, rhs=rhs, derail=bool(DERAIL_RE.search(rhs)))
            _dump(row, cp)
        logger.info(f"{dt:9s} {pid:16s} b = {row['rhs'][:56]}{'   <-- derailed' if row['derail'] else ''}")
        if row["derail"]:
            derailed.append((dt, pid))

    # §2 per derailed case: baseline · read-out @ '+' · OV-boost
    results = []
    for dt, pid in derailed:
        cp = cache_path(dt, pid, "cmp")
        if os.path.exists(cp):
            results.append(json.load(open(cp)))
            continue
        base_rhs, prompt, base_gen, p = run_obstacle(mt, heads, dt, pid)
        ro = readout_at_plus(mt, heads, prompt)
        boost_rhs, _, boost_gen, _ = run_obstacle(mt, heads, dt, pid, boost_alpha=BOOST_ALPHA)
        res = dict(
            dt=dt, pid=pid, frontier=p.description,
            base_gen=base_gen.split("\n")[0],
            base_verdict=judge_adapts(p.description, NUMERIC_OBSTACLE, base_rhs)["verdict"],
            readout=ro,
            boost_gen=boost_gen.split("\n")[0],
            boost_verdict=judge_adapts(p.description, NUMERIC_OBSTACLE, boost_rhs)["verdict"],
        )
        _dump(res, cp)
        results.append(res)
    # a subset run never overwrites the full run's results.json
    out = os.path.join(cache_dir(), "results_subset.json" if subset else "results.json")
    json.dump(results, open(out, "w"), indent=2)
    logger.info(f"saved {len(results)} derailed cases -> {out}")


if __name__ == "__main__":
    main()
