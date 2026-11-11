#!/usr/bin/env python
"""Obstacle-robustness eval.

For each QTASKS predicate g_k, build a TWO-condition prompt with the first condition PREFILLED and the
second (target) frontier forced to start with an escapable adversarial prefix (from
``scripts.evaluation.obstacle_persistence_prefills.OBSTACLE_PREFILLS``): signature + `(1) ctx (2) target` +
`    a = <ctx impl>` + `    b =` + prefix. Greedy-continue the frontier line, then a bespoke oracle
(``obstacle_persistence_judge.judge_adapts``) grades ADAPTED / SUCCUMBED / ESCAPED / BROKEN (plasticity-aware: any
valid impl of g_k that adapts around the forced stub counts). ADAPTED-rate = obstacle robustness.

One model per process (``--cuda`` pins the GPU), so several models can be tested in parallel (e.g. via
run_jobs, or one screen per GPU). Per-trial results are cached + resumable. Outputs under
``<results>/obstacle_robustness/<model_short>/``: ``trial_<hash>.json`` (per trial), ``rows.json``
(all trials), ``summary.json`` (ADAPTED-rate by level x datatype + verdict breakdown).

All 4 datatypes x every predicate x 3 levels (aligned / easy / hard) x the first 4 prefixes per
level -> 409 trials per model.

Usage (the paper runs 7 models, one per process):
    conda run -n goals python scripts/evaluation/obstacle_persistence_eval.py --cuda 0
    conda run -n goals python scripts/evaluation/obstacle_persistence_eval.py --cuda 2 \\
        --model Qwen/Qwen2.5-Coder-14B-Instruct
    # subset (writes rows_subset.json / summary_subset.json):  --datatypes 'list[str]' --predicates count_with_digit
"""
import argparse
import hashlib
import json
import logging
import os
import sys

sys.path.append(".")


def _preparse_cuda(argv):
    for i, a in enumerate(argv):
        if a == "--cuda" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--cuda="):
            return a.split("=", 1)[1]
    return None


_cuda = _preparse_cuda(sys.argv[1:])
if _cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = _cuda  # must precede torch import

import src.nrgs.data as D  # noqa: E402
from scripts.evaluation.eval_goalscope import _chat_prompt, greedy_continue  # noqa: E402
from scripts.evaluation.obstacle_persistence_judge import judge_adapts  # noqa: E402
from scripts.evaluation.obstacle_persistence_prefills import OBSTACLE_PREFILLS, ARG  # noqa: E402
from src.tokens import prepare_input  # noqa: E402
from src.utils import env_utils, experiment_utils, logging_utils  # noqa: E402

logger = logging.getLogger("obstacle_robustness")
DATATYPES = list(ARG)
LEVELS = ["aligned", "easy", "hard"]
VERDICTS = ["ADAPTED", "SUCCUMBED", "ESCAPED", "BROKEN", "UNPARSED"]
N_PER = 4  # prefixes per (predicate, level); bool `easy` has 5, so this keeps n=409
MAX_NEW = 48  # greedy budget for the frontier line (NOT in the trial cache key)


# ── prompt construction (mirrors the notebook; `mt` passed in) ──────────────────
def build_obstacle_prompt(mt, dt, pid, prefix):
    """(prompt, target_predicate): TWO listed conditions, first PREFILLED (`a = <ctx impl>`), target
    frontier `b =` + the forced obstacle prefix (applied at the 2nd condition — the probed sub-goal)."""
    bank = D.QTASKS[dt]
    arg = ARG[dt]
    preds = list(bank.predicates)
    idx = next(i for i, x in enumerate(preds) if x.id == pid)
    target = preds[idx]
    ctx = preds[(idx + 2) % len(preds)]  # a different prior sub-goal (offset 2 dodges the obstacle sibling)
    q = D._question([ctx.description, target.description], dt, arg, "python", "listed")
    sig = D._signature(dt, arg, "python")
    first = D._assign_line(ctx, arg, "python", res_name="a")
    base = "<code>\n" + sig + "\n" + first + "\n    b ="
    return _chat_prompt(mt, q, base) + prefix, target


def run_obstacle(mt, dt, pid, prefix, max_new=MAX_NEW):
    """Greedy-continue after the forced prefix; return (rhs, target_predicate). rhs = the `b =` RHS."""
    import torch

    prompt, p = build_obstacle_prompt(mt, dt, pid, prefix)
    inp = prepare_input(prompts=prompt, tokenizer=mt)
    with torch.no_grad():
        gen = greedy_continue(mt, **inp, max_new_tokens=max_new)
    rhs = (prefix + gen).split("\n")[0].strip()
    return rhs, p


# ── job list + summary (no model needed) ────────────────────────────────────────
def build_jobs():
    jobs = []
    for dt in DATATYPES:
        for pid in OBSTACLE_PREFILLS[dt]:
            spec = OBSTACLE_PREFILLS[dt][pid]
            for lvl in LEVELS:
                for prefix in spec[lvl][:N_PER]:
                    jobs.append((dt, pid, lvl, prefix))
    return jobs


def summarize(rows):
    def rate(sub, verd="ADAPTED"):
        return (sum(1 for r in sub if r["verdict"] == verd) / len(sub)) if sub else None

    by_level = {}
    for lvl in LEVELS:
        sub = [r for r in rows if r["level"] == lvl]
        by_level[lvl] = dict(
            n=len(sub),
            adapted=rate(sub),
            by_datatype={dt: rate([r for r in sub if r["dt"] == dt]) for dt in DATATYPES},
            verdicts={v: rate(sub, v) for v in VERDICTS},
        )
    return dict(n=len(rows), by_level=by_level,
                verdict_totals={v: sum(1 for r in rows if r["verdict"] == v) for v in VERDICTS})


def _fmt(x):
    return "  n/a" if x is None else f"{x:.2f}"


# ── driver ──────────────────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    ap.add_argument("--cuda", default=None, help="CUDA_VISIBLE_DEVICES (applied before torch import)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="ignore cache; recompute every trial")
    ap.add_argument("--datatypes", nargs="+", default=None, help="subset: only these datatypes (default: all)")
    ap.add_argument("--predicates", nargs="+", default=None, help="subset: only these predicate ids (default: all)")
    logging_utils.add_logging_args(ap)
    return ap


def main():
    args = parse_args().parse_args()
    logging_utils.configure(args)
    experiment_utils.set_seed(args.seed)

    jobs = [j for j in build_jobs() if (args.datatypes is None or j[0] in args.datatypes)
            and (args.predicates is None or j[1] in args.predicates)]
    assert jobs, "no trials match --datatypes / --predicates"
    subset = args.datatypes is not None or args.predicates is not None
    sfx = "_subset" if subset else ""   # a subset run never overwrites the full run's rows.json / summary.json
    slug = args.model.split("/")[-1]
    cache_dir = os.path.join(env_utils.DEFAULT_RESULTS_DIR, "obstacle_persistence", slug, "trials")

    import torch
    from src.models import ModelandTokenizer
    mt = ModelandTokenizer(model_key=args.model, dtype=torch.bfloat16,
                           device_map="auto", attn_implementation="eager")
    os.makedirs(cache_dir, exist_ok=True)

    def cache_path(dt, pid, lvl, prefix):
        key = hashlib.md5(f"{dt}|{pid}|{lvl}|{prefix}".encode()).hexdigest()[:16]
        return os.path.join(cache_dir, f"trial_{key}.json")

    def trial(dt, pid, lvl, prefix):
        cp = cache_path(dt, pid, lvl, prefix)
        if os.path.exists(cp) and not args.force:
            return json.loads(open(cp).read())
        rhs, p = run_obstacle(mt, dt, pid, prefix)
        v = judge_adapts(p.description, prefix, rhs)
        verdict, rationale = v["verdict"], v.get("rationale", "")
        row = dict(dt=dt, pid=pid, level=lvl, prefix=prefix, rhs=rhs, verdict=verdict, rationale=rationale)
        tmp = cp + ".tmp"
        open(tmp, "w").write(json.dumps(row, indent=2))
        os.replace(tmp, cp)
        return row

    n_cached = sum(os.path.exists(cache_path(*j)) for j in jobs) if not args.force else 0
    logger.info(f"{slug}: {len(jobs)} trials · {n_cached} cached, {len(jobs)-n_cached} to compute "
                f"-> {cache_dir}")
    try:
        from tqdm.auto import tqdm
        it = tqdm(jobs, desc=f"obstacle trials [{slug}]")
    except Exception:
        it = jobs
    rows = [trial(*j) for j in it]

    open(os.path.join(cache_dir, f"rows{sfx}.json"), "w").write(json.dumps(rows, indent=2))
    summ = summarize(rows)
    open(os.path.join(cache_dir, f"summary{sfx}.json"), "w").write(json.dumps(summ, indent=2))

    # console summary
    logger.info(f"=== {slug} · ADAPTED rate by level x datatype (n={summ['n']}) ===")
    header = "level     " + " ".join(f"{dt:>10s}" for dt in DATATYPES) + f" {'all':>7s}"
    logger.info(header)
    for lvl in LEVELS:
        L = summ["by_level"][lvl]
        cells = " ".join(f"{_fmt(L['by_datatype'][dt]):>10s}" for dt in DATATYPES)
        logger.info(f"{lvl:9s} {cells} {_fmt(L['adapted']):>7s}")
    logger.info(f"verdict totals: {summ['verdict_totals']}")
    logger.info(f"summary -> {os.path.join(cache_dir, f'summary{sfx}.json')}")


if __name__ == "__main__":
    main()
