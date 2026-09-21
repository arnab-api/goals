#!/usr/bin/env python
"""Obstacle-robustness eval — scriptized from notebooks/persistance/03_obstacle_robustness.ipynb.

For each QTASKS predicate g_k, build a TWO-condition prompt with the first condition PREFILLED and the
second (target) frontier forced to start with an escapable adversarial prefix (from
``scripts.evaluation.obstacle_prefills.OBSTACLE_PREFILLS``): signature + `(1) ctx (2) target` +
`    a = <ctx impl>` + `    b =` + prefix. Greedy-continue the frontier line, then a bespoke oracle
(``obstacle_eval.judge_adapts``) grades ADAPTED / SUCCUMBED / ESCAPED / BROKEN (plasticity-aware: any
valid impl of g_k that adapts around the forced stub counts). ADAPTED-rate = obstacle robustness.

One model per process (``--cuda`` pins the GPU), so several models can be tested in parallel (e.g. via
run_jobs, or one screen per GPU). Per-trial results are cached + resumable. Outputs under
``<results>/obstacle_robustness/<model_short>/``: ``trial_<hash>.json`` (per trial), ``rows.json``
(all trials), ``summary.json`` (ADAPTED-rate by level x datatype + verdict breakdown).

Usage:
    conda run -n goals python scripts/evaluation/obstacle_robustness_eval.py --cuda 0
    conda run -n goals python scripts/evaluation/obstacle_robustness_eval.py --cuda 2 \\
        --model Qwen/Qwen2.5-Coder-14B-Instruct --n-pred 4 --n-per 2
    # cheap wiring check (no model, no oracle):
    conda run -n goals python scripts/evaluation/obstacle_robustness_eval.py --dry-run
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

import src.diff.data as D  # noqa: E402
from scripts.evaluation.obstacle_prefills import OBSTACLE_PREFILLS, ARG  # noqa: E402
from src.utils import env_utils, experiment_utils, logging_utils  # noqa: E402

logger = logging.getLogger("obstacle_robustness")
LEVELS_ALL = ["aligned", "easy", "hard"]
VERDICTS = ["ADAPTED", "SUCCUMBED", "ESCAPED", "BROKEN", "UNPARSED"]


# ── prompt construction (mirrors the notebook; `mt` passed in) ──────────────────
def build_obstacle_prompt(mt, dt, pid, prefix):
    """(prompt, target_predicate): TWO listed conditions, first PREFILLED (`a = <ctx impl>`), target
    frontier `b =` + the forced obstacle prefix (applied at the 2nd condition — the probed sub-goal)."""
    from scripts.evaluation.eval_goalscope import _chat_prompt

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


def run_obstacle(mt, dt, pid, prefix, max_new=48):
    """Greedy-continue after the forced prefix; return (rhs, target_predicate). rhs = the `b =` RHS."""
    import torch
    from scripts.evaluation.eval_goalscope import greedy_continue
    from src.tokens import prepare_input

    prompt, p = build_obstacle_prompt(mt, dt, pid, prefix)
    inp = prepare_input(prompts=prompt, tokenizer=mt)
    with torch.no_grad():
        gen = greedy_continue(mt, **inp, max_new_tokens=max_new)
    rhs = (prefix + gen).split("\n")[0].strip()
    return rhs, p


# ── job list + summary (no model needed) ────────────────────────────────────────
def build_jobs(datatypes, levels, n_pred, n_per):
    jobs = []
    for dt in datatypes:
        pids = list(OBSTACLE_PREFILLS[dt])
        if n_pred:
            pids = pids[:n_pred]
        for pid in pids:
            spec = OBSTACLE_PREFILLS[dt][pid]
            for lvl in levels:
                for prefix in spec[lvl][:n_per]:
                    jobs.append((dt, pid, lvl, prefix))
    return jobs


def summarize(rows, datatypes, levels):
    def rate(sub, verd="ADAPTED"):
        return (sum(1 for r in sub if r["verdict"] == verd) / len(sub)) if sub else None

    by_level = {}
    for lvl in levels:
        sub = [r for r in rows if r["level"] == lvl]
        by_level[lvl] = dict(
            n=len(sub),
            adapted=rate(sub),
            by_datatype={dt: rate([r for r in sub if r["dt"] == dt]) for dt in datatypes},
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
    ap.add_argument("--datatypes", nargs="+", default=list(ARG), choices=list(ARG))
    ap.add_argument("--levels", nargs="+", default=LEVELS_ALL, choices=LEVELS_ALL)
    ap.add_argument("--n-pred", type=int, default=None, help="predicates per datatype (default: all)")
    ap.add_argument("--n-per", type=int, default=4, help="prefixes per (predicate, level)")
    ap.add_argument("--max-new", type=int, default=48, help="greedy budget for the frontier line")
    ap.add_argument("--oracle-model", default="claude-sonnet-4-6")
    ap.add_argument("--no-oracle", action="store_true", help="skip judging (generation only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="ignore cache; recompute every trial")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the job list + validate obstacles; NO model, NO oracle")
    logging_utils.add_logging_args(ap)
    return ap


def main():
    args = parse_args().parse_args()
    logging_utils.configure(args) if hasattr(logging_utils, "configure") else logging.basicConfig(level=logging.INFO)
    experiment_utils.set_seed(args.seed)

    jobs = build_jobs(args.datatypes, args.levels, args.n_pred, args.n_per)
    slug = args.model.split("/")[-1]
    cache_dir = os.path.join(env_utils.DEFAULT_RESULTS_DIR, "obstacle_robustness", slug)

    if args.dry_run:
        from collections import Counter
        logger.info(f"[dry-run] {len(jobs)} trials · datatypes={args.datatypes} · levels={args.levels} "
                    f"· n_pred={args.n_pred} · n_per={args.n_per}")
        logger.info(f"[dry-run] per datatype: {dict(Counter(j[0] for j in jobs))}")
        logger.info(f"[dry-run] would write to {cache_dir}")
        logger.info(f"[dry-run] sample job: {jobs[0]} -> obstacle {jobs[0][3]!r}")
        return

    import torch
    from src.models import ModelandTokenizer
    mt = ModelandTokenizer(model_key=args.model, dtype=torch.bfloat16,
                           device_map="auto", attn_implementation="eager")
    judge = None
    if not args.no_oracle:
        from scripts.evaluation.obstacle_eval import judge_adapts
        judge = judge_adapts

    os.makedirs(cache_dir, exist_ok=True)

    def cache_path(dt, pid, lvl, prefix):
        key = hashlib.md5(f"{dt}|{pid}|{lvl}|{prefix}".encode()).hexdigest()[:16]
        return os.path.join(cache_dir, f"trial_{key}.json")

    def trial(dt, pid, lvl, prefix):
        cp = cache_path(dt, pid, lvl, prefix)
        if os.path.exists(cp) and not args.force:
            return json.loads(open(cp).read())
        rhs, p = run_obstacle(mt, dt, pid, prefix, max_new=args.max_new)
        if judge is None:
            verdict, rationale = "SKIPPED", ""
        else:
            v = judge(p.description, prefix, rhs, model=args.oracle_model)
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

    open(os.path.join(cache_dir, "rows.json"), "w").write(json.dumps(rows, indent=2))
    summ = summarize(rows, args.datatypes, args.levels)
    open(os.path.join(cache_dir, "summary.json"), "w").write(json.dumps(summ, indent=2))

    # console summary
    logger.info(f"=== {slug} · ADAPTED rate by level x datatype (n={summ['n']}) ===")
    header = "level     " + " ".join(f"{dt:>10s}" for dt in args.datatypes) + f" {'all':>7s}"
    logger.info(header)
    for lvl in args.levels:
        L = summ["by_level"][lvl]
        cells = " ".join(f"{_fmt(L['by_datatype'][dt]):>10s}" for dt in args.datatypes)
        logger.info(f"{lvl:9s} {cells} {_fmt(L['adapted']):>7s}")
    logger.info(f"verdict totals: {summ['verdict_totals']}")
    logger.info(f"summary -> {os.path.join(cache_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
