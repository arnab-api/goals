#!/usr/bin/env python
"""GoalScope sub-goal-coverage benchmark — Script B: LM-judge scoring of the verbalization text
`goalscope_subgoal_generate.py` stored. Separate script (decoupled from the expensive generation
pass), same design as `goalscope_fv_judge.py`.

For each scored record, runs ONE presence-check `ask_claude` call PER SUB-GOAL in that prompt's
`subgoals` list (2 for composition: f_description, g_description; 3-4 for superposition: the
family's task_labels) -- "does this verbalization mention/describe sub-goal G, even partially?"
Reports, per record: `hits` (YES/NO/NO_INFO/MALFORMED/HEAVY_LOOP per sub-goal), `n_hit`, `k`,
`at_least_one`, `all_hit`.

Faithfulness is NOT a separate judge call -- it's a re-indexing of the same presence hits, using
metadata `goalscope_subgoal_generate.py` already computed per prompt:
  - superposition: `hits[predicted_idx]` -- did the verbalization cover the specific task the
    model's own `score_choices` argmax says it's acting on (well-defined for every prompt; there's
    no "correct" candidate to filter on, all candidates are simultaneously valid completions).
  - composition: coverage, but restricted to rows where `answer_correct` is True (only composition
    has a single correct answer, so faithfulness only makes sense conditioned on the model having
    actually gotten there) -- per user: "probably should do it when the answer is correct".

Usage:
    conda run -n goals python scripts/fv/goalscope_subgoal_judge.py --run_dir <path>
    conda run -n goals python scripts/fv/goalscope_subgoal_judge.py --all
"""
import argparse
import json
import logging
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.append(".")

import numpy as np

from scripts.evaluation.eval_goalscope import ORACLE_MODEL, _parse_verdict
from scripts.fv import goalscope_subgoal_io as fio
from src.utils import logging_utils
from src.utils.oracle_llms import ask_claude

logger = logging.getLogger(__name__)

SUBGOAL_PRESENCE_SYS = """You are grading a lightweight interpretability probe (a patchscope read-out). The DESCRIPTION below was read out of a model's hidden state at a SINGLE token while the model was working through a few-shot in-context-learning prompt. Because it is squeezed out of one token under a steering/patching setup, the description is usually NOISY: it may be wrapped in chat scaffolding (like 'Sure, but first ...'), be cut off mid-sentence, or repeat and loop. Look PAST the surface noise and grade only the SUBSTANCE.

You will be given a SUB-GOAL and the DESCRIPTION. The question is narrow: does the description mention or refer to this SPECIFIC sub-goal, even partially, implicitly, or alongside other content? The description may describe MULTIPLE things at once (e.g. several candidate tasks, or both halves of a two-step computation) -- you are only asked whether THIS ONE sub-goal is among them, not whether it's the only thing mentioned or the most prominent.

Decide in this order -- check MALFORMED and HEAVY_LOOP FIRST, and only then YES / NO / NO_INFO:

  MALFORMED  - the description is contentless: empty, pure punctuation or garbage, or repetition that carries NO relevant meaning.
  HEAVY_LOOP - the description is DOMINATED by a short repeating cycle: a unit of roughly five words/tokens or fewer that repeats over and over. Once you collapse the repetition, little or no coherent description is left. This takes PRECEDENCE over YES.
  YES        - the description mentions or clearly gestures at this sub-goal, however partial, imprecise, or garbled -- and it is NOT merely a repeating loop.
  NO         - the description is coherent but does NOT mention this sub-goal (it may mention a different, unrelated sub-goal instead).
  NO_INFO    - the description is coherent but too generic to indicate whether this sub-goal is present or not (e.g. just "answering a question").

Output only YES, NO, NO_INFO, MALFORMED, or HEAVY_LOOP.
Put your final answer inside <verdict>{ans}</verdict> tags, with no other text or explanation, to facilitate automated parsing."""


def subgoal_presence_eval(readout: str, subgoal: str, oracle_model: str = ORACLE_MODEL) -> dict:
    """dict(verdict, prompt, response). verdict in {YES, NO, NO_INFO, MALFORMED, HEAVY_LOOP, UNPARSED}."""
    if not readout or not readout.strip():
        return dict(verdict="MALFORMED", prompt=None, response=None)
    prompt = f"Sub-goal: {subgoal}\nDescription: {readout.strip()}\nAnswer:"
    resp = ask_claude(prompt=prompt, system_prompt=SUBGOAL_PRESENCE_SYS, max_tokens=512,
                      use_cache=True, model_name=oracle_model)
    return dict(verdict=_parse_verdict(resp), prompt=prompt, response=resp)


def _judge_one(r: dict, subgoals: list[str], oracle_model: str) -> dict:
    readout = r["verbalization"]
    evals = [subgoal_presence_eval(readout, sg, oracle_model=oracle_model) for sg in subgoals]
    hits = [e["verdict"] for e in evals]
    k = len(subgoals)
    n_hit = sum(h == "YES" for h in hits)
    return dict(
        **r, subgoals=subgoals, hits=hits, n_hit=n_hit, k=k,
        at_least_one=n_hit >= 1, all_hit=n_hit == k,
    )


def judge_run_dir(run_dir: Path, oracle_model: str, limit: int | None = None, workers: int = 12) -> dict:
    records = [json.loads(l) for l in fio.records_path(run_dir).read_text().splitlines() if l.strip()]
    samples = {s["prompt_seed"]: s for s in fio.load_samples(run_dir)}
    already = fio.read_judged_keys(run_dir)
    todo = [r for r in records if (r["prompt_seed"], r["pos_kind"], r["gen_offset"]) not in already]
    if limit:
        todo = todo[:limit]
    logger.info(f"[{run_dir}] {len(records)} records total | {len(already)} already judged | "
               f"judging {len(todo)} now ({workers} workers)")

    write_lock = threading.Lock()
    with open(fio.judged_path(run_dir), "a") as fh, ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_judge_one, r, samples[r["prompt_seed"]]["subgoals"], oracle_model): r
                  for r in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            result = fut.result()
            with write_lock:
                fio.append_jsonl(fh, result)
            if i % 20 == 0 or i == len(futures):
                logger.info(f"judged {i}/{len(todo)}")

    all_judged = [json.loads(l) for l in fio.judged_path(run_dir).read_text().splitlines() if l.strip()]
    summary = compute_summary(all_judged, samples, fio.read_config(run_dir)["source"])
    fio.write_summary(run_dir, summary)
    logger.info(f"saved -> {fio.judged_path(run_dir)}")
    log_summary(summary)
    return summary


def compute_summary(judged: list[dict], samples: dict, source: str) -> dict:
    def _cov_stats(rows):
        if not rows:
            return dict(n=0, coverage=float("nan"), at_least_one=float("nan"), all_hit=float("nan"))
        return dict(
            n=len(rows),
            coverage=float(np.mean([r["n_hit"] / r["k"] for r in rows])),
            at_least_one=float(np.mean([r["at_least_one"] for r in rows])),
            all_hit=float(np.mean([r["all_hit"] for r in rows])),
        )

    def _by_pos_kind():
        keys = sorted({r["pos_kind"] for r in judged})
        return {k: _cov_stats([r for r in judged if r["pos_kind"] == k]) for k in keys}

    def _by_subgoal_index(rows):
        """Per-index presence rate (index 0 = f/task0, 1 = g/task1, ...) -- for composition this
        directly separates f_description vs g_description hit rates."""
        max_k = max((r["k"] for r in rows), default=0)
        out = {}
        for i in range(max_k):
            hits = [r["hits"][i] == "YES" for r in rows if r["k"] > i]
            out[str(i)] = float(np.mean(hits)) if hits else float("nan")
        return out

    overall = _cov_stats(judged)
    overall["by_subgoal_index"] = _by_subgoal_index(judged)
    by_pos_kind = _by_pos_kind()

    faithfulness = None
    if source == "superposition":
        rows = [r for r in judged if samples[r["prompt_seed"]].get("predicted_idx") is not None]
        faithful = [r["hits"][samples[r["prompt_seed"]]["predicted_idx"]] == "YES" for r in rows]
        faithfulness = dict(n=len(rows), faithful_rate=float(np.mean(faithful)) if faithful else float("nan"),
                            definition="hits[predicted_idx] == YES (predicted_idx = argmax score_choices)")
    elif source == "composition":
        correct_seeds = {s["prompt_seed"] for s in samples.values() if s.get("answer_correct")}
        rows = [r for r in judged if r["prompt_seed"] in correct_seeds]
        faithfulness = dict(**_cov_stats(rows),
                            definition="coverage restricted to rows where answer_correct == True")

    return dict(n_records=len(judged), source=source, overall=overall,
               by_pos_kind=by_pos_kind, faithfulness=faithfulness)


def log_summary(summary: dict) -> None:
    o = summary["overall"]
    logger.info(f"[{summary['source']}] OVERALL n={o['n']} coverage={o['coverage']:.3f} "
               f"at_least_one={o['at_least_one']:.3f} all_hit={o['all_hit']:.3f}")
    logger.info(f"  by_subgoal_index: {o['by_subgoal_index']}")
    for k, m in summary["by_pos_kind"].items():
        logger.info(f"  pos_kind={k:6s} n={m['n']:<4d} coverage={m['coverage']:.3f}")
    f = summary.get("faithfulness")
    if f:
        if "faithful_rate" in f:
            logger.info(f"  faithfulness: n={f['n']} faithful_rate={f['faithful_rate']:.3f} ({f['definition']})")
        else:
            logger.info(f"  faithfulness: n={f['n']} coverage={f['coverage']:.3f} ({f['definition']})")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    ap.add_argument("--source", default=None, choices=["composition", "superposition"],
                    help="--all only: restrict to one source")
    ap.add_argument("--oracle_model", default=ORACLE_MODEL)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=12)
    logging_utils.add_logging_args(ap)
    args = ap.parse_args()
    if not args.all and not args.run_dir:
        ap.error("pass --run_dir <dir> or --all")
    if args.all and args.run_dir:
        ap.error("--run_dir and --all are mutually exclusive")
    return args


def main():
    args = parse_args()
    logging_utils.configure(args)

    if args.all:
        from src.utils import env_utils
        all_dirs = fio.list_run_dirs(env_utils.DEFAULT_RESULTS_DIR, model=args.model, source=args.source)
        ready = [d for d in all_dirs if fio.is_generation_complete(d)]
        logger.info(f"--all: {len(all_dirs)} run dir(s) found, {len(ready)} fully generated -> judging those "
                   f"(skipping {len(all_dirs) - len(ready)} still mid-generation)")
        for d in ready:
            judge_run_dir(d, args.oracle_model, args.limit, args.workers)
    else:
        judge_run_dir(Path(args.run_dir), args.oracle_model, args.limit, args.workers)


if __name__ == "__main__":
    main()
