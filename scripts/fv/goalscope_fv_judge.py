#!/usr/bin/env python
"""GoalScope verbalization eval on the function-vectors task suite — Script B: LM-judge scoring
of the raw verbalization text `goalscope_fv_generate.py` stored. Deliberately a SEPARATE script
(per design discussion) so the expensive generation pass and the API-calling judge pass are
decoupled — this can be run later, and re-run after widening the task list, without redoing
generation. Two SEPARATE judge calls per record (not combined):

  - binary "accuracy": does the verbalization correctly describe the true task's rule?
    (`fv_hard_eval`, a new rubric modeled on `scripts/evaluation/eval_goalscope.py`'s
    `hard_eval`/`_parse_verdict` <verdict> tag pattern)
  - k-way "classification": which of the candidate task labels does it best match?
    (`kway_assign`, reused as-is from `eval_goalscope.py` — already generic)

Both use `ask_claude(..., use_cache=True)`, so re-running this script only pays for genuinely new
(record, judge-call) pairs.

Usage:
    conda run -n goals python scripts/fv/goalscope_fv_judge.py --run_dir <path from goalscope_fv_generate.py>
"""
import argparse
import json
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.append(".")

import numpy as np

from scripts.evaluation.eval_goalscope import ORACLE_MODEL, _parse_verdict, kway_assign
from scripts.fv import goalscope_fv_io as fio
from scripts.fv.fv_task_registry import candidate_tasks, task_label
from src.utils import logging_utils
from src.utils.oracle_llms import ask_claude

logger = logging.getLogger(__name__)

FV_HARD_SYS = """You are grading a lightweight interpretability probe (a patchscope read-out). The DESCRIPTION below was read out of a model's hidden state at a SINGLE token while the model was working through a few-shot in-context-learning prompt that repeatedly maps an INPUT to an OUTPUT by one fixed hidden rule (e.g. "Q: flawed\\nA: perfect" for an antonym task). Because it is squeezed out of one token under a steering/patching setup, the description is usually NOISY: it may be wrapped in chat scaffolding (like 'Sure, but first ...'), be cut off mid-sentence, or repeat and loop. Look PAST the surface noise and grade only the SUBSTANCE.

You will be given the TRUE RULE and the DESCRIPTION. The question: does the description correctly identify — even loosely, even partially — what the true rule does? Grade as leniently as possible for a single-token read-out — EXCEPT that a description which has collapsed into a degenerate repeating loop is NOT a success, no matter what tokens the loop contains (see HEAVY_LOOP).

Decide in this order — check MALFORMED and HEAVY_LOOP FIRST, and only then YES / NO / NO_INFO:

  MALFORMED  - the description is contentless: empty, pure punctuation or garbage, or repetition that carries NO relevant meaning.
  HEAVY_LOOP - the description is DOMINATED by a short repeating cycle: a unit of roughly five words/tokens or fewer that repeats over and over. Once you collapse the repetition, little or no coherent description is left. This takes PRECEDENCE over YES.
  YES        - the description names the true rule, or a clearly-recognizable paraphrase of it, however partial, imprecise, or garbled — and it is NOT merely a repeating loop. Naming just the input or output DOMAIN correctly (e.g. "words", "countries") without the mapping is NOT enough on its own for YES unless it also gestures at the relationship.
  NO         - the description names a DIFFERENT, unrelated rule, or a coherent but wrong relationship between input and output.
  NO_INFO    - the description is coherent but too generic to indicate any specific rule (e.g. just "mapping one word to another" or "answering a question").

Output only YES, NO, NO_INFO, MALFORMED, or HEAVY_LOOP.
Put your final answer inside <verdict>{ans}</verdict> tags, with no other text or explanation, to facilitate automated parsing."""


def fv_hard_eval(readout: str, true_label: str, oracle_model: str = ORACLE_MODEL) -> dict:
    """dict(verdict, prompt, response). verdict in {YES, NO, NO_INFO, MALFORMED, UNPARSED}."""
    if not readout or not readout.strip():
        return dict(verdict="MALFORMED", prompt=None, response=None)
    prompt = f"True rule: {true_label}\nDescription: {readout.strip()}\nAnswer:"
    resp = ask_claude(prompt=prompt, system_prompt=FV_HARD_SYS, max_tokens=512,
                      use_cache=True, model_name=oracle_model)
    return dict(verdict=_parse_verdict(resp), prompt=prompt, response=resp)


def compute_summary(records: list[dict]) -> dict:
    has_kway = bool(records) and all(r.get("kway_true") is not None for r in records)

    def _stats(grp):
        d = dict(n=len(grp), hard_yes=float(np.mean([r["hard"] == "YES" for r in grp])),
                 heavy_loop=float(np.mean([r["hard"] == "HEAVY_LOOP" for r in grp])))
        if has_kway:
            d["kway_ok"] = float(np.mean([r["kway_pred"] == r["kway_true"] for r in grp]))
        return d

    def _group(key):
        return {str(k): _stats(g) for k in sorted({r[key] for r in records})
               if (g := [r for r in records if r[key] == k])}

    def _position_label(side: str, tok_offset: int) -> str:
        # side="Q"/"A", tok_offset=0/1 by default (window=2) -> "Q_tok"/"Q_colon"/"A_tok"/"A_colon"
        # (names true only for window=2 on single-token "Q:"/"A:" prefixes; a wider --window or a
        # task whose prefix doesn't tokenize as exactly 2 tokens still gets a safe "{side}_{offset}").
        names = {0: "tok", 1: "colon"}
        return f"{side}_{names.get(tok_offset, tok_offset)}"

    def _by_position():
        keys = sorted({(r["side"], r["tok_offset"]) for r in records})
        return {_position_label(*k): _stats(g) for k in keys
               if (g := [r for r in records if (r["side"], r["tok_offset"]) == k])}

    overall = _stats(records) if records else dict(n=0, hard_yes=float("nan"))
    # by_side/by_tok_offset are MARGINALS over by_position (e.g. by_side["A"] averages A_tok and
    # A_colon together) — kept for a quick coarse read, but by_position is the one that doesn't
    # hide interactions (A_tok and A_colon can — and do — behave very differently).
    return dict(n_records=len(records), by_task=_group("task"), by_side=_group("side"),
               by_tok_offset=_group("tok_offset"), by_position=_by_position(), overall=overall)


def log_summary(summary: dict) -> None:
    o = summary["overall"]
    k = f" kway_ok={o['kway_ok']:.3f}" if "kway_ok" in o else ""
    logger.info(f"OVERALL n={summary['n_records']} hard_yes={o.get('hard_yes', float('nan')):.3f}{k}")
    for group_name in ("by_task", "by_position", "by_side", "by_tok_offset"):
        logger.info(f"-- {group_name} --")
        for name, m in summary[group_name].items():
            k = f" kway_ok={m['kway_ok']:.3f}" if "kway_ok" in m else ""
            logger.info(f"  {name:24s} n={m['n']:<4d} hard_yes={m['hard_yes']:.3f}{k}")


def _judge_one(r: dict, menu_tasks: list[str], menu_labels: list[str], oracle_model: str) -> dict:
    """Both judge calls for one record. Pure network I/O (ask_claude) — safe to run concurrently
    across many records/threads; nothing here touches the GPU or shared mutable state."""
    true_task = r["task"]
    true_label = task_label(true_task)
    readout = r["verbalization"]
    hv = fv_hard_eval(readout, true_label, oracle_model=oracle_model)
    kv = kway_assign(readout, menu_labels)
    true_idx = menu_tasks.index(true_task) if true_task in menu_tasks else None
    return dict(
        **r, hard=hv["verdict"], hard_prompt=hv["prompt"], hard_response=hv["response"],
        kway_pred=kv["pred"], kway_true=true_idx, kway_prompt=kv["prompt"], kway_response=kv["response"],
    )


def judge_run_dir(run_dir: Path, menu_tasks: list[str], menu_labels: list[str],
                  oracle_model: str, limit: int | None = None, workers: int = 12) -> dict:
    """Judge every un-judged record in one run dir (binary + k-way), append to `judged.jsonl`,
    recompute and save `judge_summary.json`. Returns the summary dict.

    Each record's two judge calls are just `ask_claude` HTTP requests — no GPU involved — so this
    dispatches `workers` of them concurrently via a thread pool instead of one at a time. Results
    are written to `judged.jsonl` as they complete (order doesn't matter — each record is judged
    independently), under a lock since multiple threads share the one file handle."""
    records = [json.loads(l) for l in fio.records_path(run_dir).read_text().splitlines() if l.strip()]
    already = fio.read_judged_ids(run_dir)
    todo = [r for r in records
           if (r["prompt_seed"], r["demo_idx"], r["side"], r["tok_offset"]) not in already]
    if limit:
        todo = todo[:limit]
    logger.info(f"[{run_dir.parent.name}] {len(records)} records total | "
               f"{len(already)} already judged | judging {len(todo)} now ({workers} workers)")

    write_lock = threading.Lock()
    with open(fio.judged_path(run_dir), "a") as fh, ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_judge_one, r, menu_tasks, menu_labels, oracle_model) for r in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            result = fut.result()  # raises here if a record's judging ultimately failed
            with write_lock:
                fio.append_jsonl(fh, result)
            if i % 20 == 0 or i == len(futures):
                logger.info(f"judged {i}/{len(todo)}")

    all_judged = [json.loads(l) for l in fio.judged_path(run_dir).read_text().splitlines() if l.strip()]
    summary = compute_summary(all_judged)
    fio.write_summary(run_dir, summary, name="judge_summary.json")
    logger.info(f"saved -> {fio.judged_path(run_dir)}")
    log_summary(summary)
    return summary


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", default=None, help="a run dir written by goalscope_fv_generate.py")
    ap.add_argument("--all", action="store_true",
                    help="judge every FULLY-GENERATED run dir under "
                    "<results>/evaluation/goalscope_fv/<model>/ instead of just --run_dir "
                    "(skips any task still mid-generation — see goalscope_fv_io.is_generation_complete). "
                    "Safe to re-run any time: already-judged records are skipped, and a task whose "
                    "generation finishes later just gets picked up on the next --all pass.")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct", help="--all only: which model's run dirs to scan")
    ap.add_argument("--menu_tasks", nargs="+", default=None,
                    help="k-way candidate labels come from these task names; default: "
                    "fv_task_registry.candidate_tasks() (the full ~43-task registry)")
    ap.add_argument("--oracle_model", default=ORACLE_MODEL)
    ap.add_argument("--limit", type=int, default=None,
                    help="SMOKE: judge only the first N un-judged records PER run dir")
    ap.add_argument("--workers", type=int, default=12,
                    help="concurrent ask_claude calls per run dir (pure API I/O, no GPU involved "
                    "— raise this rather than accepting a sequential judge pass; watch for 429s "
                    "if you push it much higher)")
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

    menu_tasks = args.menu_tasks or candidate_tasks()
    menu_labels = [task_label(t) for t in menu_tasks]

    if args.all:
        from src.utils import env_utils
        all_dirs = fio.list_run_dirs(env_utils.DEFAULT_RESULTS_DIR, model=args.model)
        ready = [d for d in all_dirs if fio.is_generation_complete(d)]
        logger.info(f"--all: {len(all_dirs)} run dir(s) found for {args.model}, "
                   f"{len(ready)} fully generated -> judging those "
                   f"(skipping {len(all_dirs) - len(ready)} still mid-generation)")
        for d in ready:
            judge_run_dir(d, menu_tasks, menu_labels, args.oracle_model, args.limit, args.workers)
    else:
        judge_run_dir(Path(args.run_dir), menu_tasks, menu_labels, args.oracle_model, args.limit, args.workers)


if __name__ == "__main__":
    main()
