#!/usr/bin/env python
"""GoalScope verbalization eval on the function-vectors task suite — Script A ("store all the
verbalization text first"): builds 10-shot Q:/A: ICL prompts per task, runs GoalScope's goal-head
read-out at pivot positions inside the LAST 3 few-shot examples, and stores the raw verbalization
text. Deliberately does NOT call an LM judge — that is `goalscope_fv_judge.py`, a separate script
run afterward (possibly repeatedly) against the cached text.

Pivot positions per prompt: for each of the last 3 (of `--n_shots`) demos, the Q-row ("Q:" +
input value) and A-row ("A:" + output value), each capped at the first `--window` tokens (or all
of the row's tokens if shorter) — up to `2 * 3 * window` positions per prompt.

Reuses `scripts/evaluation/eval_goalscope.py`'s head-loading (`load_head_set`) and
`apply_goalscope` call shape, and `scripts/fv/fv_common.py`'s FV task loading / prompt building.
Storage follows `scripts/fv/goalscope_fv_io.py` (resumable JSONL, modeled on
`scripts/evaluation/faithfulness_io.py`).

Usage:
    # first iteration: antonym only, the default 20 prompts
    conda run -n goals python scripts/fv/goalscope_fv_generate.py --tasks antonym

    # full candidate list (fv_task_registry.candidate_tasks())
    conda run -n goals python scripts/fv/goalscope_fv_generate.py
"""
import argparse
import logging
import os
import random
import sys
import time
from glob import glob

sys.path.append(".")

# CUDA_VISIBLE_DEVICES must be set before `torch` is imported anywhere in the process (including
# transitively, via the heavy imports right below) — pre-scan argv for --gpus here rather than
# after argparse.parse_args(), which would run too late. Mirrors the --gpus convention in
# scripts/evaluation/middle_ground.py. Pins this process to one (or a few) specific GPU(s) instead
# of device_map="auto" spreading the model across every free GPU on the box, which can crowd out
# other people's jobs even when it doesn't share a GPU with them directly.
if "--gpus" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[sys.argv.index("--gpus") + 1]

import numpy as np
import torch

from scripts.evaluation.eval_goalscope import load_head_set, resolve_heads_name
from scripts.fv import goalscope_fv_io as fio
from scripts.fv.fv_common import build_icl_prompt_with_spans, load_fv_task
from scripts.fv.fv_task_registry import candidate_tasks
from src.goals.goalscope import GOALSCOPE_SYS, GOALSCOPE_SYS_GENERIC, apply_goalscope
from src.models import ModelandTokenizer
from src.tokens import prepare_input
from src.utils import env_utils, logging_utils

logger = logging.getLogger(__name__)

# The free-form "what task are you performing?" probe (notebook 19's `GOALSCOPE_PROMPTS["task_rule"]`
# — single-rule framing, matches this eval's setting of one task per run, not the interleaved-mixture
# setting `task_mixture`/`task_specific` were written for).
TASK_RULE_QUESTION = (
    "You are answering a series of questions that all follow the same hidden rule. "
    "In one short phrase, what task are you performing?"
)


def build_task_rule_probe(mt, system_prompt: str = GOALSCOPE_SYS) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": TASK_RULE_QUESTION},
        {"role": "assistant", "content": ""},
    ]
    kw = dict(add_generation_prompt=False, continue_final_message=True, tokenize=False)
    try:
        return mt.tokenizer.apply_chat_template(
            messages, chat_template_kwargs={"enable_thinking": False}, **kw
        )
    except TypeError:
        return mt.tokenizer.apply_chat_template(messages, **kw)


def default_heads_spec(model_name: str) -> str:
    """Prefer the exported JSON head set (`head_sets/<slug>/goal__lam20_thr0.9_band*.json`,
    tracked in git — same file notebook 19 loads first) over `eval_goalscope.py`'s `"goal"` spec,
    which requires a raw mask checkpoint (`results/localization/...`) that isn't present in every
    environment. Falls back to `"goal"` if no JSON is found."""
    slug = model_name.split("/")[-1]
    matches = sorted(glob(f"head_sets/{slug}/goal__lam20_thr0.9_band*.json"))
    return matches[0] if matches else "goal"


def sample_prompt(task, n_shots: int, seed: int) -> tuple[list[dict], dict]:
    """`n_shots` demos (from `task.train`, no replacement) + 1 query (from `task.valid`)."""
    rng = random.Random(seed)
    demos = rng.sample(task.train, n_shots)
    query = rng.choice(task.valid)
    return demos, query


def pivot_specs(demos: list[dict], row_spans: list[dict], offsets: list[tuple[int, int]],
                window: int) -> list[dict]:
    """For the LAST 3 demos, the Q-row/A-row token-index windows (up to `window` tokens, capped at
    the row's own length — never spilling into the next row). `offsets` is the full prompt's
    per-token (char_start, char_end), e.g. from `prepare_input(..., return_offsets_mapping=True)`."""
    specs = []
    last3 = range(max(0, len(demos) - 3), len(demos))
    for demo_idx in last3:
        for side, key in (("Q", "q"), ("A", "a")):
            c0, c1 = row_spans[demo_idx][key]
            tok_idxs = [i for i, (s, e) in enumerate(offsets) if s < c1 and e > c0]
            row_len = len(tok_idxs)
            for tok_offset, abs_pos in enumerate(tok_idxs[:window]):
                specs.append(dict(
                    demo_idx=demo_idx, demo_input=demos[demo_idx]["input"],
                    demo_output=demos[demo_idx]["output"], side=side,
                    tok_offset=tok_offset, row_len=row_len, abs_pos=abs_pos,
                ))
    return specs


def run_task(mt, task_name: str, config: dict, results_dir, probe_prompt: str,
            heads, gs_kwargs: dict) -> None:
    task = load_fv_task(task_name, seed=config["task_split_seed"])
    task_config = {**config, "task": task_name}
    run_dir = fio.run_dir_for(task_config, results_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    prev = fio.read_config(run_dir)
    if prev is not None and prev.get("selected_heads") is not None:
        old = [tuple(x) for x in prev["selected_heads"]]
        new = [tuple(int(x) for x in h) for h in heads]
        if old != new and fio.read_completed_prompt_seeds(run_dir):
            raise SystemExit(
                f"run dir {run_dir} holds samples produced with a DIFFERENT head set "
                f"({len(old)} heads, now {len(new)}). Resuming would mix them."
            )
    task_config["selected_heads"] = [[int(l), int(h)] for l, h in heads]
    task_config["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    fio.write_config(run_dir, task_config)

    completed = set(fio.read_completed_prompt_seeds(run_dir))
    kept = fio.prune_orphan_records(run_dir, completed)
    remaining = [s for s in range(config["n_prompts"]) if s not in completed]
    logger.info(f"[{task_name}] run dir -> {run_dir} | done: {len(completed)} "
               f"({kept} records kept) | remaining: {len(remaining)}")
    if not remaining:
        return

    rec_fh = open(fio.records_path(run_dir), "a")
    samp_fh = open(fio.samples_path(run_dir), "a")
    for prompt_seed in remaining:
        demos, query = sample_prompt(task, config["n_shots"], seed=config["seed"] + prompt_seed)
        prompt, row_spans = build_icl_prompt_with_spans(demos, query["input"])

        tok = prepare_input(prompts=prompt, tokenizer=mt, return_offsets_mapping=True)
        offsets = tok["offset_mapping"][0].tolist()

        specs = pivot_specs(demos, row_spans, offsets, config["window"])
        apos = sorted({s["abs_pos"] for s in specs})

        res = apply_goalscope(
            prompt, mt=mt, show=False, include_noncode=False,
            token_positions=apos, goalscope_prompt=probe_prompt, **gs_kwargs,
        )
        readouts = {r["pos"]: r for r in res}

        for s in specs:
            r = readouts.get(s["abs_pos"], {})
            fio.append_jsonl(rec_fh, dict(
                task=task_name, prompt_seed=prompt_seed, **s,
                token_text=r.get("token", ""), verbalization=r.get("patched_gen", ""),
            ))
        fio.append_jsonl(samp_fh, dict(
            task=task_name, prompt_seed=prompt_seed, prompt=prompt,
            demos=demos, query=query, n_positions_scored=len(specs),
        ))
        logger.info(f"[{task_name}] prompt {prompt_seed}: {len(specs)} positions scored")
    rec_fh.close()
    samp_fh.close()


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", default=None,
                    help="default: fv_task_registry.candidate_tasks() (the full ~43-task list)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    ap.add_argument("--gpus", default=None,
                    help="pin this process to specific GPU(s), e.g. '3' or '2,3' — sets "
                    "CUDA_VISIBLE_DEVICES before torch is imported (see top of file). Combine "
                    "with device_map='auto' (unaffected here) so the model only ever sees the "
                    "GPU(s) listed. Default: unset (torch sees every GPU on the box).")
    ap.add_argument("--heads", default=None,
                    help="goal | <path>.json (see eval_goalscope.py); default: the exported "
                    "head_sets/<slug>/goal__lam20_thr0.9_band*.json if present, else 'goal'")
    ap.add_argument("--goal_threshold", type=float, default=0.9)
    ap.add_argument("--goal_band", type=int, nargs=2, default=None, metavar=("START", "END"))
    ap.add_argument("--strategy", default="additive_norm", choices=["additive", "additive_norm", "replace"])
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--steer_layer_range", type=int, nargs=2, default=[35, 46])
    ap.add_argument("--n_steer_tokens", default="all")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--n_shots", type=int, default=10)
    ap.add_argument("--n_prompts", type=int, default=20)
    ap.add_argument("--window", type=int, default=2,
                    help="tokens per row, from the 'Q:'/'A:' prefix (capped at the row's own "
                    "length). Default 2 = just the prefix + colon token ('Q'+':' / 'A'+':') — "
                    "the smoke test showed signal is already gone by the value token (offset 2).")
    ap.add_argument("--seed", type=int, default=0, help="base seed for prompt sampling (per-prompt seed = this + prompt index)")
    ap.add_argument("--task_split_seed", type=int, default=42, help="seed for each task's train/valid split")
    ap.add_argument("--generic_sys", action="store_true",
                    help="use GOALSCOPE_SYS_GENERIC ('You are a helpful assistant.') instead of "
                    "the default code-flavored GOALSCOPE_SYS for the readout probe's system "
                    "message -- tags the run's `probe` as 'task_rule_generic' so it lands in a "
                    "separate run dir from the default 'task_rule' runs.")
    logging_utils.add_logging_args(ap)
    return ap.parse_args()


def main():
    args = parse_args()
    logging_utils.configure(args)
    torch.set_grad_enabled(False)
    results_dir = env_utils.DEFAULT_RESULTS_DIR

    tasks = args.tasks or candidate_tasks()
    n_steer = "all" if str(args.n_steer_tokens).lower() == "all" else int(args.n_steer_tokens)
    args.heads = args.heads or default_heads_spec(args.model)
    logger.info(f"--heads resolved to {args.heads!r}")

    mt = ModelandTokenizer(model_key=args.model, dtype=torch.bfloat16,
                           device_map="auto", attn_implementation="eager")
    n_heads = mt.model.config.num_attention_heads
    heads = load_head_set(args.heads, args.model, results_dir, n_heads,
                          goal_threshold=args.goal_threshold, goal_band=args.goal_band,
                          n_layer=mt.n_layer)
    heads_name = resolve_heads_name(args.heads)
    logger.info(f"heads[{heads_name}] = {len(heads)} (layer,head) pairs")

    sys_prompt = GOALSCOPE_SYS_GENERIC if args.generic_sys else GOALSCOPE_SYS
    probe_prompt = build_task_rule_probe(mt, system_prompt=sys_prompt)
    probe_name = "task_rule_generic" if args.generic_sys else "task_rule"

    config = dict(
        model=args.model, heads_spec=args.heads, heads_name=heads_name,
        strategy=args.strategy, alpha=args.alpha, steer_layer_range=list(args.steer_layer_range),
        n_steer_tokens=n_steer, max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
        n_shots=args.n_shots, n_prompts=args.n_prompts, window=args.window, probe=probe_name,
        seed=args.seed, task_split_seed=args.task_split_seed,
    )
    gs_kwargs = dict(
        selected_heads=heads, strategy=args.strategy, alpha=args.alpha,
        steer_layer_range=tuple(args.steer_layer_range), n_steer_tokens=n_steer,
        max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
    )

    for t in tasks:
        run_task(mt, t, config, results_dir, probe_prompt, heads, gs_kwargs)


if __name__ == "__main__":
    main()
