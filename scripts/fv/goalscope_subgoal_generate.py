#!/usr/bin/env python
"""GoalScope sub-goal-coverage benchmark — Script A ("store all the verbalization text first").

Two task sources, one pipeline:
  - "composition": all 18 `apoorvkh/composing-functions` tasks (composition_task_registry.py).
    Sub-goals checked = [f_description, g_description] (the two composed functions).
  - "superposition": the 5 ICL-superposition families (country, AB, word, add_translate_efs,
    add_translate_dfs — src/goals/task_superposition.py). Sub-goals checked = the family's
    task_labels (the candidate hidden rules the mixture is drawn from).

Position scheme mirrors notebooks/clock/{19,20} exactly (not the FV eval's demo-row pivot_specs
scheme, which is FV-specific): generate `--gen_tokens` from the prompt, then GoalScope the pivot
(last prompt token) + every generated token. One `apply_goalscope` call per prompt.

Faithfulness metadata (used only at judge/analysis time, not judged here):
  - superposition: `predicted_idx` = argmax of `score_choices` (teacher-forced prob per candidate)
    over the query -- which task the model is actually acting on, independent of the judge.
  - composition: `answer_correct` = does the model's own greedy-generated continuation start with
    the true `GFx`.

Usage:
    conda run -n goals python scripts/fv/goalscope_subgoal_generate.py --gpus 2
    conda run -n goals python scripts/fv/goalscope_subgoal_generate.py --sources composition --composition_tasks antonym-spanish
"""
import argparse
import logging
import os
import random
import sys
import time
from glob import glob

sys.path.append(".")

if "--gpus" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[sys.argv.index("--gpus") + 1]

import torch

from scripts.evaluation.eval_goalscope import load_head_set, resolve_heads_name
from scripts.fv import goalscope_subgoal_io as fio
from scripts.fv.composition_task_registry import ALL_TASKS as COMPOSITION_TASKS
from scripts.fv.composition_task_registry import load_task_records, task_subgoals, to_demos
from scripts.fv.fv_common import build_icl_prompt
from src.functional import generate_with_patch
from src.goals.goalscope import GOALSCOPE_SYS, apply_goalscope
from src.goals.task_superposition import build_superposition_prompt, score_choices
from src.models import ModelandTokenizer
from src.tokens import prepare_input
from src.utils import env_utils, logging_utils

logger = logging.getLogger(__name__)

SUPERPOSITION_TASKS = ["country", "AB", "word", "add_translate_efs", "add_translate_dfs"]

TASK_MIXTURE_QUESTION = (
    "You are answering a series of questions that each follow one of a few possible hidden "
    "rules. In one short phrase, what tasks are you choosing between?"
)
COMPOSITION_RULE_QUESTION = (
    "You are answering a series of questions that all follow the same hidden composition of "
    "rules. In two short phrases, what two relations/tasks, f and g, are you performing?"
)


def _chat_probe(mt, question: str) -> str:
    messages = [
        {"role": "system", "content": GOALSCOPE_SYS},
        {"role": "user", "content": question},
        {"role": "assistant", "content": ""},
    ]
    kw = dict(add_generation_prompt=False, continue_final_message=True, tokenize=False)
    try:
        return mt.tokenizer.apply_chat_template(messages, chat_template_kwargs={"enable_thinking": False}, **kw)
    except TypeError:
        return mt.tokenizer.apply_chat_template(messages, **kw)


def default_heads_spec(model_name: str) -> str:
    slug = model_name.split("/")[-1]
    matches = sorted(glob(f"head_sets/{slug}/goal__lam20_thr0.9_band*.json"))
    return matches[0] if matches else "goal"


@torch.no_grad()
def generate_clean(mt, prompt: str, gen_tokens: int) -> str:
    tok = prepare_input(prompts=prompt, tokenizer=mt, padding_side="left")
    return generate_with_patch(mt, inputs=tok, n_gen_per_prompt=1, max_new_tokens=gen_tokens,
                               do_sample=False, remove_prefix=True)[0]


def scored_positions(mt, prompt: str, gen: str) -> tuple[list[int], int]:
    """[pivot] + all generated-token positions. Returns (positions, p0) where p0 is the first
    generated token's index (pivot = p0 - 1)."""
    p0 = prepare_input(prompts=prompt, tokenizer=mt, padding_side="left")["input_ids"].shape[1]
    ptot = prepare_input(prompts=prompt + gen, tokenizer=mt, padding_side="left")["input_ids"].shape[1]
    return [p0 - 1] + list(range(p0, ptot)), p0


def run_one_prompt(mt, prompt: str, gs_kwargs: dict, probe_prompt: str) -> tuple[str, dict]:
    """Generate + GoalScope one prompt. Returns (gen, {abs_pos: {token, verbalization}})."""
    gen = generate_clean(mt, prompt, gs_kwargs.pop("gen_tokens"))
    positions, p0 = scored_positions(mt, prompt, gen)
    full = prepare_input(prompts=prompt + gen, tokenizer=mt, padding_side="left")
    res = apply_goalscope(
        prompt=full, mt=mt, show=False, include_noncode=False,
        token_positions=positions, goalscope_prompt=probe_prompt, **gs_kwargs,
    )
    readouts = {r["pos"]: r for r in res}
    out = {}
    for pos in positions:
        pos_kind = "pivot" if pos == p0 - 1 else "gen"
        gen_offset = -1 if pos_kind == "pivot" else pos - p0
        r = readouts.get(pos, {})
        out[pos] = dict(pos_kind=pos_kind, gen_offset=gen_offset,
                        token_text=r.get("token", ""), verbalization=r.get("patched_gen", ""))
    return gen, out


def run_task(mt, source: str, task_name: str, config: dict, results_dir, probe_prompt: str,
            heads, gs_kwargs: dict) -> None:
    task_config = {**config, "source": source, "task": task_name}
    run_dir = fio.run_dir_for(task_config, results_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    prev = fio.read_config(run_dir)
    if prev is not None and prev.get("selected_heads") is not None:
        old = [tuple(x) for x in prev["selected_heads"]]
        new = [tuple(int(x) for x in h) for h in heads]
        if old != new and fio.read_completed_prompt_seeds(run_dir):
            raise SystemExit(f"run dir {run_dir} holds samples produced with a DIFFERENT head set "
                             f"({len(old)} heads, now {len(new)}). Resuming would mix them.")
    task_config["selected_heads"] = [[int(l), int(h)] for l, h in heads]
    task_config["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    fio.write_config(run_dir, task_config)

    completed = set(fio.read_completed_prompt_seeds(run_dir))
    kept = fio.prune_orphan_records(run_dir, completed)
    remaining = [s for s in range(config["n_prompts"]) if s not in completed]
    logger.info(f"[{source}/{task_name}] run dir -> {run_dir} | done: {len(completed)} "
               f"({kept} records kept) | remaining: {len(remaining)}")
    if not remaining:
        return

    composition_recs = load_task_records(task_name, seed=config["task_split_seed"]) if source == "composition" else None

    rec_fh = open(fio.records_path(run_dir), "a")
    samp_fh = open(fio.samples_path(run_dir), "a")
    for prompt_seed in remaining:
        rng = random.Random(config["seed"] + prompt_seed)
        sample_meta = dict(source=source, task=task_name, prompt_seed=prompt_seed)

        if source == "superposition":
            sp = build_superposition_prompt(task_name, num_examples=config["n_shots"],
                                            seed=config["seed"] + prompt_seed, prompt_format="qa")
            prompt = sp.prompt
            subgoals = sp.task_labels
            probs = score_choices(mt, prompt, sp.output_choices)
            predicted_idx = int(max(range(len(probs)), key=lambda i: probs[i])) if probs else None
            sample_meta.update(output_choices=sp.output_choices, output_probs=probs,
                               predicted_idx=predicted_idx, task_indices=sp.task_indices)
        else:
            recs = rng.sample(composition_recs, config["n_shots"] + 1)
            demos = to_demos(recs[: config["n_shots"]])
            query = recs[config["n_shots"]]
            prompt = build_icl_prompt(demos, query["x"])
            subgoals = task_subgoals(task_name)
            sample_meta.update(x=query["x"], true_Fx=query["Fx"], true_GFx=query["GFx"])

        gs_kw = dict(gs_kwargs)
        gen, positions = run_one_prompt(mt, prompt, gs_kw, probe_prompt)

        if source == "composition":
            answer_correct = gen.strip().startswith(sample_meta["true_GFx"].strip())
            sample_meta.update(generated_continuation=gen, answer_correct=answer_correct)
        else:
            sample_meta.update(generated_continuation=gen)

        for pos, info in positions.items():
            fio.append_jsonl(rec_fh, dict(
                source=source, task=task_name, prompt_seed=prompt_seed, abs_pos=pos, **info,
            ))
        fio.append_jsonl(samp_fh, dict(prompt=prompt, subgoals=subgoals, **sample_meta))
        logger.info(f"[{source}/{task_name}] prompt {prompt_seed}: {len(positions)} positions scored, "
                   f"gen={gen[:40]!r}")
    rec_fh.close()
    samp_fh.close()


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="+", default=["composition", "superposition"],
                    choices=["composition", "superposition"])
    ap.add_argument("--composition_tasks", nargs="+", default=None, help="default: all 18")
    ap.add_argument("--superposition_tasks", nargs="+", default=None, help="default: all 5")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    ap.add_argument("--gpus", default=None)
    ap.add_argument("--heads", default=None)
    ap.add_argument("--goal_threshold", type=float, default=0.9)
    ap.add_argument("--goal_band", type=int, nargs=2, default=None, metavar=("START", "END"))
    ap.add_argument("--strategy", default="additive_norm", choices=["additive", "additive_norm", "replace"])
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--steer_layer_range", type=int, nargs=2, default=[35, 46])
    ap.add_argument("--n_steer_tokens", default="all")
    ap.add_argument("--max_new_tokens", type=int, default=48, help="GoalScope read-out length")
    ap.add_argument("--gen_tokens", type=int, default=8, help="tokens generated (clean) from the task prompt to score")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--n_shots", type=int, default=10)
    ap.add_argument("--n_prompts", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task_split_seed", type=int, default=42)
    logging_utils.add_logging_args(ap)
    return ap.parse_args()


def main():
    args = parse_args()
    logging_utils.configure(args)
    torch.set_grad_enabled(False)
    results_dir = env_utils.DEFAULT_RESULTS_DIR

    args.heads = args.heads or default_heads_spec(args.model)
    logger.info(f"--heads resolved to {args.heads!r}")

    mt = ModelandTokenizer(model_key=args.model, dtype=torch.bfloat16,
                           device_map="auto", attn_implementation="eager")
    n_heads = mt.model.config.num_attention_heads
    heads = load_head_set(args.heads, args.model, results_dir, n_heads,
                          goal_threshold=args.goal_threshold, goal_band=args.goal_band, n_layer=mt.n_layer)
    heads_name = resolve_heads_name(args.heads)
    logger.info(f"heads[{heads_name}] = {len(heads)} (layer,head) pairs")

    n_steer = "all" if str(args.n_steer_tokens).lower() == "all" else int(args.n_steer_tokens)
    base_gs_kwargs = dict(selected_heads=heads, strategy=args.strategy, alpha=args.alpha,
                          steer_layer_range=tuple(args.steer_layer_range), n_steer_tokens=n_steer,
                          max_new_tokens=args.max_new_tokens, batch_size=args.batch_size)

    probes = {}
    if "composition" in args.sources:
        probes["composition"] = _chat_probe(mt, COMPOSITION_RULE_QUESTION)
    if "superposition" in args.sources:
        probes["superposition"] = _chat_probe(mt, TASK_MIXTURE_QUESTION)

    base_config = dict(model=args.model, heads_spec=args.heads, heads_name=heads_name,
                       strategy=args.strategy, alpha=args.alpha,
                       steer_layer_range=list(args.steer_layer_range), n_steer_tokens=n_steer,
                       max_new_tokens=args.max_new_tokens, gen_tokens=args.gen_tokens,
                       batch_size=args.batch_size, n_shots=args.n_shots, n_prompts=args.n_prompts,
                       seed=args.seed, task_split_seed=args.task_split_seed)

    if "composition" in args.sources:
        tasks = args.composition_tasks or COMPOSITION_TASKS
        config = {**base_config, "probe": "composition_rule"}
        for t in tasks:
            run_task(mt, "composition", t, config, results_dir, probes["composition"], heads,
                     {**base_gs_kwargs, "gen_tokens": args.gen_tokens})

    if "superposition" in args.sources:
        tasks = args.superposition_tasks or SUPERPOSITION_TASKS
        config = {**base_config, "probe": "task_mixture"}
        for t in tasks:
            run_task(mt, "superposition", t, config, results_dir, probes["superposition"], heads,
                     {**base_gs_kwargs, "gen_tokens": args.gen_tokens})


if __name__ == "__main__":
    main()
