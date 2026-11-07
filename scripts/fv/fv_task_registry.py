#!/usr/bin/env python
"""Candidate task list for the GoalScope x function-vectors verbalization eval, + a viability
filter that reports each candidate's plain (no-intervention) few-shot accuracy so tasks the model
can't actually do in-context (e.g. character-counting tasks) can be dropped by inspection before
the expensive GoalScope pass runs on them.

Usage:
    # report accuracy for every candidate task (43 by default)
    conda run -n goals python scripts/fv/fv_task_registry.py

    # just one task, for a quick sanity check
    conda run -n goals python scripts/fv/fv_task_registry.py --tasks antonym
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.append(".")

import numpy as np
import torch

from scripts.fv.fv_common import build_icl_prompt, first_answer_token_id, list_all_tasks, load_fv_task
from src.models import ModelandTokenizer
from src.tokens import prepare_input
from src.utils import env_utils

# Tasks dropped outright: squad_val (paragraph-length input, blows up a 10-shot prompt),
# commonsense_qa (explicitly excluded by size/complexity), plus the viability-filter casualties
# (data/evaluation/goalscope_fv/Qwen2.5-Coder-32B-Instruct/task_viability.json) — the model
# essentially can't do these in-context at all (10-shot top-1 accuracy over 30 held-out queries):
# next_capital_letter (0.03), alphabetically_first_3 (0.37), alphabetically_last_3 (0.37),
# capitalize_second_letter (0.30). Everything else stays, even the merely-mediocre performers
# (english-german 0.43, person-occupation 0.43, ...) — "above chance" was judged good enough to
# give GoalScope a shot at.
EXCLUDE_TASKS = {
    "squad_val", "commonsense_qa",
    "next_capital_letter", "alphabetically_first_3", "alphabetically_last_3",
    "capitalize_second_letter",
}


def candidate_tasks() -> list[str]:
    """All `data/FV_dataset` task names minus EXCLUDE_TASKS, deduped on `_3`/`_5` size-variant
    pairs (keep only `_3` — shorter prompt, same underlying rule). Computed from the data
    directory, not hard-coded, so it tracks `data/FV_dataset` if it changes."""
    all_tasks = list_all_tasks()
    all_set = set(all_tasks)
    keep = []
    for t in all_tasks:
        if t in EXCLUDE_TASKS:
            continue
        m = re.match(r"^(.*)_5$", t)
        if m and f"{m.group(1)}_3" in all_set:
            continue  # drop the _5 sibling; _3 covers the same rule
        keep.append(t)
    return keep


_V_PATTERN = re.compile(r"^([a-z]+)_v_([a-z]+)(?:_\d+)?$")

# Hand-curated overrides for tasks whose auto-derived label was checked against the real judge
# eval results and found to be a topic/dataset-name noun rather than a rule description — the
# eval's own binary judge (which grades verbalizations against exactly this label text) scored
# these near zero partly because the TRUE-RULE text it was given wasn't actually rule-shaped, not
# only because GoalScope's readout was weak. Checked one at a time against real data + sample
# verbalizations (see agent_reports discussion) before changing — most low-scoring tasks did NOT
# get an override here because their auto-label was already accurate (e.g. the `_v_` list-pick
# tasks): for those, the low score reflects a genuine GoalScope signal weakness, not a bad label.
LABEL_OVERRIDES: dict[str, str] = {
    "conll2003_organization": "extract the organization name mentioned in the sentence",
    "ag_news": "classify the news snippet's topic (e.g. business, sports, world, sci/tech)",
    "national_parks": "identify which US state the US National Park is located in",
    "conll2003_location": "extract the location/place name mentioned in the sentence",
    "conll2003_person": "extract the person's name mentioned in the sentence",
    "capitalize_first_letter": "output the first letter of the word, capitalized",
    "capitalize_last_letter": "output the last letter of the word, capitalized",
    "lowercase_first_letter": "output the first letter of the word, lowercased",
    "lowercase_last_letter": "output the last letter of the word, lowercased",
    "choose_first_of_3": "choose first item in the list",
    "choose_last_of_3": "choose last item in the list",
    "choose_middle_of_3": "choose middle item in the list",
    "concept_v_object_3": "pick the concept, not a physical object, from the list",
    "english-french": "french translation of the word",
    "english-german": "german translation of the word",
    "english-spanish": "spanish translation of the word",
    "landmark-country": "identify which country the landmark is located in",
    "park-country": "identify which country the national park is located in",
    "prev_item": "previous or preceding item",
    "object_v_concept_3": "pick a physical object, not the concept, from the list",
    "sentiment": "identify the sentiment (positive or negative) of the statement",
    "synonym": "identify a synonym for the word",
    "word_length": "output the number of letters in the word",
}


def task_label(name: str) -> str:
    """Short human-readable label for the k-way classification judge menu. Checks
    `LABEL_OVERRIDES` first; otherwise auto-derived from the filename (e.g. `country-capital` ->
    `country -> capital`, `capitalize_first_letter` -> `capitalize first letter`) — a reasonable
    default, not hand-curated, so revisit per-task if the k-way judge shows systematic confusion
    traceable to a bad label (see `LABEL_OVERRIDES`).

    Special-cased: the `X_v_Y[_N]` extractive pairs (`adjective_v_verb_3`, `animal_v_object_3`,
    ...) — checked against the actual data, the FIRST-named category is always the one picked
    (e.g. `adjective_v_verb_3` -> output is the adjective). A plain `"adjective v verb"` doesn't
    encode that direction and is ambiguous against its own inverse-direction sibling
    (`verb_v_adjective`), so spell it out instead."""
    if name in LABEL_OVERRIDES:
        return LABEL_OVERRIDES[name]
    m = _V_PATTERN.match(name)
    if m:
        picked, other = m.groups()
        return f"pick the {picked}, not the {other}, from the list"
    return name.replace("-", " -> ").replace("_", " ")


def check_task_viability(mt, task_name: str, n_shots: int = 10, n_queries: int = 30,
                         seed: int = 42) -> dict:
    """Plain (no GoalScope intervention) n-shot top-1 accuracy on the FIRST answer token, over up
    to `n_queries` held-out queries. Mirrors the clean-hit check in
    `scripts/evaluation/eval_goalscope.py` (notebook 19 cell 18): a lightweight, already-used-in-
    this-repo proxy for "can the model do this task in-context at all"."""
    task = load_fv_task(task_name, seed=seed)
    rng = np.random.RandomState(seed)
    queries = task.valid[: min(n_queries, len(task.valid))]
    n_shots = min(n_shots, len(task.train))
    hits = 0
    for q in queries:
        demo_idx = rng.choice(len(task.train), n_shots, replace=False)
        demos = [task.train[i] for i in demo_idx]
        prompt = build_icl_prompt(demos, q["input"])
        tid = first_answer_token_id(mt, prompt, q["output"])
        inp = prepare_input(prompts=prompt, tokenizer=mt)
        with torch.inference_mode():
            logits = mt._model(**dict(inp)).logits[0, -1, :]
        hits += int(int(logits.argmax().item()) == tid)
    return dict(task=task_name, n=len(queries), accuracy=hits / max(1, len(queries)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", default=None,
                    help="subset to check; default: the full candidate list (candidate_tasks())")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    ap.add_argument("--n_shots", type=int, default=10)
    ap.add_argument("--n_queries", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None,
                    help="where to save the report JSON (default: "
                    "<results>/evaluation/goalscope_fv/<model>/task_viability.json)")
    args = ap.parse_args()

    tasks = args.tasks or candidate_tasks()
    print(f"checking {len(tasks)} task(s): {tasks}")

    mt = ModelandTokenizer(model_key=args.model, dtype=torch.bfloat16,
                           device_map="auto", attn_implementation="eager")
    torch.set_grad_enabled(False)

    rows = []
    for t in tasks:
        r = check_task_viability(mt, t, n_shots=args.n_shots, n_queries=args.n_queries, seed=args.seed)
        rows.append(r)
        print(f"  {t:32s} n={r['n']:3d} acc={r['accuracy']:.2f}")

    out_path = Path(args.out) if args.out else (
        Path(env_utils.DEFAULT_RESULTS_DIR) / "evaluation" / "goalscope_fv"
        / args.model.split("/")[-1] / "task_viability.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dict(model=args.model, n_shots=args.n_shots,
                                        n_queries=args.n_queries, seed=args.seed, tasks=rows),
                                   indent=2))
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
