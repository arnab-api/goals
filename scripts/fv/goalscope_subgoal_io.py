"""IO utils for the GoalScope sub-goal-coverage benchmark (composing-functions +
task-superposition). Modeled directly on `scripts/fv/goalscope_fv_io.py` (resumable, crash-safe
JSONL, deterministic leaf naming) but keyed by `source` ("composition"|"superposition") + `task`
instead of just `task`, and `samples.jsonl` carries a `subgoals` list (the sub-goals checked for
presence in each prompt's verbalizations) plus source-specific faithfulness metadata:
  - superposition: `output_choices`, `predicted_idx` (argmax of `score_choices` over the query)
  - composition: `true_GFx`, `generated_continuation`, `answer_correct`

Layout of a run directory::

    <results>/evaluation/goalscope_subgoal/<model>/<source>/<task>/<leaf>/
        config.json      resolved config (heads, strategy/alpha/steer_layer_range, gen_tokens,
                         n_prompts, probe, model, seed) + timestamp
        samples.jsonl    one row per prompt: prompt_seed, prompt, subgoals, + source-specific fields
        records.jsonl    one row per scored token position: prompt_seed, pos_kind, gen_offset,
                         abs_pos, token_text, verbalization
        judged.jsonl     records.jsonl rows + per-subgoal presence verdicts
        judge_summary.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

SUBGOAL_SUBDIR = os.path.join("evaluation", "goalscope_subgoal")


def model_slug(config: dict) -> str:
    return str(config.get("model", "unknown-model")).split("/")[-1]


def run_leaf_name(config: dict) -> str:
    lr = config["steer_layer_range"]
    parts = [
        f"heads={config['heads_name']}",
        f"strategy={config['strategy']}",
        f"alpha={float(config['alpha']):g}",
        f"layers={lr[0]}-{lr[1]}",
        f"gentok={config['gen_tokens']}",
        f"nprompts={config['n_prompts']}",
        f"probe={config['probe']}",
        f"seed={config['seed']}",
    ]
    return "__".join(parts)


def run_dir_for(config: dict, results_root) -> Path:
    return (Path(results_root) / SUBGOAL_SUBDIR / model_slug(config)
            / str(config["source"]) / str(config["task"]) / run_leaf_name(config))


def records_path(run_dir) -> Path:
    return Path(run_dir) / "records.jsonl"


def samples_path(run_dir) -> Path:
    return Path(run_dir) / "samples.jsonl"


def judged_path(run_dir) -> Path:
    return Path(run_dir) / "judged.jsonl"


def write_config(run_dir, config: dict) -> None:
    (Path(run_dir) / "config.json").write_text(json.dumps(config, indent=2, default=str))


def write_summary(run_dir, summary: dict, name: str = "judge_summary.json") -> None:
    (Path(run_dir) / name).write_text(json.dumps(summary, indent=2, default=str))


def append_jsonl(fh, obj: dict) -> None:
    fh.write(json.dumps(obj, default=str) + "\n")
    fh.flush()
    os.fsync(fh.fileno())


def read_config(run_dir) -> dict | None:
    p = Path(run_dir) / "config.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def read_completed_prompt_seeds(run_dir) -> list[int]:
    p = samples_path(run_dir)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line)["prompt_seed"])
        except (json.JSONDecodeError, KeyError):
            continue
    return out


def prune_orphan_records(run_dir, keep_prompt_seeds) -> int:
    p = records_path(run_dir)
    if not p.exists():
        return 0
    keep = set(keep_prompt_seeds)
    kept = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            if json.loads(line)["prompt_seed"] in keep:
                kept.append(line)
        except (json.JSONDecodeError, KeyError):
            continue
    p.write_text("".join(l + "\n" for l in kept))
    return len(kept)


def read_judged_keys(run_dir) -> set[tuple]:
    """(prompt_seed, pos_kind, gen_offset) keys already present in judged.jsonl."""
    p = judged_path(run_dir)
    if not p.exists():
        return set()
    out = set()
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            out.add((r["prompt_seed"], r["pos_kind"], r["gen_offset"]))
        except (json.JSONDecodeError, KeyError):
            continue
    return out


def load_config(run_dir) -> dict:
    return json.loads((Path(run_dir) / "config.json").read_text())


def load_summary(run_dir, name: str = "judge_summary.json") -> dict:
    return json.loads((Path(run_dir) / name).read_text())


def load_samples(run_dir) -> list[dict]:
    p = samples_path(run_dir)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def records_df(run_dir) -> pd.DataFrame:
    lines = records_path(run_dir).read_text().splitlines()
    return pd.DataFrame([json.loads(l) for l in lines if l.strip()])


def judged_df(run_dir) -> pd.DataFrame:
    lines = judged_path(run_dir).read_text().splitlines()
    return pd.DataFrame([json.loads(l) for l in lines if l.strip()])


def list_run_dirs(results_root, model: str | None = None, source: str | None = None) -> list[Path]:
    root = Path(results_root) / SUBGOAL_SUBDIR
    if not root.exists():
        return []
    dirs = sorted({p.parent for p in root.rglob("config.json")})
    if model:
        want = model.split("/")[-1]
        dirs = [d for d in dirs if want in d.parts]
    if source:
        dirs = [d for d in dirs if source in d.parts]
    return dirs


def is_generation_complete(run_dir) -> bool:
    cfg = read_config(run_dir)
    if cfg is None:
        return False
    return len(read_completed_prompt_seeds(run_dir)) >= cfg["n_prompts"]
