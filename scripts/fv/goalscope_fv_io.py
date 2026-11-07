"""IO utils for the GoalScope x function-vectors verbalization eval.

Modeled directly on ``scripts/evaluation/faithfulness_io.py``'s pattern (resumable, crash-safe
JSONL, deterministic config-hash-free leaf naming, incremental writes) but NOT a reuse of that
module: its leaf naming and config keys are hard-coded to the code/condition-predicate GoalScope
eval (``strategy``, ``steer_layer_range``, ``num_conditions``, ``datatype``, ...), which doesn't
match this eval's shape (``task``, ``n_shots``, ``window``, ``probe``).

Layout of a run directory::

    <results>/evaluation/goalscope_fv/<model>/<task>/<leaf>/
        config.json      resolved config (heads, strategy/alpha/steer_layer_range, n_shots,
                         n_prompts, window, probe, model, seed) + timestamp
        records.jsonl    one row per scored token position: task, prompt_seed, demo_idx, side,
                         tok_offset, row_len, abs_pos, token_text, verbalization
        samples.jsonl    one row per prompt: prompt_seed, prompt text, demos, query,
                         n_positions_scored (written by goalscope_fv_generate.py)
        judged.jsonl     records.jsonl rows + judge verdicts (written by goalscope_fv_judge.py)
        summary.json     accuracy / k-way-accuracy, overall + broken down by task/side/tok_offset
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

GOALSCOPE_FV_SUBDIR = os.path.join("evaluation", "goalscope_fv")


# ── path construction ──────────────────────────────────────────────────────────
def model_slug(config: dict) -> str:
    return str(config.get("model", "unknown-model")).split("/")[-1]


def run_leaf_name(config: dict) -> str:
    """Deterministic ``arg=val__arg=val`` leaf directory name for a config."""
    lr = config["steer_layer_range"]
    parts = [
        f"heads={config['heads_name']}",
        f"strategy={config['strategy']}",
        f"alpha={float(config['alpha']):g}",
        f"layers={lr[0]}-{lr[1]}",
        f"nshots={config['n_shots']}",
        f"window={config['window']}",
        f"nprompts={config['n_prompts']}",
        f"probe={config['probe']}",
        f"seed={config['seed']}",
    ]
    return "__".join(parts)


def run_dir_for(config: dict, results_root) -> Path:
    # <results>/evaluation/goalscope_fv/<model>/<task>/<leaf>
    return (Path(results_root) / GOALSCOPE_FV_SUBDIR / model_slug(config)
            / str(config["task"]) / run_leaf_name(config))


# ── paths ───────────────────────────────────────────────────────────────────────
def records_path(run_dir) -> Path:
    return Path(run_dir) / "records.jsonl"


def samples_path(run_dir) -> Path:
    return Path(run_dir) / "samples.jsonl"


def judged_path(run_dir) -> Path:
    return Path(run_dir) / "judged.jsonl"


# ── saving (incremental / resumable) ─────────────────────────────────────────────
def write_config(run_dir, config: dict) -> None:
    (Path(run_dir) / "config.json").write_text(json.dumps(config, indent=2, default=str))


def write_summary(run_dir, summary: dict, name: str = "summary.json") -> None:
    (Path(run_dir) / name).write_text(json.dumps(summary, indent=2, default=str))


def append_jsonl(fh, obj: dict) -> None:
    """Append one object as a line and flush to disk (crash-safe incremental save)."""
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
    """`prompt_seed`s of prompts already fully written to samples.jsonl (safe to skip on resume)."""
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
            continue  # tolerate a torn final line from a crash
    return out


def prune_orphan_records(run_dir, keep_prompt_seeds) -> int:
    """Drop records whose prompt never finished (crash mid-prompt) so a resumed run doesn't leave
    duplicate/half rows. Returns the number of records kept."""
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


def read_judged_ids(run_dir) -> set[tuple]:
    """(prompt_seed, demo_idx, side, tok_offset) keys already present in judged.jsonl."""
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
            out.add((r["prompt_seed"], r["demo_idx"], r["side"], r["tok_offset"]))
        except (json.JSONDecodeError, KeyError):
            continue
    return out


# ── loading ─────────────────────────────────────────────────────────────────────
def load_config(run_dir) -> dict:
    return json.loads((Path(run_dir) / "config.json").read_text())


def load_summary(run_dir, name: str = "summary.json") -> dict:
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


# ── batch discovery (for goalscope_fv_judge.py --all) ────────────────────────────
def list_run_dirs(results_root, model: str | None = None) -> list[Path]:
    """Every goalscope_fv run dir (one per task+config) under `results_root`, optionally
    filtered to one model slug or id."""
    root = Path(results_root) / GOALSCOPE_FV_SUBDIR
    if not root.exists():
        return []
    dirs = sorted({p.parent for p in root.rglob("config.json")})
    if model:
        want = model.split("/")[-1]
        dirs = [d for d in dirs if want in d.parts]
    return dirs


def is_generation_complete(run_dir) -> bool:
    """True iff every prompt `goalscope_fv_generate.py`'s config called for has a finished
    samples.jsonl row — i.e. this run dir is safe to hand to the judge."""
    cfg = read_config(run_dir)
    if cfg is None:
        return False
    return len(read_completed_prompt_seeds(run_dir)) >= cfg["n_prompts"]
