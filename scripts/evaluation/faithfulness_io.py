"""IO utils for GoalScope eval runs (``eval_goalscope.py`` / ``eval_readout_sources.py``).

Pure ``json`` — **no torch, no model import**.

Layout of a run directory::

    <results>/evaluation/goalscope/<model>/<mode>/<strategy>/<alpha>/<leaf>/
        config.json     resolved config + the selected (layer, head) list + timestamp
        records.jsonl    one row per (sample, anchor): read-out, impl_span, condition,
                         the FULL hard oracle prompt + raw response + parsed verdict
        summary.json     per-anchor + overall hard_yes
        samples.jsonl    per sample (one line each): prompt, completion, anchor token positions

Both ``records.jsonl`` and ``samples.jsonl`` are written incrementally (one flush per finished
sample) by the runner, so a run can be **resumed** from the last completed sample and inspected
while still in progress. The figure generators read ``summary.json`` directly.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

GOALSCOPE_SUBDIR = os.path.join("evaluation", "goalscope")


# ── path construction ─────────────────────────────────────────────────────────
def fmt_alpha(a) -> str:
    return f"{float(a):g}"


def run_leaf_name(config: dict) -> str:
    """Deterministic ``arg=val__arg=val`` leaf directory name for a config."""
    lr = config["steer_layer_range"]
    parts = [
        f"heads={config['heads_name']}",
        f"layers={lr[0]}-{lr[1]}",
        f"nsteer={config['n_steer_tokens']}",
        f"maxnew={config['max_new_tokens']}",
        f"span={config.get('span_source', 'rule')}",
        f"dtype={config.get('datatype', 'int')}",
        f"nspec={config['n_specs']}",
        f"nc={'-'.join(map(str, config['num_conditions']))}",
        f"seed={config['seed']}",
    ]
    # Only an EXPLICIT --goal_band lands in the path: the per-model default is a property of the
    # model (already a path component), so omitting it keeps existing run dirs stable, while a
    # deliberate band choice still gets its own dir instead of overwriting the default run.
    if config.get("goal_band_explicit") and config.get("goal_band"):
        b = config["goal_band"]
        parts.append(f"band={b[0]}-{b[1]}")
    return "__".join(parts)


def model_slug(config: dict) -> str:
    """Short model name used as a path component (``Qwen/X-32B-Instruct`` -> ``X-32B-Instruct``)."""
    return str(config.get("model", "unknown-model")).split("/")[-1]


def config_mode(config: dict) -> str:
    """``config['mode']`` with a default — tolerates pre-``--mode`` runs that stored ``null``."""
    return config.get("mode") or "faithfulness"


def run_dir_for(config: dict, results_root) -> Path:
    # <results>/evaluation/goalscope/<model>/<mode>/<strategy>/<alpha>/<leaf>
    # The model comes FIRST so every run of one model sits under one dir. Without it, two models
    # sharing a config resolved to the same run dir and the second run would "resume" the first
    # model's samples.
    return (Path(results_root) / GOALSCOPE_SUBDIR / model_slug(config) / config_mode(config)
            / str(config["strategy"]) / fmt_alpha(config["alpha"]) / run_leaf_name(config))


# ── paths ─────────────────────────────────────────────────────────────────────
def records_path(run_dir) -> Path:
    return Path(run_dir) / "records.jsonl"


def samples_path(run_dir) -> Path:
    return Path(run_dir) / "samples.jsonl"


# ── saving (incremental / resumable) ───────────────────────────────────────────
def write_config(run_dir, config: dict) -> None:
    (Path(run_dir) / "config.json").write_text(json.dumps(config, indent=2, default=str))


def write_summary(run_dir, summary: dict) -> None:
    (Path(run_dir) / "summary.json").write_text(json.dumps(summary, indent=2, default=str))


def append_jsonl(fh, obj: dict) -> None:
    """Append one object as a line and flush to disk (crash-safe incremental save)."""
    fh.write(json.dumps(obj, default=str) + "\n")
    fh.flush()
    os.fsync(fh.fileno())


def read_config(run_dir) -> dict | None:
    """The config of a previous run in this dir, or None if there isn't one (or it is unreadable).

    Used to verify that a resume is resuming the *same* experiment: `run_leaf_name` is built from a
    whitelist of keys, so any config field outside that list (notably the resolved head set) can
    differ between two runs that share a run dir."""
    p = Path(run_dir) / "config.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def read_completed_spec_ids(run_dir) -> list[str]:
    """spec_ids of samples already fully written to samples.jsonl (i.e. safe to skip on resume)."""
    p = samples_path(run_dir)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line)["spec_id"])
        except (json.JSONDecodeError, KeyError):
            continue  # tolerate a torn final line from a crash
    return out


def prune_orphan_records(run_dir, keep_spec_ids) -> int:
    """Drop records whose sample never finished (spec not in `keep_spec_ids`) so a resumed run
    doesn't leave duplicate/half rows. Returns the number of records kept."""
    p = records_path(run_dir)
    if not p.exists():
        return 0
    keep = set(keep_spec_ids)
    kept = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            if json.loads(line)["spec"] in keep:
                kept.append(line)
        except (json.JSONDecodeError, KeyError):
            continue
    p.write_text("".join(l + "\n" for l in kept))
    return len(kept)
