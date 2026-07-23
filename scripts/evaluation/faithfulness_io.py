"""IO + parsing utils for GoalScope goal-head *faithfulness* eval runs.

Pure ``json`` / ``pandas`` — **no torch, no model import** — so it is cheap to import in a notebook
just to read results produced by ``run_goalscope_faithfulness.py``.

Layout of a run directory::

    <results>/evaluation/goalscope/faithfulness/<strategy>/<alpha>/<leaf>/
        config.json     resolved config + the selected (layer, head) list + timestamp
        records.jsonl    one row per (sample, condition, anchor): read-out, impl_span, condition,
                         the FULL hard/kway oracle prompt + raw response + parsed verdict
        summary.json     per-anchor + overall hard_yes / kway_ok
        samples.json     per sample: prompt, completion, oracle step labels, anchor token positions

Typical use::

    from scripts.evaluation import faithfulness_io as fio
    runs = fio.list_runs(RESULTS_DIR)
    run  = fio.load_run(runs[-1])          # dict(config, records, summary, samples, dir)
    fio.summary_df(runs[-1])               # per-anchor table for one run
    fio.grid_df(RESULTS_DIR)               # every run's per-anchor summary, for sweeping designs
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

FAITH_SUBDIR = os.path.join("evaluation", "goalscope", "faithfulness")


# ── path construction (shared by the runner and the loaders) ──────────────────
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
    return "__".join(parts)


def run_dir_for(config: dict, results_root) -> Path:
    return (Path(results_root) / FAITH_SUBDIR / str(config["strategy"])
            / fmt_alpha(config["alpha"]) / run_leaf_name(config))


def faithfulness_root(results_root) -> Path:
    return Path(results_root) / FAITH_SUBDIR


# ── saving ────────────────────────────────────────────────────────────────────
def save_run(run_dir, config: dict, records: list[dict], summary: dict, samples: list[dict]) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    with open(run_dir / "records.jsonl", "w") as fh:
        for r in records:
            fh.write(json.dumps(r, default=str) + "\n")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (run_dir / "samples.json").write_text(json.dumps(samples, indent=2, default=str))
    return run_dir


# ── loading ─────────────────────────────────────────────────────────────────--
def load_config(run_dir) -> dict:
    return json.loads((Path(run_dir) / "config.json").read_text())


def load_summary(run_dir) -> dict:
    return json.loads((Path(run_dir) / "summary.json").read_text())


def load_samples(run_dir) -> list[dict]:
    return json.loads((Path(run_dir) / "samples.json").read_text())


def records_df(run_dir) -> pd.DataFrame:
    """Per-(sample, condition, anchor) records, including full oracle prompts + responses."""
    lines = (Path(run_dir) / "records.jsonl").read_text().splitlines()
    return pd.DataFrame([json.loads(l) for l in lines if l.strip()])


def load_run(run_dir) -> dict:
    run_dir = Path(run_dir)
    return dict(config=load_config(run_dir), records=records_df(run_dir),
                summary=load_summary(run_dir), samples=load_samples(run_dir), dir=str(run_dir))


def list_runs(results_root) -> list[Path]:
    """Every run dir (has config.json) under the faithfulness root, sorted by path."""
    root = faithfulness_root(results_root)
    if not root.exists():
        return []
    return sorted({p.parent for p in root.rglob("config.json")})


# ── tabular views ──────────────────────────────────────────────────────────────
_CFG_COLS = ["strategy", "alpha", "heads_name", "n_heads", "steer_layer_range",
             "n_steer_tokens", "max_new_tokens", "seed"]


def summary_df(run_dir) -> pd.DataFrame:
    """One run's per-anchor summary, with the run's config columns attached."""
    cfg, s = load_config(run_dir), load_summary(run_dir)
    cfg_vals = {c: cfg.get(c) for c in _CFG_COLS}
    rows = [dict(anchor=anchor, **cfg_vals, **metrics) for anchor, metrics in s["by_anchor"].items()]
    rows.append(dict(anchor="OVERALL", **cfg_vals, **s["overall"], n=s["n_records"]))
    return pd.DataFrame(rows)


def grid_df(results_root) -> pd.DataFrame:
    """Every run's per-anchor summary concatenated — for sweeping design choices."""
    frames = [summary_df(d) for d in list_runs(results_root)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def recompute_from_records(run_dir) -> pd.DataFrame:
    """Per-anchor hard_yes / kway_ok recomputed from records.jsonl — e.g. after you re-label
    verdicts by hand while polishing the oracle prompts."""
    df = records_df(run_dir)
    df["hard_yes"] = (df["hard"] == "YES").astype(float)
    df["kway_ok"] = (df["kway_pred"] == df["kway_true"]).astype(float)
    return (df.groupby("anchor")
            .agg(n=("hard_yes", "size"), hard_yes=("hard_yes", "mean"), kway_ok=("kway_ok", "mean")))
