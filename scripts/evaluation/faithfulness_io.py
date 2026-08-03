"""IO + parsing utils for GoalScope goal-head *faithfulness* eval runs.

Pure ``json`` / ``pandas`` — **no torch, no model import** — so it is cheap to import in a notebook
just to read results produced by ``eval_goalscope.py`` (either ``--mode``).

Also serves ``eval_act_oracles.py`` (activation-oracle read-outs): it writes the *same* record /
summary schema into a sibling tree, so every loader here works on it — see
``act_oracle_run_dir_for`` / ``list_act_oracle_runs`` / ``act_oracle_grid_df``.

Layout of a run directory::

    <results>/evaluation/goalscope/<mode>/<strategy>/<alpha>/<leaf>/        (eval_goalscope)
    <results>/evaluation/act_oracle/<mode>/<target-model>/<leaf>/          (eval_act_oracles)
        config.json     resolved config + the selected (layer, head) list + timestamp
        records.jsonl    one row per (sample, condition, anchor): read-out, impl_span, condition,
                         the FULL hard/kway oracle prompt + raw response + parsed verdict
        summary.json     per-anchor + overall hard_yes / kway_ok
        samples.jsonl    per sample (one line each): prompt, completion, oracle step labels,
                         anchor token positions

Both ``records.jsonl`` and ``samples.jsonl`` are written incrementally (one flush per finished
sample) by the runner, so a run can be **resumed** from the last completed sample and inspected
while still in progress.

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

GOALSCOPE_SUBDIR = os.path.join("evaluation", "goalscope")
# per-mode root under the goalscope tree; mode "faithfulness" keeps the original path.
FAITH_SUBDIR = os.path.join(GOALSCOPE_SUBDIR, "faithfulness")  # back-compat default
# activation-oracle runs (``eval_act_oracles.py``) live in a sibling tree; same record schema,
# so every loader below works on them unchanged — only the path/leaf layout differs.
ACT_ORACLE_SUBDIR = os.path.join("evaluation", "act_oracle")


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
    # <results>/evaluation/goalscope/<mode>/<strategy>/<alpha>/<leaf>; mode defaults to
    # "faithfulness" so pre-mode runs keep their exact path.
    mode = config.get("mode", "faithfulness")
    return (Path(results_root) / GOALSCOPE_SUBDIR / mode / str(config["strategy"])
            / fmt_alpha(config["alpha"]) / run_leaf_name(config))


def faithfulness_root(results_root, mode: str = "faithfulness") -> Path:
    return Path(results_root) / GOALSCOPE_SUBDIR / mode


# ── activation-oracle paths (``eval_act_oracles.py``) ─────────────────────────
def act_oracle_leaf_name(config: dict) -> str:
    """Deterministic leaf directory name for an activation-oracle run config."""
    parts = [
        f"L={config['extract_layer']}",
        f"style={config.get('prompt_style', 'paper')}",
        f"q={config['question_tag']}",
        f"coeff={float(config['steering_coeff']):g}",
        f"maxnew={config['oracle_max_new_tokens']}",
        f"span={config.get('span_source', 'rule')}",
        f"dtype={config.get('datatype', 'int')}",
        f"nspec={config['n_specs']}",
        f"nc={'-'.join(map(str, config['num_conditions']))}",
        f"seed={config['seed']}",
    ]
    return "__".join(parts)


def act_oracle_run_dir_for(config: dict, results_root) -> Path:
    # <results>/evaluation/act_oracle/<mode>/<target-model>/<leaf>
    return (Path(results_root) / ACT_ORACLE_SUBDIR / config.get("mode", "faithfulness")
            / str(config["model"]).split("/")[-1] / act_oracle_leaf_name(config))


def act_oracle_root(results_root, mode: str = "faithfulness") -> Path:
    return Path(results_root) / ACT_ORACLE_SUBDIR / mode


# ── paths ─────────────────────────────────────────────────────────────────────
def records_path(run_dir) -> Path:
    return Path(run_dir) / "records.jsonl"


def samples_path(run_dir) -> Path:
    return Path(run_dir) / "samples.jsonl"


# ── saving (batch) ──────────────────────────────────────────────────────────────
def save_run(run_dir, config: dict, records: list[dict], summary: dict, samples: list[dict]) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_config(run_dir, config)
    with open(records_path(run_dir), "w") as fh:
        for r in records:
            fh.write(json.dumps(r, default=str) + "\n")
    write_summary(run_dir, summary)
    with open(samples_path(run_dir), "w") as fh:
        for s in samples:
            fh.write(json.dumps(s, default=str) + "\n")
    return run_dir


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


# ── loading ─────────────────────────────────────────────────────────────────--
def load_config(run_dir) -> dict:
    return json.loads((Path(run_dir) / "config.json").read_text())


def load_summary(run_dir) -> dict:
    return json.loads((Path(run_dir) / "summary.json").read_text())


def load_samples(run_dir) -> list[dict]:
    p = samples_path(run_dir)
    if p.exists():  # current incremental format (one sample per line)
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return json.loads((Path(run_dir) / "samples.json").read_text())  # legacy batch format


def records_df(run_dir) -> pd.DataFrame:
    """Per-(sample, condition, anchor) records, including full oracle prompts + responses."""
    lines = (Path(run_dir) / "records.jsonl").read_text().splitlines()
    return pd.DataFrame([json.loads(l) for l in lines if l.strip()])


def load_run(run_dir) -> dict:
    run_dir = Path(run_dir)
    return dict(config=load_config(run_dir), records=records_df(run_dir),
                summary=load_summary(run_dir), samples=load_samples(run_dir), dir=str(run_dir))


def _list_runs_under(root) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted({p.parent for p in root.rglob("config.json")})


def list_runs(results_root, mode: str = "faithfulness") -> list[Path]:
    """Every run dir (has config.json) under the goalscope <mode> root, sorted by path."""
    return _list_runs_under(faithfulness_root(results_root, mode))


def list_act_oracle_runs(results_root, mode: str = "faithfulness") -> list[Path]:
    """Every activation-oracle run dir under the act_oracle <mode> root, sorted by path."""
    return _list_runs_under(act_oracle_root(results_root, mode))


# ── tabular views ──────────────────────────────────────────────────────────────
_CFG_COLS = ["strategy", "alpha", "heads_name", "n_heads", "steer_layer_range",
             "n_steer_tokens", "max_new_tokens", "seed"]
# activation-oracle runs have no heads/steering; these are the knobs that vary instead.
_AO_CFG_COLS = ["model", "extract_layer", "layer_pct", "prompt_style", "question_tag",
                "steering_coeff", "oracle_max_new_tokens", "oracle_lora", "seed"]


def summary_df(run_dir) -> pd.DataFrame:
    """One run's per-anchor summary, with the run's config columns attached.
    Works for both goalscope and activation-oracle runs (config columns differ)."""
    cfg, s = load_config(run_dir), load_summary(run_dir)
    cols = _AO_CFG_COLS if cfg.get("eval_kind") == "act_oracle" else _CFG_COLS
    cfg_vals = {c: cfg.get(c) for c in cols}
    rows = [dict(anchor=anchor, **cfg_vals, **metrics) for anchor, metrics in s["by_anchor"].items()]
    rows.append(dict(anchor="OVERALL", **cfg_vals, **s["overall"], n=s["n_records"]))
    return pd.DataFrame(rows)


def grid_df(results_root) -> pd.DataFrame:
    """Every run's per-anchor summary concatenated — for sweeping design choices."""
    frames = [summary_df(d) for d in list_runs(results_root)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def act_oracle_grid_df(results_root, mode: str = "faithfulness") -> pd.DataFrame:
    """Every activation-oracle run's per-anchor summary — for sweeping layer / question / coeff."""
    frames = [summary_df(d) for d in list_act_oracle_runs(results_root, mode)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def recompute_from_records(run_dir) -> pd.DataFrame:
    """Per-anchor hard_yes / kway_ok recomputed from records.jsonl — e.g. after you re-label
    verdicts by hand while polishing the oracle prompts."""
    df = records_df(run_dir)
    df["hard_yes"] = (df["hard"] == "YES").astype(float)
    df["kway_ok"] = (df["kway_pred"] == df["kway_true"]).astype(float)
    return (df.groupby("anchor")
            .agg(n=("hard_yes", "size"), hard_yes=("hard_yes", "mean"), kway_ok=("kway_ok", "mean")))
