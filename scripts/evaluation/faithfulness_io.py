"""IO + parsing utils for GoalScope goal-head *faithfulness* eval runs.

Pure ``json`` / ``pandas`` — **no torch, no model import** — so it is cheap to import in a notebook
just to read results produced by ``eval_goalscope.py`` (either ``--mode``).

Also serves ``eval_act_oracles.py`` (activation-oracle read-outs): it writes the *same* record /
summary schema into a sibling tree, so every loader here works on it — see
``act_oracle_run_dir_for`` / ``list_act_oracle_runs`` / ``act_oracle_grid_df``.

Layout of a run directory::

    <results>/evaluation/goalscope/<model>/<mode>/<strategy>/<alpha>/<leaf>/   (eval_goalscope)
    <results>/evaluation/act_oracle/<model>/<mode>/<leaf>/                     (eval_act_oracles)

Both trees group by model first, so one model's runs live under one directory.
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
    # Only an EXPLICIT --goal_band lands in the path: the per-model default is a property of the
    # model (already a path component), so omitting it keeps existing run dirs stable, while a
    # deliberate band sweep still gets its own dir instead of overwriting the default run.
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


def goalscope_root(results_root) -> Path:
    """Root of the goalscope eval tree — the parent of the per-model dirs."""
    return Path(results_root) / GOALSCOPE_SUBDIR


def faithfulness_root(results_root, mode: str = "faithfulness") -> Path:
    """Back-compat: the tree root, so ``run_dir.relative_to(faithfulness_root(RES))`` still works
    (it now yields ``<model>/<mode>/...`` instead of ``<strategy>/...``) and ``root / run_name``
    still round-trips. ``mode`` is accepted but ignored — it is no longer a top-level directory."""
    return goalscope_root(results_root)


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
    # <results>/evaluation/act_oracle/<target-model>/<mode>/<leaf> — model first, matching
    # run_dir_for, so both trees group by model.
    return (Path(results_root) / ACT_ORACLE_SUBDIR / model_slug(config) / config_mode(config)
            / act_oracle_leaf_name(config))


def act_oracle_root(results_root, mode: str = "faithfulness") -> Path:
    """Root of the act-oracle tree — the parent of the per-model dirs. ``mode`` is accepted but
    ignored (it is no longer a top-level directory); kept so callers need not change."""
    return Path(results_root) / ACT_ORACLE_SUBDIR


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


def _select_runs(root, mode: str, model: str | None) -> list[Path]:
    """Run dirs under `root` whose own config.json matches `mode` (and `model`, if given).

    Filtering on the config rather than on path structure means this finds runs written under the
    current ``<model>/<mode>/...`` layout *and* any left at a legacy path.
    """
    want = model.split("/")[-1] if model else None
    out = []
    for d in _list_runs_under(root):
        try:
            cfg = load_config(d)
        except (json.JSONDecodeError, OSError):
            continue
        if config_mode(cfg) != mode:
            continue
        if want and model_slug(cfg) != want:
            continue
        out.append(d)
    return sorted(out)


def list_runs(results_root, mode: str = "faithfulness", model: str | None = None) -> list[Path]:
    """Every goalscope run dir for `mode` (and optionally one `model`), sorted by path.
    `model` may be a full id or a slug (``Qwen/Qwen2.5-32B-Instruct`` or ``Qwen2.5-32B-Instruct``)."""
    return _select_runs(goalscope_root(results_root), mode, model)


def list_act_oracle_runs(results_root, mode: str = "faithfulness",
                         model: str | None = None) -> list[Path]:
    """Every activation-oracle run dir for `mode` (and optionally one `model`), sorted by path."""
    return _select_runs(act_oracle_root(results_root), mode, model)


# ── tabular views ──────────────────────────────────────────────────────────────
_CFG_COLS = ["model", "strategy", "alpha", "heads_name", "n_heads", "steer_layer_range",
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


def act_oracle_grid_df(results_root, mode: str = "faithfulness",
                       model: str | None = None) -> pd.DataFrame:
    """Every activation-oracle run's per-anchor summary — for sweeping layer / question / coeff."""
    frames = [summary_df(d) for d in list_act_oracle_runs(results_root, mode, model)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def recompute_from_records(run_dir) -> pd.DataFrame:
    """Per-anchor hard_yes / kway_ok recomputed from records.jsonl — e.g. after you re-label
    verdicts by hand while polishing the oracle prompts."""
    df = records_df(run_dir)
    df["hard_yes"] = (df["hard"] == "YES").astype(float)
    df["kway_ok"] = (df["kway_pred"] == df["kway_true"]).astype(float)
    return (df.groupby("anchor")
            .agg(n=("hard_yes", "size"), hard_yes=("hard_yes", "mean"), kway_ok=("kway_ok", "mean")))
