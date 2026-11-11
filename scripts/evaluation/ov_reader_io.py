"""IO + parsing utils for OV-reader eval runs (``scripts/evaluation/eval_ov_reader.py``).

Pure ``json`` / ``pandas`` — **no torch, no model import** — so it is cheap to import in a notebook
to read/plot results while a run is still in progress.

Layout of the tree::

    <results>/evaluation/ov-reader/<model>/<layer_idx>/<mode>/<leaf>/

``<layer_idx>`` is either the integer probe layer ``L`` (an OVReader run) or the literal string
``"goalscope"`` (the GoalScope skyline). Every such dir is a standalone run dir::

    config.json     resolved config (+ the selected heads, + this dir's layer_idx) + timestamp
    records.jsonl   one row per (datapoint, pivot): readout, impl_span, frontier, hard verdict
    samples.jsonl   one row per datapoint: prompt, completion, impl_span, pivot positions
    summary.json    per-pivot + overall hard_yes / heavy_loop

All dirs of one run (goalscope + every probe layer) share the SAME ``<mode>/<leaf>`` and differ only
in ``<layer_idx>``, so cross-layer comparison = reading each layer dir's summary (see
``across_layers_df``). ``<datatype>`` is baked into the leaf, so parallel invocations over disjoint
``--datatypes`` never collide.

Record/sample IO (append, resume, prune) is delegated to ``faithfulness_io`` — same primitives.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.evaluation import faithfulness_io as fio

OV_SUBDIR = "ov_reader"   # results/ov_reader/<model>/<layer>/<mode>/<leaf>
SKYLINE_IDX = "goalscope"  # layer_idx for the GoalScope skyline dir
PIVOT_ORDER = ["prev_end", "indent", "var", "eq", "first_impl"]

# re-export the incremental/resume primitives so the runner imports them from one place
append_jsonl = fio.append_jsonl
records_path = fio.records_path
samples_path = fio.samples_path
write_config = fio.write_config
write_summary = fio.write_summary
read_completed_spec_ids = fio.read_completed_spec_ids
prune_orphan_records = fio.prune_orphan_records


# ── path construction (shared by the runner and the loaders) ──────────────────
def fmt_alpha(a) -> str:
    return f"{float(a):g}"


def model_slug(config: dict) -> str:
    return str(config.get("model", "unknown-model")).split("/")[-1]


def leaf_name(config: dict) -> str:
    """Deterministic ``arg=val__…`` leaf. Holds every data/steer axis EXCEPT layer_idx and mode
    (which are path components) and n_samples (a resume just extends the same dir)."""
    parts = [
        f"heads={config['heads_name']}",
        f"nheads={config['n_heads']}",
        f"alpha={fmt_alpha(config['alpha'])}",
        f"skyalpha={fmt_alpha(config['skyline_alpha'])}",
        f"dtype={config['datatype']}",
        f"lang={config['lang']}",
        f"nc={'-'.join(map(str, config['n_conditions']))}",
        f"maxnew={config['max_new_tokens']}",
        f"gen={config['gen_max_new_tokens']}",
        f"span={config.get('span_source', 'rule')}",
        f"seed={config['seed']}",
    ]
    if config.get("mode") == "causality" and config.get("causal_stub") == "v2":
        parts.append("stub=func")   # v2 causality probe; v1 leaves keep the old names
    return "__".join(parts)


def run_dir_for(config: dict, layer_idx, results_root) -> Path:
    """<results>/ov_reader/<model>/<layer_idx>/<mode>/<leaf>."""
    return (Path(results_root) / OV_SUBDIR / model_slug(config) / str(layer_idx)
            / str(config["mode"]) / leaf_name(config))


def ov_root(results_root) -> Path:
    return Path(results_root) / OV_SUBDIR


# ── summary (grouped by pivot; mirrors eval_goalscope.compute_summary by anchor) ──
def compute_summary(records: list[dict], n_samples: int) -> dict:
    """Per-pivot + overall hard_yes / heavy_loop."""

    def _stats(grp):
        return dict(
            n=len(grp),
            hard_yes=float(np.mean([r["hard"] == "YES" for r in grp])),
            heavy_loop=float(np.mean([r["hard"] == "HEAVY_LOOP" for r in grp])),
        )

    by_pivot = {
        p: _stats(grp)
        for p in PIVOT_ORDER
        if (grp := [r for r in records if r["pivot"] == p])
    }
    if records:
        overall = dict(
            hard_yes=float(np.mean([r["hard"] == "YES" for r in records])),
            heavy_loop=float(np.mean([r["hard"] == "HEAVY_LOOP" for r in records])),
        )
    else:
        overall = dict(hard_yes=float("nan"), heavy_loop=float("nan"))
    return dict(n_records=len(records), n_samples=n_samples, by_pivot=by_pivot, overall=overall)


# ── loading ───────────────────────────────────────────────────────────────────
def load_config(run_dir) -> dict:
    return json.loads((Path(run_dir) / "config.json").read_text())


def load_summary(run_dir) -> dict:
    return json.loads((Path(run_dir) / "summary.json").read_text())


def _layer_dirs_for(config: dict, results_root) -> dict:
    """{layer_idx: dir} for every existing sibling dir sharing this config's <mode>/<leaf>.
    layer_idx is ``"goalscope"`` or an int; the returned dict is ordered goalscope-first then by layer."""
    model_dir = ov_root(results_root) / model_slug(config)
    leaf, mode = leaf_name(config), str(config["mode"])
    found = {}
    if model_dir.exists():
        for layer_dir in model_dir.iterdir():
            d = layer_dir / mode / leaf
            if (d / "summary.json").exists():
                idx = layer_dir.name
                found[idx if idx == SKYLINE_IDX else int(idx)] = d
    order = [SKYLINE_IDX] + sorted(k for k in found if k != SKYLINE_IDX)
    return {k: found[k] for k in order if k in found}


def across_layers_df(config: dict, results_root) -> pd.DataFrame:
    """Tidy per-(layer_idx, pivot) hard_yes/heavy_loop for one run, goalscope included — the table
    for 'does the goal show up earlier than the goal-head layers, and at which pivot'."""
    rows = []
    for idx, d in _layer_dirs_for(config, results_root).items():
        s = load_summary(d)
        for pivot, m in s["by_pivot"].items():
            rows.append(dict(layer_idx=idx, pivot=pivot, **m))
        rows.append(dict(layer_idx=idx, pivot="OVERALL", **s["overall"], n=s["n_records"]))
    return pd.DataFrame(rows)


def _list_run_dirs(results_root) -> list[Path]:
    root = ov_root(results_root)
    return sorted({p.parent for p in root.rglob("config.json")}) if root.exists() else []


def list_runs(results_root, mode: str | None = None, model: str | None = None) -> list[Path]:
    """Every OV-reader run dir (one per layer_idx), optionally filtered by mode / model."""
    want = model.split("/")[-1] if model else None
    out = []
    for d in _list_run_dirs(results_root):
        try:
            cfg = load_config(d)
        except (json.JSONDecodeError, OSError):
            continue
        if mode is not None and str(cfg.get("mode")) != mode:
            continue
        if want and model_slug(cfg) != want:
            continue
        out.append(d)
    return sorted(out)


def leaf_configs(results_root, mode: str | None = None, model: str | None = None) -> list[dict]:
    """One representative config per distinct <model>/<mode>/<leaf> run (deduped across layer dirs) —
    feed each to ``across_layers_df`` to get its full layer sweep."""
    seen, out = set(), []
    for d in list_runs(results_root, mode=mode, model=model):
        cfg = load_config(d)
        key = (model_slug(cfg), str(cfg.get("mode")), leaf_name(cfg))
        if key in seen:
            continue
        seen.add(key)
        out.append(cfg)
    return out


def grid_df(results_root, mode: str | None = None, model: str | None = None) -> pd.DataFrame:
    """Every run's full layer sweep concatenated, with the identifying config columns attached."""
    cols = ["model", "mode", "datatype", "lang", "heads_name", "n_heads", "alpha",
            "skyline_alpha", "n_conditions", "seed"]
    frames = []
    for cfg in leaf_configs(results_root, mode=mode, model=model):
        df = across_layers_df(cfg, results_root)
        if df.empty:
            continue
        for c in cols:
            v = cfg.get(c)
            df[c] = "-".join(map(str, v)) if isinstance(v, list) else v
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
