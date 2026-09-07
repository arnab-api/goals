"""Re-grade saved GoalScope runs with the CURRENT oracle rubric — no model needed.

Every read-out is already stored in ``records.jsonl``; the only thing that changes when we
refine the oracle prompt is the *verdict*. This script walks the goalscope result tree, re-runs
ONLY the hard oracle (``hard_eval``) over the stored read-outs, and rewrites the summary — the
attention/steering forward passes are never repeated.

Single source of truth for the prompt
--------------------------------------
It imports ``FAITH_HARD_SYS`` / ``CAUSAL_HARD_SYS`` / ``hard_eval`` / ``compute_summary`` straight
from ``eval_goalscope.py``. So the workflow is:

  1. refine the rubric in ``notebooks/clock/__goal_oracle_refinement.ipynb``
  2. paste the final prompts into ``eval_goalscope.py`` (and extend ``_parse_verdict`` for any new
     verdict class, e.g. HEAVY_LOOP)
  3. run this script  ->  every saved run is re-graded with that exact prompt, and every NEW run
     from ``eval_goalscope.py`` already uses it too.

k-way verdicts are left untouched (that rubric did not change).

Backup / idempotency
--------------------
On the first re-eval of a dir, the pristine files are preserved:
    records.jsonl -> records_oldeval.jsonl
    summary.json  -> summary_oldeval.json
and the new verdicts are written to the canonical ``records.jsonl`` / ``summary.json`` that the
parsers (``faithfulness_io``, notebook 12.2) load. Re-grading ALWAYS reads the pristine
``records_oldeval.jsonl`` when it exists, so re-running is safe and repeatable. Each record keeps
its previous verdict in a ``hard_oldeval`` field for easy A/B. A run whose config
``hard_rubric_tag`` already equals the current prompt's tag is skipped (nothing to do) unless
``--force``.

Usage
-----
    # what would be re-graded, and how many oracle calls it costs (no API hits):
    python -m scripts.evaluation.reeval_goalscope --dry-run

    # re-grade everything (after updating the prompt in eval_goalscope.py):
    python -m scripts.evaluation.reeval_goalscope --workers 16

    # a subset:
    python -m scripts.evaluation.reeval_goalscope --model Qwen2.5-Coder-32B-Instruct --mode causality

    # write the book-keeping manifest of every run in the tree:
    python -m scripts.evaluation.reeval_goalscope --manifest agent_reports/claude/goalscope_reeval_manifest.md --dry-run
"""

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.append(".")

from tqdm.auto import tqdm

from scripts.evaluation import faithfulness_io as fio
from scripts.evaluation.eval_goalscope import (ANCHOR_ORDER, CAUSAL_HARD_SYS,
                                               FAITH_HARD_SYS, compute_summary,
                                               hard_eval)
from src.utils import env_utils


# ── the current rubric's fingerprint, exactly as eval_goalscope stamps it into config ──────────
def current_rubric_tag(mode: str) -> str:
    sys_prompt = CAUSAL_HARD_SYS if mode == "causality" else FAITH_HARD_SYS
    return hashlib.md5(sys_prompt.encode()).hexdigest()[:8]


# ── discovery ──────────────────────────────────────────────────────────────────────────────────
def discover_runs(results_dir) -> list[Path]:
    """Every goalscope run dir that has a records file, sorted for stable ordering."""
    root = fio.goalscope_root(results_dir)
    dirs = {p.parent for p in root.rglob("records.jsonl")}
    dirs |= {p.parent for p in root.rglob("records_oldeval.jsonl")}  # already-backed-up runs
    return sorted(dirs)


def run_meta(run_dir: Path) -> dict:
    """Lightweight descriptor for the manifest / filtering (config + record count)."""
    cfg = fio.read_config(run_dir) or {}
    src = run_dir / "records_oldeval.jsonl"
    if not src.exists():
        src = fio.records_path(run_dir)
    n_records = sum(1 for l in src.read_text().splitlines() if l.strip()) if src.exists() else 0
    mode = fio.config_mode(cfg)
    return dict(
        dir=run_dir,
        model=str(cfg.get("model", "?")).split("/")[-1],
        mode=mode,
        strategy=cfg.get("strategy"),
        alpha=cfg.get("alpha"),
        heads_name=cfg.get("heads_name"),
        n_heads=cfg.get("n_heads"),
        n_specs=cfg.get("n_specs"),
        stored_tag=cfg.get("hard_rubric_tag"),
        current_tag=current_rubric_tag(mode),
        n_records=n_records,
        reevaled=(run_dir / "records_oldeval.jsonl").exists(),
    )


def load_pristine_records(run_dir: Path) -> list[dict]:
    """Read-outs to grade — always from the pristine backup when it exists."""
    src = run_dir / "records_oldeval.jsonl"
    if not src.exists():
        src = fio.records_path(run_dir)
    return [json.loads(l) for l in src.read_text().splitlines() if l.strip()]


# ── per-dir change report: hard_yes v1->v2 + heavy_loop, overall and per pivot token ─────────
def change_report(records: list[dict]) -> str:
    def rate(rs, field, val):
        return (sum(r.get(field) == val for r in rs) / len(rs)) if rs else float("nan")

    groups = [(a, [r for r in records if r["anchor"] == a]) for a in ANCHOR_ORDER]
    groups = [(a, g) for a, g in groups if g] + [("OVERALL", records)]
    pad = "            "
    lines = [f"{pad}{'pivot':<10} {'n':>6}   {'hard_yes v1->v2':<22} {'heavy_loop':>10}"]
    for a, g in groups:
        oy, ny = rate(g, "hard_oldeval", "YES"), rate(g, "hard", "YES")
        hl = rate(g, "hard", "HEAVY_LOOP")
        lines.append(f"{pad}{a:<10} {len(g):>6}   "
                     f"{oy:5.3f} -> {ny:5.3f} ({ny - oy:+.3f})   {hl:10.3f}")
    return "\n".join(lines)


# ── re-grading one run ───────────────────────────────────────────────────────────────────────
def reeval_one(run_dir: Path, workers: int, force: bool) -> str:
    cfg = fio.read_config(run_dir) or {}
    mode = fio.config_mode(cfg)
    tag = current_rubric_tag(mode)

    if not force and cfg.get("hard_rubric_tag") == tag and (run_dir / "records_oldeval.jsonl").exists():
        return dict(status="skip (already graded with current rubric)", calls=0, regraded=False)

    records = load_pristine_records(run_dir)
    if not records:
        return dict(status="skip (no records)", calls=0, regraded=False)

    # ── dedup: identical (read-out, impl_span) -> one oracle call, fanned back out ──
    # (hard_eval short-circuits an empty read-out to MALFORMED with no API hit, so those keys
    # are graded but not counted as calls.)
    keys = {(r.get("readout", ""), r["impl_span"]) for r in records}
    n_calls = sum(1 for ro, _ in keys if ro.strip())
    graded: dict[tuple, dict] = {}

    def _grade(key):
        readout, impl = key
        return key, hard_eval(readout, impl, mode)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for key, res in tqdm(ex.map(_grade, keys), total=len(keys),
                             desc=f"  grading {run_dir.name[:38]}", leave=False,
                             unit="ro"):
            graded[key] = res

    for r in records:
        r["hard_oldeval"] = r.get("hard")
        res = graded[(r.get("readout", ""), r["impl_span"])]
        r["hard"] = res["verdict"]
        r["hard_prompt"] = res["prompt"]
        r["hard_response"] = res["response"]

    # ── preserve pristine files (once), then write the canonical new ones ──
    rec_path, old_rec = fio.records_path(run_dir), run_dir / "records_oldeval.jsonl"
    if not old_rec.exists() and rec_path.exists():
        rec_path.rename(old_rec)
    sum_path, old_sum = run_dir / "summary.json", run_dir / "summary_oldeval.json"
    if not old_sum.exists() and sum_path.exists():
        sum_path.rename(old_sum)

    with open(rec_path, "w") as fh:
        for r in records:
            fh.write(json.dumps(r, default=str) + "\n")

    n_samples = len(fio.read_completed_spec_ids(run_dir)) or len({r["spec"] for r in records})
    summary = compute_summary(records, n_samples=n_samples)
    fio.write_summary(run_dir, summary)

    dt = time.time() - t0
    cfg["hard_rubric_tag_prev"] = cfg.get("hard_rubric_tag")
    cfg["hard_rubric_tag"] = tag
    cfg["reeval"] = dict(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                         n_records=len(records), oracle_calls=n_calls,
                         seconds=round(dt, 1))
    fio.write_config(run_dir, cfg)

    flips = sum(r["hard"] != r["hard_oldeval"] for r in records)
    yes = sum(r["hard"] == "YES" for r in records)
    return dict(status=(f"{len(records)} records ({n_calls} calls, {dt:.0f}s) | "
                        f"hard_yes {yes / len(records):.3f} | {flips} verdicts changed"),
                report=change_report(records), calls=n_calls, regraded=True)


# ── manifest (book-keeping) ────────────────────────────────────────────────────────────────
def write_manifest(metas: list[dict], path: Path) -> None:
    lines = [
        "# GoalScope re-evaluation manifest",
        "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')}_  ·  {len(metas)} runs under "
        "`results/evaluation/goalscope`.",
        "",
        "Every run below is re-graded with the current `eval_goalscope` oracle rubric. `stored`/"
        "`current` are the `hard_rubric_tag` md5 prefixes; a run is a no-op when they match and "
        "it has already been backed up.",
        "",
        "| model | mode | strategy | α | heads | n_heads | n_specs | records | stored→current | done |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m in metas:
        lines.append(
            f"| {m['model']} | {m['mode']} | {m['strategy']} | {m['alpha']} | {m['heads_name']} "
            f"| {m['n_heads']} | {m['n_specs']} | {m['n_records']} "
            f"| {m['stored_tag']}→{m['current_tag']} | {'✓' if m['reevaled'] else ''} |"
        )
    total_recs = sum(m["n_records"] for m in metas)
    lines += ["", f"**Total records:** {total_recs:,} across {len(metas)} runs "
              "(oracle calls are fewer after de-duplicating identical read-outs)."]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


# ── named comparison axes (subsets worth re-grading on their own) ────────────────────────────
# Each maps to a predicate over run_meta dicts. `baseline` = notebook 12.2 §1: full-residual
# patchscope vs the 122-head goal encoding, both metrics -> 2 configs x 2 metrics = 4 runs.
AXES = {
    "baseline": lambda m: (
        m["model"] == "Qwen2.5-Coder-32B-Instruct"
        and m["n_specs"] == 512
        and (m["strategy"], m["heads_name"]) in {
            ("additive_norm", "goal"),                 # goal-head encoding (122 heads)
            ("resid_norm", "resid:L25-56:mean"),       # full-residual patchscope
        }
    ),
}


# ── driver ─────────────────────────────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=env_utils.DEFAULT_RESULTS_DIR)
    ap.add_argument("--model", default=None, help="substring filter on model slug")
    ap.add_argument("--mode", default=None, choices=["faithfulness", "causality"])
    ap.add_argument("--strategy", default=None,
                    help="substring filter: additive | additive_norm | replace | resid_norm")
    ap.add_argument("--axis", default=None, choices=sorted(AXES),
                    help="named comparison axis. 'baseline' = residual vs 122-head goal encoding "
                         "(both metrics) = the 4 runs behind notebook 12.2 §1")
    ap.add_argument("--heads-name", default=None,
                    help="substring filter on heads_name (e.g. 'goal', 'resid', 'json:goal__lam20')")
    ap.add_argument("--n-specs", type=int, default=None, help="filter on n_specs (e.g. 512)")
    ap.add_argument("--only", action="append", default=[],
                    help="keep runs whose dir path contains this substring (repeatable)")
    ap.add_argument("--workers", type=int, default=8, help="parallel oracle requests")
    ap.add_argument("--force", action="store_true",
                    help="re-grade even runs whose stored rubric tag already matches")
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be re-graded (+ oracle-call counts); no API calls")
    ap.add_argument("--manifest", default=None,
                    help="also write the book-keeping manifest to this markdown path")
    return ap.parse_args()


def main():
    args = parse_args()
    runs = discover_runs(args.results_dir)
    metas = [run_meta(d) for d in runs]

    def keep(m):
        if args.model and args.model not in m["model"]:
            return False
        if args.mode and m["mode"] != args.mode:
            return False
        if args.strategy and (m["strategy"] or "") != args.strategy and args.strategy not in (m["strategy"] or ""):
            return False
        if args.heads_name and args.heads_name not in (m["heads_name"] or ""):
            return False
        if args.n_specs is not None and m["n_specs"] != args.n_specs:
            return False
        if args.axis and not AXES[args.axis](m):
            return False
        if args.only and not all(s in str(m["dir"]) for s in args.only):
            return False
        return True

    metas = [m for m in metas if keep(m)]
    print(f"{len(metas)} run dirs matched under {fio.goalscope_root(args.results_dir)}")

    if args.manifest:
        write_manifest(metas, Path(args.manifest))
        print(f"manifest -> {args.manifest}")

    if args.dry_run:
        total_calls = 0
        for m in metas:
            recs = load_pristine_records(m["dir"])
            uniq = len({(r.get("readout", ""), r["impl_span"]) for r in recs
                        if r.get("readout", "").strip()})
            total_calls += uniq
            todo = "done" if (m["stored_tag"] == m["current_tag"] and m["reevaled"]) else "TO RE-GRADE"
            print(f"  [{todo:11s}] {m['model']:28s} {m['mode']:12s} {m['strategy']:13s} "
                  f"a={m['alpha']} heads={m['heads_name']:26s} "
                  f"records={m['n_records']:>6d} uniq_calls={uniq:>6d}")
        print(f"\ntotal unique oracle calls if all re-graded: {total_calls:,} "
              f"(cached after the first pass)")
        return

    t_start = time.time()
    total_calls, n_regraded = 0, 0
    for i, m in enumerate(metas, 1):
        # header BEFORE grading, so a long dir shows what is in flight (its bar follows)
        print(f"[{i}/{len(metas)}] ▶ {m['model']} · {m['mode']} · {m['strategy']} "
              f"a={m['alpha']} heads={m['heads_name']} ({m['n_records']} records)", flush=True)
        res = reeval_one(m["dir"], args.workers, args.force)
        total_calls += res["calls"]
        n_regraded += int(res["regraded"])
        print(f"          {res['status']}", flush=True)
        if res.get("report"):
            print(res["report"], flush=True)
    dt = time.time() - t_start
    print(f"\ndone: {n_regraded}/{len(metas)} runs re-graded, {len(metas) - n_regraded} skipped | "
          f"{total_calls:,} oracle calls | {dt / 60:.1f} min")


if __name__ == "__main__":
    main()
