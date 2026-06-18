#!/usr/bin/env python
"""Kill jobs from a baulab batch.

Kill specific jobs by name, or every job in a batch. A killed job is **not** auto-requeued
(unlike an OOM / externally-killed one) — that's the whole point.

    python run_jobs/baulab/kill.py --names qwen14b_cma qwen32b_cma   # kill these jobs
    python run_jobs/baulab/kill.py --all                            # kill the whole latest batch
    python run_jobs/baulab/kill.py --all --run-id 20260617_171530   # ... of a specific batch
    python run_jobs/baulab/kill.py --list                          # show batches + run ids

How it works
------------
Each target is recorded in a per-run kill list (`<run_id>.kill`). If the scheduler is still
running it reads that list on its next poll and settles the job as `killed` (dropping it from
the wait queue if it hadn't been placed yet). For responsiveness — and so kills work even when
the scheduler has already exited — kill.py *also* directly quits the screen of any currently
placed target. Placed jobs run in detached screens, so quitting the screen stops the driver and
the job immediately.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dispatch import (  # noqa: E402
    load_credentials,
    manifests_dir,
    print_runs,
    quit_screen,
    read_kill_list,
    summarize_runs,
    write_kill_list,
)
from rich.console import Console  # noqa: E402

TERMINAL = {"done", "error", "killed", "cancelled"}


def latest_run_id() -> str | None:
    runs = summarize_runs()
    return runs[0]["run_id"] if runs else None


def load_manifest(run_id: str) -> dict | None:
    p = manifests_dir() / f"{run_id}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--names", nargs="*", default=[], help="job names to kill")
    ap.add_argument("--all", action="store_true", help="kill every job in the batch")
    ap.add_argument("--run-id", default=None, help="batch to target (default: most recent)")
    ap.add_argument("--list", action="store_true", help="list batches + run ids, then exit")
    args = ap.parse_args()

    console = Console()
    if args.list:
        print_runs(console)
        return
    if not args.all and not args.names:
        ap.error("pass --names <job> ... or --all (or --list)")

    run_id = args.run_id or latest_run_id()
    if not run_id:
        console.print("[red]no batches found.[/] Nothing to kill.")
        sys.exit(1)
    manifest = load_manifest(run_id)
    if not manifest:
        console.print(f"[red]no manifest for run {run_id}.[/]")
        sys.exit(1)

    jobs = {j["name"]: j for j in manifest["jobs"]}
    if args.all:
        targets = {n for n, j in jobs.items() if j.get("state") not in TERMINAL}
    else:
        unknown = [n for n in args.names if n not in jobs]
        if unknown:
            console.print(f"[red]not in run {run_id}:[/] {', '.join(unknown)}")
            console.print(f"jobs in this run: {', '.join(jobs)}")
            sys.exit(1)
        targets = {n for n in args.names if jobs[n].get("state") not in TERMINAL}

    already = [n for n in (args.names or jobs) if jobs[n].get("state") in TERMINAL]
    if not targets:
        console.print(f"[yellow]nothing to kill[/] in run {run_id} "
                      f"(already settled: {', '.join(already) or 'none'}).")
        return

    # 1) record the request so a live scheduler settles these as `killed` (and won't requeue them).
    write_kill_list(run_id, read_kill_list(run_id) | targets)
    console.print(f"recorded kill request for run [bold]{run_id}[/]: {', '.join(sorted(targets))}")

    # 2) directly quit the screens of placed targets (immediate; also covers a dead scheduler).
    user, pw = load_credentials()
    for name in sorted(targets):
        job = jobs[name]
        if job.get("status_path") and job.get("screen"):
            quit_screen({"host": job["host"], "gpus": job["gpus"], "screen": job["screen"],
                         "is_local": job.get("is_local")}, user, pw)
            console.print(f"  ✗ quit screen for [bold]{name}[/] on {job['host']}")
        else:
            console.print(f"  • {name} was still queued (no screen) — will be dropped by the scheduler")

    if already:
        console.print(f"[dim]skipped (already settled): {', '.join(already)}[/]")
    console.print("\nwatch:  python run_jobs/baulab/monitor.py "
                  + (f"--run-id {run_id}" if args.run_id else ""))


if __name__ == "__main__":
    main()
