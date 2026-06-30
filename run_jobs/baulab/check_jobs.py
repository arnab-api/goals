#!/usr/bin/env python
"""Integrity-check a baulab jobs JSON file *before* dispatching it.

    python run_jobs/baulab/check_jobs.py job_list/my_jobs.json
    python -m run_jobs.baulab.check_jobs job_list/my_jobs.json

Exits 0 if the file is valid, 1 if any integrity error is found (and prints them all).
`dispatch.py` runs the exact same `validate_jobs` check at launch, so this script is just a
fast, model-free way to catch problems (typos, bad/circular dependencies) ahead of time.

What it checks (see `validate_jobs` in dispatch.py):
  - top-level is a non-empty JSON list of objects
  - each job has a non-empty string `name` (unique) and `command`
  - no unknown fields on any job (typo guard — `name`, `command`, `dependencies` only)
  - every `dependencies` entry names a real job; no self-edges, duplicates, or cycles
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dispatch import job_deps, validate_jobs  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402


def execution_waves(jobs: list[dict]) -> list[list[str]]:
    """Group job names into dependency waves (Kahn layering): wave 0 has no deps, wave k
    runs once every job in earlier waves is done. Purely informational."""
    deps_of = {j["name"]: list(dict.fromkeys(job_deps(j))) for j in jobs}
    done: set[str] = set()
    waves: list[list[str]] = []
    remaining = set(deps_of)
    while remaining:
        ready = sorted(n for n in remaining if set(deps_of[n]) <= done)
        if not ready:                          # should never happen post-validation
            break
        waves.append(ready)
        done |= set(ready)
        remaining -= set(ready)
    return waves


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="path to a jobs JSON file")
    args = ap.parse_args()
    console = Console()

    p = Path(args.path)
    if not p.exists():
        console.print(f"[bold red]✗[/] file not found: {p}")
        sys.exit(1)
    try:
        jobs = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        console.print(f"[bold red]✗ invalid JSON[/] in {p}: {e}")
        sys.exit(1)

    errors = validate_jobs(jobs)
    if errors:
        console.print(f"[bold red]✗ {p} — {len(errors)} integrity error(s):[/]")
        for msg in errors:
            console.print(f"  [red]•[/] {msg}")
        sys.exit(1)

    # valid — show a short summary + the dependency waves.
    n_dep_edges = sum(len(job_deps(j)) for j in jobs)
    console.print(f"[bold green]✓ {p} is valid[/] — {len(jobs)} jobs, {n_dep_edges} dependency edge(s).")

    waves = execution_waves(jobs)
    table = Table(title="execution waves (dependency order)", show_header=True)
    table.add_column("wave", justify="right", style="cyan")
    table.add_column("jobs that can start once earlier waves finish")
    for i, wave in enumerate(waves):
        table.add_row(str(i), ", ".join(wave))
    console.print(table)
    sys.exit(0)


if __name__ == "__main__":
    main()
