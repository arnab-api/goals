#!/usr/bin/env python
"""Live terminal monitor for a baulab batch submission.

Reads the local manifest written by `dispatch.py` (defaults to the most recent run), then every
few seconds shows each job's status. Placed jobs are polled at their assigned host — by reading
`status.json` directly for the local worker, or `sshpass ssh <host> cat <status_path>` for
remote ones; jobs still in the wait queue show `queued`.

    job name | machine (host:gpus) | status | detail (error brief / elapsed)

Statuses: queued (in the wait queue) / waiting (driver waiting on GPU mem) / running / done /
error. For errors the driver's one-line classification (CUDA OOM, a Python `SomeError:` line,
exit 137, ...) is shown in the detail column.

Usage
-----
    python run_jobs/baulab/monitor.py                 # latest batch, exits when all finish
    python run_jobs/baulab/monitor.py --run-id 20260617_171530
    python run_jobs/baulab/monitor.py --interval 5 --watch   # keep refreshing after finish
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from rich.box import SIMPLE_HEAVY
from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dispatch import SSH_OPTS, load_credentials, manifests_dir  # noqa: E402

TERMINAL = {"done", "error"}
STATE_STYLE = {
    "queued": "dim",
    "waiting": "yellow",
    "running": "cyan",
    "done": "green",
    "error": "bold red",
    "no-status": "magenta",
}


def latest_manifest() -> Path | None:
    d = manifests_dir()
    runs = sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime) if d.exists() else []
    return runs[-1] if runs else None


def fetch_job(job: dict, user: str, pw: str) -> dict | None:
    """A placed job's status record (the single entry in its status.json), or None."""
    path = job.get("status_path")
    if not path:
        return None
    if job.get("is_local"):
        try:
            data = json.loads(Path(os.path.expanduser(path)).read_text())
        except (OSError, json.JSONDecodeError):
            return None
    else:
        cmd = f"sshpass -p {shlex.quote(pw)} ssh {SSH_OPTS} {user}@{job['host']} cat {path}"
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable="/bin/bash")
        if res.returncode != 0 or not res.stdout.strip():
            return None
        try:
            data = json.loads(res.stdout)
        except json.JSONDecodeError:
            return None
    jobs = data.get("jobs") or []
    return jobs[0] if jobs else None


def _fmt_elapsed(sec: float) -> str:
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def _detail(rec: dict, now: float) -> str:
    state = rec["state"]
    if state == "error":
        return rec.get("error") or "failed"
    if state == "running" and rec.get("started"):
        return _fmt_elapsed(now - rec["started"])
    if state == "done" and rec.get("started") and rec.get("ended"):
        return f"took {_fmt_elapsed(rec['ended'] - rec['started'])}"
    if state == "waiting":
        return "for GPU memory"
    return ""


def resolve(job: dict, user: str, pw: str, now: float) -> tuple[str, str, str]:
    """(machine, state, detail) for one manifest job, polling its host if placed."""
    if not job.get("status_path"):
        return "—", "queued", "in wait queue"
    machine = f"{job['host']}:{','.join(map(str, job['gpus']))}"
    rec = fetch_job(job, user, pw)
    if rec is None:
        # placed but status not readable yet (driver starting) or host unreachable.
        return machine, job.get("state", "no-status"), ""
    return machine, rec["state"], _detail(rec, now)


def render(manifest: dict, rows: list[tuple[str, str, str, str]]) -> Group:
    table = Table(box=SIMPLE_HEAVY, expand=False, pad_edge=False)
    table.add_column("job", style="bold", no_wrap=True)
    table.add_column("machine", no_wrap=True)
    table.add_column("status", no_wrap=True)
    table.add_column("detail", overflow="fold", max_width=60)

    counts: dict[str, int] = {}
    for name, machine, state, detail in rows:
        counts[state] = counts.get(state, 0) + 1
        table.add_row(name, machine, Text(state, style=STATE_STYLE.get(state, "white")), detail)

    order = ["running", "waiting", "queued", "done", "error", "no-status"]
    parts = [f"[{STATE_STYLE.get(k, 'white')}]{k} {counts[k]}[/]" for k in order if counts.get(k)]
    header = Text.from_markup(
        f"run [bold]{manifest['run_id']}[/] · submitted {manifest['created']} · "
        + "  ".join(parts) + f"  ·  refreshed {time.strftime('%H:%M:%S')}"
    )
    return Group(header, table)


def collect(manifest: dict, user: str, pw: str) -> list[tuple[str, str, str, str]]:
    now = time.time()
    rows = []
    for job in manifest["jobs"]:
        machine, state, detail = resolve(job, user, pw, now)
        rows.append((job["name"], machine, state, detail))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", default=None, help="batch to monitor (default: most recent)")
    ap.add_argument("--interval", type=float, default=3.0, help="refresh seconds (default: 3)")
    ap.add_argument("--watch", action="store_true", help="keep refreshing even after all jobs finish")
    args = ap.parse_args()

    console = Console()
    mpath = (manifests_dir() / f"{args.run_id}.json") if args.run_id else latest_manifest()
    if not mpath or not mpath.exists():
        console.print("[red]no submission manifest found.[/] Run dispatch.py first.")
        sys.exit(1)
    user, pw = load_credentials()

    with Live(console=console, screen=True, auto_refresh=False) as live:
        while True:
            manifest = json.loads(mpath.read_text())  # re-read: scheduler updates it live
            rows = collect(manifest, user, pw)
            live.update(render(manifest, rows))
            live.refresh()
            if all(state in TERMINAL for _, _, state, _ in rows) and not args.watch:
                break
            time.sleep(args.interval)

    manifest = json.loads(mpath.read_text())
    rows = collect(manifest, user, pw)
    console.print(render(manifest, rows))
    console.print("[dim]all jobs finished.[/]" if not args.watch else "")


if __name__ == "__main__":
    main()
