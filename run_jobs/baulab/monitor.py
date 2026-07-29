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
import re
import sys
import time
from pathlib import Path

from rich.box import SIMPLE_HEAVY
from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dispatch import (  # noqa: E402
    STATE_STYLE,
    _run_on_host,
    load_credentials,
    manifests_dir,
    print_runs,
    use_real_terminal_size,
)

# `stale` is monitor-derived (driver pid dead) — terminal for display/--watch-exit purposes, but
# NOT a MANIFEST_TERMINAL (the scheduler never writes it).
TERMINAL = {"done", "error", "killed", "cancelled", "stale"}
MANIFEST_TERMINAL = {"done", "error", "killed", "cancelled"}  # scheduler-authoritative states


def latest_manifest() -> Path | None:
    d = manifests_dir()
    runs = sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime) if d.exists() else []
    return runs[-1] if runs else None


def probe_job(job: dict, user: str, pw: str) -> tuple[dict | None, bool | None]:
    """Read the job's status.json AND its liveness. Returns ``(rec, alive)`` where ``rec`` is the
    status.json job entry (state/started/ended, or None if unreadable) and ``alive`` is True/False
    (None = unknown / host unreachable).

    Liveness prefers the driver `pid` recorded in status.json (unique per run -> collision-proof,
    since the screen *name* is the reused job name) via `kill -0`; falls back to the screen session
    for OLD runs whose driver never recorded a pid."""
    path = job.get("status_path")
    if not path:
        return None, None
    host = job.get("host")
    key = "LOCAL" if job.get("is_local") else host
    screen = job.get("screen") or ""
    cmd = f"cat {path} 2>/dev/null; echo '<<<CCSEP>>>'; screen -ls 2>/dev/null || true"
    res = _run_on_host(key, cmd, host, user, pw)
    if key != "LOCAL" and res.returncode == 255:  # host unreachable -> unknown liveness
        return None, None
    status_txt, _, screen_txt = res.stdout.partition("<<<CCSEP>>>")
    rec, pid = None, None
    try:
        data = json.loads(status_txt)
        rec = data["jobs"][0]
        pid = int(data["pid"]) if data.get("pid") is not None else None
    except (json.JSONDecodeError, KeyError, IndexError, ValueError, TypeError):
        pass
    if pid:  # collision-proof liveness via this run's own driver pid
        alive = bool(_run_on_host(key, f"kill -0 {pid} 2>/dev/null && echo ok", host, user, pw).stdout.strip())
    elif screen:  # old run (no pid): best-effort screen-name liveness
        alive = bool(re.search(rf"\.{re.escape(screen)}\s", screen_txt))
    else:
        alive = None
    return rec, alive


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


def _retries(job: dict) -> str:
    return ", ".join(f"{h['host']}({h['reason']})" for h in (job.get("history") or []))


def resolve(job: dict, user: str, pw: str, now: float) -> tuple[str, str, str, str]:
    """(machine, state, detail, retries) for one manifest job, polling its host if placed."""
    retries = _retries(job)
    mstate = job.get("state")
    if not job.get("status_path"):
        # never placed (or placed-then-removed): the manifest is authoritative.
        detail = {"queued": "in wait queue"}.get(mstate, job.get("error") or "")
        machine = "—"
        if job.get("host"):
            machine = f"{job['host'].split('.')[0]}:{','.join(map(str, job['gpus']))}"
        return machine, mstate or "queued", detail, retries
    machine = f"{job['host'].split('.')[0]}:{','.join(map(str, job['gpus']))}"
    if mstate in MANIFEST_TERMINAL:
        # scheduler already settled this one (incl. killed/cancelled / retries-exhausted).
        return machine, mstate, job.get("error") or "", retries
    rec, alive = probe_job(job, user, pw)
    if rec is None:
        # placed but status not readable yet (driver starting) or host unreachable.
        return machine, mstate or "no-status", "", retries
    state = rec["state"]  # status.json is authoritative (corrects a manifest lagging behind error/done)
    # liveness: a status.json still claiming a live state but whose driver is gone is a phantom
    # (hard-killed / rebooted host that never wrote a terminal state) -> report `stale`.
    if state not in TERMINAL and alive is False:
        elapsed = _detail(rec, now)
        return machine, "stale", "driver gone" + (f" · {elapsed}" if elapsed else ""), retries
    return machine, state, _detail(rec, now), retries


def render(manifest: dict, rows: list[tuple[str, str, str, str, str]]) -> Group:
    table = Table(box=SIMPLE_HEAVY, expand=False, pad_edge=False)
    table.add_column("job", style="bold", no_wrap=True)
    table.add_column("machine", no_wrap=True)
    table.add_column("status", no_wrap=True)
    table.add_column("retries", no_wrap=True, style="yellow")
    table.add_column("detail", overflow="fold", max_width=52)

    counts: dict[str, int] = {}
    for name, machine, state, detail, retries in rows:
        counts[state] = counts.get(state, 0) + 1
        table.add_row(name, machine, Text(state, style=STATE_STYLE.get(state, "white")), retries, detail)

    order = ["running", "waiting", "queued", "done", "error", "killed", "cancelled", "stale", "no-status"]
    parts = [f"[{STATE_STYLE.get(k, 'white')}]{k} {counts[k]}[/]" for k in order if counts.get(k)]
    header = Text.from_markup(
        f"run [bold]{manifest['run_id']}[/] · submitted {manifest['created']} · "
        + "  ".join(parts) + f"  ·  refreshed {time.strftime('%H:%M:%S')}"
    )
    return Group(header, table)


def collect(manifest: dict, user: str, pw: str) -> list[tuple[str, str, str, str, str]]:
    now = time.time()
    rows = []
    for job in manifest["jobs"]:
        machine, state, detail, retries = resolve(job, user, pw, now)
        rows.append((job["name"], machine, state, detail, retries))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", default=None, help="batch to monitor (default: most recent)")
    ap.add_argument("--interval", type=float, default=30.0, help="refresh seconds (default: 30)")
    ap.add_argument("--watch", action="store_true", help="keep refreshing even after all jobs finish")
    ap.add_argument("--list", action="store_true", help="list past batches + run ids, then exit")
    args = ap.parse_args()

    use_real_terminal_size()
    console = Console()
    if args.list:
        # probe screen liveness so hard-killed/rebooted runs show `stale`, not a phantom `running`.
        try:
            user, pw = load_credentials()
        except Exception:  # noqa: BLE001 — no creds: remote probes fail to "unreachable", left as-is
            user, pw = None, None
        print_runs(console, probe=True, user=user, pw=pw)
        return
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
            if all(state in TERMINAL for _, _, state, _, _ in rows) and not args.watch:
                break
            time.sleep(args.interval)

    manifest = json.loads(mpath.read_text())
    rows = collect(manifest, user, pw)
    console.print(render(manifest, rows))
    console.print("[dim]all jobs finished.[/]" if not args.watch else "")


if __name__ == "__main__":
    main()
