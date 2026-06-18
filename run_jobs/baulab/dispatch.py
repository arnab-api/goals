#!/usr/bin/env python
"""Schedule a list of jobs across baulab machines with a central greedy scheduler.

A *worker* is a single `MachineConfig(name=<host>, cuda=[<gpu indices>])` entry and runs at
most ONE of our jobs at a time (using all of its GPUs). Listing the same host twice with
different GPU sets makes it two independent workers, e.g.

    MACHINES = [
        MachineConfig("kyoto", cuda=[0, 1]),   # worker kyoto_gpu0-1  (one job on GPUs 0,1)
        MachineConfig("kyoto", cuda=[2, 3]),   # worker kyoto_gpu2-3  (separate, concurrent)
    ]

Scheduling model
----------------
A global FIFO wait queue holds every job. The scheduler keeps placing the front job onto the
*currently best* free machine, so a job never sits behind a busy machine when an idle one
exists:

1. A machine *satisfies the constraints* for a job when its specified GPUs have total capacity
   >= MIN_GPU_MEMORY and FREE memory now >= MIN_GPU_MEMORY, and it is not already running one of
   our jobs.
2. Among the satisfying machines, the one with the **most free memory** wins; the job is
   deployed there immediately (driver + screen). Jobs that don't fit stay in the queue.
3. Every POLL_INTERVAL_MIN the scheduler re-probes idle machines (catching memory freed by
   *other people's* jobs too) and places as many front-of-queue jobs as fit. Naive polling.

Auto-requeue
------------
Every STATUS_POLL_SEC the scheduler polls each placed job. Two failure modes are assumed to be
"someone else contended the machine" and are put **back on the global queue** (so the retry can
land on a *different* host), up to MAX_RETRY times:
  * **OOM** — the job hit CUDA OOM;
  * **killed** — the job (or its whole screen) was killed by a signal: oom-killer, or a person
    reclaiming their machine. Detected from the driver's exit code *or* from the screen session
    vanishing while the job was still mid-run.
Any *other* error (a real bug) is terminal. A job you kill yourself (kill.py) is never requeued.

Live monitor + lifetime
-----------------------
The scheduler is **long-lived**: it stays up until every job is terminal (so it can reap and
requeue), rendering a live dashboard right in this terminal. Killing the scheduler (Ctrl-C)
cancels only the still-*queued* jobs — placed jobs run on in their own detached screens, and
`monitor.py` can still watch them. Each job's `status.json` (read by monitor.py) and a local
job-centric manifest in `.remote_jobs/manifests/<run_id>.json` are kept up to date.

Usage
-----
    python run_jobs/baulab/dispatch.py --jobs run_jobs/baulab/jobs.example.json
    python run_jobs/baulab/dispatch.py --jobs my_jobs.json --dry-run   # probe + plan, no launch
    python run_jobs/baulab/dispatch.py --list                          # past batches + run ids

Edit MACHINES and the CONFIG block below in-place. Credentials come from env.yml
(BAULAB_USER / BAULAB_PASS).
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from rich.box import SIMPLE_HEAVY
from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ──────────────────────────────────────────────────────────────────────────────
# EDIT ME — machines and config.
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class MachineConfig:
    """One worker: a host plus the GPU indices it may use. Repeat a host with different
    `cuda` sets to give it several independent queues."""

    name: str
    cuda: list[int] = field(default_factory=lambda: [0])


MACHINES = [
    MachineConfig(name="umibozu", cuda=[0]),
    MachineConfig(name="tokyo", cuda=[0]),
    # MachineConfig(name="hawaii", cuda=[0]),
    # MachineConfig(name="karasuno", cuda=[0]),
    MachineConfig(name="kobe", cuda=[0]),
    MachineConfig(name="macondo", cuda=[0]),
    MachineConfig(name="ei", cuda=[0]),
    # MachineConfig(name="andromeda", cuda=[0]), #! 20.04
    MachineConfig(name="kawasaki", cuda=[0]),
    # MachineConfig(name="sendai", cuda=[0]),   #! 20.04
    # MachineConfig(name="kumamoto", cuda=[0]), 
    # MachineConfig(name="hamada", cuda=[0]),
    MachineConfig(name="fukuyama", cuda=[0]),
    # MachineConfig(name="osaka", cuda=[0]), #! 20.04
    MachineConfig(name="cancun", cuda=[0]),
    MachineConfig(name="kameoka", cuda=[0]),
    # MachineConfig(name="bippu", cuda=[0]), #! 20.04
    # MachineConfig(name="hokkaido", cuda=[0]), #! 20.04
    MachineConfig(name="kyoto", cuda=[0]),
    MachineConfig(name="tottori", cuda=[0]),
    # MachineConfig(name="nagoya.research.khoury.northeastern.edu", cuda=list(range(8))),
    # MachineConfig(name="hakone.research.khoury.northeastern.edu", cuda=list(range(8)))
]

# SAFE_MACHINES are the *preferred* tier — tried FIRST for every job (e.g. machines you own, where
# your jobs won't be pre-empted by someone reclaiming the box). Only when **no** safe machine
# currently satisfies a job's requirements does the scheduler fall back to MACHINES. You may list a
# host here even if it's also in MACHINES — duplicates are de-duped into the safe tier, so you
# don't have to edit MACHINES.
SAFE_MACHINES = [
    MachineConfig(name="umibozu", cuda=[0]),
    MachineConfig(name="tokyo", cuda=[0])
]

# --- gates / behaviour --------------------------------------------------------
MIN_GPU_MEMORY = 45        # GB; a job needs this much FREE (and capacity) across a worker's GPUs
POLL_INTERVAL_MIN = 10     # idle-machine probe + placement cadence
MAX_RETRY = 5              # auto-requeue attempts for OOM / externally-killed jobs
AUTO_CONTINUE = True       # (per-job driver flag; one job per worker so largely cosmetic)

# --- live-display cadences --------------------------------------------------
# STATUS_POLL_SEC is the only one that hits the network (ssh to each placed job), so it's kept
# coarse to avoid overloading the master box; RENDER_INTERVAL_SEC is a local-only repaint.
STATUS_POLL_SEC = 8 * 60   # how often to ssh-poll placed jobs' status + handle requeue/kills
RENDER_INTERVAL_SEC = 2    # dashboard repaint (local; just animates elapsed clocks)

# --- where things live --------------------------------------------------------
CONDA_ENV = "goals"
LOCAL_PROJECT_DIR = str(_PROJECT_ROOT)        # used when name == "local"
REMOTE_PROJECT_DIR = "~/Codes/Projects/goals"  # used on remote hosts (tilde ok)
REMOTE_STAGING_SUBDIR = ".remote_jobs"        # under the project dir on each host

# SSH hardening so automation never blocks on a prompt.
SSH_OPTS = "-o StrictHostKeyChecking=accept-new -o LogLevel=ERROR -o ConnectTimeout=15"

# Failure kinds the scheduler will auto-requeue (everything else is terminal).
RETRYABLE_KINDS = {"oom", "killed"}
TERMINAL = {"done", "error", "killed", "cancelled"}


# ──────────────────────────────────────────────────────────────────────────────
# Worker model.
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Worker:
    host: str
    cuda: list[int]
    tier: str = "fallback"   # "safe" (preferred) or "fallback"

    @property
    def is_local(self) -> bool:
        return self.host == "local"

    @property
    def worker_id(self) -> str:
        safe_host = self.host.split(".")[0].replace("/", "_")
        return f"{safe_host}_gpu{'-'.join(map(str, self.cuda))}"

    @property
    def project_dir(self) -> str:
        return LOCAL_PROJECT_DIR if self.is_local else REMOTE_PROJECT_DIR


def use_real_terminal_size() -> None:
    """Drop COLUMNS/LINES from the env so rich measures the real terminal via ioctl.

    zsh exports COLUMNS (bash usually doesn't); a stale/narrow value makes rich render the
    table to that width and fold the rightmost flexible column (``detail``) down to nothing,
    while the fixed columns still show. Popping these lets rich follow the actual window.
    """
    for v in ("COLUMNS", "LINES"):
        os.environ.pop(v, None)


def load_credentials() -> tuple[str, str]:
    with open(_PROJECT_ROOT / "env.yml") as f:
        cfg = yaml.safe_load(f)
    user = cfg.get("BAULAB_USER")
    pw = cfg.get("BAULAB_PASS")
    if not user or not pw:
        print(
            "WARNING: BAULAB_USER / BAULAB_PASS not set in env.yml — only the 'local' "
            "worker will work (remote SSH/scp will fail).",
            file=sys.stderr,
        )
    return user, pw


# ──────────────────────────────────────────────────────────────────────────────
# Remote exec helpers (sshpass for remote, plain subprocess for local).
# ──────────────────────────────────────────────────────────────────────────────


def _ssh_prefix(user: str, pw: str, host: str) -> str:
    return f"sshpass -p {shlex.quote(pw)} ssh {SSH_OPTS} {user}@{host}"


def run_on(worker: Worker, command: str, user: str, pw: str) -> subprocess.CompletedProcess:
    """Run a shell `command` on the worker's host (local or via ssh). Captures output."""
    if worker.is_local:
        full = command
    else:
        full = f"{_ssh_prefix(user, pw, worker.host)} {shlex.quote(command)}"
    return subprocess.run(full, shell=True, capture_output=True, text=True, executable="/bin/bash")


def copy_dir_to(worker: Worker, local_dir: Path, remote_dir: str, user: str, pw: str) -> None:
    """Place every file in `local_dir` at `remote_dir` on the worker's host."""
    if worker.is_local:
        target = Path(os.path.expanduser(remote_dir))
        target.mkdir(parents=True, exist_ok=True)
        for f in local_dir.iterdir():
            subprocess.run(["cp", str(f), str(target / f.name)], check=True)
        return
    run_on(worker, f"mkdir -p {remote_dir}", user, pw)
    scp = (
        f"sshpass -p {shlex.quote(pw)} scp {SSH_OPTS} "
        f"{shlex.quote(str(local_dir))}/* {user}@{worker.host}:{remote_dir}/"
    )
    res = subprocess.run(scp, shell=True, capture_output=True, text=True, executable="/bin/bash")
    if res.returncode != 0:
        raise RuntimeError(f"scp to {worker.host} failed: {res.stderr.strip()}")


# ──────────────────────────────────────────────────────────────────────────────
# Probe + gating.
# ──────────────────────────────────────────────────────────────────────────────

_NVIDIA_QUERY = "nvidia-smi --query-gpu=index,memory.total,memory.free --format=csv,noheader,nounits"


def probe_gpus(worker: Worker, user: str, pw: str) -> dict[int, tuple[float, float]]:
    """Return {gpu_index: (total_gb, free_gb)} for the host, or {} if unreachable / no GPU."""
    res = run_on(worker, _NVIDIA_QUERY, user, pw)
    if res.returncode != 0:
        return {}
    out: dict[int, tuple[float, float]] = {}
    for line in res.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            idx, total, free = (x.strip() for x in line.split(","))
            out[int(idx)] = (float(total) / 1024.0, float(free) / 1024.0)
        except ValueError:
            continue
    return out


def free_across(gpus: dict[int, tuple[float, float]], cuda: list[int]) -> float:
    """Total FREE GB across the worker's GPUs (0 for any GPU not reported)."""
    return sum(gpus.get(g, (0.0, 0.0))[1] for g in cuda)


def static_gate(worker: Worker, gpus: dict[int, tuple[float, float]]) -> tuple[bool, str]:
    """Constraints independent of moment-to-moment free memory (GPU presence + capacity)."""
    if not gpus:
        return False, "unreachable or no GPUs"
    missing = [g for g in worker.cuda if g not in gpus]
    if missing:
        return False, f"requested GPUs {missing} not present (has {sorted(gpus)})"
    capacity = sum(gpus[g][0] for g in worker.cuda)
    # Capacity gate is about nominal card size, so round to whole GiB: a 48GB card reports
    # 47.99 GiB and must still pass MIN_GPU_MEMORY=48. (Free-memory check below is exact.)
    if round(capacity) < MIN_GPU_MEMORY:
        return False, f"total capacity {capacity:.1f}GB < MIN_GPU_MEMORY={MIN_GPU_MEMORY}GB"
    free = free_across(gpus, worker.cuda)
    return True, f"capacity {capacity:.1f}GB, free {free:.1f}GB now"


# ──────────────────────────────────────────────────────────────────────────────
# Staging + deploy (one job -> one worker).
# ──────────────────────────────────────────────────────────────────────────────


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def staging_paths(worker: Worker, run_id: str, job_name: str) -> tuple[str, str, str]:
    """(remote_dir_rel, remote_dir_abs, status_path) for ONE job's staging area.

    Per-job dirs (not per-worker) so a worker reused across jobs never clobbers an earlier
    job's status.json. Paths keep the worker's tilde for remote hosts (expanded by the remote
    shell) and are absolute for the local worker.
    """
    rel = f"{REMOTE_STAGING_SUBDIR}/{run_id}/jobs/{_safe(job_name)}"
    abs_ = f"{worker.project_dir}/{rel}"
    return rel, abs_, f"{abs_}/status.json"


def build_run_sh(worker: Worker, remote_dir_rel: str) -> str:
    """The bash wrapper screen launches: activate conda, cd project, run the driver."""
    return (
        "#!/bin/bash\n"
        "source ~/.bashrc\n"
        'eval "$(conda shell.bash hook)"\n'
        f"conda activate {CONDA_ENV}\n"
        f"cd {worker.project_dir}\n"
        f"python {remote_dir_rel}/driver.py --config {remote_dir_rel}/config.json "
        f"|& tee {remote_dir_rel}/_driver.log\n"
    )


def deploy_job(worker: Worker, job: dict, run_id: str, user: str, pw: str) -> dict:
    """Stage + launch ONE job on `worker` in its own screen. Returns an assignment record."""
    rel, abs_, status_path = staging_paths(worker, run_id, job["name"])
    screen = _safe(f"{job['name']}")
    config = {
        "run_id": run_id,
        "worker_id": worker.worker_id,
        "host": worker.host,
        "gpus": worker.cuda,
        "min_gpu_memory_gb": MIN_GPU_MEMORY,
        "poll_interval_sec": POLL_INTERVAL_MIN * 60,
        "auto_continue": AUTO_CONTINUE,
        "jobs": [job],
    }
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "driver.py").write_text((Path(__file__).parent / "driver.py").read_text())
        (tmp / "config.json").write_text(json.dumps(config, indent=2))
        (tmp / "run.sh").write_text(build_run_sh(worker, rel))
        copy_dir_to(worker, tmp, abs_, user, pw)

    res = run_on(worker, f"screen -dmS {screen} bash -i {abs_}/run.sh", user, pw)
    if res.returncode != 0:
        raise RuntimeError(f"launch on {worker.host} failed: {res.stderr.strip()}")
    return {
        "worker_id": worker.worker_id,
        "host": worker.host,
        "is_local": worker.is_local,
        "gpus": worker.cuda,
        "screen": screen,
        "remote_dir": abs_,
        "status_path": status_path,
    }


def quit_screen(assignment: dict, user: str, pw: str) -> None:
    """Stop a placed job by quitting its detached screen session (driver + job die)."""
    worker = Worker(host=assignment["host"], cuda=assignment["gpus"])
    run_on(worker, f"screen -S {assignment['screen']} -X quit", user, pw)


def poll_job(assignment: dict, user: str, pw: str) -> tuple[str | None, str | None, str | None, bool | None]:
    """One round-trip per placed job: ``(state, error, error_kind, screen_alive)``.

    A single ssh reads status.json AND checks whether the job's screen session still exists,
    so we can tell "still running" from "the whole screen was killed while mid-run". Returns
    all-None when the host is unreachable (transient — caller leaves the job running).
    """
    sp = assignment["status_path"]
    screen = assignment["screen"]
    worker = Worker(host=assignment["host"], cuda=assignment["gpus"])
    cmd = f"cat {sp} 2>/dev/null; echo '<<<CCSEP>>>'; screen -ls 2>/dev/null || true"
    res = run_on(worker, cmd, user, pw)
    if not worker.is_local and res.returncode == 255:
        return None, None, None, None  # ssh/connection failure -> unreachable
    status_txt, _, screen_txt = res.stdout.partition("<<<CCSEP>>>")
    state = err = kind = None
    try:
        job = json.loads(status_txt)["jobs"][0]
        state, err, kind = job["state"], job.get("error"), job.get("error_kind")
    except (json.JSONDecodeError, KeyError, IndexError):
        pass
    alive = bool(re.search(rf"\.{re.escape(screen)}\s", screen_txt))
    return state, err, kind, alive


# ──────────────────────────────────────────────────────────────────────────────
# Local state: manifest (read by monitor.py) + the kill-request side channel.
# ──────────────────────────────────────────────────────────────────────────────


def manifests_dir() -> Path:
    return _PROJECT_ROOT / REMOTE_STAGING_SUBDIR / "manifests"


def kill_list_path(run_id: str) -> Path:
    """Names requested for kill by kill.py (a ``.kill`` file, so it's outside the manifest glob)."""
    return manifests_dir() / f"{run_id}.kill"


def read_kill_list(run_id: str) -> set[str]:
    p = kill_list_path(run_id)
    if not p.exists():
        return set()
    try:
        return set(json.loads(p.read_text()))
    except (OSError, json.JSONDecodeError):
        return set()


def write_kill_list(run_id: str, names: set[str]) -> None:
    p = kill_list_path(run_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".kill.tmp")
    tmp.write_text(json.dumps(sorted(names)))
    os.replace(tmp, p)


def write_manifest(run_id: str, created: str, jobs_out: list[dict]) -> Path:
    """Persist every job's placement + last-known state + retry history (always on the local box)."""
    manifest = {
        "run_id": run_id,
        "created": created,
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {"min_gpu_memory": MIN_GPU_MEMORY, "poll_interval_min": POLL_INTERVAL_MIN,
                   "max_retry": MAX_RETRY},
        "jobs": jobs_out,
    }
    out = manifests_dir() / f"{run_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    os.replace(tmp, out)
    return out


def summarize_runs() -> list[dict]:
    """Newest-first summary of every batch with a manifest (for `--list` / kill.py)."""
    d = manifests_dir()
    runs: list[dict] = []
    if not d.exists():
        return runs
    for p in sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            m = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        counts: dict[str, int] = {}
        for j in m.get("jobs", []):
            counts[j.get("state", "?")] = counts.get(j.get("state", "?"), 0) + 1
        runs.append({"run_id": m.get("run_id", p.stem), "created": m.get("created", "?"),
                     "updated": m.get("updated", "?"), "n": len(m.get("jobs", [])), "counts": counts})
    return runs


def load_jobs(path: str) -> list[dict]:
    with open(path) as f:
        jobs = json.load(f)
    if not isinstance(jobs, list):
        raise ValueError(f"{path} must contain a JSON list of jobs")
    names = set()
    for j in jobs:
        if "name" not in j or "command" not in j:
            raise ValueError(f"each job needs 'name' and 'command'; got {j}")
        if j["name"] in names:
            raise ValueError(f"duplicate job name {j['name']!r} (names must be unique)")
        names.add(j["name"])
    return jobs


# ──────────────────────────────────────────────────────────────────────────────
# Dashboard rendering.
# ──────────────────────────────────────────────────────────────────────────────

STATE_STYLE = {
    "queued": "dim",
    "waiting": "yellow",
    "running": "cyan",
    "done": "green",
    "error": "bold red",
    "killed": "red",
    "cancelled": "dim red",
    "no-status": "magenta",
}
_STATE_ORDER = ["running", "waiting", "queued", "done", "error", "killed", "cancelled", "no-status"]


def _reason(kind: str | None) -> str:
    return {"oom": "OOM", "killed": "killed"}.get(kind or "", kind or "err")


def _fmt_elapsed(sec: float) -> str:
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def render_dashboard(meta: dict, rows: list[tuple], events: list[str]) -> Group:
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

    parts = [f"[{STATE_STYLE.get(k, 'white')}]{k} {counts[k]}[/]" for k in _STATE_ORDER if counts.get(k)]
    header = Text.from_markup(
        f"scheduler · run [bold]{meta['run_id']}[/] · submitted {meta['created']} · "
        + "  ".join(parts) + f"  ·  {time.strftime('%H:%M:%S')}"
    )
    evlog = Text("\n".join(events[-8:]), style="dim")
    return Group(header, table, Text("recent events:", style="bold dim"), evlog)


# ──────────────────────────────────────────────────────────────────────────────
# Scheduler.
# ──────────────────────────────────────────────────────────────────────────────


def schedule(run_id: str, jobs: list[dict], pool: list[Worker], user: str, pw: str) -> None:
    created = time.strftime("%Y-%m-%d %H:%M:%S")
    job_by_name = {j["name"]: j for j in jobs}
    order = [j["name"] for j in jobs]

    queue: deque[str] = deque(order)                # FIFO of job names waiting for a machine
    running: dict[str, dict] = {}                   # worker_id -> {name, assignment, deployed_at}
    terminal: dict[str, dict] = {}                  # name -> {state, error, error_kind, assignment}
    tries = {n: 0 for n in order}
    history: dict[str, list] = {n: [] for n in order}   # name -> [{host, reason}, ...]
    cur_state = {n: "queued" for n in order}
    events: deque[str] = deque(maxlen=20)

    # clear any stale kill requests from a previous run with this id.
    write_kill_list(run_id, set())

    def ev(msg: str) -> None:
        events.append(f"{time.strftime('%H:%M:%S')}  {msg}")

    def placed_by_name() -> dict[str, dict]:
        return {info["name"]: info for info in running.values()}

    def snapshot() -> list[dict]:
        placed = placed_by_name()
        out = []
        for name in order:
            rec = {"name": name, "try_count": tries[name],
                   "history": [dict(h) for h in history[name]]}
            if name in placed:
                a = placed[name]["assignment"]
                rec.update(state=cur_state[name], host=a["host"], gpus=a["gpus"],
                           worker_id=a["worker_id"], is_local=a["is_local"],
                           status_path=a["status_path"], screen=a["screen"])
            elif name in terminal:
                t = terminal[name]
                a = t.get("assignment")
                rec.update(state=t["state"], error=t.get("error"), error_kind=t.get("error_kind"))
                if a:
                    rec.update(host=a["host"], gpus=a["gpus"], worker_id=a["worker_id"],
                               is_local=a["is_local"], status_path=a["status_path"], screen=a["screen"])
                else:
                    rec.update(host=None, gpus=None, worker_id=None, is_local=None,
                               status_path=None, screen=None)
            else:
                rec.update(state="queued", host=None, gpus=None, worker_id=None,
                           is_local=None, status_path=None, screen=None)
            out.append(rec)
        return out

    def persist() -> None:
        write_manifest(run_id, created, snapshot())

    def requeue(name: str, a: dict, kind: str) -> None:
        tries[name] += 1
        history[name].append({"host": a["host"].split(".")[0], "reason": _reason(kind)})
        queue.append(name)
        cur_state[name] = "queued"
        ev(f"requeued {name} after {_reason(kind)} on {a['host']} (try {tries[name]}/{MAX_RETRY})")

    def reap() -> None:
        # (a) honour user kill requests first, so an intentional kill never gets requeued.
        killset = read_kill_list(run_id)
        if killset:
            for name in [n for n in list(queue) if n in killset]:
                queue.remove(name)
                terminal[name] = {"state": "killed", "error": "killed by user (was queued)",
                                  "error_kind": "killed", "assignment": None}
                cur_state[name] = "killed"
                ev(f"killed {name} (removed from queue)")
            for wid, info in list(running.items()):
                if info["name"] in killset:
                    a = info["assignment"]
                    quit_screen(a, user, pw)
                    terminal[info["name"]] = {"state": "killed", "error": "killed by user",
                                              "error_kind": "killed", "assignment": a}
                    cur_state[info["name"]] = "killed"
                    del running[wid]
                    ev(f"killed {info['name']} on {a['host']}")

        # (b) poll the still-running jobs; reap / requeue as needed.
        for wid, info in list(running.items()):
            name = info["name"]
            a = info["assignment"]
            state, err, kind, alive = poll_job(a, user, pw)
            if state in TERMINAL:
                if state == "error" and kind in RETRYABLE_KINDS and tries[name] < MAX_RETRY:
                    requeue(name, a, kind)
                    del running[wid]
                else:
                    if state == "error" and kind in RETRYABLE_KINDS:
                        err = f"{_reason(kind)} ×{tries[name]} (gave up)"
                    terminal[name] = {"state": state, "error": err, "error_kind": kind, "assignment": a}
                    cur_state[name] = state
                    del running[wid]
                    ev(f"reaped {name} -> {state}" + (f" ({err})" if err else "") + f"; freed {a['host']}")
            elif alive is False:
                # status not terminal but the screen is gone -> the driver was killed externally.
                if tries[name] < MAX_RETRY:
                    requeue(name, a, "killed")
                    del running[wid]
                else:
                    terminal[name] = {"state": "error", "error": f"killed ×{tries[name]} (gave up)",
                                      "error_kind": "killed", "assignment": a}
                    cur_state[name] = "error"
                    del running[wid]
                    ev(f"{name} killed on {a['host']} (retries exhausted)")
            elif state in ("running", "waiting"):
                cur_state[name] = state
            # state None + alive True/None -> driver starting or host blip; leave running.
        persist()

    def candidates(workers: list[Worker]) -> list[tuple[Worker, float]]:
        cands: list[tuple[Worker, float]] = []
        for w in workers:
            gpus = probe_gpus(w, user, pw)
            ok, _ = static_gate(w, gpus)
            if not ok:
                continue
            free = free_across(gpus, w.cuda)
            if free >= MIN_GPU_MEMORY:
                cands.append((w, free))
        cands.sort(key=lambda c: c[1], reverse=True)  # most free first
        return cands

    def place_on(cands: list[tuple[Worker, float]]) -> None:
        for w, free in cands:
            if not queue:
                break
            name = queue[0]
            try:
                assignment = deploy_job(w, job_by_name[name], run_id, user, pw)
            except Exception as e:  # noqa: BLE001 — skip this machine, retry the job next tick
                ev(f"deploy {name} -> {w.host} failed: {e}")
                continue
            queue.popleft()
            running[w.worker_id] = {"name": name, "assignment": assignment, "deployed_at": time.time()}
            cur_state[name] = "running"
            ev(f"placed {name} on [{w.tier}] {w.host}:{','.join(map(str, w.cuda))} (free {free:.0f}GB)")

    def place() -> None:
        idle = [w for w in pool if w.worker_id not in running]
        # Tier 1 — prefer SAFE_MACHINES.
        place_on(candidates([w for w in idle if w.tier == "safe"]))
        # Tier 2 — fall back to MACHINES only for jobs no safe machine could take.
        if queue:
            idle_fb = [w for w in idle if w.tier == "fallback" and w.worker_id not in running]
            place_on(candidates(idle_fb))
        persist()

    def row(rec: dict, now: float) -> tuple:
        name, state = rec["name"], rec["state"]
        machine = "—"
        if rec.get("host"):
            machine = f"{rec['host'].split('.')[0]}:{','.join(map(str, rec['gpus']))}"
        if state == "running":
            info = placed_by_name().get(name)
            detail = _fmt_elapsed(now - info["deployed_at"]) if info else "running"
        elif state == "waiting":
            detail = "for GPU memory"
        elif state == "queued":
            detail = "in wait queue"
        elif state == "done":
            detail = "finished"
        else:
            detail = rec.get("error") or state
        retries = ", ".join(f"{h['host']}({h['reason']})" for h in rec["history"])
        return (name, machine, state, detail, retries)

    persist()
    meta = {"run_id": run_id, "created": created}
    place_interval = POLL_INTERVAL_MIN * 60
    last_status = last_place = 0.0
    cancelled = False
    console = Console()
    try:
        with Live(console=console, screen=True, auto_refresh=False) as live:
            while True:
                now = time.time()
                do_place = (now - last_place) >= place_interval
                if do_place or (now - last_status) >= STATUS_POLL_SEC:
                    reap()                 # reap (and process kills) before placing -> frees machines
                    last_status = time.time()
                if do_place:
                    place()
                    last_place = time.time()
                now = time.time()
                rows = [row(rec, now) for rec in snapshot()]
                live.update(render_dashboard(meta, rows, list(events)))
                live.refresh()
                if not queue and not running:
                    break
                time.sleep(RENDER_INTERVAL_SEC)
    except KeyboardInterrupt:
        cancelled = True

    if cancelled:
        for name in list(queue):
            terminal[name] = {"state": "cancelled", "error": "scheduler stopped before placement",
                              "error_kind": None, "assignment": None}
            cur_state[name] = "cancelled"
        queue.clear()
    persist()

    n_done = sum(1 for r in snapshot() if r["state"] == "done")
    n_bad = sum(1 for r in snapshot() if r["state"] in ("error", "killed"))
    n_running = len(running)
    console.print("\n" + "=" * 80)
    if cancelled:
        console.print("[yellow]scheduler stopped.[/] Queued jobs cancelled; "
                      f"{n_running} placed job(s) keep running in their own screens.")
    else:
        console.print(f"all jobs settled: [green]{n_done} done[/], {n_bad} error/killed.")
    console.print("monitor:  python run_jobs/baulab/monitor.py")
    console.print("kill:     python run_jobs/baulab/kill.py --all   (or --names <job> ...)")
    console.print("=" * 80)


def print_runs(console: Console) -> None:
    runs = summarize_runs()
    if not runs:
        console.print("[dim]no batches found.[/]")
        return
    table = Table(box=SIMPLE_HEAVY, title="batch history (newest first)")
    table.add_column("run id", style="bold")
    table.add_column("submitted")
    table.add_column("jobs", justify="right")
    table.add_column("status")
    for r in runs:
        summary = "  ".join(f"[{STATE_STYLE.get(k, 'white')}]{k} {v}[/]" for k, v in r["counts"].items())
        table.add_row(r["run_id"], r["created"], str(r["n"]), summary)
    console.print(table)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", help="path to a JSON list of {name, command} jobs")
    ap.add_argument("--run-id", default=None, help="staging subdir name (default: timestamp)")
    ap.add_argument("--dry-run", action="store_true", help="probe + plan only; no scp / launch")
    ap.add_argument("--list", action="store_true", help="list past batches + run ids, then exit")
    args = ap.parse_args()

    use_real_terminal_size()
    if args.list:
        print_runs(Console())
        return
    if not args.jobs:
        ap.error("--jobs is required (or use --list)")

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    user, pw = load_credentials()
    jobs = load_jobs(args.jobs)

    # Two tiers: SAFE_MACHINES (preferred) then MACHINES, de-duping any host listed in both.
    safe_workers = [Worker(host=m.name, cuda=list(m.cuda), tier="safe") for m in SAFE_MACHINES]
    safe_ids = {w.worker_id for w in safe_workers}
    fb_workers = [Worker(host=m.name, cuda=list(m.cuda), tier="fallback")
                  for m in MACHINES if Worker(host=m.name, cuda=list(m.cuda)).worker_id not in safe_ids]
    workers = safe_workers + fb_workers

    print("=" * 80)
    print(f"run_id={run_id} | jobs={len(jobs)} | safe={len(safe_workers)} fallback={len(fb_workers)} | "
          f"MIN_GPU_MEMORY={MIN_GPU_MEMORY}GB | poll={POLL_INTERVAL_MIN}min | max_retry={MAX_RETRY}")
    print("=" * 80)

    # Static gate (GPU presence + capacity + reachability) — drops permanently-unusable machines.
    print("\nprobing machines (safe tier first)...")
    pool: list[Worker] = []
    for w in workers:
        gpus = probe_gpus(w, user, pw)
        ok, reason = static_gate(w, gpus)
        print(f"  {'✓' if ok else '✗ SKIP'}  [{w.tier:8}] {w.worker_id:<26} ({w.host}, gpus={w.cuda}): {reason}")
        if ok:
            pool.append(w)

    if not pool:
        print("\nno usable machines — nothing to schedule.")
        sys.exit(1)

    n_safe = sum(1 for w in pool if w.tier == "safe")
    print(f"\n{len(pool)} machines in the pool ({n_safe} safe, {len(pool) - n_safe} fallback); "
          f"{len(jobs)} jobs to place.")
    if args.dry_run:
        print("dry-run: would start the scheduler now (place onto free machines, queue the rest).")
        return

    schedule(run_id, jobs, pool, user, pw)


if __name__ == "__main__":
    main()
