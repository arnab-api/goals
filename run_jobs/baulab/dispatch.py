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

Machines + scheduler tunables live in a local, gitignored CONFIG.py (copy CONFIG.example.py).
Credentials come from auth.yml next to this module (BAULAB_USER + BAULAB_PASS, or
ssh_login: true for passwordless ssh).
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
from dataclasses import dataclass
from pathlib import Path

import yaml
from rich.box import SIMPLE_HEAVY
from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ──────────────────────────────────────────────────────────────────────────────
# Per-client config comes from a local, gitignored CONFIG.py (each client edits its own).
# ──────────────────────────────────────────────────────────────────────────────
# The `MACHINES`/`SAFE_MACHINES` inventory AND the scheduler tunables below all live there,
# because different clients own different boxes and may want different gates/cadences. (The
# `MachineConfig` dataclass it uses lives in datatypes.py.) Copy the template on a new client:
#   cp run_jobs/baulab/CONFIG.example.py run_jobs/baulab/CONFIG.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from CONFIG import (  # (local, gitignored)
        AUTO_CONTINUE,
        CONDA_ENV,
        LOCAL_PROJECT_DIR,
        MACHINES,
        MAX_RETRY,
        MIN_GPU_MEMORY,
        POLL_INTERVAL_MIN,
        REMOTE_PROJECT_DIR,
        REMOTE_STAGING_SUBDIR,
        RENDER_INTERVAL_SEC,
        SAFE_MACHINES,
        STATUS_POLL_SEC,
    )
except ModuleNotFoundError as e:
    if e.name != "CONFIG":
        raise
    sys.exit(
        "run_jobs/baulab: no CONFIG.py found next to dispatch.py.\n"
        "  cp run_jobs/baulab/CONFIG.example.py run_jobs/baulab/CONFIG.py\n"
        "then edit it: this client's machines (MACHINES / SAFE_MACHINES) and scheduler tunables."
    )
except ImportError as e:
    sys.exit(
        f"run_jobs/baulab: CONFIG.py is missing a required setting — {e}.\n"
        "Re-sync it against CONFIG.example.py (it gained the scheduler tunables: MIN_GPU_MEMORY, "
        "POLL_INTERVAL_MIN, MAX_RETRY, AUTO_CONTINUE, STATUS_POLL_SEC, RENDER_INTERVAL_SEC, "
        "CONDA_ENV, LOCAL_PROJECT_DIR, REMOTE_PROJECT_DIR, REMOTE_STAGING_SUBDIR)."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Scheduler invariants (logic, not deployment config — keep here).
# ──────────────────────────────────────────────────────────────────────────────

# SSH hardening so automation never blocks on a prompt. RequestTTY=no + RemoteCommand=none
# neutralise any per-host `RequestTTY`/`RemoteCommand` directives in the user's ~/.ssh/config
# (which force an interactive login shell and make ssh refuse to run our passed command —
# "Cannot execute command-line and remote command").
SSH_OPTS = (
    "-o StrictHostKeyChecking=accept-new -o LogLevel=ERROR -o ConnectTimeout=15 "
    "-o RequestTTY=no -o RemoteCommand=none"
)

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


# Auth lives next to this module so run_jobs/baulab is self-contained (no dependency on
# the project-root env.yml). auth.yml is gitignored; see auth.example.yml for the template.
_AUTH_PATH = Path(__file__).resolve().parent / "auth.yml"


def load_credentials() -> tuple[str | None, str | None]:
    """Read BAULAB auth from run_jobs/baulab/auth.yml.

    Returns (user, pw). When `ssh_login: true`, `pw` is None and remote exec uses plain
    (passwordless) ssh — i.e. key/agent-based login must be configured for every target
    host, and BAULAB_PASS is not needed at all. Otherwise `pw` is BAULAB_PASS (sshpass).
    """
    if not _AUTH_PATH.exists():
        print(
            f"WARNING: {_AUTH_PATH} not found — only the 'local' worker will work "
            f"(remote SSH/scp will fail). Copy auth.example.yml -> auth.yml and fill it in.",
            file=sys.stderr,
        )
        return None, None
    with open(_AUTH_PATH) as f:
        cfg = yaml.safe_load(f) or {}
    user = cfg.get("BAULAB_USER")
    ssh_login = bool(cfg.get("ssh_login", False))
    pw = None if ssh_login else cfg.get("BAULAB_PASS")
    if not user:
        print(
            "WARNING: BAULAB_USER not set in auth.yml — only the 'local' worker will work "
            "(remote SSH/scp will fail).",
            file=sys.stderr,
        )
    elif not ssh_login and not pw:
        print(
            "WARNING: BAULAB_PASS not set and ssh_login is false in auth.yml — remote "
            "SSH/scp will fail. Either set ssh_login: true (passwordless ssh configured "
            "for all target hosts) or provide BAULAB_PASS.",
            file=sys.stderr,
        )
    return user, pw


# ──────────────────────────────────────────────────────────────────────────────
# Remote exec helpers (sshpass for remote, plain subprocess for local).
# ──────────────────────────────────────────────────────────────────────────────


def _ssh_prefix(user: str, pw: str | None, host: str) -> str:
    # pw is None when auth.yml sets ssh_login: true → rely on passwordless (key/agent) ssh.
    if pw:
        return f"sshpass -p {shlex.quote(pw)} ssh {SSH_OPTS} {user}@{host}"
    return f"ssh {SSH_OPTS} {user}@{host}"


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
    scp_bin = f"sshpass -p {shlex.quote(pw)} scp" if pw else "scp"
    scp = (
        f"{scp_bin} {SSH_OPTS} "
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


_SUMMARY_TERMINAL = {"done", "error", "killed", "cancelled"}  # trusted as-is; never re-probed


def _run_on_host(host_key: str, cmd: str, host: str, user: str | None, pw: str | None):
    """Run cmd locally (host_key=='LOCAL') or via ssh; returns CompletedProcess."""
    if host_key == "LOCAL":
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, executable="/bin/bash")
    return subprocess.run(f"{_ssh_prefix(user, pw, host)} {shlex.quote(cmd)}",
                          shell=True, capture_output=True, text=True, executable="/bin/bash")


def _probe_placed(jobs: list[dict], user: str | None, pw: str | None) -> dict[str, tuple[str | None, bool | None]]:
    """Per placed job (keyed by status_path) -> (status_json_state, alive). Batches ssh per host.

    Signals, in priority order:
      * status.json state — per-run, authoritative for what the driver last wrote (corrects a
        manifest that lagged behind an error/done).
      * alive — liveness for a status.json still frozen at 'running' (hard death). Uses the driver
        `pid` recorded in status.json (unique per run -> collision-proof) via `kill -0`; falls back
        to the screen session for OLD runs whose driver never recorded a pid. The screen *name* is
        the job name (reused across runs), so screen-only liveness is ambiguous — hence pid first."""
    by_host: dict[str, list[dict]] = {}
    for j in jobs:
        by_host.setdefault("LOCAL" if j.get("is_local") else j.get("host"), []).append(j)

    result: dict[str, tuple[str | None, bool | None]] = {}
    for key, hjobs in by_host.items():
        if not key:
            continue
        host = hjobs[0].get("host")
        # pass 1: read every status.json on this host + one screen -ls
        parts = [f"echo '<<<F:{i}>>>'; cat {j['status_path']} 2>/dev/null" for i, j in enumerate(hjobs)]
        inner = "; ".join(parts) + "; echo '<<<SCREENS>>>'; screen -ls 2>/dev/null || true"
        res = _run_on_host(key, inner, host, user, pw)
        if key != "LOCAL" and res.returncode == 255:  # host unreachable -> unknown for all its jobs
            for j in hjobs:
                result[j["status_path"]] = (None, None)
            continue
        body, _, screen_txt = res.stdout.partition("<<<SCREENS>>>")
        chunks = re.split(r"<<<F:(\d+)>>>", body)  # [pre, idx, content, idx, content, ...]
        contents = {int(chunks[i]): chunks[i + 1] for i in range(1, len(chunks) - 1, 2)}

        states: dict[int, str | None] = {}
        pids: dict[int, int | None] = {}
        for i in range(len(hjobs)):
            try:
                data = json.loads(contents.get(i, ""))
                states[i] = data["jobs"][0]["state"]
                pids[i] = int(data["pid"]) if data.get("pid") is not None else None
            except (json.JSONDecodeError, KeyError, IndexError, ValueError, TypeError):
                states[i], pids[i] = None, None

        # pass 2 (only if any pids to check): kill -0 them in one shot -> set of live pids
        live_pids: set[int] = set()
        want = sorted({p for p in pids.values() if p})
        if want:
            probe = "; ".join(f"kill -0 {p} 2>/dev/null && echo {p}" for p in want)
            live_pids = {int(x) for x in _run_on_host(key, probe, host, user, pw).stdout.split()}

        for i, j in enumerate(hjobs):
            scr = j.get("screen") or ""
            if pids[i]:  # collision-proof: this run's own driver pid
                alive = pids[i] in live_pids
            elif scr:    # old run (no pid): best-effort screen-name liveness
                alive = bool(re.search(rf"\.{re.escape(scr)}\s", screen_txt))
            else:
                alive = None
            result[j["status_path"]] = (states[i], alive)
    return result


def summarize_runs(probe: bool = False, user: str | None = None, pw: str | None = None) -> list[dict]:
    """Newest-first summary of every batch with a manifest (for `--list` / kill.py).

    probe=True: for jobs the manifest still shows non-terminal, read the authoritative per-run
    status.json (correcting a manifest that lagged behind an error/done) and, if that too is still
    'running' but the screen session is gone, count it as `stale` — so hard-killed / rebooted runs
    stop showing a phantom `running`. Batched one ssh per host, so --list stays fast."""
    d = manifests_dir()
    runs: list[dict] = []
    if not d.exists():
        return runs
    manifests = []
    for p in d.glob("*.json"):
        try:
            m = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        m.setdefault("run_id", p.stem)
        manifests.append(m)
    # newest first by run_id (== submission timestamp YYYYMMDD_HHMMSS, lexicographically sortable) —
    # NOT file mtime, which any manifest rewrite (a state update / manual reconcile) would disturb.
    manifests.sort(key=lambda m: m["run_id"], reverse=True)

    probed: dict[str, tuple[str | None, bool | None]] = {}
    if probe:
        placed = [j for m in manifests for j in m.get("jobs", [])
                  if j.get("status_path") and j.get("state") not in _SUMMARY_TERMINAL]
        probed = _probe_placed(placed, user, pw)

    for m in manifests:
        counts: dict[str, int] = {}
        for j in m.get("jobs", []):
            state = j.get("state", "?")
            if probe and j.get("status_path") and state not in _SUMMARY_TERMINAL:
                sjson_state, alive = probed.get(j["status_path"], (None, None))
                state = sjson_state or state  # status.json is authoritative when readable
                if state not in _SUMMARY_TERMINAL and alive is False:
                    state = "stale"  # still 'running' but its screen session is gone -> phantom
            counts[state] = counts.get(state, 0) + 1
        runs.append({"run_id": m.get("run_id", "?"), "created": m.get("created", "?"),
                     "updated": m.get("updated", "?"), "n": len(m.get("jobs", [])), "counts": counts})
    return runs


def job_deps(job: dict) -> list[str]:
    """The job's dependency names. Missing or empty 'dependencies' both mean 'no deps'."""
    deps = job.get("dependencies") or []
    if not isinstance(deps, list) or not all(isinstance(d, str) for d in deps):
        raise ValueError(f"job {job.get('name')!r}: 'dependencies' must be a list of job names")
    return deps


def dep_cancel_closure(
    order: list[str], deps_of: dict[str, list[str]], terminal_state: dict[str, str | None]
) -> dict[str, str]:
    """Jobs that must be cancelled because a dependency reached a non-'done' terminal state.

    Returns {job_name: offending_dependency}. Transitive: a job cancelled here counts as a
    non-'done' terminal for *its* dependents, so whole chains collapse. `terminal_state` maps
    each job to its current terminal state (or None if not terminal). Pure / side-effect free.
    """
    bad = {n for n in order if terminal_state.get(n) in (TERMINAL - {"done"})}
    result: dict[str, str] = {}
    changed = True
    while changed:
        changed = False
        for name in order:
            if terminal_state.get(name) is not None or name in result:
                continue
            dead = next((d for d in deps_of[name] if d in bad or d in result), None)
            if dead is not None:
                result[name] = dead
                bad.add(name)        # now also "failed" for the benefit of its own dependents
                changed = True
    return result


ALLOWED_JOB_FIELDS = {"name", "command", "dependencies"}


def _find_dependency_cycles(deps_of: dict[str, list[str]]) -> list[list[str]]:
    """All distinct dependency cycles (each as a path ``a -> b -> ... -> a``).

    Only traverses edges whose target exists in `deps_of`; unknown-name edges are reported
    separately. Each cycle (by its node set) is reported once.
    """
    WHITE, GREY, BLACK = 0, 1, 2
    color = {n: WHITE for n in deps_of}
    cycles: list[list[str]] = []
    seen: set[frozenset[str]] = set()

    def visit(n: str, stack: list[str]) -> None:
        color[n] = GREY
        for d in deps_of[n]:
            if d not in color:
                continue                       # unknown dep — reported elsewhere
            if color[d] == GREY:
                cyc = stack[stack.index(d):] + [d]
                key = frozenset(cyc[:-1])
                if key not in seen:
                    seen.add(key)
                    cycles.append(cyc)
            elif color[d] == WHITE:
                visit(d, stack + [d])
        color[n] = BLACK

    for n in deps_of:
        if color[n] == WHITE:
            visit(n, [n])
    return cycles


def validate_jobs(jobs: object) -> list[str]:
    """Return a list of integrity errors for a parsed jobs list — empty means valid.

    Checks: top-level is a non-empty list; each job is an object with non-empty string
    `name` (unique) and `command`; no unknown fields (typo guard); `dependencies` is a list
    of known job names with no self-edges, no duplicates, and no cycles (deadlocks).
    Collects *all* problems rather than failing on the first.
    """
    errors: list[str] = []
    if not isinstance(jobs, list):
        return [f"top-level JSON must be a list of job objects (got {type(jobs).__name__})"]
    if not jobs:
        return ["job list is empty — nothing to run"]

    names: set[str] = set()
    for i, j in enumerate(jobs):
        where = f"job #{i}"
        if not isinstance(j, dict):
            errors.append(f"{where}: must be a JSON object, got {type(j).__name__}")
            continue
        name = j.get("name")
        if isinstance(name, str) and name:
            where = f"job #{i} ({name!r})"
        # name
        if "name" not in j:
            errors.append(f"{where}: missing required field 'name'")
        elif not isinstance(name, str) or not name.strip():
            errors.append(f"{where}: 'name' must be a non-empty string")
        elif name in names:
            errors.append(f"{where}: duplicate job name {name!r} (names must be unique)")
        else:
            names.add(name)
        # command
        if "command" not in j:
            errors.append(f"{where}: missing required field 'command'")
        elif not isinstance(j["command"], str) or not j["command"].strip():
            errors.append(f"{where}: 'command' must be a non-empty string")
        # unknown fields (typo guard — no fuzzy resolution, just reject)
        unknown = set(j) - ALLOWED_JOB_FIELDS
        if unknown:
            errors.append(f"{where}: unknown field(s) {sorted(unknown)} "
                          f"(allowed: {sorted(ALLOWED_JOB_FIELDS)})")
        # dependencies type
        if "dependencies" in j:
            deps = j["dependencies"]
            if not isinstance(deps, list) or not all(isinstance(d, str) for d in deps):
                errors.append(f"{where}: 'dependencies' must be a list of job-name strings")

    # dependency-reference checks (only over well-formed, named jobs).
    deps_of: dict[str, list[str]] = {}
    for j in jobs:
        if not isinstance(j, dict):
            continue
        name = j.get("name")
        if not isinstance(name, str) or not name or name not in names:
            continue
        deps = j.get("dependencies")
        if deps is None:
            deps_of[name] = []
            continue
        if not isinstance(deps, list) or not all(isinstance(d, str) for d in deps):
            continue                            # type error already reported
        dups = sorted({d for d in deps if deps.count(d) > 1})
        if dups:
            errors.append(f"job {name!r}: duplicate entries in 'dependencies': {dups}")
        for d in deps:
            if d == name:
                errors.append(f"job {name!r}: depends on itself (deadlock)")
            elif d not in names:
                errors.append(f"job {name!r}: depends on unknown job {d!r}")
        deps_of[name] = [d for d in dict.fromkeys(deps) if d in names and d != name]

    for cyc in _find_dependency_cycles(deps_of):
        errors.append(f"dependency cycle (deadlock): {' -> '.join(cyc)}")
    return errors


def load_jobs(path: str) -> list[dict]:
    with open(path) as f:
        jobs = json.load(f)
    errors = validate_jobs(jobs)
    if errors:
        bullet = "\n  - ".join(errors)
        raise ValueError(f"{path}: {len(errors)} integrity error(s):\n  - {bullet}")
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
    "stale": "bright_black",  # monitor-derived: status.json says live but the driver pid is dead
    "no-status": "magenta",
}
_STATE_ORDER = ["running", "waiting", "queued", "done", "error", "killed", "cancelled", "stale", "no-status"]


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

    deps_of: dict[str, list[str]] = {j["name"]: job_deps(j) for j in jobs}   # name -> deps it waits on

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

    def deps_ready(name: str) -> bool:
        """True iff every dependency has finished successfully (terminal state 'done')."""
        return all(terminal.get(d, {}).get("state") == "done" for d in deps_of[name])

    def blocked_deps(name: str) -> list[str]:
        """Dependencies of `name` that have not (yet) succeeded — for display."""
        return [d for d in deps_of[name] if terminal.get(d, {}).get("state") != "done"]

    def propagate_cancellations() -> None:
        """Cancel any pending job whose dependency reached a non-'done' terminal state
        (transitively). A running dependent is unexpected — its deps were all done — but if
        one exists (e.g. via direct kill of a sibling) its screen is torn down too."""
        tstate = {n: terminal.get(n, {}).get("state") for n in order}
        for name, dead in dep_cancel_closure(order, deps_of, tstate).items():
            dstate = terminal[dead]["state"] if dead in terminal else "cancelled"
            if name in queue:
                queue.remove(name)
            for wid, info in list(running.items()):
                if info["name"] == name:
                    quit_screen(info["assignment"], user, pw)
                    del running[wid]
            terminal[name] = {"state": "cancelled",
                              "error": f"dependency {dead!r} {dstate}",
                              "error_kind": "dep_failed", "assignment": None}
            cur_state[name] = "cancelled"
            ev(f"cancelled {name} (dependency {dead} {dstate})")

    def snapshot() -> list[dict]:
        placed = placed_by_name()
        out = []
        for name in order:
            rec = {"name": name, "try_count": tries[name],
                   "history": [dict(h) for h in history[name]],
                   "dependencies": list(deps_of[name])}
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
                           is_local=None, status_path=None, screen=None,
                           blocked_on=blocked_deps(name))
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
        # cascade: any job whose dependency just failed/was-killed gets cancelled (transitively).
        propagate_cancellations()
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

    def next_eligible(skip: set[str]) -> str | None:
        """First queued job (FIFO order) whose dependencies are all done, ignoring `skip`."""
        for name in queue:
            if name not in skip and deps_ready(name):
                return name
        return None

    def place_on(cands: list[tuple[Worker, float]]) -> None:
        failed: set[str] = set()   # deploy failed this tick -> don't retry on another worker now
        for w, free in cands:
            name = next_eligible(failed)
            if name is None:        # nothing both eligible and not-yet-failed left to place
                break
            try:
                assignment = deploy_job(w, job_by_name[name], run_id, user, pw)
            except Exception as e:  # noqa: BLE001 — skip this machine, retry the job next tick
                ev(f"deploy {name} -> {w.host} failed: {e}")
                failed.add(name)
                continue
            queue.remove(name)
            running[w.worker_id] = {"name": name, "assignment": assignment, "deployed_at": time.time()}
            cur_state[name] = "running"
            ev(f"placed {name} on [{w.tier}] {w.host}:{','.join(map(str, w.cuda))} (free {free:.0f}GB)")

    def place() -> None:
        if next_eligible(set()) is None:   # all queued jobs (if any) blocked on deps — don't probe
            persist()
            return
        idle = [w for w in pool if w.worker_id not in running]
        # Tier 1 — prefer SAFE_MACHINES.
        place_on(candidates([w for w in idle if w.tier == "safe"]))
        # Tier 2 — fall back to MACHINES only for jobs no safe machine could take.
        if next_eligible(set()) is not None:
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
            blocked = rec.get("blocked_on") or []
            detail = f"waiting on: {', '.join(blocked)}" if blocked else "in wait queue"
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


def print_runs(console: Console, probe: bool = False,
               user: str | None = None, pw: str | None = None) -> None:
    runs = summarize_runs(probe=probe, user=user, pw=pw)
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

    # Integrity-check the jobs file first — refuse to run *anything* if it doesn't pass.
    try:
        jobs = load_jobs(args.jobs)
    except (ValueError, json.JSONDecodeError) as e:
        print(f"\njobs file failed integrity check — nothing was launched.\n{e}", file=sys.stderr)
        print("\n(run `python run_jobs/baulab/check_jobs.py <file>` to re-check.)", file=sys.stderr)
        sys.exit(1)

    # Two tiers: SAFE_MACHINES (preferred) then MACHINES, de-duping any host listed in both.
    safe_workers = [Worker(host=m.name, cuda=list(m.cuda), tier="safe") for m in SAFE_MACHINES]
    safe_ids = {w.worker_id for w in safe_workers}
    fb_workers = [Worker(host=m.name, cuda=list(m.cuda), tier="fallback")
                  for m in MACHINES if Worker(host=m.name, cuda=list(m.cuda)).worker_id not in safe_ids]
    workers = safe_workers + fb_workers

    print("=" * 80)
    print(f"run_id={run_id} | jobs={len(jobs)} | safe={len(safe_workers)} fallback={len(fb_workers)} | "
          f"MIN_GPU_MEMORY={MIN_GPU_MEMORY}GB | poll={POLL_INTERVAL_MIN}min | max_retry={MAX_RETRY}")
    deps_listed = [(j["name"], job_deps(j)) for j in jobs if job_deps(j)]
    if deps_listed:
        print("dependencies:")
        for name, deps in deps_listed:
            print(f"  {name}  <-  {', '.join(deps)}")
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
