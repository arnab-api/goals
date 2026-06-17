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
3. Every POLL_INTERVAL_MIN the scheduler reaps finished jobs (freeing their machines), re-probes
   idle machines (catching memory freed by *other people's* jobs too), and places as many
   front-of-queue jobs as fit — until the queue drains. Naive polling, no remote agents.

The scheduler is **long-lived**: keep it running (a local screen/tmux/nohup) until the queue
drains. It can exit once everything is *placed* — running jobs finish on their own. Each
deployed job runs in its own detached `screen`, records `status.json` (read by monitor.py), and
the scheduler maintains a local job-centric manifest in `.remote_jobs/manifests/<run_id>.json`.

Usage
-----
    python run_jobs/baulab/dispatch.py --jobs run_jobs/baulab/jobs.example.json
    python run_jobs/baulab/dispatch.py --jobs my_jobs.json --dry-run   # probe + plan, no launch

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
from dataclasses import dataclass, field
from pathlib import Path

import yaml

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
    MachineConfig(name="andromeda", cuda=[0]),
    MachineConfig(name="kawasaki", cuda=[0]),
    MachineConfig(name="sendai", cuda=[0]),
    # MachineConfig(name="kumamoto", cuda=[0]),
    # MachineConfig(name="hamada", cuda=[0]),
    MachineConfig(name="fukuyama", cuda=[0]),
    MachineConfig(name="osaka", cuda=[0]),
    MachineConfig(name="cancun", cuda=[0]),
    MachineConfig(name="kameoka", cuda=[0]),
    MachineConfig(name="bippu", cuda=[0]),
    MachineConfig(name="hokkaido", cuda=[0]),
    MachineConfig(name="kyoto", cuda=[0, 1]),
    MachineConfig(name="tottori", cuda=[0]),
    # MachineConfig(name="nagoya.research.khoury.northeastern.edu", cuda=list(range(8))),
    # MachineConfig(name="hakone.research.khoury.northeastern.edu", cuda=list(range(8)))
]

# --- gates / behaviour --------------------------------------------------------
MIN_GPU_MEMORY = 48        # GB; a job needs this much FREE (and capacity) across a worker's GPUs
POLL_INTERVAL_MIN = 10     # scheduler reap + placement cadence
AUTO_CONTINUE = True       # (per-job driver flag; one job per worker so largely cosmetic)

# --- where things live --------------------------------------------------------
CONDA_ENV = "goals"
LOCAL_PROJECT_DIR = str(_PROJECT_ROOT)        # used when name == "local"
REMOTE_PROJECT_DIR = "~/Codes/Projects/goals"  # used on remote hosts (tilde ok)
REMOTE_STAGING_SUBDIR = ".remote_jobs"        # under the project dir on each host

# SSH hardening so automation never blocks on a prompt.
SSH_OPTS = "-o StrictHostKeyChecking=accept-new -o LogLevel=ERROR -o ConnectTimeout=15"

TERMINAL = {"done", "error"}


# ──────────────────────────────────────────────────────────────────────────────
# Worker model.
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Worker:
    host: str
    cuda: list[int]

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
    """Constraints independent of moment-to-moment free memory (GPU count + capacity)."""
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


def fetch_job_state(assignment: dict, user: str, pw: str) -> tuple[str | None, str | None]:
    """(state, error) of a deployed job from its status.json, or (None, None) if not readable."""
    path = assignment["status_path"]
    if assignment["is_local"]:
        try:
            data = json.loads(Path(os.path.expanduser(path)).read_text())
        except (OSError, json.JSONDecodeError):
            return None, None
    else:
        cmd = f"sshpass -p {shlex.quote(pw)} ssh {SSH_OPTS} {user}@{assignment['host']} cat {path}"
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable="/bin/bash")
        if res.returncode != 0 or not res.stdout.strip():
            return None, None
        try:
            data = json.loads(res.stdout)
        except json.JSONDecodeError:
            return None, None
    job = data["jobs"][0]
    return job["state"], job.get("error")


# ──────────────────────────────────────────────────────────────────────────────
# Local state (job-centric manifest the monitor reads).
# ──────────────────────────────────────────────────────────────────────────────


def manifests_dir() -> Path:
    return _PROJECT_ROOT / REMOTE_STAGING_SUBDIR / "manifests"


def write_state(run_id: str, created: str, all_jobs: list[dict], running: dict,
                done: list[dict], last_state: dict[str, str]) -> Path:
    """Persist every job's current placement + last-known state (always on the local box)."""
    running_by_name = {info["job"]["name"]: info for info in running.values()}
    done_by_name = {d["job"]["name"]: d for d in done}

    jobs_out = []
    for job in all_jobs:
        name = job["name"]
        if name in running_by_name:
            a = running_by_name[name]["assignment"]
            jobs_out.append({"name": name, "state": last_state.get(name, "running"),
                             "host": a["host"], "gpus": a["gpus"], "worker_id": a["worker_id"],
                             "is_local": a["is_local"], "status_path": a["status_path"]})
        elif name in done_by_name:
            a = done_by_name[name]["assignment"]
            jobs_out.append({"name": name, "state": done_by_name[name]["state"],
                             "host": a["host"], "gpus": a["gpus"], "worker_id": a["worker_id"],
                             "is_local": a["is_local"], "status_path": a["status_path"]})
        else:
            jobs_out.append({"name": name, "state": "queued", "host": None, "gpus": None,
                             "worker_id": None, "is_local": None, "status_path": None})

    manifest = {
        "run_id": run_id,
        "created": created,
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {"min_gpu_memory": MIN_GPU_MEMORY, "poll_interval_min": POLL_INTERVAL_MIN},
        "jobs": jobs_out,
    }
    out = manifests_dir() / f"{run_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    os.replace(tmp, out)
    return out


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
# Scheduler.
# ──────────────────────────────────────────────────────────────────────────────


def schedule(run_id: str, jobs: list[dict], pool: list[Worker], user: str, pw: str) -> None:
    created = time.strftime("%Y-%m-%d %H:%M:%S")
    queued = list(jobs)                       # FIFO wait queue
    running: dict[str, dict] = {}             # worker_id -> {job, assignment, deployed_at}
    done: list[dict] = []                     # {job, assignment, state}
    last_state: dict[str, str] = {}

    poll_sec = POLL_INTERVAL_MIN * 60
    tick = 0
    while queued:
        tick += 1
        # 1) reap finished jobs -> free their machines.
        for wid, info in list(running.items()):
            state, err = fetch_job_state(info["assignment"], user, pw)
            if state in TERMINAL:
                info["state"] = state
                info["error"] = err
                done.append(info)
                last_state[info["job"]["name"]] = state
                del running[wid]
                print(f"  [{time.strftime('%H:%M:%S')}] reaped {info['job']['name']} "
                      f"-> {state}{(' (' + err + ')') if err else ''}; freed {info['assignment']['host']}")
            elif state:
                last_state[info["job"]["name"]] = state

        # 2) probe idle machines, keep those that satisfy constraints + free now.
        idle = [w for w in pool if w.worker_id not in running]
        candidates: list[tuple[Worker, float]] = []
        for w in idle:
            gpus = probe_gpus(w, user, pw)
            ok, _ = static_gate(w, gpus)
            if not ok:
                continue
            free = free_across(gpus, w.cuda)
            if free >= MIN_GPU_MEMORY:
                candidates.append((w, free))
        candidates.sort(key=lambda c: c[1], reverse=True)  # most free first

        # 3) place front-of-queue jobs; the most-free machine takes the front job.
        for w, free in candidates:
            if not queued:
                break
            job = queued[0]
            try:
                assignment = deploy_job(w, job, run_id, user, pw)
            except Exception as e:  # noqa: BLE001 — skip this machine, retry job next tick
                print(f"  [{time.strftime('%H:%M:%S')}] deploy {job['name']} -> {w.host} failed: {e}")
                continue
            queued.pop(0)
            running[w.worker_id] = {"job": job, "assignment": assignment, "deployed_at": time.time()}
            last_state[job["name"]] = "running"
            print(f"  [{time.strftime('%H:%M:%S')}] placed {job['name']} on "
                  f"{w.host}:{','.join(map(str, w.cuda))} (free {free:.0f}GB)")

        write_state(run_id, created, jobs, running, done, last_state)
        print(f"  [tick {tick}] queued={len(queued)} running={len(running)} done={len(done)}")
        if not queued:
            break
        print(f"  ...all idle machines busy/insufficient; re-checking in {POLL_INTERVAL_MIN} min.")
        time.sleep(poll_sec)

    write_state(run_id, created, jobs, running, done, last_state)
    print("\n" + "=" * 80)
    print(f"all {len(jobs)} jobs placed. {len(running)} still running, {len(done)} finished.")
    print("monitor:  python run_jobs/baulab/monitor.py")
    print("scheduler can now exit; running jobs finish on their own.")
    print("=" * 80)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", required=True, help="path to a JSON list of {name, command} jobs")
    ap.add_argument("--run-id", default=None, help="staging subdir name (default: timestamp)")
    ap.add_argument("--dry-run", action="store_true", help="probe + plan only; no scp / launch")
    args = ap.parse_args()

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    user, pw = load_credentials()
    jobs = load_jobs(args.jobs)
    workers = [Worker(host=m.name, cuda=list(m.cuda)) for m in MACHINES]

    print("=" * 80)
    print(f"run_id={run_id} | jobs={len(jobs)} | machines={len(workers)} | "
          f"MIN_GPU_MEMORY={MIN_GPU_MEMORY}GB | poll={POLL_INTERVAL_MIN}min")
    print("=" * 80)

    # Static gate (GPU count + capacity + reachability) — drops permanently-unusable machines.
    print("\nprobing machines...")
    pool: list[Worker] = []
    for w in workers:
        gpus = probe_gpus(w, user, pw)
        ok, reason = static_gate(w, gpus)
        print(f"  {'✓' if ok else '✗ SKIP'}  {w.worker_id:<28} ({w.host}, gpus={w.cuda}): {reason}")
        if ok:
            pool.append(w)

    if not pool:
        print("\nno usable machines — nothing to schedule.")
        sys.exit(1)

    print(f"\n{len(pool)} machines in the pool; {len(jobs)} jobs to place.")
    if args.dry_run:
        print("dry-run: would start the scheduler now (place onto free machines, queue the rest).")
        return

    schedule(run_id, jobs, pool, user, pw)


if __name__ == "__main__":
    main()
