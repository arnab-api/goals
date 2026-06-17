#!/usr/bin/env python
"""Self-contained per-worker job driver.

`dispatch.py` pushes a copy of this file (plus a `config.json`) to each worker and launches
it once inside a `screen` session. A *worker* is a single (host, GPU-set) pair — so the same
physical host can run several independent drivers, one per GPU group.

Each driver runs ONE serial queue of jobs on its fixed GPU set. Before launching a job it
polls `nvidia-smi` until the total FREE memory across its GPUs is at least
`min_gpu_memory_gb`, re-checking every `poll_interval_sec` (the naive run_monitor.py loop).
Every job command runs under bash with `set -o pipefail` and `CUDA_VISIBLE_DEVICES` pinned to
this worker's GPUs.

As it runs, the driver records each job's state (queued -> waiting -> running -> done|error)
in two places the monitor reads:
    * a machine-readable `status.json` next to this config (atomic writes), and
    * `@@CC_STATUS@@ {json}` marker lines on stdout (captured into `_driver.log`).
On failure it captures the job's output tail and classifies the error (CUDA OOM, a Python
`SomeError:` line, OOM-kill exit 137, generic non-zero, ...).

Config (JSON, passed via --config):
    {
      "run_id": "20260617_171530",
      "worker_id": "kyoto_gpu0-1",
      "host": "kyoto",
      "gpus": [0, 1],
      "min_gpu_memory_gb": 48,
      "poll_interval_sec": 600,
      "auto_continue": true,
      "jobs": [{"name": "...", "command": "..."}, ...]
    }

This module has NO third-party dependencies so it can run under any python on the remote.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

STATUS_MARKER = "@@CC_STATUS@@"


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


# ──────────────────────────────────────────────────────────────────────────────
# GPU memory.
# ──────────────────────────────────────────────────────────────────────────────


def gpu_free_memory_gb(gpus: list[int]) -> float:
    """Total FREE memory (GB) summed across the physical GPU indices in `gpus`."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        free_mb = [float(x.strip()) for x in result.stdout.splitlines() if x.strip()]
        return sum(free_mb[g] for g in gpus) / 1024.0
    except Exception as e:  # noqa: BLE001  (naive: treat any failure as "no memory yet")
        print(f"[{_now()}] error reading nvidia-smi: {e}", flush=True)
        return 0.0


def wait_for_gpu_memory(gpus: list[int], threshold_gb: float, poll_sec: int) -> None:
    """Block until FREE memory across `gpus` >= threshold, polling every `poll_sec`."""
    gpu_str = ",".join(map(str, gpus))
    while True:
        free = gpu_free_memory_gb(gpus)
        if free >= threshold_gb:
            print(
                f"[{_now()}] GPUs [{gpu_str}] have {free:.1f}GB free "
                f"(>= {threshold_gb}GB) — launching.",
                flush=True,
            )
            return
        print(
            f"[{_now()}] GPUs [{gpu_str}] free={free:.1f}GB < {threshold_gb}GB; "
            f"waiting {poll_sec / 60:.0f} min...",
            flush=True,
        )
        time.sleep(poll_sec)


# ──────────────────────────────────────────────────────────────────────────────
# Status tracking.
# ──────────────────────────────────────────────────────────────────────────────


class StatusBoard:
    """Per-worker status the monitor reads. Writes status.json atomically + emits markers."""

    def __init__(self, path: Path, run_id: str, worker_id: str, host: str, gpus: list[int],
                 jobs: list[dict]):
        self.path = path
        self.meta = {"run_id": run_id, "worker_id": worker_id, "host": host, "gpus": gpus}
        self.jobs = [
            {"seq": i + 1, "name": j["name"], "state": "queued",
             "error": None, "started": None, "ended": None}
            for i, j in enumerate(jobs)
        ]
        self.flush()

    def update(self, idx: int, *, state: str, error: str | None = None,
               started: float | None = None, ended: float | None = None) -> None:
        job = self.jobs[idx]
        job["state"] = state
        if error is not None:
            job["error"] = error
        if started is not None:
            job["started"] = started
        if ended is not None:
            job["ended"] = ended
        # marker line -> _driver.log (the monitor can fall back to grepping these).
        print(f"{STATUS_MARKER} " + json.dumps(
            {**self.meta, "seq": job["seq"], "name": job["name"],
             "state": state, "error": job["error"], "ts": time.time()}
        ), flush=True)
        self.flush()

    def flush(self) -> None:
        payload = {**self.meta, "updated": time.time(), "jobs": self.jobs}
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, self.path)  # atomic on POSIX


_ERR_LINE = re.compile(r"^[\w.]*(?:Error|Exception|Warning):.*$|^[\w.]*Error$")


def classify_error(tail: list[str], returncode: int) -> str:
    """A short human description of why a job failed, from its output tail + exit code."""
    text = "\n".join(tail)
    low = text.lower()
    if "out of memory" in low or "cuda out of memory" in low or "cublas" in low and "alloc" in low:
        return "CUDA OOM"
    if returncode == 137:
        return "killed (exit 137 — likely OOM/oom-killer)"
    if returncode == 139:
        return "segfault (exit 139)"
    # Last Python exception line, if any.
    for line in reversed(tail):
        s = line.strip()
        if _ERR_LINE.match(s):
            return (s[:120] + "…") if len(s) > 120 else s
    if "Traceback (most recent call last)" in text:
        return "Python traceback (see log)"
    return f"non-zero exit {returncode}"


# ──────────────────────────────────────────────────────────────────────────────
# Job execution (stream live to screen + capture a runlog for error extraction).
# ──────────────────────────────────────────────────────────────────────────────


def run_job(job: dict, gpus: list[int], runlog: Path, job_num: int, total: int) -> tuple[bool, str | None]:
    name = job["name"]
    command = job["command"]
    cuda = ",".join(map(str, gpus))

    print("\n" + "=" * 80, flush=True)
    print(f"[{_now()}] JOB {job_num}/{total}: {name}  (CUDA_VISIBLE_DEVICES={cuda})", flush=True)
    print(f"  {command}", flush=True)
    print("=" * 80, flush=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda
    # `set -o pipefail` so `python ... | tee` returns python's exit code, not tee's.
    full = f"set -o pipefail; {command}"

    tail: deque[str] = deque(maxlen=200)
    with open(runlog, "w") as lf:
        proc = subprocess.Popen(
            full, shell=True, executable="/bin/bash", env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        for line in proc.stdout:  # live: stream to screen AND capture
            sys.stdout.write(line)
            sys.stdout.flush()
            lf.write(line)
            tail.append(line.rstrip("\n"))
        proc.wait()

    if proc.returncode == 0:
        print(f"[{_now()}] ✓ job {job_num}/{total} ({name}) done.", flush=True)
        return True, None
    err = classify_error(list(tail), proc.returncode)
    print(f"[{_now()}] ✗ job {job_num}/{total} ({name}) failed: {err}", flush=True)
    return False, err


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="path to this worker's config.json")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    with open(cfg_path) as f:
        cfg = json.load(f)

    worker_id = cfg.get("worker_id", "worker")
    run_id = cfg.get("run_id", "run")
    host = cfg.get("host", "?")
    gpus = cfg["gpus"]
    threshold = float(cfg["min_gpu_memory_gb"])
    poll_sec = int(cfg.get("poll_interval_sec", 600))
    auto_continue = bool(cfg.get("auto_continue", True))
    jobs = cfg["jobs"]

    board = StatusBoard(cfg_path.parent / "status.json", run_id, worker_id, host, gpus, jobs)

    print(">>> worker driver <<<", flush=True)
    print(f"worker_id     : {worker_id}", flush=True)
    print(f"gpus          : {gpus}", flush=True)
    print(f"min_gpu_memory: {threshold}GB (total free across gpus)", flush=True)
    print(f"poll_interval : {poll_sec / 60:.0f} min", flush=True)
    print(f"jobs in queue : {len(jobs)}", flush=True)

    for i, job in enumerate(jobs):
        print(f"\n{'=' * 80}\n[{_now()}] preparing {i + 1}/{len(jobs)}: {job['name']}\n{'=' * 80}", flush=True)
        board.update(i, state="waiting")
        wait_for_gpu_memory(gpus, threshold, poll_sec)
        board.update(i, state="running", started=time.time())
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", job["name"])
        runlog = cfg_path.parent / f"job_{i + 1:02d}_{safe}.log"
        ok, err = run_job(job, gpus, runlog, i + 1, len(jobs))
        board.update(i, state="done" if ok else "error", error=err, ended=time.time())
        if not ok and not auto_continue:
            print(f"[{_now()}] stopping queue (auto_continue=False).", flush=True)
            break

    print(f"\n[{_now()}] queue '{worker_id}' finished ({len(jobs)} jobs).", flush=True)


if __name__ == "__main__":
    main()
