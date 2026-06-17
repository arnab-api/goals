# baulab job dispatcher

Schedule a list of jobs across baulab machines with a **central greedy scheduler**. A *worker*
is a `(host, GPU-set)` entry that runs at most **one** of our jobs at a time. A global FIFO wait
queue holds all jobs; the scheduler keeps placing the front job onto the **currently best free
machine**, so a job never sits behind a busy machine when an idle one exists.

```
dispatch.py   scheduler (long-lived): machines + config at top; reads jobs from a JSON file,
              and keeps placing front-of-queue jobs onto satisfying machines, polling every
              POLL_INTERVAL_MIN. Writes a local job-centric manifest.
driver.py     per-job driver pushed to the chosen host (no third-party deps): runs one job,
              records status (status.json + @@CC_STATUS@@ log markers), classifies errors.
monitor.py    live terminal monitor: reads the latest manifest, polls each placed job's status
              (ssh for remote, file read for local) and renders a refreshing table.
```

## 1. Define jobs (your responsibility)

A JSON list of `{name, command}` with **unique names**. **Do not put `CUDA_VISIBLE_DEVICES` in
the command** — the driver pins it to the chosen machine's GPUs. Commands run from the project
dir, so `|& tee logs/...` paths are relative to it.

```json
[
  {"name": "qwen14b_cma", "command": "python -m scripts.clock.sweep_multiline_patch --model=\"Qwen/Qwen2.5-Coder-14B-Instruct\" --n-pairs=100 |& tee logs/cma14.log"}
]
```

See `jobs.example.json`.

## 2. Define machines + config (edit `dispatch.py`)

```python
MACHINES = [
    MachineConfig("kyoto", cuda=[0, 1]),   # one job at a time, on GPUs 0,1
    MachineConfig("kyoto", cuda=[2, 3]),   # same host, independent slot (runs concurrently)
    MachineConfig("local", cuda=[0]),
]

MIN_GPU_MEMORY  = 48    # GB; a job needs this much FREE (and capacity) across a machine's GPUs
POLL_INTERVAL_MIN = 10  # scheduler reap + placement cadence
```

A machine **satisfies the constraints** for a job when:
- total *capacity* across its specified GPUs ≥ `MIN_GPU_MEMORY` (nominal-rounded, so real 48GB cards pass);
- **free memory now ≥ `MIN_GPU_MEMORY`**;
- it isn't already running one of our jobs.

Among satisfying machines, the one with the **most free memory** wins. Jobs that don't fit stay
in the queue and are retried every `POLL_INTERVAL_MIN` (which also catches machines freed by
*other people's* jobs ending). One job per entry, using all its GPUs — to run several jobs on
one host concurrently, add multiple entries (`cuda=[0]`, `cuda=[1]`, ...).

Credentials (`BAULAB_USER`, `BAULAB_PASS`) are read from `env.yml`. Needs `sshpass` + `screen`
on the local box and `screen` on each remote.

## 3. Run the scheduler

```bash
# plan only — static-gate the machines, no launch
python run_jobs/baulab/dispatch.py --jobs run_jobs/baulab/jobs.example.json --dry-run

# for real — long-lived; keep it alive until the queue drains (run it in a screen/tmux/nohup)
screen -dmS scheduler python run_jobs/baulab/dispatch.py --jobs my_jobs.json
```

The scheduler is **long-lived**: it keeps polling/placing until every job has been placed, then
exits (already-running jobs finish on their own). Each placed job runs in its own detached
`screen` named after the job.

## 4. Monitor

```bash
python run_jobs/baulab/monitor.py                  # latest batch; exits when all jobs finish
python run_jobs/baulab/monitor.py --run-id <id>    # a specific batch
python run_jobs/baulab/monitor.py --interval 5 --watch   # keep refreshing after finish
```

A refreshing table of **job · machine · status · detail**. Statuses:

| status      | meaning                                            |
|-------------|----------------------------------------------------|
| `queued`    | still in the wait queue (not placed yet)           |
| `waiting`   | placed; driver waiting on GPU memory (safety net)  |
| `running`   | executing now (detail = elapsed)                   |
| `done`      | finished cleanly (detail = wall time)              |
| `error`     | failed; detail = brief classification (CUDA OOM, a Python `SomeError:` line, exit 137, ...) |
| `no-status` | placed but status not readable yet (driver starting) / host unreachable |

The monitor reads `.remote_jobs/manifests/<run_id>.json` (written by the scheduler) for the job
list + assignments, then `ssh <host> cat <status_path>` for each placed job (local jobs read
directly). Drivers write `status.json` atomically, so reads never see a half-written file.

## 5. Manage by hand

```bash
ssh <host>; screen -ls; screen -r <job_name>          # attach to a job (Ctrl-A D to detach)
# driver log: <project>/.remote_jobs/<run_id>/jobs/<job_name>/_driver.log
```

Staged files per job live in `<project>/.remote_jobs/<run_id>/jobs/<job_name>/`
(`driver.py`, `config.json`, `run.sh`, `status.json`, `_driver.log`, `job_NN_<name>.log`).
