# baulab job dispatcher

Schedule a list of jobs across baulab machines with a **central greedy scheduler**. A *worker*
is a `(host, GPU-set)` entry that runs at most **one** of our jobs at a time. A global FIFO wait
queue holds all jobs; the scheduler keeps placing the front job onto the **currently best free
machine**, so a job never sits behind a busy machine when an idle one exists.

```
dispatch.py   scheduler (long-lived): machines + config at top; reads jobs from a JSON file,
              keeps placing front-of-queue jobs onto satisfying machines, reaps finished ones,
              and auto-requeues OOM / externally-killed jobs. Renders a live dashboard right in
              its own terminal and writes a local job-centric manifest.
driver.py     per-job driver pushed to the chosen host (no third-party deps): runs one job,
              records status (status.json + @@CC_STATUS@@ log markers), classifies errors into
              kinds (oom / killed / other) that drive the requeue policy.
monitor.py    live terminal monitor: reads a manifest, polls each placed job's status (ssh for
              remote, file read for local) and renders a refreshing table. Works even after the
              scheduler exits. `--list` shows past batches.
kill.py       kill jobs by name (`--names`) or a whole batch (`--all`). Killed jobs are NOT
              auto-requeued.
setup.sh      one-time setup: installs the CLI tools (sshpass, screen) into the conda env and
              seeds CONFIG.py from the template.
CONFIG.py     per-client config — machine inventory (MACHINES + SAFE_MACHINES) AND scheduler
              tunables (MIN_GPU_MEMORY, POLL_INTERVAL_MIN, …); gitignored, copied from
              CONFIG.example.py. (Credentials are separate: auth.yml.)
```

## 0. Setup (once per client)

```bash
bash run_jobs/baulab/setup.sh          # or: ENV_NAME=myenv bash run_jobs/baulab/setup.sh
```

This installs **`sshpass`** (feed the ssh password non-interactively) and **`screen`** (detached
per-job sessions) into the `goals` conda env, and copies `CONFIG.example.py` → `CONFIG.py` and
`auth.example.yml` → `auth.yml` if you don't have them yet. `sshpass` is only needed on the box you
launch the scheduler from (and only when `ssh_login: false`); `screen` must also exist on every
remote worker. Then edit `CONFIG.py` (step 2) and fill in `auth.yml`.

**Auth (`run_jobs/baulab/auth.yml`, gitignored).** Credentials live next to the module so it's
self-contained — nothing is read from the project-root `env.yml`. Two modes:
- `ssh_login: false` (default): set `BAULAB_USER` + `BAULAB_PASS`; remote exec uses `sshpass`.
- `ssh_login: true`: set `BAULAB_USER` only. Remote exec uses plain passwordless `ssh`
  (key/agent-based login must already work for **every** target host — test `ssh <host> true`).
  `BAULAB_PASS` is then ignored and `sshpass` is not needed at all.

> **ssh-config gotcha.** If your `~/.ssh/config` sets `RequestTTY yes` + `RemoteCommand ...` for a
> host (forcing an interactive login shell), plain `ssh <host> <cmd>` fails with *"Cannot execute
> command-line and remote command."* The dispatcher already neutralises this with
> `-o RequestTTY=no -o RemoteCommand=none` in `SSH_OPTS`, so such hosts work regardless. (For the
> box you're *running on*, prefer `name="local"` — it skips ssh entirely.)

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

### Dependencies (optional)

A job may declare `"dependencies": [<job names>]` — it won't be placed until **every** listed
dependency has finished **successfully** (`done`). A missing or empty `dependencies` field means
"no dependencies". Jobs without (satisfied) deps are placed in list order, FIFO; a job blocked on
deps is simply skipped over until its deps complete, so it never stalls independent jobs.

```json
[
  {"name": "build_data", "command": "..."},
  {"name": "train",    "command": "...", "dependencies": ["build_data"]},
  {"name": "cma",      "command": "...", "dependencies": []},
  {"name": "clean_up", "command": "...", "dependencies": ["build_data", "train", "cma"]}
]
```

If a dependency ends in any non-`done` terminal state — `error`, `killed` (via `kill.py` or after
exhausting `MAX_RETRY`), or `cancelled` — all of its dependents are **cancelled transitively**
(state `cancelled`, detail `dependency '<dep>' <state>`) and never run. Dependency names are
validated up front: unknown names, self-edges, and cycles are rejected before anything launches.
The dashboard shows a blocked job's outstanding deps as `waiting on: <deps>`.

### Integrity check

`dispatch.py` validates the jobs file before launching anything and refuses to start if it
doesn't pass. You can run the same check ahead of time (model-free, instant):

```bash
python run_jobs/baulab/check_jobs.py job_list/my_jobs.json
```

It reports **all** problems at once (exit 1 if any), and on success prints the dependency
*execution waves*. Checks: top-level is a non-empty list of objects; every job has a non-empty
unique `name` and a `command`; **no unknown fields** (typo guard — only `name`, `command`,
`dependencies` are allowed); every `dependencies` entry names a real job, with no self-edges,
duplicates, or cycles (deadlocks).

## 2. Define machines + config

Machines **and** the scheduler tunables live in a **local, gitignored `CONFIG.py`** — different
clients own different boxes and may want different gates/cadences, so this is per-client and not
version-controlled. Copy the template once:

```bash
cp run_jobs/baulab/CONFIG.example.py run_jobs/baulab/CONFIG.py
```

then edit `CONFIG.py`:

```python
SAFE_MACHINES = [          # preferred tier — tried FIRST for every job (e.g. machines you own)
    MachineConfig("umibozu", cuda=[0]),
]

MACHINES = [               # fallback tier — used only when no safe machine currently fits
    MachineConfig("kyoto", cuda=[0, 1]),   # one job at a time, on GPUs 0,1
    MachineConfig("kyoto", cuda=[2, 3]),   # same host, independent slot (runs concurrently)
    MachineConfig("local", cuda=[0]),
]

# ── scheduler tunables (same file) ──
MIN_GPU_MEMORY    = 45   # GB; a job needs this much FREE (and capacity) across a machine's GPUs
POLL_INTERVAL_MIN = 10   # idle-machine probe + placement cadence
MAX_RETRY         = 5    # auto-requeue attempts for OOM / externally-killed jobs
# … plus AUTO_CONTINUE, STATUS_POLL_SEC, RENDER_INTERVAL_SEC, CONDA_ENV, *_PROJECT_DIR, etc.
```

`dispatch.py` imports the machine lists and the tunables from `CONFIG.py` (and exits with a
copy-the-template hint if it's missing, or names the missing setting if `CONFIG.py` is stale).
Only the scheduler *invariants* (`SSH_OPTS`, retry/terminal state sets) stay in `dispatch.py`.

### Machine tiers

The scheduler tries the **`SAFE_MACHINES`** tier first for every job, and only falls back to
**`MACHINES`** for jobs that no safe machine can currently take. Use the safe tier for boxes you
own (where your jobs won't be pre-empted by an owner reclaiming the machine); the fallback tier is
where the OOM / killed **auto-requeue** earns its keep. You can list a host in `SAFE_MACHINES`
even if it's also in `MACHINES` — duplicates are de-duped into the safe tier, so you don't have to
edit `MACHINES`. An empty `SAFE_MACHINES` just means "treat every machine equally".

A machine **satisfies the constraints** for a job when:
- total *capacity* across its specified GPUs ≥ `MIN_GPU_MEMORY` (nominal-rounded, so real 48GB cards pass);
- **free memory now ≥ `MIN_GPU_MEMORY`**;
- it isn't already running one of our jobs.

> ⚠️ The free-memory check is **exact** (only capacity is rounded). An *idle* 48 GB card reports
> ~47.4 GB free (driver/ECC overhead), so `MIN_GPU_MEMORY = 48` will never place onto a single
> 48 GB card. Set `MIN_GPU_MEMORY` to your job's real footprint (e.g. 40) so idle cards qualify.

Within the chosen tier, among satisfying machines the one with the **most free memory** wins. Jobs
that don't fit stay in the queue and are retried every `POLL_INTERVAL_MIN` (which also catches
machines freed by *other people's* jobs ending). One job per entry, using all its GPUs — to run
several jobs on one host concurrently, add multiple entries (`cuda=[0]`, `cuda=[1]`, ...).

Credentials are read from `run_jobs/baulab/auth.yml` (see step 0). Needs `screen` on each remote,
and `sshpass` on the local box unless `ssh_login: true` (passwordless ssh) — `bash
run_jobs/baulab/setup.sh` installs them.

## 3. Run the scheduler

```bash
# plan only — static-gate the machines, no launch
python run_jobs/baulab/dispatch.py --jobs run_jobs/baulab/jobs.example.json --dry-run

# for real — long-lived; keep it in the foreground of a terminal (or a screen/tmux you re-attach)
python run_jobs/baulab/dispatch.py --jobs my_jobs.json
```

The scheduler is **long-lived**: it stays up until every job is *terminal* (done / error /
killed), because it has to keep reaping and **requeueing** OOM / killed jobs. It renders a live
dashboard right in its own terminal — no need to launch the monitor separately while it runs.
Each placed job runs in its own detached `screen` named after the job.

**Ctrl-C** stops the scheduler and cancels only the still-*queued* jobs. Jobs already placed keep
running in their own screens (they're detached from the scheduler); re-attach with `monitor.py`.

### Auto-requeue

Two failure modes are assumed to mean "someone else contended this machine", so the job is put
**back on the global queue** (it may then land on a *different* host), up to `MAX_RETRY` times:

- **OOM** — the job hit CUDA OOM.
- **killed** — the job or its whole `screen` was killed by a signal (the kernel oom-killer, or a
  person reclaiming their assigned machine — baulab hosts are owned, and an owner's job takes
  priority). Detected from the driver's exit code, *or* from the job's screen session vanishing
  while it was still mid-run.

Any *other* error (a genuine bug) is terminal and never requeued. Retry history shows up in the
`retries` column as `umibozu(OOM), tokyo(killed)`. After `MAX_RETRY` attempts the job is left as
`error` (`OOM ×5 (gave up)`).

## 4. Monitor

```bash
python run_jobs/baulab/monitor.py                  # latest batch; exits when all jobs finish
python run_jobs/baulab/monitor.py --run-id <id>    # a specific batch
python run_jobs/baulab/monitor.py --interval 5 --watch   # keep refreshing after finish
python run_jobs/baulab/monitor.py --list           # past batches + their run ids
```

A refreshing table of **job · machine · status · retries · detail**. Statuses:

| status      | meaning                                            |
|-------------|----------------------------------------------------|
| `queued`    | still in the wait queue (not placed yet)           |
| `waiting`   | placed; driver waiting on GPU memory (safety net)  |
| `running`   | executing now (detail = elapsed)                   |
| `done`      | finished cleanly                                   |
| `error`     | failed; detail = brief classification (CUDA OOM, a Python `SomeError:` line, exit 137, retries-exhausted, ...) |
| `killed`    | killed by you via `kill.py` (not requeued)         |
| `cancelled` | was still queued when you Ctrl-C'd the scheduler   |
| `no-status` | placed but status not readable yet (driver starting) / host unreachable |

The monitor reads `.remote_jobs/manifests/<run_id>.json` for the job list + assignments + retry
history, then `ssh <host> cat <status_path>` for each placed job (local jobs read directly). It
**keeps working after the scheduler exits** — placed jobs keep updating their own `status.json`.
Drivers write `status.json` atomically, so reads never see a half-written file.

## 5. Kill jobs

```bash
python run_jobs/baulab/kill.py --names qwen14b_cma qwen32b_cma   # kill specific jobs
python run_jobs/baulab/kill.py --all                            # kill the whole latest batch
python run_jobs/baulab/kill.py --all --run-id <id>              # ... of a specific batch
python run_jobs/baulab/kill.py --list                          # past batches + their run ids
```

A killed job is recorded in a per-run kill list and its `screen` is quit immediately. The
scheduler (if still running) settles it as `killed` and drops it from the queue if it hadn't
been placed yet. Killed jobs are **never** auto-requeued — that's the difference from an OOM.

## 6. Manage by hand

```bash
ssh <host>; screen -ls; screen -r <job_name>          # attach to a job (Ctrl-A D to detach)
# driver log: <project>/.remote_jobs/<run_id>/jobs/<job_name>/_driver.log
```

Staged files per job live in `<project>/.remote_jobs/<run_id>/jobs/<job_name>/`
(`driver.py`, `config.json`, `run.sh`, `status.json`, `_driver.log`, `job_NN_<name>.log`).
