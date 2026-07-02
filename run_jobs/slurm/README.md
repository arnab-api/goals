# run_jobs/slurm

Minimal utility for rendering + submitting a list of jobs on a SLURM cluster.
Supports **Delta** and **NURC (Northeastern RC, `agi` partition)** via `--on`.
Not a scheduler like `run_jobs/baulab` — it just renders `.sh` scripts from a
per-cluster template and calls `sbatch` on each one.

## Usage

```bash
# render + submit every job in jobs.json on Delta
python -m run_jobs.slurm.submit --on delta --jobs jobs.json

# same, on NURC's agi partition
python -m run_jobs.slurm.submit --on nurc --jobs jobs.json

# render only, don't call sbatch
python -m run_jobs.slurm.submit --on delta --jobs jobs.json --dry-run

# submit a subset by name
python -m run_jobs.slurm.submit --on nurc --jobs jobs.json --only finetuning cache_qwen_states

# override an SBATCH field for every job in this run
python -m run_jobs.slurm.submit --on nurc --jobs jobs.json --gpus 4 --time 12:00:00
```

Each invocation writes to a fresh timestamped run dir, one subdir per job:
`generated/<cluster>/<run_id>/<name>/<name>.sh`, where `run_id` defaults to
`%Y%m%d_%H%M%S` (override with `--run-id`). Jobs are submitted with
`sbatch --chdir <job_dir> ...`, so the template's relative `slurm-%j.out` /
`slurm-%j.err` land in that same per-job dir next to the rendered script.
The whole `generated/` tree is gitignored and left on disk after submission
for inspection/resubmission.

## Defaults

Each cluster has its own scope in `defaults.yml`:

```yaml
delta:
  mem: 256g
  gpus_per_task: 2
  partition: gpuH200x8       # or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8 gpuH200x8
  account: bezl-delta-gpu
  time: "48:00:00"
  cwd: /projects/bezl/goals

nurc:
  # `agi` doesn't require --account (SLURM picks the default association)
  partition: agi
  nodes: 1
  cpus_per_task: 64
  gpus: 2
  time: "48:00:00"
  cwd: /projects/baulab/sensharma.a/goals
```

Note the SBATCH schemas differ: Delta uses `--mem`, `--gpus-per-task`, and
requires `--account`; NURC uses `--nodes`, `--cpus-per-task`, `--gpus`, and
its `agi` partition works without `--account` (SLURM picks the default
association). That's why each cluster gets its own template
(`template_delta.sh`, `template_nurc.sh`).

### Field precedence (lowest → highest)

1. `defaults.yml[cluster]`
2. Fields set inside a job dict in the JSON
3. `--<field>` CLI overrides (apply to every job in the run)

## Job list format

A JSON list of job dicts. `name` and `command` are required; every other
field falls back to the cluster's defaults or a CLI override. Any of the
override fields can also be pinned per-job:

```json
[
  {
    "name": "finetuning",
    "gpus_per_task": 4,
    "time": "15:00:00",
    "command": "python run_finetuning.py"
  }
]
```

Copy `jobs.example.json` to `jobs.json` and edit as needed — `jobs.json` is
not tracked in git, so it's safe to leave real, in-flight job configs there.

## Environment

Both templates activate the uv-managed `${cwd}/.venv` (the project's conda
`goals` env has been retired). `python -m …` in a job command resolves to
`.venv/bin/python`. After any `pyproject.toml` change, run `uv sync` on the
cluster's login node once; compute nodes just consume the shared `.venv`.

The templates also print a `== venv sanity ==` block at job start showing
`which python` and `sys.executable`, so a wrong-python situation is visible
in the first few lines of every SLURM `.out` file.

## Files

- `template_delta.sh` / `template_nurc.sh` — SBATCH boilerplate per cluster, filled in via `string.Template`.
- `defaults.yml` — per-cluster default SBATCH fields.
- `jobs.example.json` — example job list.
- `submit.py` — renders + submits (entry point: `python -m run_jobs.slurm.submit`).
