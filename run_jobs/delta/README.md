# run_jobs/delta

Minimal utility for running a list of jobs on the Delta slurm cluster. Not a
scheduler like `run_jobs/baulab` — it just renders `.sh` scripts from a
template and calls `sbatch` on each one.

## Usage

```bash
# render + submit every job in jobs.json
python submit_jobs.py jobs.json

# render only, don't call sbatch
python submit_jobs.py jobs.json --dry-run

# submit a subset by name
python submit_jobs.py jobs.json --only finetuning cache_qwen_states
```

Each invocation writes to a fresh timestamped run dir (same convention as
`run_jobs/baulab`'s `.remote_jobs/<run_id>/`), one subdir per job:
`generated/<run_id>/<name>/<name>.sh`, where `run_id` defaults to
`%Y%m%d_%H%M%S` (override with `--run-id`). Jobs are submitted with
`sbatch --chdir <job_dir> ...`, so the template's relative `slurm-%j.out` /
`slurm-%j.err` land in that same per-job dir, next to the rendered script.
The whole `generated/` tree is gitignored and left on disk after submission
for inspection/resubmission.

## Job list format

A JSON list of job dicts. `name` and `command` are required; everything else
falls back to the defaults in `submit_jobs.py` (`DEFAULTS`): `mem=256g`,
`gpus_per_task=1`, `partition=gpuH200x8`, `account=bezl-delta-gpu`,
`time=48:00:00`, `conda_env=goals`, `cwd=/projects/bezl/goals`. Override any
of these per-job (see `jobs.example.json`'s `finetuning` entry).

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

Copy `jobs.example.json` (which reproduces the old hardcoded scripts) to
`jobs.json` and edit as needed — `jobs.json` is not tracked in git, so it's
safe to leave real, in-flight job configs there.

## Files

- `template.sh` — SBATCH boilerplate, filled in via `string.Template`.
- `jobs.example.json` — example job list.
- `submit_jobs.py` — renders + submits.
