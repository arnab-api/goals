#!/usr/bin/env python3
"""Render a list of jobs into sbatch scripts and submit them on Delta.

Usage:
    python submit_jobs.py jobs.json
    python submit_jobs.py jobs.json --dry-run
    python submit_jobs.py jobs.json --only finetuning cache_qwen_states
"""

import argparse
import json
import string
import subprocess
import sys
import time
from pathlib import Path

# Delta GPU partitions (`sinfo` / `accounts` to see what's available to your project):
#   gpuA100x4   gpuA40x4   gpuA100x8   gpuMI100x8   gpuH200x8
DEFAULTS = {
    "mem": "256g",
    "gpus_per_task": 1,
    "partition": "gpuH200x8",
    "account": "bezl-delta-gpu",
    "time": "48:00:00",
    "conda_env": "goals",
    "cwd": "/projects/bezl/goals",
}

HERE = Path(__file__).resolve().parent
TEMPLATE_PATH = HERE / "template.sh"
SCRIPTS_DIR = HERE / "generated"


def render_job(job: dict, template: string.Template) -> str:
    missing = [k for k in ("name", "command") if k not in job]
    if missing:
        raise ValueError(f"job is missing required field(s): {missing} -- {job}")
    fields = {**DEFAULTS, **job}
    return template.substitute(fields)


def submit(script_path: Path, job_dir: Path, dry_run: bool) -> None:
    # --chdir makes the template's relative `slurm-%j.out`/`slurm-%j.err` land in job_dir,
    # alongside the rendered script, instead of wherever this process happens to run from.
    cmd = ["sbatch", "--chdir", str(job_dir), str(script_path)]
    if dry_run:
        print(f"[dry-run] would submit: {' '.join(cmd)}")
        return
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[FAILED] {script_path.name}: {result.stderr.strip()}", file=sys.stderr)
    else:
        print(f"[submitted] {script_path.name}: {result.stdout.strip()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jobs_file", type=Path, help="JSON file with a list of job configs")
    parser.add_argument(
        "--only", nargs="+", default=None, help="only render/submit jobs with these names"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="render scripts but do not call sbatch"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=SCRIPTS_DIR, help="base dir for rendered .sh scripts"
    )
    parser.add_argument(
        "--run-id", default=None, help="subdir name under --out-dir (default: timestamp)"
    )
    args = parser.parse_args()

    jobs = json.loads(args.jobs_file.read_text())
    if args.only:
        jobs = [j for j in jobs if j["name"] in args.only]
        found = {j["name"] for j in jobs}
        for name in args.only:
            if name not in found:
                print(f"warning: no job named {name!r} in {args.jobs_file}", file=sys.stderr)

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    run_dir = args.out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run_id={run_id} | jobs={len(jobs)} | scripts -> {run_dir}")

    template = string.Template(TEMPLATE_PATH.read_text())

    for job in jobs:
        rendered = render_job(job, template)
        job_dir = run_dir / job["name"]
        job_dir.mkdir(parents=True, exist_ok=True)
        script_path = job_dir / f"{job['name']}.sh"
        script_path.write_text(rendered)
        script_path.chmod(0o755)
        submit(script_path, job_dir, args.dry_run)


if __name__ == "__main__":
    main()
