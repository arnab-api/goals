#!/usr/bin/env python3
"""Render a list of jobs into sbatch scripts and submit them on a SLURM cluster.

Usage:
    python -m run_jobs.slurm.submit --on delta --jobs jobs.json
    python -m run_jobs.slurm.submit --on nurc  --jobs jobs.json --dry-run
    python -m run_jobs.slurm.submit --on delta --jobs jobs.json --only finetuning cache_qwen_states
    python -m run_jobs.slurm.submit --on nurc  --jobs jobs.json --gpus 4 --time 12:00:00

Field precedence (lowest -> highest):
    defaults.yml[cluster]  <  fields inside each job dict  <  --<field> CLI overrides
"""

import argparse
import json
import string
import subprocess
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
DEFAULTS_PATH = HERE / "defaults.yml"
SCRIPTS_DIR = HERE / "generated"

CLUSTERS = ("delta", "nurc")

# SBATCH fields that can be overridden per-job (via JSON) or on the CLI
# (via --<field>). Union of what every template consumes; each template
# only substitutes the ones it references, so extras are silently ignored.
OVERRIDE_FIELDS = (
    "mem", "gpus_per_task", "gpus", "nodes", "cpus_per_task",
    "partition", "account", "time", "cwd",
)


def template_path_for(cluster: str) -> Path:
    return HERE / f"template_{cluster}.sh"


def load_defaults() -> dict:
    return yaml.safe_load(DEFAULTS_PATH.read_text())


def render_job(job: dict, template: string.Template, cluster_defaults: dict, cli_overrides: dict) -> str:
    missing = [k for k in ("name", "command") if k not in job]
    if missing:
        raise ValueError(f"job is missing required field(s): {missing} -- {job}")
    fields = {**cluster_defaults, **job, **cli_overrides}
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


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--on", choices=CLUSTERS, required=True, help="target SLURM cluster")
    p.add_argument("--jobs", type=Path, required=True, help="JSON file with a list of job configs")
    p.add_argument("--only", nargs="+", default=None, help="only render/submit jobs with these names")
    p.add_argument("--dry-run", action="store_true", help="render scripts but do not call sbatch")
    p.add_argument("--out-dir", type=Path, default=SCRIPTS_DIR,
                   help="base dir for rendered .sh scripts (default: run_jobs/slurm/generated/)")
    p.add_argument("--run-id", default=None,
                   help="subdir name under --out-dir/<cluster>/ (default: %%Y%%m%%d_%%H%%M%%S)")
    # per-field overrides that beat both defaults.yml and per-job JSON fields
    for f in OVERRIDE_FIELDS:
        p.add_argument(f"--{f.replace('_', '-')}", default=None, help=f"override {f} for every job in this run")
    return p.parse_args()


def main():
    args = parse_args()

    all_defaults = load_defaults()
    if args.on not in all_defaults:
        raise SystemExit(f"defaults.yml has no scope for cluster {args.on!r}")
    cluster_defaults = all_defaults[args.on]

    tpl_path = template_path_for(args.on)
    if not tpl_path.exists():
        raise SystemExit(f"no template for cluster {args.on!r}: expected {tpl_path}")
    template = string.Template(tpl_path.read_text())

    cli_overrides = {f: getattr(args, f) for f in OVERRIDE_FIELDS if getattr(args, f) is not None}

    jobs = json.loads(args.jobs.read_text())
    if args.only:
        jobs = [j for j in jobs if j["name"] in args.only]
        found = {j["name"] for j in jobs}
        for name in args.only:
            if name not in found:
                print(f"warning: no job named {name!r} in {args.jobs}", file=sys.stderr)

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    run_dir = args.out_dir / args.on / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"cluster={args.on} run_id={run_id} | jobs={len(jobs)} | scripts -> {run_dir}")
    if cli_overrides:
        print(f"CLI overrides (apply to every job): {cli_overrides}")

    for job in jobs:
        rendered = render_job(job, template, cluster_defaults, cli_overrides)
        job_dir = run_dir / job["name"]
        job_dir.mkdir(parents=True, exist_ok=True)
        script_path = job_dir / f"{job['name']}.sh"
        script_path.write_text(rendered)
        script_path.chmod(0o755)
        submit(script_path, job_dir, args.dry_run)


if __name__ == "__main__":
    main()
