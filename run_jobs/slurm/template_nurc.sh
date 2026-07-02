#!/bin/bash
#SBATCH --partition=${partition}
#SBATCH --nodes=${nodes}
#SBATCH --cpus-per-task=${cpus_per_task}
#SBATCH --gpus=${gpus}
#SBATCH --job-name=${name}
#SBATCH --time=${time}
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

# initialize bash
source ~/.bashrc

# cd to project directory FIRST — .venv is resolved relative to it
cd ${cwd}

# activate the uv-managed venv (single source of truth for this project's Python).
# We deliberately do NOT `conda activate` here — mixing conda + .venv causes
# PATH-shadowing bugs where the effective python is one env while pip-installed
# packages land in another.
# Assumes `.venv` is up to date; run `uv sync` on the login node after any
# pyproject.toml change to refresh it.
source .venv/bin/activate

# diagnostic: prove which python is on PATH before any real work runs
echo "== venv sanity =="
which python
python -c "import sys; print('sys.executable:', sys.executable)"
echo "================="

#export your required environment variables below
#################################################
export WANDB_API_KEY=$$(cat ~/keys/wandb.key)

# run the script (bare `python -m ...` in the command resolves to .venv/bin/python)
${command}
