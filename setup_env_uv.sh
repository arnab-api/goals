#!/usr/bin/env bash
# Creates and populates the `goals` environment with uv (the non-conda path).
#
# Usage:
#   bash setup_env_uv.sh
#
# Requirements:
#   - uv          (https://docs.astral.sh/uv/  ->  curl -LsSf https://astral.sh/uv/install.sh | sh)
#   - CUDA 12.x drivers installed on the host
#
# Dependencies, the PyTorch CUDA index, and the nnsight and nnterp git pins are declared in
# pyproject.toml. To use a different CUDA version, edit the [[tool.uv.index]] url and the
# matching [tool.uv.sources] name there, then re-run this script.

set -euo pipefail

PYTHON_VERSION="3.11"
ENV_NAME="goals"

cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv not found on PATH." >&2
    echo "install it with:  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

echo "==> Creating virtual environment (.venv) with Python ${PYTHON_VERSION}"
# uv downloads a managed CPython if ${PYTHON_VERSION} isn't already available.
uv venv --python "${PYTHON_VERSION}"

echo "==> Resolving + installing dependencies from pyproject.toml (writes uv.lock)"
# `uv sync` installs torch/torchvision from the CUDA index and nnsight and nnterp from git.
uv sync

echo "==> Registering Jupyter kernel as '${ENV_NAME}'"
uv run python -m ipykernel install --user --name "${ENV_NAME}" --display-name "Python (${ENV_NAME})"

echo ""
echo "Done! The environment lives in ./.venv"
echo "Run commands with:   uv run python <script>.py        (no activation needed)"
echo "      or activate:   source .venv/bin/activate"
echo "Then set up env.yml from env_demo.yml with your API keys and paths."
