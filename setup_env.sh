#!/usr/bin/env bash
# Creates and populates the `goals` conda environment.
#
# Usage:
#   bash setup_env.sh
#
# Requirements:
#   - conda (miniconda or anaconda)
#   - CUDA 12.x drivers installed on the host
#
# If you have a different CUDA version, change the --index-url below.
# Find the right URL at: https://pytorch.org/get-started/locally/

set -euo pipefail

ENV_NAME="goals"
PYTHON_VERSION="3.11"

echo "==> Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}"
conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"

# Activate inside the script via conda run (avoids needing `conda init` in the shell)
RUN="conda run -n ${ENV_NAME} --no-capture-output"

echo "==> Installing PyTorch (CUDA 12.x)"
# Adjust the index URL if your CUDA version differs:
#   CUDA 11.8: https://download.pytorch.org/whl/cu118
#   CUDA 12.1: https://download.pytorch.org/whl/cu121
#   CUDA 12.4: https://download.pytorch.org/whl/cu124
$RUN pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu124

echo "==> Installing core ML / interpretability libraries"
$RUN pip install \
    transformers \
    accelerate \
    nnsight \
    datasets \
    huggingface-hub \
    safetensors \
    tokenizers \
    sentencepiece \
    einops \
    bitsandbytes

echo "==> Installing baukit (from source — required, not on PyPI)"
$RUN pip install git+https://github.com/davidbau/baukit

echo "==> Installing scientific stack"
$RUN pip install \
    numpy \
    scipy \
    scikit-learn \
    pandas \
    matplotlib

echo "==> Installing notebook / dev tools"
$RUN pip install \
    ipykernel \
    ipywidgets \
    jupyterlab \
    rich \
    tqdm \
    pyyaml

echo "==> Installing API clients & experiment tracking"
$RUN pip install \
    anthropic \
    openai \
    wandb

echo "==> Installing project utility libraries"
$RUN pip install \
    dataclasses-json \
    astor \
    line-profiler \
    goodfire

echo "==> Registering Jupyter kernel as '${ENV_NAME}'"
$RUN python -m ipykernel install --user --name "${ENV_NAME}" --display-name "Python (${ENV_NAME})"

echo ""
echo "Done! Activate with:  conda activate ${ENV_NAME}"
echo "Then set up env.yml from env_demo.yml with your API keys and paths."
