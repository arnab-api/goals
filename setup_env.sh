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
PYTHON_VERSION="3.12"

echo "==> Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}"
conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"

# Activate inside the script via conda run (avoids needing `conda init` in the shell)
RUN="conda run -n ${ENV_NAME} --no-capture-output"

echo "==> Installing PyTorch (CUDA 12.8)"
# The code needs torch>=2.7, which needs CUDA >= 12.6 wheels (cu124 caps at torch 2.6). Adjust if needed:
#   CUDA 12.6: https://download.pytorch.org/whl/cu126
#   CUDA 12.8: https://download.pytorch.org/whl/cu128
$RUN pip install "torch>=2.7" torchvision \
    --index-url https://download.pytorch.org/whl/cu128

echo "==> Installing core ML / interpretability libraries"
# nnterp: the v2.0.0 commit pyproject.toml pins (ndif-team/nnterp#124, #127, #128) until a release carries them
$RUN pip install \
    "transformers>=5.17" \
    accelerate \
    "nnsight>=0.8.2" \
    "nnterp @ git+https://github.com/ndif-team/nnterp@3fb2cc02b853c0c444e64b8cf75a7743b6f1e661" \
    peft \
    datasets \
    huggingface-hub \
    safetensors \
    tokenizers \
    sentencepiece \
    "circuitsvis>=1.43.3"

echo "==> Installing scientific stack"
$RUN pip install \
    numpy \
    pandas \
    matplotlib

echo "==> Installing notebook / dev tools"
$RUN pip install \
    ipykernel \
    ipywidgets \
    jupyterlab \
    nbformat \
    nbconvert \
    rich \
    tqdm \
    pyyaml

echo "==> Installing API clients (LM judges: Claude)"
$RUN pip install \
    anthropic

echo "==> Installing project utility libraries"
$RUN pip install \
    dataclasses-json

echo "==> Registering Jupyter kernel as '${ENV_NAME}'"
$RUN python -m ipykernel install --user --name "${ENV_NAME}" --display-name "Python (${ENV_NAME})"

echo ""
echo "Done! Activate with:  conda activate ${ENV_NAME}"
echo "Then set up env.yml from env_demo.yml with your API keys and paths."
