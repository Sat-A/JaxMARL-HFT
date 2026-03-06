#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONDA_ENV="${CONDA_ENV:-jaxmarl_hft}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-true}"
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

CONFIG_NAME="${CONFIG_NAME:-ippo_rnn_JAXMARL_2player}"
WANDB_MODE="${WANDB_MODE:-disabled}"

python3 gymnax_exchange/jaxrl/MARL/ippo_rnn_JAXMARL.py \
  --config-name="$CONFIG_NAME" \
  WANDB_MODE="$WANDB_MODE" \
  "$@"
