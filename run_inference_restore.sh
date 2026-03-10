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

CONFIG_NAME="${CONFIG_NAME:-2player_config}"
RESTORE_PROJECT="${RESTORE_PROJECT:-2PLayer}"
RESTORE_RUN="${RESTORE_RUN:-dummy-2vdmzbye}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-$SCRIPT_DIR/checkpoints/MARLCheckpoints}"
EVAL_TIME_PERIOD="${EVAL_TIME_PERIOD:-2022}"

python3 gymnax_exchange/jaxrl/MARL/baseline_eval/baseline_JAXMARL.py \
  --config-name="$CONFIG_NAME" \
  WANDB_MODE=disabled \
  RESTORE_PROJECT="$RESTORE_PROJECT" \
  RESTORE_RUN="$RESTORE_RUN" \
  "EvalTimePeriod=\"$EVAL_TIME_PERIOD\"" \
  +CHECKPOINT_BASE_DIR="$CHECKPOINT_BASE_DIR" \
  "$@"
