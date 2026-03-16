#!/usr/bin/env bash
#SBATCH --job-name=jaxhft-smoke-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/s5e/satyamaga.s5e/JaxMARL-HFT}"
cd "${REPO_ROOT}"
mkdir -p slurm/logs

CONDA_ENV="${CONDA_ENV:-lobs5}"
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  set +u
  conda activate "${CONDA_ENV}"
  set -u
elif [ -f "/home/s5e/satyamaga.s5e/miniforge3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "/home/s5e/satyamaga.s5e/miniforge3/etc/profile.d/conda.sh"
  set +u
  conda activate "${CONDA_ENV}"
  set -u
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" gymnax_exchange/jaxrl/MARL/ippo_rnn_JAXMARL.py \
  --config-name "${CONFIG_NAME:-ippo_rnn_JAXMARL_2player_cluster_smoke}" \
  WANDB_MODE="${WANDB_MODE:-disabled}" \
  NUM_ENVS="${NUM_ENVS:-8}" \
  NUM_STEPS="${NUM_STEPS:-16}" \
  TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-4096}"
