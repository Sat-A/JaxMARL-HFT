#!/usr/bin/env bash
#SBATCH --job-name=jaxhft-smoke-rollout
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
export LOBS5_ROOT="${LOBS5_ROOT:-/home/s5e/satyamaga.s5e/LOBS5}"
export WORLD_MODEL_CKPT="${WORLD_MODEL_CKPT:-/lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_H1-scaling-law/checkpoints/j2514440_bkotgtm5_2514440}"
export LOB_PREPROC_DATA_DIR="${LOB_PREPROC_DATA_DIR:-/lus/lfs1aip2/projects/s5e/lob_preproc/GOOG}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.50}"
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" minimal_agent_generative_step.py \
  --fast_startup \
  --n_cond_msgs "${N_COND_MSGS:-8}" \
  --n_steps "${N_STEPS:-3}" \
  --sample_top_n 1 \
  --sample_index "${SAMPLE_INDEX:-0}" \
  --action_policy "${ACTION_POLICY:-market_making}" \
  --start_date "${START_DATE:-2026-01-01}" \
  --end_date "${END_DATE:-2026-01-31}" \
  --run_name "slurm_smoke_rollout_${SLURM_JOB_ID:-manual}"
