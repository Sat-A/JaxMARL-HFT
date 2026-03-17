#!/usr/bin/env bash
#SBATCH --job-name=genwm-train-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/home/s5e/satyamaga.s5e/JaxMARL-HFT}"
cd "${REPO_ROOT}"
mkdir -p slurm/logs

CONDA_ENV="${CONDA_ENV:-lobs5}"
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  set +u
  conda activate "${CONDA_ENV}"
  set -u
elif [ -f "/home/s5e/satyamaga.s5e/miniforge3/etc/profile.d/conda.sh" ]; then
  source "/home/s5e/satyamaga.s5e/miniforge3/etc/profile.d/conda.sh"
  set +u
  conda activate "${CONDA_ENV}"
  set -u
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TMPDIR="/tmp"
mkdir -p "${TMPDIR}"
export CUDA_CACHE_PATH="/tmp/.nv/ComputeCache"
mkdir -p "${CUDA_CACHE_PATH}"
python run_gen_worldmodel_pg_train.py \
  --fast_startup \
  --policy_arch "${POLICY_ARCH:-ippo_rnn}" \
  --checkpoint_restore_topology "${CHECKPOINT_RESTORE_TOPOLOGY:-single-device-remap}" \
  --mm_action_space "${MM_ACTION_SPACE:-bobStrategy}" \
  --mm_bob_v0 "${MM_BOB_V0:-10}" \
  --mm_fixed_quant_value "${MM_FIXED_QUANT_VALUE:-10}" \
  --lr "${LR:-3e-4}" \
  --entropy_coef "${ENTROPY_COEF:-1e-3}" \
  --value_coef "${VALUE_COEF:-0.5}" \
  --n_envs "${N_ENVS:-1}" \
  --n_updates "${N_UPDATES:-1}" \
  --n_steps "${N_STEPS:-2}" \
  --n_cond_msgs "${N_COND_MSGS:-8}" \
  --run_name "genwm_train_smoke_${SLURM_JOB_ID}" \
  --seed "${SEED:-42}"
