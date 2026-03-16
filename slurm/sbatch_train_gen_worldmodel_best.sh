#!/usr/bin/env bash
#SBATCH --job-name=genwm-train-best
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
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
N_ENVS_BEST="${N_ENVS_BEST:-8}"
N_UPDATES="${N_UPDATES:-10}"
N_STEPS="${N_STEPS:-10}"

for i in 0 1 2 3; do
  seed=$((42+i))
  CUDA_VISIBLE_DEVICES="${i}" TMPDIR="${TMPDIR}" python run_gen_worldmodel_pg_train.py \
    --fast_startup \
    --gpu_id "${i}" \
    --n_envs "${N_ENVS_BEST}" \
    --n_updates "${N_UPDATES}" \
    --n_steps "${N_STEPS}" \
    --seed "${seed}" \
    --run_name "train_best_seed${seed}_job${SLURM_JOB_ID}" &
done
wait

python3.11 aggregate_gen_worldmodel_pnl.py --glob "outputs/gen_worldmodel_pg_train/train_best_seed*_job${SLURM_JOB_ID}/summary.json" --output "outputs/gen_worldmodel_pg_train/train_best_aggregate_${SLURM_JOB_ID}.json"
