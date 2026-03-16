#!/usr/bin/env bash
#SBATCH --job-name=genwm-sweep-1node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
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
export PYTHON_BIN="${PYTHON_BIN:-python}"
export N_ENVS_CANDIDATES="${N_ENVS_CANDIDATES:-2 4 8}"
export TMPDIR="/tmp"
mkdir -p "${TMPDIR}"
export CUDA_CACHE_PATH="/tmp/.nv/ComputeCache"
mkdir -p "${CUDA_CACHE_PATH}"
export SWEEP_TAG="${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)"

for gpu in 0 1 2 3; do
  GPU_ID="${gpu}" N_UPDATES="${N_UPDATES:-2}" N_STEPS="${N_STEPS:-8}" SWEEP_TAG="${SWEEP_TAG}" \
    bash run_sweep_gen_worldmodel_train_single_node.sh &
done
wait

AGG_OUT="outputs/gen_worldmodel_pg_train/single_node_sweep_aggregate_${SLURM_JOB_ID:-manual}.json"
python3.11 aggregate_gen_worldmodel_pnl.py \
  --glob "outputs/gen_worldmodel_pg_train/sweep_gen_train_${SWEEP_TAG}_g*_p*_n*/summary.json" \
  --output "${AGG_OUT}"
cp "${AGG_OUT}" outputs/gen_worldmodel_pg_train/single_node_sweep_aggregate.json
