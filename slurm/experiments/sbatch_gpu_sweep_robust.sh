#!/usr/bin/env bash
#SBATCH --job-name=gpu-sweep-robust
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
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
export N_ENVS_CANDIDATES="${N_ENVS_CANDIDATES:-1 2 4 8}"
export POLICY_ARCH="${POLICY_ARCH:-ippo_rnn}"
export CHECKPOINT_RESTORE_TOPOLOGY="${CHECKPOINT_RESTORE_TOPOLOGY:-auto}"
export TMPDIR="/tmp"
mkdir -p "${TMPDIR}"
export CUDA_CACHE_PATH="/tmp/.nv/ComputeCache"
mkdir -p "${CUDA_CACHE_PATH}"
export SWEEP_TAG="${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)"
export MAX_PARALLEL_GPUS="${MAX_PARALLEL_GPUS:-1}"

echo "========================================"
echo "Robust GPU Sweep with Safe Concurrency"
echo "SWEEP_TAG: ${SWEEP_TAG}"
echo "POLICY_ARCH: ${POLICY_ARCH}"
echo "CHECKPOINT_RESTORE_TOPOLOGY: ${CHECKPOINT_RESTORE_TOPOLOGY}"
echo "N_ENVS_CANDIDATES: ${N_ENVS_CANDIDATES}"
echo "MAX_PARALLEL_GPUS: ${MAX_PARALLEL_GPUS}"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Time: $(date)"
echo ""

# Single GPU sweep with MAX_PARALLEL_GPUS=1 (serialized)
export GPU_IDS="0"

pids=()
for gpu in ${GPU_IDS}; do
  while [[ "$(jobs -pr | wc -l)" -ge "${MAX_PARALLEL_GPUS}" ]]; do
    sleep 1
  done
  echo "Starting sweep on GPU ${gpu}..."
  GPU_ID="${gpu}" N_UPDATES="${N_UPDATES:-2}" N_STEPS="${N_STEPS:-8}" SWEEP_TAG="${SWEEP_TAG}" \
    bash run_sweep_gen_worldmodel_train_single_node.sh &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

if [[ "${failed}" -ne 0 ]]; then
  echo "ERROR: GPU sweep worker failed" >&2
  exit 1
fi

echo ""
echo "========================================"
echo "Aggregating results..."
echo "========================================"
AGG_OUT="outputs/gen_worldmodel_pg_train/single_node_sweep_aggregate_${SLURM_JOB_ID:-manual}.json"
python3.11 aggregate_gen_worldmodel_pnl.py \
  --glob "outputs/gen_worldmodel_pg_train/sweep_gen_train_${SWEEP_TAG}_g*_p*_n*/summary.json" \
  --output "${AGG_OUT}"
cp "${AGG_OUT}" outputs/gen_worldmodel_pg_train/single_node_sweep_aggregate.json

echo "Aggregate saved to: ${AGG_OUT}"
echo "Sweep complete!"
