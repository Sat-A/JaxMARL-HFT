#!/usr/bin/env bash
#SBATCH --job-name=ippo-rnn-profile-mini
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=00:45:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/s5e/satyamaga.s5e/JaxMARL-HFT}"
cd "${REPO_ROOT}"
mkdir -p slurm/logs

# Activate conda environment
CONDA_ENV="${CONDA_ENV:-lobs5}"
if [ -f "/home/s5e/satyamaga.s5e/miniforge3/etc/profile.d/conda.sh" ]; then
  source "/home/s5e/satyamaga.s5e/miniforge3/etc/profile.d/conda.sh"
  set +u
  conda activate "${CONDA_ENV}"
  set -u
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export N_ENVS_CANDIDATES="${N_ENVS_CANDIDATES:-1 2 4 8}"
export POLICY_ARCH="${POLICY_ARCH:-ippo_rnn}"
export TMPDIR="/tmp"
mkdir -p "${TMPDIR}"
export CUDA_CACHE_PATH="/tmp/.nv/ComputeCache"
mkdir -p "${CUDA_CACHE_PATH}"
export SWEEP_TAG="${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)"

# Mini profile sweep: n_envs candidates on single GPU (GPU 0)
echo "========================================"
echo "IPPO-RNN Generative Trainer Profile Search"
echo "SWEEP_TAG: ${SWEEP_TAG}"
echo "POLICY_ARCH: ${POLICY_ARCH}"
echo "N_ENVS_CANDIDATES: ${N_ENVS_CANDIDATES}"
echo "========================================"

# Run on GPU 0 only for this mini sweep
GPU_ID="0"
N_UPDATES="${N_UPDATES:-2}"
N_STEPS="${N_STEPS:-8}"
N_COND_MSGS="${N_COND_MSGS:-8}"

echo "Starting sweep on GPU ${GPU_ID}..."
GPU_ID="${GPU_ID}" N_UPDATES="${N_UPDATES}" N_STEPS="${N_STEPS}" SWEEP_TAG="${SWEEP_TAG}" \
  bash run_sweep_gen_worldmodel_train_single_node.sh

# Aggregate results
echo ""
echo "========================================"
echo "Aggregating results..."
echo "========================================"

AGG_GLOB="outputs/gen_worldmodel_pg_train/sweep_gen_train_${SWEEP_TAG}_g${GPU_ID}_p*_n*/summary.json"
AGG_OUT="outputs/gen_worldmodel_pg_train/ippo_rnn_profile_search_${SLURM_JOB_ID:-manual}.json"

echo "Searching for summaries: ${AGG_GLOB}"
matching_files=$(ls ${AGG_GLOB} 2>/dev/null || true | wc -l)
echo "Found ${matching_files} summary files"

if ls ${AGG_GLOB} 1> /dev/null 2>&1; then
  python3.11 aggregate_gen_worldmodel_pnl.py \
    --glob "${AGG_GLOB}" \
    --output "${AGG_OUT}"
  
  echo ""
  echo "Aggregate results saved to: ${AGG_OUT}"
  echo ""
  cat "${AGG_OUT}"
else
  echo "No summary files found. This may indicate a training failure."
  exit 1
fi

echo ""
echo "========================================"
echo "Profile search complete"
echo "========================================"
