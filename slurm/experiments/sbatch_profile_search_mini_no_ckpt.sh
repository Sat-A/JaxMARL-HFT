#!/usr/bin/env bash
#SBATCH --job-name=ippo-rnn-profile-no-ckpt
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

# Mini profile sweep: n_envs candidates on single GPU (GPU 0) WITHOUT checkpoint
echo "========================================"
echo "IPPO-RNN Generative Trainer Profile Search"
echo "(Without pre-trained checkpoint - fresh initialization)"
echo "SWEEP_TAG: ${SWEEP_TAG}"
echo "POLICY_ARCH: ${POLICY_ARCH}"
echo "N_ENVS_CANDIDATES: ${N_ENVS_CANDIDATES}"
echo "========================================"

GPU_ID="0"
N_UPDATES="${N_UPDATES:-2}"
N_STEPS="${N_STEPS:-8}"
N_COND_MSGS="${N_COND_MSGS:-8}"

for n_envs in ${N_ENVS_CANDIDATES}; do
  run_name="sweep_gen_train_${SWEEP_TAG}_n${n_envs}"
  echo "[SWEEP] n_envs=${n_envs} gpu=${GPU_ID}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" TMPDIR="${TMPDIR}" "${PYTHON_BIN}" run_gen_worldmodel_pg_train.py \
    --n_envs "${n_envs}" \
    --n_updates "${N_UPDATES}" \
    --n_steps "${N_STEPS}" \
    --n_cond_msgs "${N_COND_MSGS}" \
    --policy_arch "${POLICY_ARCH}" \
    --gpu_id "${GPU_ID}" \
    --run_name "${run_name}" || continue

  summary="outputs/gen_worldmodel_pg_train/${run_name}/summary.json"
  if [[ -f "${summary}" ]]; then
    cp "${summary}" "${SWEEP_TAG}_${run_name}_summary.json"
    echo "[SWEEP] n_envs=${n_envs} summary created"
  fi
done

echo ""
echo "========================================"
echo "Aggregating results..."
echo "========================================"

AGG_GLOB="outputs/gen_worldmodel_pg_train/sweep_gen_train_${SWEEP_TAG}_n*/summary.json"
AGG_OUT="outputs/gen_worldmodel_pg_train/ippo_rnn_profile_search_no_ckpt_${SLURM_JOB_ID:-manual}.json"

echo "Searching for summaries: ${AGG_GLOB}"

if ls ${AGG_GLOB} 1> /dev/null 2>&1; then
  python3.11 aggregate_gen_worldmodel_pnl.py \
    --glob "${AGG_GLOB}" \
    --output "${AGG_OUT}"
  
  echo ""
  echo "Aggregate results saved to: ${AGG_OUT}"
  echo ""
  python3.11 -m json.tool "${AGG_OUT}"
else
  echo "No summary files found."
  exit 1
fi

echo ""
echo "========================================"
echo "Profile search complete"
echo "========================================"
