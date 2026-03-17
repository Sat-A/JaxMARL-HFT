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
export N_ENVS_CANDIDATES="${N_ENVS_CANDIDATES:-1 2 4 8}"
export POLICY_ARCH="${POLICY_ARCH:-ippo_rnn}"
export CHECKPOINT_RESTORE_TOPOLOGY="${CHECKPOINT_RESTORE_TOPOLOGY:-single-device-remap}"
export TMPDIR="/tmp"
mkdir -p "${TMPDIR}"
export CUDA_CACHE_PATH="/tmp/.nv/ComputeCache"
mkdir -p "${CUDA_CACHE_PATH}"
export GPU_IDS="${GPU_IDS:-0 1 2 3}"
export MAX_PARALLEL_GPUS="${MAX_PARALLEL_GPUS:-1}"
export RETRY_PER_PROFILE="${RETRY_PER_PROFILE:-2}"

SCOPE_KEY="${POLICY_ARCH}_u${N_UPDATES:-2}_s${N_STEPS:-8}_m${N_COND_MSGS:-8}"
TAG_FILE="outputs/gen_worldmodel_pg_train/last_sweep_tag_${SCOPE_KEY}.txt"
if [[ -n "${SWEEP_TAG:-}" ]]; then
  export SWEEP_TAG
elif [[ "${RESUME_SWEEP:-1}" == "1" && -f "${TAG_FILE}" ]]; then
  export SWEEP_TAG
  SWEEP_TAG="$(cat "${TAG_FILE}")"
  echo "[SWEEP] Resuming prior sweep tag: ${SWEEP_TAG}"
else
  export SWEEP_TAG="${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)"
  echo "[SWEEP] Starting new sweep tag: ${SWEEP_TAG}"
fi
mkdir -p "$(dirname "${TAG_FILE}")"
echo "${SWEEP_TAG}" > "${TAG_FILE}"

if ! [[ "${MAX_PARALLEL_GPUS}" =~ ^[0-9]+$ ]] || [[ "${MAX_PARALLEL_GPUS}" -lt 1 ]]; then
  echo "ERROR: MAX_PARALLEL_GPUS must be a positive integer, got '${MAX_PARALLEL_GPUS}'" >&2
  exit 1
fi

total_gpus=$(wc -w <<<"${GPU_IDS}")
if [[ "${MAX_PARALLEL_GPUS}" -gt "${total_gpus}" ]]; then
  echo "[SWEEP] MAX_PARALLEL_GPUS=${MAX_PARALLEL_GPUS} exceeds GPU count (${total_gpus}); capping."
  MAX_PARALLEL_GPUS="${total_gpus}"
fi

echo "[SWEEP] GPU_IDS='${GPU_IDS}' MAX_PARALLEL_GPUS=${MAX_PARALLEL_GPUS}"
echo "[SWEEP] CHECKPOINT_RESTORE_TOPOLOGY=${CHECKPOINT_RESTORE_TOPOLOGY} RETRY_PER_PROFILE=${RETRY_PER_PROFILE}"

pids=()
for gpu in ${GPU_IDS}; do
  while [[ "$(jobs -pr | wc -l)" -ge "${MAX_PARALLEL_GPUS}" ]]; do
    sleep 1
  done
  GPU_ID="${gpu}" N_UPDATES="${N_UPDATES:-2}" N_STEPS="${N_STEPS:-8}" N_COND_MSGS="${N_COND_MSGS:-8}" \
    SWEEP_TAG="${SWEEP_TAG}" CHECKPOINT_RESTORE_TOPOLOGY="${CHECKPOINT_RESTORE_TOPOLOGY}" \
    RETRY_PER_PROFILE="${RETRY_PER_PROFILE}" \
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
  echo "ERROR: one or more GPU sweep workers failed" >&2
  exit 1
fi

AGG_OUT="outputs/gen_worldmodel_pg_train/single_node_sweep_aggregate_${SLURM_JOB_ID:-manual}.json"
SUMMARY_GLOB="outputs/gen_worldmodel_pg_train/sweep_gen_train_${SWEEP_TAG}_g*_n*/summary.json"
N_SUMMARIES=$(find outputs/gen_worldmodel_pg_train -path "outputs/gen_worldmodel_pg_train/sweep_gen_train_${SWEEP_TAG}_g*_n*/summary.json" | wc -l)
echo "[SWEEP] Found ${N_SUMMARIES} summary files for tag ${SWEEP_TAG}"
if [[ "${N_SUMMARIES}" -gt 0 ]]; then
  python3.11 aggregate_gen_worldmodel_pnl.py --glob "${SUMMARY_GLOB}" --output "${AGG_OUT}"
  cp "${AGG_OUT}" outputs/gen_worldmodel_pg_train/single_node_sweep_aggregate.json
else
  echo "ERROR: no summary files found for tag ${SWEEP_TAG}" >&2
  exit 1
fi
