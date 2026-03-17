#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  if command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN="python3.11"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "ERROR: no python interpreter found" >&2
    exit 1
  fi
fi
GPU_ID="${GPU_ID:-0}"
TMPDIR="${TMPDIR:-/tmp}"
mkdir -p "${TMPDIR}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-/tmp/.nv/ComputeCache}"
mkdir -p "${CUDA_CACHE_PATH}"

N_ENVS_CANDIDATES="${N_ENVS_CANDIDATES:-1 2 4 8}"
N_UPDATES="${N_UPDATES:-2}"
N_STEPS="${N_STEPS:-8}"
N_COND_MSGS="${N_COND_MSGS:-8}"
POLICY_ARCH="${POLICY_ARCH:-ippo_rnn}"
CHECKPOINT_RESTORE_TOPOLOGY="${CHECKPOINT_RESTORE_TOPOLOGY:-single-device-remap}"
MM_ACTION_SPACE="${MM_ACTION_SPACE:-bobStrategy}"
MM_BOB_V0="${MM_BOB_V0:-10}"
MM_FIXED_QUANT_VALUE="${MM_FIXED_QUANT_VALUE:-10}"
LR="${LR:-3e-4}"
ENTROPY_COEF="${ENTROPY_COEF:-1e-3}"
VALUE_COEF="${VALUE_COEF:-0.5}"
SWEEP_TAG="${SWEEP_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_SUFFIX="${RUN_SUFFIX:-}"
RETRY_PER_PROFILE="${RETRY_PER_PROFILE:-2}"

if [[ -n "${RUN_SUFFIX}" ]]; then
  RUN_TAG="${SWEEP_TAG}_g${GPU_ID}_${RUN_SUFFIX}"
else
  RUN_TAG="${SWEEP_TAG}_g${GPU_ID}"
fi
RESULTS_DIR="outputs/sweep_gen_worldmodel_train_${RUN_TAG}"
STATE_FILE="${RESULTS_DIR}/state.tsv"
FAILED_FILE="${RESULTS_DIR}/failed_profiles.txt"
mkdir -p "${RESULTS_DIR}"
touch "${STATE_FILE}"

best_n_envs=0
best_tput=0
completed_profiles=0

on_interrupt() {
  echo -e "INTERRUPTED\t$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${STATE_FILE}"
  echo "[SWEEP] Interrupted. Resume with SWEEP_TAG=${SWEEP_TAG} GPU_ID=${GPU_ID}."
}
trap on_interrupt INT TERM

for n_envs in ${N_ENVS_CANDIDATES}; do
  run_name="sweep_gen_train_${RUN_TAG}_n${n_envs}"
  summary="outputs/gen_worldmodel_pg_train/${run_name}/summary.json"

  if [[ -f "${summary}" ]]; then
    echo "[SWEEP] n_envs=${n_envs} gpu=${GPU_ID} summary exists, skipping."
    echo -e "SKIP\t${n_envs}\t${summary}" >> "${STATE_FILE}"
  else
    attempt=1
    success=0
    while [[ "${attempt}" -le "${RETRY_PER_PROFILE}" ]]; do
      echo "[SWEEP] n_envs=${n_envs} gpu=${GPU_ID} attempt=${attempt}/${RETRY_PER_PROFILE}"
      echo -e "START\t${n_envs}\tattempt=${attempt}" >> "${STATE_FILE}"
      if CUDA_VISIBLE_DEVICES="${GPU_ID}" TMPDIR="${TMPDIR}" "${PYTHON_BIN}" run_gen_worldmodel_pg_train.py \
        --fast_startup \
        --n_envs "${n_envs}" \
        --n_updates "${N_UPDATES}" \
        --n_steps "${N_STEPS}" \
        --n_cond_msgs "${N_COND_MSGS}" \
        --policy_arch "${POLICY_ARCH}" \
        --checkpoint_restore_topology "${CHECKPOINT_RESTORE_TOPOLOGY}" \
        --mm_action_space "${MM_ACTION_SPACE}" \
        --mm_bob_v0 "${MM_BOB_V0}" \
        --mm_fixed_quant_value "${MM_FIXED_QUANT_VALUE}" \
        --lr "${LR}" \
        --entropy_coef "${ENTROPY_COEF}" \
        --value_coef "${VALUE_COEF}" \
        --gpu_id "${GPU_ID}" \
        --run_name "${run_name}"; then
        success=1
      fi
      if [[ "${success}" -eq 1 && -f "${summary}" ]]; then
        echo -e "DONE\t${n_envs}\t${summary}" >> "${STATE_FILE}"
        break
      fi
      echo -e "FAIL\t${n_envs}\tattempt=${attempt}" >> "${STATE_FILE}"
      attempt=$((attempt + 1))
      sleep 2
    done
  fi

  if [[ -f "${summary}" ]]; then
    completed_profiles=$((completed_profiles + 1))
    cp "${summary}" "${RESULTS_DIR}/${run_name}_summary.json"
    tput=$(python3.11 - <<PY
import json
with open('${summary}') as f:
    d=json.load(f)
print(float(d.get('throughput',{}).get('updates_mean_steps_per_sec',0.0)))
PY
)
    echo "[SWEEP] n_envs=${n_envs} mean_steps_per_sec=${tput}"
    better=$(python3.11 - <<PY
print(1 if float('${tput}') > float('${best_tput}') else 0)
PY
)
    if [[ "${better}" == "1" ]]; then
      best_tput="${tput}"
      best_n_envs="${n_envs}"
    fi
  else
    echo "[SWEEP] n_envs=${n_envs} failed after retries on gpu=${GPU_ID}"
    echo "n_envs=${n_envs}" >> "${FAILED_FILE}"
  fi
done

echo "best_n_envs=${best_n_envs}" | tee "${RESULTS_DIR}/best_profile.txt"
echo "best_mean_steps_per_sec=${best_tput}" | tee -a "${RESULTS_DIR}/best_profile.txt"
echo "completed_profiles=${completed_profiles}" | tee -a "${RESULTS_DIR}/best_profile.txt"

if [[ "${completed_profiles}" -eq 0 ]]; then
  echo "[SWEEP] No successful profiles completed on gpu=${GPU_ID}" >&2
  exit 1
fi
