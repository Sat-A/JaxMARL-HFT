#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
TMPDIR="/tmp"
mkdir -p "${TMPDIR}"
export CUDA_CACHE_PATH="/tmp/.nv/ComputeCache"
mkdir -p "${CUDA_CACHE_PATH}"
N_ENVS_CANDIDATES="${N_ENVS_CANDIDATES:-2 4 8 16}"
N_UPDATES="${N_UPDATES:-2}"
N_STEPS="${N_STEPS:-8}"
N_COND_MSGS="${N_COND_MSGS:-8}"
SWEEP_TAG="${SWEEP_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_TAG="${SWEEP_TAG}_g${GPU_ID}_p${$}"
RESULTS_DIR="outputs/sweep_gen_worldmodel_train_${RUN_TAG}"
mkdir -p "${RESULTS_DIR}"

best_n_envs=0
best_tput=0

for n_envs in ${N_ENVS_CANDIDATES}; do
  run_name="sweep_gen_train_${RUN_TAG}_n${n_envs}"
  echo "[SWEEP] n_envs=${n_envs} gpu=${GPU_ID}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" TMPDIR="${TMPDIR}" "${PYTHON_BIN}" run_gen_worldmodel_pg_train.py \
    --fast_startup \
    --n_envs "${n_envs}" \
    --n_updates "${N_UPDATES}" \
    --n_steps "${N_STEPS}" \
    --n_cond_msgs "${N_COND_MSGS}" \
    --gpu_id "${GPU_ID}" \
    --run_name "${run_name}" || continue

  summary="outputs/gen_worldmodel_pg_train/${run_name}/summary.json"
  if [[ -f "${summary}" ]]; then
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
  fi

done

echo "best_n_envs=${best_n_envs}" | tee "${RESULTS_DIR}/best_profile.txt"
echo "best_mean_steps_per_sec=${best_tput}" | tee -a "${RESULTS_DIR}/best_profile.txt"
