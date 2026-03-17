#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <profile.env> <smoke|sweep|train-best> [--submit]" >&2
  exit 1
fi

PROFILE_PATH="$1"
MODE="$2"
SUBMIT_FLAG="${3:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ ! -f "${PROFILE_PATH}" ]]; then
  echo "ERROR: profile file not found: ${PROFILE_PATH}" >&2
  exit 1
fi

# Export profile key-value pairs into current environment.
set -a
source "${PROFILE_PATH}"
set +a

case "${MODE}" in
  smoke)
    JOB_SCRIPT="slurm/sbatch_smoke_gen_worldmodel_train.sh"
    ;;
  sweep)
    JOB_SCRIPT="slurm/sbatch_sweep_gen_worldmodel_single_node.sh"
    ;;
  train-best)
    JOB_SCRIPT="slurm/sbatch_train_gen_worldmodel_best.sh"
    ;;
  *)
    echo "ERROR: mode must be one of: smoke, sweep, train-best" >&2
    exit 1
    ;;
esac

echo "Profile loaded: ${PROFILE_PATH}"
echo "Mode: ${MODE}"
echo "Job script: ${JOB_SCRIPT}"
echo "Key settings:"
echo "  POLICY_ARCH=${POLICY_ARCH:-unset}"
echo "  CHECKPOINT_RESTORE_TOPOLOGY=${CHECKPOINT_RESTORE_TOPOLOGY:-unset}"
echo "  MM_ACTION_SPACE=${MM_ACTION_SPACE:-unset}"
echo "  MM_BOB_V0=${MM_BOB_V0:-unset}"
echo "  LR=${LR:-unset} ENTROPY_COEF=${ENTROPY_COEF:-unset} VALUE_COEF=${VALUE_COEF:-unset}"

if [[ "${SUBMIT_FLAG}" == "--submit" ]]; then
  sbatch "${JOB_SCRIPT}"
else
  echo
  echo "Dry-run only. To submit:"
  echo "  $0 ${PROFILE_PATH} ${MODE} --submit"
fi
