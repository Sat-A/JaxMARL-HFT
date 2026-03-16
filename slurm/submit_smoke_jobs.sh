#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/s5e/satyamaga.s5e/JaxMARL-HFT}"
cd "${REPO_ROOT}"

MAX_ACTIVE_JOBS="${MAX_ACTIVE_JOBS:-5}"
active_jobs="$(squeue -u "${USER}" -h | wc -l)"

if (( active_jobs >= MAX_ACTIVE_JOBS )); then
  echo "Refusing submission: active jobs=${active_jobs}, limit=${MAX_ACTIVE_JOBS}."
  exit 1
fi

remaining_slots=$((MAX_ACTIVE_JOBS - active_jobs))
echo "Active jobs: ${active_jobs}. Remaining slots: ${remaining_slots}."

scripts=(
  "slurm/sbatch_smoke_worldmodel_rollout.sh"
  "slurm/sbatch_smoke_train.sh"
)

submitted=0
for script in "${scripts[@]}"; do
  if (( submitted >= remaining_slots )); then
    echo "Reached submission cap for this run (${remaining_slots})."
    break
  fi
  sbatch "${script}"
  submitted=$((submitted + 1))
done

echo "Submitted ${submitted} job(s)."
