#!/usr/bin/env bash
set -euo pipefail

cd /home/s5e/satyamaga.s5e/JaxMARL-HFT
source .venv/bin/activate

export N_ENVS_CANDIDATES="1 2 4 8"
export POLICY_ARCH="ippo_rnn"
export CHECKPOINT_RESTORE_TOPOLOGY="auto"
export N_UPDATES="2"
export N_STEPS="8"
export N_COND_MSGS="8"
SWEEP_TAG="single_node_profile_$(date +%s)"
export SWEEP_TAG
export PYTHON_BIN="python"

mkdir -p slurm/logs

echo "Starting single-node profile search..."
echo "N_ENVS_CANDIDATES=${N_ENVS_CANDIDATES}"
echo "POLICY_ARCH=${POLICY_ARCH}"
echo "CHECKPOINT_RESTORE_TOPOLOGY=${CHECKPOINT_RESTORE_TOPOLOGY}"
echo "SWEEP_TAG=${SWEEP_TAG}"
echo "Python: $(which python)"
echo ""

for gpu in 0; do
  echo "Running GPU ${gpu}..."
  GPU_ID="${gpu}" \
    N_UPDATES="${N_UPDATES}" \
    N_STEPS="${N_STEPS}" \
    N_COND_MSGS="${N_COND_MSGS}" \
    POLICY_ARCH="${POLICY_ARCH}" \
    CHECKPOINT_RESTORE_TOPOLOGY="${CHECKPOINT_RESTORE_TOPOLOGY}" \
    N_ENVS_CANDIDATES="${N_ENVS_CANDIDATES}" \
    SWEEP_TAG="${SWEEP_TAG}" \
    bash run_sweep_gen_worldmodel_train_single_node.sh
done

echo ""
echo "Aggregating results..."
AGG_OUT="outputs/gen_worldmodel_pg_train/single_node_sweep_aggregate_${SWEEP_TAG}.json"
python aggregate_gen_worldmodel_pnl.py \
  --glob "outputs/gen_worldmodel_pg_train/sweep_gen_train_${SWEEP_TAG}_g*_p*/summary.json" \
  --output "${AGG_OUT}"
cp "${AGG_OUT}" outputs/gen_worldmodel_pg_train/single_node_sweep_aggregate.json
echo ""
echo "Results saved to: ${AGG_OUT}"
