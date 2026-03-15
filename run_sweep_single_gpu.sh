#!/usr/bin/env bash
# Phase 4 — Single-GPU throughput sweep for learned MM world-model rollout.
#
# Stage A: stability probe — small candidate batch sizes (n_envs)
# Stage B: (manual) refinement around boundary found in Stage A
# Stage C: repeat best 2-3 settings across seeds

set -euo pipefail

PYTHON="${PYTHON:-/homes/80/satyam/miniconda3/envs/jaxmarl_hft/bin/python}"
SCRIPT="run_learned_mm_worldmodel_rollout.py"
GPU_ID="${GPU_ID:-0}"
N_STEPS="${N_STEPS:-25}"        # enough steps for steady-state signal (>10)
N_COND_MSGS="${N_COND_MSGS:-8}"
SAMPLE_TOP_N="${SAMPLE_TOP_N:-1}"
SAMPLE_INDEX="${SAMPLE_INDEX:-0}"
POLICY_CKPT_DIR="${POLICY_CKPT_DIR:-}"
JIT_MESSAGE_BUILD="${JIT_MESSAGE_BUILD:-0}"
RESULTS_DIR="outputs/sweep_single_gpu_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$RESULTS_DIR"
SWEEP_LOG="$RESULTS_DIR/sweep_log.txt"

log() { echo "[SWEEP] $*" | tee -a "$SWEEP_LOG"; }

run_config() {
    local n_envs="$1"
    local seed="$2"
    local run_name="sweep_n${n_envs}_s${seed}"
    local jit_flag=""
    local policy_ckpt_args=()

    if [[ "${JIT_MESSAGE_BUILD}" == "1" ]]; then
        jit_flag="--jit_message_build"
    fi

    if [[ -n "${POLICY_CKPT_DIR}" ]]; then
        policy_ckpt_args=(--policy_ckpt_dir "${POLICY_CKPT_DIR}")
    fi

    log "Running n_envs=${n_envs} seed=${seed} -> ${run_name}"

    CUDA_VISIBLE_DEVICES="${GPU_ID}" timeout 1800 \
        "${PYTHON}" "${SCRIPT}" \
            --fast_startup \
            --n_cond_msgs "${N_COND_MSGS}" \
            --sample_index "${SAMPLE_INDEX}" \
            --sample_top_n "${SAMPLE_TOP_N}" \
            --n_steps "${N_STEPS}" \
            --seed "${seed}" \
            --gpu_id "${GPU_ID}" \
            --policy_deterministic \
            --allow_obs_pad \
            --n_envs "${n_envs}" \
            "${policy_ckpt_args[@]}" \
            ${jit_flag} \
            --run_name "${run_name}" \
        && echo "EXIT:0" >> "$SWEEP_LOG" \
        || { echo "EXIT:$? (n_envs=${n_envs} seed=${seed})" >> "$SWEEP_LOG"; return 1; }
}

summarize_result() {
    local run_name="$1"
    local summary_file
    summary_file=$(find outputs/learned_mm_worldmodel_rollout/"${run_name}" -name "summary.json" 2>/dev/null | head -1)
    if [[ -f "$summary_file" ]]; then
        python3 -c "
import json, sys
d = json.load(open('$summary_file'))
t = d.get('throughput', {})
td = d.get('timing_breakdown', {})
mem = d.get('memory_telemetry', {})
print(f\"  n_envs={d.get('n_envs',1)} steps={d.get('n_steps','?')} samples/s={t.get('samples_per_sec',0):.3f}\")
print(f\"  step_p50={td.get('step_latency_ms_p50',0):.1f}ms p95={td.get('step_latency_ms_p95',0):.1f}ms\")
print(f\"  policy_p50={td.get('policy_latency_ms_p50',0):.2f}ms gen_p50={td.get('generate_latency_ms_p50',0):.2f}ms\")
print(f\"  peak_mem={mem.get('peak_used_mib',0):.0f}MiB total_mib={mem.get('post_init',{}).get('total_mib',0):.0f}MiB\")
" 2>/dev/null | tee -a "$SWEEP_LOG" || true
        cp "$summary_file" "$RESULTS_DIR/${run_name}_summary.json"
    fi
}

log "=== Phase 4 Stage A: Stability probe ==="
log "GPU=${GPU_ID}  n_steps=${N_STEPS}  n_cond_msgs=${N_COND_MSGS}"
log ""

# Stage A: probe n_envs = 1, 2, 4, 8, 16, 32
STAGE_A_ENVS=(1 2 4 8 16 32)
STABLE_MAX=1
for n_envs in "${STAGE_A_ENVS[@]}"; do
    log "--- Stage A: n_envs=${n_envs} seed=42 ---"
    if run_config "$n_envs" 42; then
        summarize_result "sweep_n${n_envs}_s42"
        STABLE_MAX="$n_envs"
        log "STABLE: n_envs=${n_envs} passed"
    else
        log "FAILED/OOM: n_envs=${n_envs} — stopping Stage A at stable_max=${STABLE_MAX}"
        break
    fi
    log ""
done

log ""
log "=== Stage A complete. Stable max n_envs=${STABLE_MAX} ==="
log ""

# Stage C: repeat stable_max across 3 seeds for variance estimate
log "=== Phase 4 Stage C: Variance validation at n_envs=${STABLE_MAX} ==="
for seed in 42 123 777; do
    log "--- Stage C: n_envs=${STABLE_MAX} seed=${seed} ---"
    if run_config "$STABLE_MAX" "$seed"; then
        summarize_result "sweep_n${STABLE_MAX}_s${seed}"
    else
        log "FAILED: n_envs=${STABLE_MAX} seed=${seed}"
    fi
    log ""
done

log ""
log "=== Sweep complete. Results in ${RESULTS_DIR}/ ==="
log "To view summary: cat ${RESULTS_DIR}/sweep_log.txt"
