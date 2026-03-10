#!/usr/bin/env bash
set -euo pipefail

# Precompute/prime one-step inference on selected GPUs.
# This pays model load + compile cost up front so subsequent jobs in the same
# workflow/session are throughput-optimized.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONDA_ENV="${CONDA_ENV:-jaxmarl_hft}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

CKPT_PATH="${CKPT_PATH:-/scratch/local/homes/groups/finance/data/checkpoints/lobs5_v2/twilight-sound-77_s42sujip}"
DATA_DIR="${DATA_DIR:-/scratch/local/homes/groups/finance/data/processed_data/GOOG/2022}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
N_COND_MSGS="${N_COND_MSGS:-64}"
CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$SCRIPT_DIR/.cache/jax_compilation_speedtest}"

# One lightweight index per GPU to trigger compile/autotune on each device.
python run_one_step_inference_multigpu.py \
  --ckpt_path "$CKPT_PATH" \
  --data_dir "$DATA_DIR" \
  --stock GOOG \
  --sample_indices 0,1,2,3 \
  --gpu_ids "$GPU_IDS" \
  --n_cond_msgs "$N_COND_MSGS" \
  --sample_top_n 1 \
  --run_name_prefix warmup \
  --compile_cache_dir "$CACHE_DIR" \
  --fast_startup
