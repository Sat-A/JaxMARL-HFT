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

CKPT_PATH="${CKPT_PATH:-/lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_H1-scaling-law/checkpoints/j2514440_bkotgtm5_2514440}"
DATA_DIR="${DATA_DIR:-/lus/lfs1aip2/projects/s5e/lob_preproc/GOOG}"
LOBS5_ROOT="${LOBS5_ROOT:-/home/s5e/satyamaga.s5e/LOBS5}"
START_DATE="${START_DATE:-2026-01-01}"
END_DATE="${END_DATE:-2026-01-31}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
N_COND_MSGS="${N_COND_MSGS:-64}"
CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$SCRIPT_DIR/.cache/jax_compilation_speedtest}"

# One lightweight index per GPU to trigger compile/autotune on each device.
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
"${PYTHON_BIN}" run_one_step_inference_multigpu.py \
  --ckpt_path "$CKPT_PATH" \
  --data_dir "$DATA_DIR" \
  --stock GOOG \
  --lobs5_root "$LOBS5_ROOT" \
  --sample_indices 0,1,2,3 \
  --gpu_ids "$GPU_IDS" \
  --n_cond_msgs "$N_COND_MSGS" \
  --sample_top_n 1 \
  --start_date "$START_DATE" \
  --end_date "$END_DATE" \
  --run_name_prefix warmup \
  --compile_cache_dir "$CACHE_DIR" \
  --fast_startup
