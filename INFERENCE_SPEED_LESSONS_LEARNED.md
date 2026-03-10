# Inference Speed Lessons Learned

Date: 2026-03-10
Project: JaxMARL-HFT one-step LOBS5 inference workflow

## Executive Summary

The main latency issue is not token generation. Most runtime is spent in startup and JAX/XLA compile-autotune work that repeats for each fresh process.

## Measured Results

### Single-sample comparison (same settings, only generated steps changed)

- Settings:
  - checkpoint: twilight-sound-77_s42sujip (step 23)
  - data: GOOG 2022
  - n_cond_msgs: 64
  - sampling: greedy (sample_top_n=1)
  - fast startup: enabled

| Scenario | Generated Steps | Sample Time (s) | Total Time (s) |
|---|---:|---:|---:|
| Baseline | 1 | 123.662 | 185.326 |
| Comparison | 5 | 124.477 | 187.995 |

Derived:
- Additional sample-time from 1 to 5 steps: 0.815 s
- Additional total-time from 1 to 5 steps: 2.669 s
- Approx incremental sample-time per extra step: 0.204 s/step

Interpretation:
- Extra generation steps add only a small cost.
- Fixed overhead dominates runtime.

### Throughput experiments

- 4 samples in one process (compile amortized):
  - total about 204 s for 4 outputs
  - effective about 51 s per sample
- 4 samples on 4 GPUs in parallel processes:
  - wall time about 194 s for 4 outputs

Interpretation:
- Multi-GPU improves throughput, not single-request latency.
- Running more work per process amortizes compile cost and helps effective per-sample speed.

## Root Causes of Slow Runtime

1. Per-process JAX/XLA trace, lowering, compile, and autotuning overhead.
2. Model restore and initialization cost each run.
3. Long conditioning path processing before generation.
4. Fresh-process execution pattern repeats expensive setup.

## What Helped

1. Fast startup runtime defaults:
   - XLA_PYTHON_CLIENT_PREALLOCATE=false
   - XLA_PYTHON_CLIENT_MEM_FRACTION=0.50
2. Compile cache configuration:
   - JAX_COMPILATION_CACHE_DIR set and reused
3. Batch in one process:
   - process multiple sample indices in a single invocation
4. Multi-GPU parallel launcher:
   - run separate jobs pinned to different GPUs
5. Greedy decoding for speed:
   - sample_top_n=1

## What Did Not Help Enough by Itself

1. Compilation cache across fresh process restarts did not materially reduce total latency in this path.

## Practical Recommendations

1. If goal is lowest latency for one request:
   - Use a persistent worker process so compile happens once, then serve many requests.
2. If goal is highest throughput:
   - Use multi-GPU parallel execution with one process per GPU.
3. If quality allows:
   - Keep n_cond_msgs lower (for example 64 vs 500) and use greedy decode.
4. For reproducible speed benchmarking:
   - Keep checkpoint, sample index, n_cond_msgs, and sample_top_n fixed.

## Artifacts and Scripts Added During Optimization

- run_one_step_inference.py
  - Added n_gen_msgs, n_samples, batch_size, sample_indices, fast_startup, compile_cache_dir
  - Added detailed timing breakdown in run summary
- run_one_step_inference.sh
  - Startup defaults tuned for faster launch
- run_one_step_inference_multigpu.py
  - Parallel one-step jobs across selected GPUs
- run_one_step_warmup_multigpu.sh
  - Warmup/precompute helper across GPUs

## Future Work (Highest Impact)

1. Implement persistent inference server mode in-process (single compile warmup, repeated requests).
2. Keep worker pinned to a dedicated GPU and queue requests.
3. Optional: profile inside LOBS5 generate path to reduce compile graph size where possible.
