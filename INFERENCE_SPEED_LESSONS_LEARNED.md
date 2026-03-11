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

## Random-Policy Long-Run Stress Tests

### 100-Step Note

- A 100-step random-policy run was started and progressed substantially (well past step 60), showing at least one regime change in spread/midprice behavior mid-rollout.
- The job was intentionally terminated on user request before final artifact flush — treat that trace as partial, not a complete benchmark.

### Book-Crash Safeguards (added 2026-03-11)

- **Root cause**: random actions can produce limit orders with price <= 0 (agent quoting relative to a stale or zero midprice). A zero-price bid enters the book and clears the bid side, making `best_bid = -1`. The next step then computes midprice = (−1 + ask) / 2, an artificial ~50% crash, and also quotes the next action relative to that garbage midprice — compound cascade.
- **Fix applied** in `minimal_agent_generative_step.py`:
  1. `_sanitize_action_msgs`: converts any type-1 (limit-add) order with price <= 0 to a no-op before simulator ingestion.
  2. Agent-action revert guard: if best_bid or best_ask goes <= 0 after applying agent messages, roll back to pre-action state.
  3. Generated-message revert guard: same check after the LOBS5-generated message is applied.
- **Lesson**: always guard simulator state validity at both ingestion points (agent orders and generator output) when running unsupervised random or exploratory policies. The historical-sequence replay already filtered for bid > 0 && ask > 0 but the rollout loop did not.

### 1000-Step Verification Run

- Command: `CUDA_VISIBLE_DEVICES=4,5,6,7 python minimal_agent_generative_step.py --fast_startup --n_cond_msgs 8 --sample_index 0 --sample_top_n 1 --n_steps 1000 --action_policy random --seed 42 --run_name verify_1kstep_random_gpus4567_safeguards`
- Purpose: verify that the book-crash safeguards keep the simulator stable across 1000 random-action steps.
- Run in progress; results will be reflected in the output directory once complete.
