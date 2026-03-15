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

---

# Learned-Policy World-Model Rollout — Phase 4 Throughput Sweep

Date: 2026-03-15
Script: run_learned_mm_worldmodel_rollout.py
GPU: Single A100 (46068 MiB)
Settings: n_cond_msgs=8, greedy decode (sample_top_n=1), seed=42, n_steps=15

## Critical Discovery: Policy JIT Fix

Before fix: `policy.act()` called `train_state.apply_fn` in Python (not JIT-compiled).
After fix: `_apply_jit = jax.jit(train_state.apply_fn)` set once in `__init__`.

| Metric | Before (eager) | After (JIT) | Improvement |
|--------|---------------:|------------:|-------------|
| policy_latency p50 | 230 ms | 1.7 ms | **133× faster** |
| step_latency p50 (steady-state) | 514 ms | 247 ms | **2× faster** |

## Stage A Sweep Results — n_envs=1,2,4 (single GPU, sequential multi-trajectory)

| n_envs | agg steps/s | step_p50 (ms) | policy_p50 (ms) | build_p50 (ms) | gen_p50 (ms) | peak_mem (MiB) |
|-------:|------------:|--------------:|----------------:|---------------:|-------------:|---------------:|
| 1      | 0.092       | 247           | 1.72            | 121            | 21.3         | 36873          |
| 2      | 0.173       | 266           | 2.26            | 139            | 22.6         | 36873          |
| 4      | 0.324       | 263           | 1.78            | 125            | 21.1         | 36873          |

Key observations:
- Aggregate throughput scales linearly with n_envs (confirmed sequential independence).
- Peak memory stays flat: per-trajectory overhead is negligible (~14 MiB) vs 36.8 GB model.
- OOM boundary not reached at n_envs=4; higher values (8, 16) expected stable.
- First-step warmup (JAX JIT compile): ~151 seconds, amortized across all n_envs.
- p95 decreases with n_envs: warmup cost is fixed while total work grows.

## Steady-State Step Breakdown (n_envs=1, JIT-fixed, ms)

| Phase | p50 | Notes |
|-------|----:|-------|
| policy inference | 1.7 | JIT-compiled GRU, near-zero overhead |
| message build | 121 | mm_agent.get_messages() + _build_world_state(), JAX ops |
| simulator apply | 1.7 | JaxLOB process_orders_array, fast |
| world-model generate | 21 | LOBS5 SSM sequential generation |
| postprocess/IO | 5 | sim update + state bookkeeping |
| unaccounted overhead | ~96 | _best_quotes calls, Python dispatch, JAX blocking |
| **total step** | **247** | |

Current bottleneck: `message_build` at ~121 ms. Next optimization target.

## Phase 5 Decisions

- **Production candidate batch size**: n_envs=4 (proven stable, 4× throughput vs bsz=1).
- **Safe fallback**: n_envs=1.
- **Memory constraint**: Not active — model dominates at 80% GPU utilization.
- **Memory headroom**: ~9 GB available for future batching if model is kept fixed.
- **Scaling out**: Use one process per GPU (as previously recommended), set n_envs≥4 per process.
- **Next optimization**: JIT `mm_agent.get_messages()` call chain to cut 121 ms build phase.

## Artifacts Added

- run_learned_mm_worldmodel_rollout.py — full learned-policy async rollout script
  - LearnedPolicyAdapter: checkpoint restore, GRU policy inference, JIT-compiled act()
  - act_with_state() + fresh_hidden() for stateless multi-env policy management
  - --n_envs flag for parallel trajectory count (sequential multi-env loop)
  - full timing instrumentation, memory telemetry, action histograms, PnL reporting
- run_sweep_single_gpu.sh — Phase 4 sweep driver (Stage A stability + Stage C variance)

## Quick-Start Commands

```bash
# Smoke test (10 steps, single env)
CUDA_VISIBLE_DEVICES=0 python run_learned_mm_worldmodel_rollout.py \
  --fast_startup --n_cond_msgs 8 --n_steps 10 --n_envs 1 \
  --policy_deterministic --allow_obs_pad --run_name smoke_test

# Throughput mode (n_envs=4, single GPU)
CUDA_VISIBLE_DEVICES=0 python run_learned_mm_worldmodel_rollout.py \
  --fast_startup --n_cond_msgs 8 --n_steps 25 --n_envs 4 \
  --policy_deterministic --allow_obs_pad --run_name throughput_n4

# Full Phase 4 sweep
cd /homes/80/satyam/JaxMARL-HFT && bash run_sweep_single_gpu.sh
```

## Known Risks and Next Experiments

1. Action diversity is zero with `dummy-2vdmzbye` checkpoint (argmax of random weights).
   Replace with a trained checkpoint to get meaningful trade incidence and quality signals.
2. Trade incidence is 0 — needs real policy weights to validate fill-rate guardrail.
3. message_build at 121 ms (49% of step time): candidate for JIT wrap around get_messages().
4. Stage C (seed variance) not yet run for n_envs>1 — run sweep script to collect.
5. n_envs=8+ not yet probed — likely stable given flat memory profile, run to confirm.


## Heuristic vs Learned Policy Comparison (Structural)

| Property | Heuristic (random/fixed) | Learned (dummy ckpt) | Learned (trained ckpt) |
|---|---|---|---|
| action diversity | full (uniform random) | zero (argmax of random weights = constant) | expected non-trivial |
| trade incidence | occasional | 0 | needed for quality validation |
| PnL signal | stochastic | 0 | needed for Phase 5 |
| throughput | same as learned | same | same |

Until a trained checkpoint replaces `dummy-2vdmzbye`, all quality metrics (PnL, fill rate, action diversity) remain uninformative. Throughput numbers are valid regardless of checkpoint.

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

### 1000-Step Market-Making Run (GPU 0 only)

- Date: 2026-03-15
- Command:
   - `CUDA_VISIBLE_DEVICES=0 /homes/80/satyam/miniconda3/envs/jaxmarl_hft/bin/python minimal_agent_generative_step.py --fast_startup --n_cond_msgs 8 --sample_index 0 --sample_top_n 1 --n_steps 1000 --action_policy market_making --seed 42 --run_name verify_1kstep_market_making_gpu0`
- Policy/config:
   - `action_policy=market_making`
   - MM action space switched to `bobStrategy` in `minimal_agent_generative_step.py`
- Output dir:
   - `/scratch/local/homes/80/satyam/JaxMARL-HFT/outputs/minimal_agent_generative_step/verify_1kstep_market_making_gpu0_action0_sample0`

Measured runtime (from `verification_summary.json`):
- Total runtime: `415.237 s`
- Rollout total: `314.370 s` (`0.314 s/step`)
- Rollout generate total: `166.533 s` (`0.167 s/step`)
- Rollout action total: `135.158 s` (`0.135 s/step`)
- Rollout post total: `10.465 s` (`0.010 s/step`)

PnL snapshot:
- Total PnL: `0.0`
- Cash PnL: `0.0`
- Inventory MTM: `0.0`
- Agent trades: `0`

Top bottlenecks:
1. `rollout_generate`: `166.533 s` (`40.11%` of total)
2. `rollout_action`: `135.158 s` (`32.55%` of total)
3. `rollout_post`: `10.465 s` (`2.52%` of total)

Notes:
- Run completed successfully for all 1000 steps on GPU 0.
- Action trace remained all zeros, with no executed MM trades in this trajectory; resulting PnL stayed flat.
- Compared with prior observations, generation remains a major bottleneck, and action processing is also substantial in long rollouts.

## Investigation: Why No Trades in 1k Market-Making Run

Date: 2026-03-15

Findings from `verify_1kstep_market_making_gpu0_action0_sample0`:

1. Empirical run artifacts indicate a near-static rollout:

## Next Phase Bootstrap (2026-03-15)

Implementation started for `plan-learnedPolicyNextPhase.prompt.md` with the following concrete changes:

1. Throughput sweep expansion
   - `run_sweep_single_gpu.sh` Stage A now probes `n_envs=(1 2 4 8 16 32)` before selecting `stable_max`.
   - This closes the previous gap where Stage A stopped at `n_envs=8`.

2. message_build JIT entry point
   - `run_learned_mm_worldmodel_rollout.py` now supports `--jit_message_build`.
   - When enabled, world-state construction + `mm_agent.get_messages(...)` run through a JIT-compiled function.
   - Safety behavior: if JIT compile fails, the script logs a warning and falls back to the existing eager path.
   - Summary telemetry now records:
     - `timing_breakdown.message_build_jit_enabled`
     - `timing_breakdown.message_build_jit_compile_sec`

3. Production command templates
   - README now includes a “Production Deployment” section with:
     - quality smoke command for trained checkpoints,
     - throughput candidate command using `--jit_message_build`,
     - OOM boundary sweep command.

Pending measurements to finalize this phase:
- Verify `message_build_latency_ms_p50` reduction with `--jit_message_build`.
- Run Stage A/B/C sweep with trained checkpoint to produce quality + throughput selection.
- Confirm seed variance at `stable_max` is below 15%.

## Supercomputer Start Note (2026-03-15)

When moving to a stronger multi-node system, start in this sequence:

1. Validate one-node end-to-end first (real checkpoint + `--jit_message_build`, `n_envs=1`, `n_steps=50`).
2. Recompute Stage A stability boundary on that hardware (`n_envs=1,2,4,8,16,32`) before scaling out.
3. Scale to multi-node with one process per GPU and fixed `stable_max` from Step 2.
4. Use disjoint seeds/sample ranges per process and merge all `summary.json` artifacts for aggregate throughput/variance.

Tomorrow default starting point:
- Use `policy_ckpt_dir=/scratch/local/homes/groups/finance/data/checkpoints/MARLCheckpoints/2PLayer/whole-sweep-1`
- Start with `n_envs=32` + `--jit_message_build`
- Track three metrics first: `samples_per_sec`, `message_build_latency_ms_p50`, and Stage C variance at stable max.
   - `action_trace` contained only action `0`.
   - `step_trace.csv` had no best-quote changes after action (`orderbook_changed_after_action=False` on all steps).
   - Generated message `event_type` was always `4` in `generated_message.csv`.
   - `agent_trade_count=0`, total PnL remained `0.0`.

2. Root-cause in action/cancel handling for this minimal script path:
   - `minimal_agent_generative_step.py` calls `mm_agent.get_messages(...)` and receives `(action_msgs, cancel_msgs, extras)`.
   - The script applies only `action_msgs` to the book and ignores `cancel_msgs`.
   - Inside `mm_env.py`, `get_messages` already nets actions vs cancel messages via `_filter_messages`.
   - With repeated same-price quotes, `_filter_messages` reduces later `action_msgs` to zero-quantity dummy rows.
   - Result in this run: only first step had nonzero limit-add messages; from step 2 onward, action messages were effectively no-ops.

3. Learning status clarification:
   - This minimal script is **inference-only** and does not run PPO/IPPO updates.
   - It does not create/apply gradients or update policy parameters.
   - The parallel learning logic lives in `gymnax_exchange/jaxrl/MARL/ippo_rnn_JAXMARL.py` (and pmap variant), where `TrainState.apply_gradients(...)` is executed in the training loop.

Practical takeaway:
- The 1k run used a fixed heuristic market-making policy wrapper in an inference loop, not a learning agent. For meaningful MM fills in this path, action and cancellation integration should match environment step semantics (apply cancel+action together) or bypass `_filter_messages` assumptions when cancels are not consumed.

## Learned MM World-Model Rollout Telemetry Upgrade

Date: 2026-03-15

Code changes completed in `run_learned_mm_worldmodel_rollout.py`:
- Added full phase timing split telemetry:
   - policy inference
   - message build/filter
   - simulator apply
   - world-model generate
   - postprocess
- Added throughput counters:
   - total samples processed
   - samples/sec and steps/sec
   - generated messages/sec
- Added policy behavior counters:
   - action histogram
   - placed order message count
   - trade incidence and fill-rate proxy
   - inventory and PnL traces
   - final PnL snapshot
- Added memory telemetry checkpoints via `nvidia-smi`:
   - post-init
   - post-first-step
   - peak used MiB

Validation run:
- Command: `cd /homes/80/satyam/JaxMARL-HFT && python -m py_compile run_learned_mm_worldmodel_rollout.py`
- Result: pass (exit code `0`)

Operational takeaway:
- The learned-policy rollout script now emits the core instrumentation required for Phase 3 analysis and for comparing warmup vs steady-state behavior before running the single-GPU batch-size sweep protocol.

## Learned MM Smoke Run (10 steps, GPU 0)

Date: 2026-03-15

Initial runtime issue observed:
- First smoke run failed with `ScopeParamShapeError` due to action-head mismatch:
  - checkpoint policy head width = `13`
  - script default `--policy_action_dim` = `5`
- Fix applied in `run_learned_mm_worldmodel_rollout.py`:
  - infer action dim from restored params at `params["params"]["SingleActionOutput_0"]["Dense_0"]["kernel"].shape[1]`
  - rebuild `ActorCriticRNN` with inferred action dim
  - rebind `train_state.apply_fn` to the rebuilt network

Verification run (post-fix):
- Command:
  - `CUDA_VISIBLE_DEVICES=0 /homes/80/satyam/miniconda3/envs/jaxmarl_hft/bin/python run_learned_mm_worldmodel_rollout.py --fast_startup --n_cond_msgs 8 --sample_index 0 --sample_top_n 1 --n_steps 10 --seed 42 --gpu_id 0 --policy_deterministic --allow_obs_pad --run_name learned_mm_smoke_10step_gpu0`
- Result: pass (run completed, summary emitted)
- Summary artifact:
  - `/scratch/local/homes/80/satyam/JaxMARL-HFT/outputs/learned_mm_worldmodel_rollout/learned_mm_smoke_10step_gpu0/summary.json`

Key measured outcomes:
- `policy_action_dim=13` (auto-inferred from checkpoint)
- action histogram: `{2: 10}`
- throughput: `0.0645 samples/sec`
- warmup first step latency: `143838.15 ms`
- steady-state step latency p50/p95: `514.04 / 4487.62 ms`
- generate latency p50: `20.46 ms` (first-step compile dominates p95)
- placed order msgs: `2`, agent trades: `0`, fill-rate proxy: `0.0`
- memory: post-init `36859 MiB`, post-first-step `36867 MiB`, peak `36873 MiB`

Practical takeaway:
- Integration now runs end-to-end with checkpoint-compatible action dimensions and emits all Phase 3 telemetry; the dominant cost remains first-step compile/warmup, so throughput decisions should rely on steady-state metrics and repeated-seed runs.
