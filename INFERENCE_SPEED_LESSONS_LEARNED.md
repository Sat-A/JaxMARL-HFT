# Skills & Lessons Learned: Inference Speed (Cluster Edition)

Date: 2026-03-16  
Project: JaxMARL-HFT world-model inference and learned-policy rollout

## Executive Summary

For this stack, end-to-end latency is dominated by process startup and first-step JAX/XLA warmup, not by per-step token generation. Throughput improves materially when each process performs more work (`n_envs`, longer rollouts) and when runs stay on a warm process.

## Key Findings

- **Warmup dominates first step**: first-step compile can take minutes.
- **Steady-state is much faster**: policy inference becomes low-latency after JIT.
- **World-model memory dominates**: GPU memory is mostly consumed by LOBS5; per-trajectory overhead is small.
- **Multi-GPU helps throughput, not single-request latency**.

## What Works Reliably

- `--fast_startup` for lower startup reservation.
- Compilation cache reuse (`JAX_COMPILATION_CACHE_DIR`).
- Batch trajectories per process (`--n_envs` > 1 when stable).
- Greedy sampling (`sample_top_n=1`) for speed-focused runs.
- Short smoke runs before larger sweeps.

## Current Cluster Defaults

- LOBS5 repo: `/home/s5e/satyamaga.s5e/LOBS5`
- World-model checkpoint:  
  `/lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_H1-scaling-law/checkpoints/j2514440_bkotgtm5_2514440`
- Dataset root: `/lus/lfs1aip2/projects/s5e/lob_preproc/GOOG`
- Test window: `2026-01-01` to `2026-01-31` (enforced by date flags)

## Recommended Validation Sequence

1. Run one short single-node smoke rollout.
2. Verify summary artifact and job logs.
3. Run bounded sweep on one node.
4. Scale only after stability is confirmed.

## Slurm-First Commands

```bash
# bounded submission (respects max 5 active jobs)
bash slurm/submit_smoke_jobs.sh
```

```bash
# rollout smoke
sbatch slurm/sbatch_smoke_worldmodel_rollout.sh
```

```bash
# training smoke
sbatch slurm/sbatch_smoke_train.sh
```

## Direct Script Smoke (without Slurm wrapper)

```bash
python run_learned_mm_worldmodel_rollout.py \
  --fast_startup \
  --n_cond_msgs 8 \
  --n_steps 10 \
  --n_envs 1 \
  --start_date 2026-01-01 \
  --end_date 2026-01-31 \
  --policy_deterministic \
  --allow_obs_pad \
  --run_name learned_smoke
```

## Operational Constraints

- Submit **Slurm jobs only**.
- Use **one node per job**.
- Keep **at most 5 active jobs** at a time.

## Next Optimization Targets

- Reduce message-build overhead in rollout loop.
- Add persistent worker mode for repeated inference requests.
- Extend quality checks once trained policy checkpoints are selected for production candidates.
