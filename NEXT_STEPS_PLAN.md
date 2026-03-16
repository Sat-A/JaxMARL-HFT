# Next Steps Plan (Concise)

## Goal
Train a generative-world-model trading agent to achieve consistently positive average PnL while preserving high single-node throughput.

## Phase 1: Throughput Revalidation (Single Node)
1. Re-run `slurm/sbatch_sweep_gen_worldmodel_single_node.sh` with longer settings (`n_updates`, `n_steps`) to confirm the best `n_envs` under realistic training load.
2. Select the profile using both throughput and stability (no job/runtime failures), not throughput alone.

## Phase 2: Policy Quality Training
1. Launch `slurm/sbatch_train_gen_worldmodel_best.sh` with the selected `N_ENVS_BEST`.
2. Increase training horizon (more updates and larger seed set) and aggregate results using `aggregate_gen_worldmodel_pnl.py`.
3. Track mean/median/std of `pnl_total` and keep the best checkpoint profile by average PnL.

## Phase 3: Quality Gates
1. Validate best checkpoint with rollout scripts on the Jan-2026 holdout window.
2. Require non-zero trade incidence and positive average PnL across multiple seeds before promotion.

## Phase 4: Multi-Node Scale-Out (After Single-Node Lock-In)
1. Launch one process per GPU per node with disjoint seeds and fixed config.
2. Aggregate node-level and global summaries from all `summary.json` outputs.
3. Compare variance and throughput scaling efficiency before expanding run count.
