# Next Steps Plan (Current)

## Goal
Train IPPO-style logic on the generative world-model simulator and promote only profiles that satisfy **quality + stability + throughput**.

## What is done
1. Added checkpoint restore topology controls in `run_gen_worldmodel_pg_train.py`:
   - `--checkpoint_restore_topology {auto, strict, single-device-remap}`
   - auto fallback metadata logged in summary.
2. Added local recurrent IPPO-style policy path (`--policy_arch ippo_rnn`) without importing `ippo_rnn_JAXMARL.py` (avoids `hydra` dependency).
3. Hardened sweep execution for interruption safety:
   - `MAX_PARALLEL_GPUS` throttling in Slurm sweep wrapper.
   - per-profile retry (`RETRY_PER_PROFILE`), progress state logging, resumable tags (`RESUME_SWEEP=1`), and skip completed profiles.

## Current blocker status
1. Latest resilient sweep attempts still produced no completed profile summaries (`best_n_envs=0`).
2. Error logs continue to show checkpoint topology mismatch traces and GPU solver pressure in failed runs.
3. Current aggregate file remains from older successful run snapshot, not from latest fixed sweep pipeline.

## Immediate next steps
1. Run a **single-GPU, single-profile control job** (no multi-GPU wrapper) to isolate remaining restore/runtime failure:
   - `GPU_ID=0 N_ENVS_CANDIDATES="1" CHECKPOINT_RESTORE_TOPOLOGY=single-device-remap POLICY_ARCH=ippo_rnn bash run_sweep_gen_worldmodel_train_single_node.sh`
2. If control job succeeds, expand to full sweep with safe settings:
   - `MAX_PARALLEL_GPUS=1 RESUME_SWEEP=1 CHECKPOINT_RESTORE_TOPOLOGY=auto sbatch slurm/sbatch_sweep_gen_worldmodel_single_node.sh`
3. Promote profile only after new aggregate JSON is produced from the latest sweep tag.

## After sweep unblocks
1. Launch long-run training: `slurm/sbatch_train_gen_worldmodel_best.sh` with selected `N_ENVS_BEST`.
2. Apply holdout quality gates (Jan-2026): positive average PnL and non-zero trade incidence across seeds.
3. Then run throughput A/B optimization and optional W&B long-run tracking.
