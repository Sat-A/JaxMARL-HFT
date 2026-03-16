# Copilot Instructions for JaxMARL-HFT

This document provides essential context for AI assistants working on this GPU-accelerated multi-agent reinforcement learning framework for high-frequency trading.

## Setup & Environment

**Before running any code:**
```bash
export PYTHONPATH="$(pwd):$PYTHONPATH"
conda create -n jaxmarl_hft python=3.10
conda activate jaxmarl_hft
pip install "jax[cuda12]"
pip install -r requirements.txt
```

**Data Requirements:**
- LOBSTER limit order book data as matched CSV pairs (message + orderbook files)
- Expected directory structure: `data/rawLOBSTER/<STOCK>/<TIME_PERIOD>/`
- Files must be named: `<STOCK>_<DATE>_34200000_57600000_{message,orderbook}_10.csv`

**Current cluster defaults (supercomputer):**
- LOBS5 repo: `/home/s5e/satyamaga.s5e/LOBS5`
- World-model checkpoint: `/lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_H1-scaling-law/checkpoints/j2514440_bkotgtm5_2514440`
- Dataset root: `/lus/lfs1aip2/projects/s5e/lob_preproc/GOOG`
- Test date window: `2026-01-01` to `2026-01-31` (use `--start_date/--end_date`)
- Slurm constraints: one node per job, max 5 active jobs.

**Current operational status (2026-03-16):**
- Cluster smoke rollout and smoke generative training are validated.
- Single-node 4-GPU sweep + best-profile multi-seed training pipeline is validated.
- Current best throughput profile in generative training: `n_envs=1` (longer run set).

## High-Level Architecture

### Three-Layer Environment Stack

1. **BaseLOBEnv** (`gymnax_exchange/jaxen/base_env.py`)
   - Handles LOBSTER data loading and caching (NPZ format)
   - Manages world state: order book, market messages, episode resets
   - JIT-compiled state transitions
   - Generates preprocessed reset states (saved as PKL files)

2. **Agent Environments** (inherit from BaseLOBEnv)
   - **MarketMakingAgent** (`gymnax_exchange/jaxen/mm_env.py`): Liquidity provision with spread/skew-based actions
   - **ExecutionAgent** (`gymnax_exchange/jaxen/exec_env.py`): Large order execution with quantity/price action spaces
   - Each maintains per-agent state (inventory, PnL, action history)

3. **MARLEnv** (`gymnax_exchange/jaxen/marl_env.py`)
   - Multi-agent orchestrator that composes all agent types
   - Distributes observations and rewards to agents
   - Returns step contract: `(obs_list, state, reward_list, dones, info)`

### Configuration System

**Dataclass Hierarchy** (`gymnax_exchange/jaxob/jaxob_config.py`):
- `MultiAgentConfig` (root) → contains:
  - `World_EnvironmentConfig`: Global paths, LOBSTER settings, market hours, episode length
  - `dict_of_agents_configs`: Maps agent types to their configs
    - `MarketMaking_EnvironmentConfig`: Action/obs/reward spaces for MM agents
    - `Execution_EnvironmentConfig`: Task types, action/reward spaces for execution agents

**Settings Precedence** (highest to lowest):
1. WandB sweep overrides (`SWEEP_PARAMETERS` in YAML)
2. Hydra CLI arguments (e.g., `python script.py NUM_ENVS=256`)
3. RL YAML config (`config/rl_configs/*.yaml`)
4. Environment JSON config (`config/env_configs/*.json`)
5. Dataclass defaults

**Config Loading Flow:**
1. Hydra loads YAML → DictConfig
2. `load_config_from_file()` reads env JSON → MultiAgentConfig dataclass
3. `OmegaConf.merge()` overlays YAML onto dataclass config
4. Training script receives merged config

### Data & State Flow

**Preprocessing (first run only):**
1. LOBSTER CSVs loaded → cleaned and cached as NPZ
2. Reset states precomputed and cached as PKL
3. Subsequent runs load from cache (1000x speedup)

**Episode Structure:**
- Messages streamed from LOBSTER at configurable resolution
- Each step: world processes N data messages → agents take actions → new messages posted
- Episode length and data windows controlled by `world_config` (e.g., `start_resolution`, `n_data_msg_per_step`)

**Training Loop** (IPPO-RNN in `ippo_rnn_JAXMARL.py`):
1. Parallel rollouts across `NUM_ENVS` environments
2. Per-agent-type networks with separate GAE/PPO updates
3. GRU-based recurrent policy (state preserved across steps)
4. Orbax checkpoints saved to `checkpoints/MARLCheckpoints/<PROJECT>/<RUN>/`

## Key Conventions

### JAX-Specific Patterns

- **JIT Compilation**: Static decorators on methods that don't change. Use `@partial(jax.jit, static_argnums=(0,))` for instance methods.
- **Vectorization**: `jax.vmap` used for parallel agent stepping within environments; `vmap` composition for episode batching.
- **XLA Memory**: Environment variables set to 95% allocation with preallocation (`XLA_PYTHON_CLIENT_MEM_FRACTION`, `XLA_PYTHON_CLIENT_PREALLOCATE`).
- **Config Updates**: JAX config updates (e.g., disable JIT for debugging) must happen before JAX imports.

### Configuration Patterns

- **Agent Type Identification**: `MarketMaking_EnvironmentConfig` vs `Execution_EnvironmentConfig` instances; use `isinstance()` to branch.
- **List-Valued Hyperparameters**: Fields like `LR`, `GAMMA`, `GAE_LAMBDA` are lists in YAML, indexed by agent type; length must match `NUM_AGENTS_PER_TYPE`.
- **Action/Obs Spaces**: Determined by agent config fields (e.g., `action_space="bobRL"`, `observation_space="engineered"`). Post-init logic computes derived fields like `n_actions`.
- **Reward Functions**: Named string selectors (e.g., `reward_function="spooner_asym_damped2"`); logic often branches on this string.

### File & Path Conventions

- **Relative Imports**: Use `gymnax_exchange.` prefix (e.g., `from gymnax_exchange.jaxen import marl_env`).
- **Output Artifacts**: Saved under `<alphatradePath>/`:
  - `saved_npz/`: Preprocessed LOBSTER cache
  - `pre_reset_states/`: Precomputed reset PKL files
  - `checkpoints/MARLCheckpoints/`: Orbax training checkpoints
  - `outputs/`: Experiment results (sweeps, rollouts, verification)
- **Config Files**: JSON for environment, YAML for RL training (Hydra-managed).

### Observation & Reward Naming

- **Observations**: Agent obs includes world state snapshot + agent-local state; returned as lists per agent type.
- **Rewards**: Per-agent, per-type; returned as lists (same structure as obs).
- **Inventory Tracking**: Market makers track bid/ask inventory separately; execution agents track remaining task size.
- **PnL Calculation**: End-of-episode mark-to-market using midprice at episode end; supports multiple PnL components (cash, inventory, total).

## Common Workflows

### Running a Single Training Experiment

```bash
# 2-agent market maker + execution (recommended starting point)
python3 gymnax_exchange/jaxrl/MARL/ippo_rnn_JAXMARL.py \
  --config-name="ippo_rnn_JAXMARL_2player" \
  WANDB_MODE="disabled" \
  NUM_ENVS=256
```

**Config Selection Quick Reference:**
- `ippo_rnn_JAXMARL_2player.yaml` + `2_player_fq_fqc.json`: Multi-agent MM+Exec
- `ippo_rnn_JAXMARL_mm_*.yaml` + `mm_*.json`: Market making only
- `ippo_rnn_JAXMARL_exec*.yaml` + `exec_*.json`: Execution only

### Debugging Steps

**Disable JIT for stack traces:**
```python
# In Python REPL or script, before imports:
import jax
jax.config.update('jax_disable_jit', False)  # or True to disable
```

**Common OOM issues:**
- Reduce `NUM_ENVS` first (most memory impact)
- Then reduce `GRU_HIDDEN_DIM` or `FC_DIM_SIZE`
- Check LOBSTER data cache not corrupted (delete `saved_npz/` and retry)

**Config/Data Path Errors:**
- Verify `alphatradePath` points to repo root
- Verify `dataPath` points to parent of `rawLOBSTER/`
- Confirm `stock` and `timePeriod` match actual data folders
- Run with `-v` flag or enable Hydra's `hydra.verbose` for config resolution

### Verification & Rollout Scripts

**One-step minimal check** (validates data + world model):
```bash
python minimal_agent_generative_step.py --fast_startup --n_cond_msgs 8 --sample_index 0 --agent_action 0
```

**Learned policy rollout** (production candidate validation):
```bash
python run_learned_mm_worldmodel_rollout.py \
  --policy_ckpt_dir checkpoints/MARLCheckpoints/<project>/<run> \
  --policy_deterministic --n_steps 100 --n_envs 4
```

**Full sweep** (throughput boundary search):
```bash
bash run_sweep_single_gpu.sh
```

**Generative-world-model policy training (PG baseline):**
```bash
python run_gen_worldmodel_pg_train.py --fast_startup --n_envs 4 --n_updates 5 --n_steps 10
```

**Single-node 4-GPU sweep + train scripts:**
- `slurm/sbatch_sweep_gen_worldmodel_single_node.sh`
- `slurm/sbatch_train_gen_worldmodel_best.sh`
- `aggregate_gen_worldmodel_pnl.py` for mean/median/std PnL summaries

## Testing & Linting

**No standard test suite exists** — testing is primarily through integration verification scripts:
- `minimal_agent_generative_step.py`: Single-step validation
- `run_learned_mm_worldmodel_rollout.py`: Policy rollout validation
- `run_sweep_single_gpu.sh`: Throughput/stability sweep

**Docker-based testing** (if available):
```bash
make test
```

**Linting**: No automated linting configured. Manual code review recommended for style consistency.

## Docker Workflow (Alternative)

```bash
make build                           # Build image
make run                             # Interactive shell
make ppo_2player gpu=0               # Run 2-player training on GPU 0
```

**Note**: Image assumes x86_64/amd64 only; repo mounted at `/home/myuser/`, data at `/home/myuser/data/`.

## When Making Code Changes

### Backward Compatibility
- Config changes: Ensure old JSON/YAML configs still load (add defaults in dataclasses).
- State/checkpoint format: Orbax is versioned; backwards compatibility not guaranteed between versions.
- Environment interface: `step()` signature change requires updating training loop.

### Common Edit Patterns

**Adding new agent type:**
1. Create new dataclass in `jaxob_config.py`
2. Create agent environment class in new file (follow MM/Exec structure)
3. Add type check in `MARLEnv.__init__()` to instantiate new agent
4. Add config JSON examples in `config/env_configs/`

**Adding new action/obs space:**
1. Add string selector to agent config dataclass (e.g., `action_space_new: str = "my_space"`)
2. Implement space logic in agent environment (usually in `action_space()` and observation encoding methods)
3. Update post-init logic if space alters derived fields (n_actions, observation dim)

**Adding reward function:**
1. Add option to `reward_function` string selector in agent config
2. Implement reward calculation in agent's step logic (usually branching on config string)
3. Validate against test rollouts (use `minimal_agent_generative_step.py`)

## Important Caveats

- **Book Crashes**: Random actions can corrupt order book if prices are invalid; safeguards exist (price sanitization, agent action revert, generated message revert).
- **Memory Dominance**: LOBS5 world model (~36.8 GB) dominates memory; per-trajectory overhead negligible.
- **Long JIT Compile Times**: First run of policy may take minutes for compilation; subsequent runs cached.
- **Seed Reproducibility**: JAX PRNG seeding is deterministic but environment reset randomness can vary across runs if seed not explicitly controlled in config.

## References

- **README.md**: Public-facing setup, data requirements, and run commands.
- **This file (`.github/copilot-instructions.md`)**: Canonical internal engineering guidance (includes the former quick-guide content).
- **INFERENCE_SPEED_LESSONS_LEARNED.md**: Skills/lessons-learned playbook for inference and rollout operations.
- **config/**: Example JSON (env setup) and YAML (training hyperparams) files.
- **Orbax Checkpoints**: Snapshots of train state; see `orbax-checkpoint==0.11.18` in requirements.txt.
