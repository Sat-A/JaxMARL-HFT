# JaxMARL-HFT Agent Quick Guide

## 1) What This Repo Is
JAX-based multi-agent RL for limit-order-book trading simulation (market making + execution), using LOBSTER historical data and PPO-RNN training.

Core paths:
- `gymnax_exchange/jaxrl/MARL/ippo_rnn_JAXMARL.py` (main training entrypoint)
- `gymnax_exchange/jaxen/marl_env.py` (multi-agent env orchestrator)
- `gymnax_exchange/jaxen/base_env.py` (LOBSTER data windows + base world state)
- `gymnax_exchange/jaxen/mm_env.py` (market-making agent logic)
- `gymnax_exchange/jaxen/exec_env.py` (execution agent logic)
- `gymnax_exchange/jaxob/jaxob_config.py` (all config dataclasses)
- `config/env_configs/*.json` (env/world+agent setup)
- `config/rl_configs/*.yaml` (training hyperparams + WandB/sweep)

## 2) How It Works (Runtime Pipeline)
1. RL YAML is loaded by Hydra (`ippo_rnn_JAXMARL.py`).
2. Env JSON (`ENV_CONFIG`) is loaded into `MultiAgentConfig` (`config_io.py`).
3. YAML + env dataclass are merged (`OmegaConf.merge`).
4. `MARLEnv` builds:
   - `BaseLOBEnv` world/data stream
   - one agent instance per configured type (MM/Execution)
5. Training loop (IPPO-RNN):
   - rollout (`NUM_STEPS` x `NUM_ENVS`)
   - GAE + PPO updates per agent-type network
   - optional eval
   - Orbax checkpoints periodically

## 3) Minimal Usage Method
### Local
- Install deps (`README.md`, `requirements.txt`)
- Set `PYTHONPATH`
- Ensure LOBSTER files exist under:
  - `data/rawLOBSTER/<STOCK>/<TIME_PERIOD>/*message*.csv`
  - `data/rawLOBSTER/<STOCK>/<TIME_PERIOD>/*orderbook*.csv`
- Pick:
  - one RL config: `config/rl_configs/*.yaml`
  - one env config: `config/env_configs/*.json`
- Run:
  - `python3 gymnax_exchange/jaxrl/MARL/ippo_rnn_JAXMARL.py --config-name=<yaml_name_without_ext> WANDB_MODE=disabled`

### Docker
- `make build`
- `make run` or `make ppo_2player gpu=0`
(see `Dockerfile`, `Makefile`)

## 4) Inputs
### Required external input
- LOBSTER CSV pairs per day (message + orderbook), correctly named and colocated.

### Required config input
- World data paths and market slice:
  - `world_config.alphatradePath`
  - `world_config.dataPath`
  - `world_config.stock`
  - `world_config.timePeriod`
- RL scale params:
  - `NUM_ENVS`, `NUM_STEPS`, `TOTAL_TIMESTEPS`, `NUM_AGENTS_PER_TYPE`
- Agent setup:
  - MM: action/obs/reward modes
  - Execution: task/action/obs/reward modes

## 5) Outputs
### Runtime artifacts
- Preprocessed dataset cache:
  - `<alphatradePath>/saved_npz/loaded_lobster_*.npz`
- Precomputed reset states:
  - `<alphatradePath>/pre_reset_states/ResetStates_*.pkl`
- Training checkpoints (Orbax):
  - `<alphatradePath>/checkpoints/MARLCheckpoints/<PROJECT>/<RUN>/`
- WandB logs (if enabled)
- Baseline eval trajectory dumps (when enabled):
  - `trajectories/traj_batch_<combo>__<timestamp>.pkl`

### Env step contract (`MARLEnv.step_env`)
Returns:
- `agent_obs_list`
- `new_multi_state`
- `agent_reward_list`
- `dones` (`{"__all__": ..., "agents": ...}`)
- `info` (`{"world": ..., "agents": ...}`)

## 6) Settings Hierarchy (Important)
Precedence (highest to lowest):
1. WandB sweep overrides (`SWEEP_PARAMETERS`)
2. Hydra CLI overrides
3. RL YAML (`config/rl_configs/*.yaml`)
4. Env JSON (`ENV_CONFIG`)
5. Dataclass defaults (`jaxob_config.py`)

## 7) Config Knobs That Matter Most
- Throughput/memory: `NUM_ENVS`, `NUM_STEPS`, `GRU_HIDDEN_DIM`, `FC_DIM_SIZE`
- Episode/data slicing: `n_data_msg_per_step`, `ep_type`, `episode_time`, `start_resolution`
- Agent behavior:
  - MM: `action_space`, `reward_function`, inventory penalties
  - Execution: `task`, `action_space`, `reward_function`, `task_size`
- Logging/eval: `WANDB_MODE`, `CALC_EVAL`, `Timing`

## 8) Fast Orientation by Scenario
- 2-agent MM+Exec: `ippo_rnn_JAXMARL_2player.yaml` + `2_player_fq_fqc.json`
- Exec-only: `ippo_rnn_JAXMARL_exec*.yaml` + `exec_*.json`
- MM-only: `ippo_rnn_JAXMARL_mm_*.yaml` + `mm_*.json`
- Multi-device variant: `ippo_rnn_JAXMARL_pmap.py` + `PMAP_*.yaml`

## 9) Common Failure Causes
- Wrong data path or stock/timePeriod mismatch
- Missing LOBSTER file pairs
- Invalid `NUM_AGENTS_PER_TYPE` vs list-valued hyperparams
- Over-aggressive `NUM_ENVS` causing GPU OOM
- Missing WandB credentials when `WANDB_MODE=online`
