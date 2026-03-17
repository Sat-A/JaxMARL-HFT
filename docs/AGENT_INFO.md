# Agent Info: Training Agents, Policies, Tunables, and Goals

This document is the practical reference for **what agents exist**, **what policy families are used**, **which initialization/tuning parameters matter**, and **what each agent is trying to optimize**.

## 1) Agent types in this codebase

### MarketMaking agent (`MarketMakingAgent`)
- Config type: `MarketMaking_EnvironmentConfig`
- Core objective: provide liquidity around the mid-price while managing inventory risk and maximizing reward/PnL.
- Where defined:
  - `gymnax_exchange/jaxob/jaxob_config.py` (`MarketMaking_EnvironmentConfig`)
  - `gymnax_exchange/jaxen/mm_env.py`

### Execution agent (`ExecutionAgent`)
- Config type: `Execution_EnvironmentConfig`
- Core objective: execute buy/sell tasks (often fixed size) with good execution quality under time/price constraints.
- Where defined:
  - `gymnax_exchange/jaxob/jaxob_config.py` (`Execution_EnvironmentConfig`)
  - `gymnax_exchange/jaxen/exec_env.py`

### Multi-agent wrapper (`MARLEnv`)
- Composes MM + Execution agent instances and runs them against one shared world/orderbook.
- Core objective: orchestrate heterogeneous agents with aligned step/reset contracts for MARL training.
- Where defined:
  - `gymnax_exchange/jaxen/marl_env.py`

## 2) Policy families currently used

## A) Original MARL IPPO-RNN trainer (baseline/original path)
- Entrypoint: `gymnax_exchange/jaxrl/MARL/ippo_rnn_JAXMARL.py`
- Policy style: recurrent actor-critic (GRU-based PPO/IPPO).
- Typical use: full/original JaxMARL-HFT training stack with Hydra/YAML config.

## B) Generative world-model trainer (active fast iteration path)
- Entrypoint: `run_gen_worldmodel_pg_train.py`
- Policy options (`--policy_arch`):
  - `mlp`: simple feed-forward policy head
  - `ippo_rnn`: local recurrent actor-critic (GRU), used to mirror IPPO-style behavior without hydra dependency
- Typical use: policy-gradient-style control loop inside world-model rollouts.

## C) Action-space policy modes for MM behavior generation
These are not neural architectures; they define how discrete actions map to order-placement logic:
- `fixed_quants`, `fixed_prices`, `AvSt`, `bobStrategy`, `bobRL`, `spread_skew`, `directional_trading`, `simple`
- In generative trainer, this is controlled with `--mm_action_space`.

## 3) Tunable initialization parameters (most important)

## A) MarketMaking config tunables
From `MarketMaking_EnvironmentConfig`:
- **Structure/space selection**
  - `action_space`, `observation_space`, `reward_function`
- **Action mapping controls**
  - `bob_v0` (changes action cardinality for `bobRL`: 3/5/11/21 actions)
  - `fixed_quant_value`, `spread_multiplier`, `skew_multiplier`, `n_ticks_offset`, `tenth_action`
- **Risk + reward shaping**
  - `inv_penalty`, `inv_penalty_lambda`, `inv_penalty_quadratic_factor`, `inv_penalty_threshold`
  - `inventoryPnL_eta`, `inventoryPnL_gamma`, `reward_scaling_quo`
  - `reference_price`, `unwind_price`, `unwind_price_penalty`, `rebate_bps`
- **Safety**
  - `auto_liquidate_threshold`, `auto_liquidate_alpha`

## B) Execution config tunables
From `Execution_EnvironmentConfig`:
- `task` (`random|buy|sell`)
- `action_type` (`delta|pure`)
- `action_space`, `observation_space`, `reward_function`
- `task_size`, `fixed_quant_value`, `n_ticks_in_book`
- `reward_lambda`, `doom_price_penalty`, `reward_scaling_quo`, `reference_price`

## C) Generative trainer policy/init tunables
From `run_gen_worldmodel_pg_train.py` CLI:
- **Policy architecture**
  - `--policy_arch {mlp,ippo_rnn}`
  - `--fc_dim_size`, `--gru_hidden_dim`
- **Optimization**
  - `--lr`, `--entropy_coef`, `--value_coef`, `--baseline_momentum`
- **Rollout/training size**
  - `--n_envs`, `--n_steps`, `--n_updates`, `--seed`
- **MM behavior controls**
  - `--mm_action_space`, `--mm_bob_v0`, `--mm_fixed_quant_value`
- **Checkpoint restore behavior**
  - `--checkpoint_restore_topology {auto,strict,single-device-remap}`

## D) Original IPPO YAML-level tunables
From `config/rl_configs/ippo_rnn_JAXMARL_2player.yaml` and `ippo_rnn_JAXMARL_mm_BOB.yaml`:
- `LR`, `NUM_ENVS`, `NUM_STEPS`, `TOTAL_TIMESTEPS`
- `GRU_HIDDEN_DIM`, `FC_DIM_SIZE`
- `UPDATE_EPOCHS`, `NUM_MINIBATCHES`
- `GAMMA`, `GAE_LAMBDA`, `CLIP_EPS`
- `ENT_COEF`, `VF_COEF`, `MAX_GRAD_NORM`
- `NUM_AGENTS_PER_TYPE`, `ENV_CONFIG`

## 4) Goal of each agent/policy combination

- **MM + `bobStrategy` / `spread_skew`**  
  Goal: stable market making behavior with interpretable spread/inventory control.

- **MM + `bobRL`**  
  Goal: richer discrete policy control (more actions as `bob_v0` increases), potentially higher expressiveness.

- **Execution + fixed-quantity/fixed-price families**  
  Goal: complete execution tasks efficiently while balancing price impact and completion risk.

- **`policy_arch=mlp` (generative trainer)**  
  Goal: fastest/simple baseline for control in world-model loop; lower representational capacity.

- **`policy_arch=ippo_rnn` (generative trainer/original IPPO)**  
  Goal: leverage temporal state/memory for partial observability and sequential orderbook dynamics.

## 5) Practical defaults (today)

- Throughput-stable profile currently: `n_envs=2` for the recent single-GPU sweeps.
- Recommended operational defaults for generative training:
  - `policy_arch=ippo_rnn`
  - `checkpoint_restore_topology=single-device-remap` on cluster
  - start with smoke (`n_updates`/`n_steps` small), then scale.
- Promotion rule: do not promote on throughput alone; require non-zero trade incidence and acceptable PnL.

