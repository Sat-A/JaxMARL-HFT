# JaxMARL-HFT: GPU-Accelerated Multi-Agent Reinforcement Learning for High-Frequency Trading

A JAX-based framework for multi-agent reinforcement learning for high-frequency trading, based on the [JAX-LOB simulator](https://github.com/KangOxford/jax-lob) and an extension of [JaxMARL](https://github.com/FLAIROx/JaxMARL) to the financial trading domain.

## Key Features

- **GPU-Accelerated**: Built on JAX for high-performance parallel computation with JIT compilation
- **Two levels of Parallelization**: Parallel processing across episodes and agent types using `vmap`
- **Multi-Agent RL**: Supports market making, execution, and directional trading agents
- **LOBSTER Data Integration**: Real market data support with efficient GPU memory usage
- **Scalable**: Handles thousands of parallel environments
- **Heterogeneous Agents**: Supports different observation/action spaces

## Getting Started

### 1. Clone and install

```bash
conda create -n jaxmarl_hft python=3.10
conda activate jaxmarl_hft
pip install "jax[cuda12]"
pip install -r requirements.txt
```

Requires Python 3.8+ and a CUDA-capable GPU.

### 2. Get LOBSTER data

You need [LOBSTER](https://data.lobsterdata.com/info/WhatIsLOBSTER.php) limit order book data. Each trading day needs a matched pair of `message` and `orderbook` CSV files.

Create the following directory structure inside the repo (or anywhere — you'll point to it in the config):

```
data/rawLOBSTER/<STOCK>/<TIME_PERIOD>/
├── <STOCK>_<DATE>_34200000_57600000_message_10.csv
└── <STOCK>_<DATE>_34200000_57600000_orderbook_10.csv
```

For example, for GOOG data from 2022:
```
data/rawLOBSTER/GOOG/2022/
├── GOOG_2022-01-03_34200000_57600000_message_10.csv
├── GOOG_2022-01-03_34200000_57600000_orderbook_10.csv
└── ...
```

### 3. Edit the environment config

Pick an environment config from `config/env_configs/` (recommended starting point: `2_player_fq_fqc.json` for multi-agent market making + execution) and set these fields in the `world_config` section:

```json
"alphatradePath": "/absolute/path/to/JaxMARL-HFT",
"dataPath": "/absolute/path/to/JaxMARL-HFT/data",
"stock": "GOOG",
"timePeriod": "2022"
```

`alphatradePath` is the repo root (used for caching and checkpoints), `dataPath` is the parent of `rawLOBSTER/`, `stock` matches your data folder name, and `timePeriod` is the subfolder under `rawLOBSTER/<stock>/`.

Also set `TimePeriod` in your training config (`config/rl_configs/*.yaml`) to match.

### 4. Run training

```bash
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Multi-agent (market making + execution)
python3 gymnax_exchange/jaxrl/MARL/ippo_rnn_JAXMARL.py \
    --config-name="ippo_rnn_JAXMARL_2player" \
    WANDB_MODE="disabled"
```

The first run preprocesses the LOBSTER data and caches it. Subsequent runs are much faster. Additional training configs are in `config/rl_configs/`. You can override any config value from the command line using [Hydra](https://hydra.cc/) syntax (e.g. `TOTAL_TIMESTEPS=50000 NUM_ENVS=64`).

### 5. WandB (optional)

To enable [Weights & Biases](https://wandb.ai/) experiment tracking, run `wandb login` and then add `WANDB_MODE="online" ENTITY="your-wandb-entity" PROJECT="your-project-name"` to the training command. These can also be set directly in the YAML configs (`config/rl_configs/*.yaml`). The YAML configs support [WandB sweeps](https://docs.wandb.ai/guides/sweeps) — when a `SWEEP_PARAMETERS` section is present and `WANDB_MODE` is not `"disabled"`, training automatically creates a sweep.

### 6. One-Step Agent + Generative Verification

For a minimal process check that combines:
1. historical-message startup,
2. one market-maker action,
3. one generative model world-update step,

use:

```bash
python minimal_agent_generative_step.py \
  --fast_startup \
  --n_cond_msgs 8 \
  --sample_index 0 \
  --agent_action 0 \
  --run_name verify_a0
```

Artifacts are saved under:

`outputs/minimal_agent_generative_step/<run_name>_action<id>_sample<idx>/`

including:
- `agent_action_messages.csv`
- `generated_message.csv`
- `verification_summary.json`
- `verification_report.txt`

### 7. Multi-Step Fixed/Random Policy Verification

The same script supports multi-step rollouts with either a fixed action or random actions:

```bash
# 20-step random actions
python minimal_agent_generative_step.py \
  --fast_startup \
  --n_cond_msgs 8 \
  --sample_index 0 \
  --n_steps 20 \
  --action_policy random \
  --seed 42 \
  --run_name verify_20step_random
```

```bash
# 100-step random actions
python minimal_agent_generative_step.py \
  --fast_startup \
  --n_cond_msgs 8 \
  --sample_index 0 \
  --n_steps 100 \
  --action_policy random \
  --seed 42 \
  --run_name verify_100step_random
```

```bash
# 1000-step random actions (GPUs 4-7)
CUDA_VISIBLE_DEVICES=4,5,6,7 python minimal_agent_generative_step.py \
  --fast_startup \
  --n_cond_msgs 8 \
  --sample_index 0 \
  --n_steps 1000 \
  --action_policy random \
  --seed 42 \
  --run_name verify_1kstep_random
```

Additional artifacts include:
- `step_trace.csv` (per-step action and market state)
- `midprice_trajectory.png` (historical + generated midprice)
- `action_midprice_trajectory.png` (actions and midprice combined)

`verification_summary.json` now includes end-of-run `agent_pnl` (cash, inventory mark-to-market, and total PnL in tick-normalized units).

### Book-Crash Safeguards

Random actions can occasionally post orders with invalid (zero or negative) prices, which clears one side of the order book and corrupts the midprice for all subsequent steps. Three safeguards are applied inside the rollout loop:

1. **Price sanitisation** — any limit order with `price <= 0` is converted to a no-op before it reaches the simulator.
2. **Agent action revert** — if the agent's messages leave `best_bid <= 0` or `best_ask <= 0`, the post-action state is rolled back to the pre-action snapshot and the step continues from there.
3. **Generated message revert** — same check after the LOBS5 generated message is applied.

Both revert events are printed as diagnostics during the rollout.

### 8. Learned-Policy World-Model Rollout

`run_learned_mm_worldmodel_rollout.py` replaces the heuristic action selector with a trained IPPO/GRU policy loaded from an Orbax checkpoint. It runs full multi-step rollouts with per-step timing instrumentation and supports multiple independent trajectories in a single process.

```bash
# Smoke test — 10 steps, single trajectory
CUDA_VISIBLE_DEVICES=0 python run_learned_mm_worldmodel_rollout.py \
  --fast_startup \
  --n_cond_msgs 8 \
  --n_steps 10 \
  --n_envs 1 \
  --policy_deterministic \
  --allow_obs_pad \
  --run_name smoke_test
```

```bash
# Throughput mode — 4 independent trajectories, single GPU
CUDA_VISIBLE_DEVICES=0 python run_learned_mm_worldmodel_rollout.py \
  --fast_startup \
  --n_cond_msgs 8 \
  --n_steps 25 \
  --n_envs 4 \
  --policy_deterministic \
  --allow_obs_pad \
  --run_name throughput_n4
```

**Key flags:**

| Flag | Default | Description |
|---|---|---|
| `--policy_ckpt_dir` | `checkpoints/MARLCheckpoints/2PLayer/dummy-2vdmzbye` | Orbax checkpoint directory for the IPPO policy |
| `--policy_config` | `config/rl_configs/ippo_rnn_JAXMARL_mm_BOB.yaml` | YAML config used during training |
| `--policy_model_index` | `1` | Index into the model list saved in the checkpoint |
| `--n_envs` | `1` | Number of independent parallel trajectories (sequential loop, independent state per env) |
| `--policy_deterministic` | off | Use argmax over logits instead of sampling |
| `--allow_obs_pad` | off | Pad/truncate observation vector if checkpoint obs dim differs from built features |

**Measured throughput (A100, single GPU, n_cond_msgs=8, greedy decode):**

| n_envs | agg steps/s | step p50 (ms) | peak memory |
|-------:|------------:|--------------:|------------:|
| 1 | 0.09 | 247 | 36.9 GB |
| 2 | 0.17 | 266 | 36.9 GB |
| 4 | 0.32 | 263 | 36.9 GB |

Memory is dominated by the LOBS5 world model (~36.8 GB). Per-trajectory overhead is negligible, so higher `n_envs` values are expected stable. The policy GRU inference runs at ~1.7 ms per step after JIT compilation.

Full Phase 4 sweep (Stage A stability probe + Stage C seed variance):

```bash
bash run_sweep_single_gpu.sh
```

Results are written under `outputs/sweep_single_gpu_<timestamp>/`.

### 9. Production Deployment (Learned Policy Rollout)

Use a trained policy checkpoint (not `dummy-2vdmzbye`) for quality validation and deployment decisions.

```bash
# Quality smoke (trained checkpoint, single env)
CUDA_VISIBLE_DEVICES=0 python run_learned_mm_worldmodel_rollout.py \
  --fast_startup \
  --policy_ckpt_dir checkpoints/MARLCheckpoints/trained_smoke \
  --policy_deterministic \
  --allow_obs_pad \
  --n_cond_msgs 8 \
  --n_steps 25 \
  --n_envs 1 \
  --run_name quality_smoke_trained
```

```bash
# Throughput candidate (replace N with highest stable n_envs from sweep)
CUDA_VISIBLE_DEVICES=0 python run_learned_mm_worldmodel_rollout.py \
  --fast_startup \
  --policy_ckpt_dir checkpoints/MARLCheckpoints/trained_smoke \
  --policy_deterministic \
  --allow_obs_pad \
  --n_cond_msgs 8 \
  --n_steps 50 \
  --n_envs N \
  --jit_message_build \
  --run_name prod_candidate_nN
```

```bash
# Sweep to OOM/stability boundary (now probes 1,2,4,8,16,32)
bash run_sweep_single_gpu.sh
```

Notes:
- `--jit_message_build` is opt-in and falls back to eager message-build automatically if JIT compile fails.
- Production `n_envs` should be selected as highest stable throughput setting with quality guardrails (action diversity + trade incidence > 0).

### 10. Supercomputer Multi-Node Kickoff (Tomorrow)

Start with this order on a stronger multi-node cluster:

1. **Single-node validation first (15-30 min)**
  - Confirm environment + paths + checkpoint access on one node.
  - Run one short real-checkpoint quality baseline:

```bash
CUDA_VISIBLE_DEVICES=0 python run_learned_mm_worldmodel_rollout.py \
  --fast_startup \
  --policy_ckpt_dir /scratch/local/homes/groups/finance/data/checkpoints/MARLCheckpoints/2PLayer/whole-sweep-1 \
  --policy_deterministic \
  --allow_obs_pad \
  --jit_message_build \
  --n_cond_msgs 8 \
  --n_steps 50 \
  --n_envs 1 \
  --seed 42 \
  --run_name sc_quality_seed42
```

2. **Single-node throughput boundary**
  - Reconfirm stable max on the new GPU type:

```bash
POLICY_CKPT_DIR=/scratch/local/homes/groups/finance/data/checkpoints/MARLCheckpoints/2PLayer/whole-sweep-1 \
JIT_MESSAGE_BUILD=1 N_STEPS=25 N_COND_MSGS=8 GPU_ID=0 \
bash run_sweep_single_gpu.sh
```

3. **Then scale out across nodes**
  - Launch one process per GPU, each with the same stable `n_envs` and distinct `--seed`/`--sample_index` ranges.
  - Keep each process long-lived to amortize JAX warmup.
  - Aggregate all `summary.json` files under `outputs/` and compare throughput + variance.

Practical default for tomorrow:
- Start from `n_envs=32` + `--jit_message_build` (current best throughput on this setup), then tune upward/downward per GPU memory and stability on the new node.

## Docker Setup (alternative)

For **x86_64/amd64 only** (base image: `nvcr.io/nvidia/jax`). Edit the `Makefile` to set `DATADIR` to your LOBSTER data directory, then:

```bash
make build              # build image
make run                # interactive shell
make ppo_2player gpu=0  # run training on GPU 0
```

The repo is mounted at `/home/myuser/` and data at `/home/myuser/data/`, so the default env config paths work without modification. For WandB, set `export WANDB_API_KEY=<your-key>` before running.

## Agent Types

### Market Making Agents
- **Purpose**: Provide liquidity by posting bid/ask orders
- **Action Spaces**: Multiple discrete action spaces (spread_skew, fixed_quants, AvSt, directional_trading, simple)
- **Reward Functions**: Various PnL-based rewards with configurable inventory penalties

### Execution Agents
- **Purpose**: Execute large orders with minimal market impact
- **Action Spaces**: Discrete quantity selection at reference prices (fixed_quants, fixed_prices, complex variants)
- **Reward Functions**: Slippage-based with configurable end-of-episode penalties

### Directional Trading
- **Purpose**: Simple directional trading strategy
- **Action Spaces**: Bid/ask at best prices or no action
- **Reward Function**: Portfolio value
- **Note:** Uses the same class as the market making agent

## Repository Structure

```
config/
├── env_configs/          # Environment JSON configurations
└── rl_configs/           # Training YAML configurations
gymnax_exchange/
├── jaxen/                # Environment implementations
│   ├── marl_env.py       # Multi-agent RL environment
│   ├── mm_env.py         # Market making (and directional trading) environment
│   ├── exec_env.py       # Execution environment
│   └── from_JAXMARL/     # Multi-agent base classes and spaces
├── jaxrl/                # Reinforcement learning algorithms
│   └── MARL/             # IPPO implementation and baseline evaluation
├── jaxob/                # Order book implementation
├── jaxlobster/           # LOBSTER data integration
└── utils/                # Shared utilities
```

## Configuration

The framework uses a comprehensive configuration system with dataclasses for different components:

### Core Configuration Classes

- **`MultiAgentConfig`**: Main configuration combining world and agent settings
- **`World_EnvironmentConfig`**: Global environment parameters (data paths, episode settings, market hours)
- **`MarketMaking_EnvironmentConfig`**: Market making and directional trading agent configuration (action spaces, reward functions, observation spaces)
- **`Execution_EnvironmentConfig`**: Execution agent configuration (task types, action spaces, reward parameters)

### Training Configuration

Edit YAML files in `config/rl_configs/` to customize:
- Number of parallel environments (default: 4096)
- Training parameters (steps, learning rates, etc.)
- Agent configurations (action spaces, reward functions)
- Market data settings (resolution, episode length)

Environment configurations are in `config/env_configs/`.

## Citation

If you use JaxMARL-HFT in your research, please cite:

```bibtex
@inproceedings{mohl2025jaxmarlhft,
  title={JaxMARL-HFT: GPU-Accelerated Large-Scale Multi-Agent Reinforcement Learning for High-Frequency Trading},
  author={Mohl, Valentin and Frey, Sascha and Leyland, Reuben and Li, Kang and Nigmatulin, George and Cucuringu, Mihai and Zohren, Stefan and Foerster, Jakob and Calinescu, Anisoara},
  booktitle={Proceedings of the 6th ACM International Conference on AI in Finance (ICAIF)},
  pages={18--26},
  year={2025},
  doi={10.1145/3768292.3770416}
}
```

## Acknowledgements

JaxMARL-HFT builds on:
- [JaxMARL](https://github.com/FLAIROx/JaxMARL) — Multi-agent RL environments and algorithms in JAX
- [JAX-LOB](https://github.com/KangOxford/jax-lob) — GPU-accelerated limit order book simulator

## Disclaimer

This software is provided for **research and educational purposes only**. It is not intended for live trading, financial decision-making, or any form of real-money deployment. The authors and contributors make no warranties regarding the accuracy, reliability, or suitability of this software for any particular purpose.

**The authors assume no responsibility or liability for any financial losses, damages, or other consequences arising from the use of this software.** Use at your own risk.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
