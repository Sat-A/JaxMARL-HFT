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
