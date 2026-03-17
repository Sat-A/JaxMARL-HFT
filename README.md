# JaxMARL-HFT: GPU-Accelerated Multi-Agent Reinforcement Learning for High-Frequency Trading

A JAX-based framework for multi-agent reinforcement learning for high-frequency trading, based on the [JAX-LOB simulator](https://github.com/KangOxford/jax-lob) and an extension of [JaxMARL](https://github.com/FLAIROx/JaxMARL) to the financial trading domain.

## Key Features

- **GPU-Accelerated**: Built on JAX for high-performance parallel computation with JIT compilation
- **Two levels of Parallelization**: Parallel processing across episodes and agent types using `vmap`
- **Multi-Agent RL**: Supports market making, execution, and directional trading agents
- **LOBSTER Data Integration**: Real market data support with efficient GPU memory usage
- **Scalable**: Handles thousands of parallel environments
- **Heterogeneous Agents**: Supports different observation/action spaces

## Latest status and high-signal findings

- The current active work is **IPPO-style policy training on the generative world-model simulator**.
- Restore portability is implemented (`checkpoint_restore_topology=auto|strict|single-device-remap`).
- Sweep execution supports resilient resumption (`MAX_PARALLEL_GPUS`, `RESUME_SWEEP`, `RETRY_PER_PROFILE`).
- Historical best throughput profile remains **`n_envs=1`** (best observed single run: **3.6465 steps/sec**, job `2894876` snapshot).
- Recent cluster runs are now completing, but policy quality is not yet acceptable: latest train-best aggregate (`train_best_aggregate_2915160.json`) shows **~3.66 mean steps/sec** with **zero trade incidence and zero PnL** across seeds.
- Current interpretation: training is in a **no-trade / no-reward regime**; throughput is validated, agent quality is not.
- Detailed operational notes and reports are in `docs/reports/` and next-step tracking is in `docs/plans/`.

## Quick Start

### 1) Install

```bash
conda create -n jaxmarl_hft python=3.10
conda activate jaxmarl_hft
pip install "jax[cuda12]"
pip install -r requirements.txt
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

### 1-minute entrance points (recommended)

Use these as the default workflow. Everything else is advanced.

1. **Smoke train on one GPU** (sanity check):
```bash
sbatch slurm/sbatch_smoke_gen_worldmodel_train.sh
```

2. **Profile sweep** (`n_envs` search, resilient):
```bash
MAX_PARALLEL_GPUS=1 RESUME_SWEEP=1 POLICY_ARCH=ippo_rnn \
CHECKPOINT_RESTORE_TOPOLOGY=single-device-remap \
sbatch slurm/sbatch_sweep_gen_worldmodel_single_node.sh
```

3. **Train best profile** (multi-seed):
```bash
N_ENVS_BEST=2 POLICY_ARCH=ippo_rnn \
CHECKPOINT_RESTORE_TOPOLOGY=single-device-remap \
sbatch slurm/sbatch_train_gen_worldmodel_best.sh
```

4. **Read results**:
```bash
ls -1t outputs/gen_worldmodel_pg_train/*aggregate*.json | head
```

### 2) Data layout

You need LOBSTER message/orderbook CSV pairs:

```text
data/rawLOBSTER/<STOCK>/<TIME_PERIOD>/
  <STOCK>_<DATE>_34200000_57600000_message_10.csv
  <STOCK>_<DATE>_34200000_57600000_orderbook_10.csv
```

### 3) Two run tracks

### Track A: Original framework training (baseline path)

Use this when you want the original JaxMARL-HFT training flow.

1. Set paths in an env config (e.g. `config/env_configs/2_player_fq_fqc.json`):
   - `alphatradePath` = repo root
   - `dataPath` = parent of `rawLOBSTER/`
   - `stock`, `timePeriod` to match data
2. Run IPPO-RNN training:

```bash
python3 gymnax_exchange/jaxrl/MARL/ippo_rnn_JAXMARL.py \
  --config-name="ippo_rnn_JAXMARL_2player" \
  WANDB_MODE="disabled"
```

Optional WandB:

```bash
wandb login
python3 gymnax_exchange/jaxrl/MARL/ippo_rnn_JAXMARL.py \
  --config-name="ippo_rnn_JAXMARL_2player" \
  WANDB_MODE="online" ENTITY="<entity>" PROJECT="<project>"
```

### Track B: Current generative-world-model experiments (active work)

Use this path for the ongoing IPPO-style-on-generative-simulator experiments.

### Minimal local control run

```bash
GPU_ID=0 N_ENVS_CANDIDATES="1" \
CHECKPOINT_RESTORE_TOPOLOGY=single-device-remap \
POLICY_ARCH=ippo_rnn \
bash run_sweep_gen_worldmodel_train_single_node.sh
```

### Resilient Slurm sweep (recommended)

```bash
MAX_PARALLEL_GPUS=1 \
RESUME_SWEEP=1 \
RETRY_PER_PROFILE=2 \
CHECKPOINT_RESTORE_TOPOLOGY=auto \
POLICY_ARCH=ippo_rnn \
sbatch slurm/sbatch_sweep_gen_worldmodel_single_node.sh
```

### Train best profile after sweep

```bash
N_ENVS_BEST=<best_n_envs> sbatch slurm/sbatch_train_gen_worldmodel_best.sh
```

### Profile-driven submission (new, recommended for experimentation)

Define agent/training knobs once in an env profile and submit by mode:

```bash
# Dry-run (no job submitted)
bash scripts/experiments/submit_genwm_profile.sh \
  config/gen_worldmodel_profiles/aggressive_pnl.env smoke

# Submit smoke/sweep/train-best with the same profile
bash scripts/experiments/submit_genwm_profile.sh \
  config/gen_worldmodel_profiles/aggressive_pnl.env smoke --submit
bash scripts/experiments/submit_genwm_profile.sh \
  config/gen_worldmodel_profiles/aggressive_pnl.env sweep --submit
bash scripts/experiments/submit_genwm_profile.sh \
  config/gen_worldmodel_profiles/aggressive_pnl.env train-best --submit
```

Notes:
- `policy_arch`: `mlp` (legacy) or `ippo_rnn` (current experimental default).
- `checkpoint_restore_topology`: `strict`, `auto` (default), or `single-device-remap`.
- Sweep outputs live under `outputs/gen_worldmodel_pg_train/` and include `summary.json` plus aggregate files when successful.

### Promotion rule (keep it simple)

- Do not promote a profile on throughput alone.
- Require both:
  - stable completion + aggregate artifact
  - non-zero trade incidence and positive/acceptable PnL on holdout checks

## Docker Setup (alternative)

For **x86_64/amd64 only** (base image: `nvcr.io/nvidia/jax`). Edit the `Makefile` to set `DATADIR` to your LOBSTER data directory, then:

```bash
make build              # build image
make run                # interactive shell
make ppo_2player gpu=0  # run training on GPU 0
```

The repo is mounted at `/home/myuser/` and data at `/home/myuser/data/`, so the default env config paths work without modification. For WandB, set `export WANDB_API_KEY=<your-key>` before running.

## Repository map (minimal)

- `config/env_configs/` and `config/rl_configs/`: environment + training configs
- `gymnax_exchange/jaxrl/MARL/ippo_rnn_JAXMARL.py`: original IPPO training entrypoint
- `run_gen_worldmodel_pg_train.py`: generative-world-model training/experiment entrypoint
- `slurm/`: cluster job wrappers (smoke, sweep, train-best)
- `docs/AGENT_INFO.md`: agent/policy types, tunables, and optimization goals

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
