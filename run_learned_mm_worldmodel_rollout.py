#!/usr/bin/env python3
"""Learned-policy market-making rollout with one-step world-model generation.

This is a Phase 1/2 implementation starter:
- Loads a learned MARL policy checkpoint (IPPO-style GRU policy)
- Runs per-step policy inference to choose MM action
- Applies action + cancel semantics via MarketMakingAgent.get_messages
- Runs one generated world-model message per step and updates simulator state
- Reports timing/throughput and action histogram for smoke validation
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as oxcp
from flax.training.train_state import TrainState
from flax.training import orbax_utils
from omegaconf import OmegaConf

from run_one_step_inference import (
    _add_python_paths,
    _enable_legacy_token_mode_22,
    _ensure_model_args_defaults,
    _latest_checkpoint_step,
    _load_metadata_robust,
    _prepare_date_filtered_data_dir,
    _restore_params_only,
)
from minimal_agent_generative_step import (
    _best_quotes,
    _build_world_state,
    _compute_agent_pnl_from_trades,
    _configure_runtime,
    _midprice_from_quotes,
    _sanitize_action_msgs,
)

from gymnax_exchange.jaxrl.MARL.baseline_eval.baseline_JAXMARL import ActorCriticRNN, ScannedRNN


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_LOBS5_ROOT = Path(
    os.environ.get("LOBS5_ROOT", "/home/s5e/satyamaga.s5e/LOBS5")
)
DEFAULT_LOBS5_CKPT = Path(
    os.environ.get(
        "WORLD_MODEL_CKPT",
        "/lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_H1-scaling-law/checkpoints/j2514440_bkotgtm5_2514440",
    )
)
DEFAULT_DATA = Path(
    os.environ.get("LOB_PREPROC_DATA_DIR", "/lus/lfs1aip2/projects/s5e/lob_preproc/GOOG")
)
DEFAULT_MARL_CKPT = REPO_ROOT / "checkpoints" / "MARLCheckpoints" / "2PLayer" / "dummy-2vdmzbye"
DEFAULT_MARL_CONFIG = REPO_ROOT / "config" / "rl_configs" / "ippo_rnn_JAXMARL_mm_BOB.yaml"


class LearnedPolicyAdapter:
    def __init__(
        self,
        checkpoint_dir: Path,
        config_path: Path,
        obs_dim: int,
        action_dim: int,
        seed: int,
        checkpoint_step: int | None = None,
        deterministic: bool = True,
        model_index: int = 0,
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.config_path = config_path
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.deterministic = bool(deterministic)
        self.model_index = int(model_index)
        self.rng = jax.random.PRNGKey(seed)

        cfg_raw = OmegaConf.to_container(OmegaConf.load(str(config_path)), resolve=True)
        if not isinstance(cfg_raw, dict):
            raise ValueError(f"Invalid policy config format: {config_path}")
        self.config = self._normalize_config(cfg_raw)

        self.network = ActorCriticRNN(self.action_dim, config=self.config)
        self.hidden = ScannedRNN.initialize_carry(1, int(self.config["GRU_HIDDEN_DIM"]))

        self.train_state = self._build_template_train_state()
        self._restore(checkpoint_step)

        inferred_action_dim = self._infer_action_dim_from_params(self.train_state.params)
        if inferred_action_dim is not None and inferred_action_dim != self.action_dim:
            self.action_dim = int(inferred_action_dim)
            self.network = ActorCriticRNN(self.action_dim, config=self.config)
            self.train_state = self.train_state.replace(apply_fn=self.network.apply)

        inferred_obs_dim = self._infer_obs_dim_from_params(self.train_state.params)
        if inferred_obs_dim is not None and inferred_obs_dim != self.obs_dim:
            self.obs_dim = int(inferred_obs_dim)
        self._apply_jit = jax.jit(self.train_state.apply_fn)

    @staticmethod
    def _as_scalar(value: Any) -> Any:
        if isinstance(value, (list, tuple)) and value:
            return value[0]
        return value

    def _normalize_config(self, cfg: dict[str, Any]) -> dict[str, Any]:
        out = dict(cfg)
        out["NUM_ENVS"] = 1
        out["GRU_HIDDEN_DIM"] = int(self._as_scalar(out.get("GRU_HIDDEN_DIM", 256)))
        out["FC_DIM_SIZE"] = int(self._as_scalar(out.get("FC_DIM_SIZE", 256)))
        out["MAX_GRAD_NORM"] = float(self._as_scalar(out.get("MAX_GRAD_NORM", 0.5)))
        out["ANNEAL_LR"] = bool(self._as_scalar(out.get("ANNEAL_LR", False)))
        out["LR"] = float(self._as_scalar(out.get("LR", 1e-4)))
        return out

    def _build_template_train_state(self) -> TrainState:
        self.rng, init_rng = jax.random.split(self.rng)
        init_x = (
            jnp.zeros((1, 1, self.obs_dim), dtype=jnp.float32),
            jnp.zeros((1, 1), dtype=jnp.bool_),
        )
        params = self.network.init(init_rng, self.hidden, init_x)
        tx = optax.sgd(learning_rate=0.0)
        return TrainState.create(apply_fn=self.network.apply, params=params, tx=tx)

    @staticmethod
    def _infer_obs_dim_from_params(params: Any) -> int | None:
        try:
            kernel = params["params"]["Dense_0"]["kernel"]
            if hasattr(kernel, "shape") and len(kernel.shape) == 2:
                return int(kernel.shape[0])
        except Exception:
            return None
        return None

    @staticmethod
    def _infer_action_dim_from_params(params: Any) -> int | None:
        try:
            kernel = params["params"]["SingleActionOutput_0"]["Dense_0"]["kernel"]
            if hasattr(kernel, "shape") and len(kernel.shape) == 2:
                return int(kernel.shape[1])
        except Exception:
            return None
        return None

    def _restore(self, checkpoint_step: int | None) -> None:
        checkpointer = oxcp.PyTreeCheckpointer()
        manager = oxcp.CheckpointManager(str(self.checkpoint_dir), checkpointer)
        step = checkpoint_step if checkpoint_step is not None else manager.latest_step()
        if step is None:
            raise RuntimeError(f"No checkpoint step found in {self.checkpoint_dir}")

        # First try schema-free restore. This is robust across different checkpoint
        # tree layouts used by training/eval scripts.
        try:
            restored_raw = manager.restore(int(step))
            if isinstance(restored_raw, dict) and "model" in restored_raw:
                model_states = restored_raw["model"]
                if isinstance(model_states, (list, tuple)) and len(model_states) > 0:
                    idx = max(0, min(self.model_index, len(model_states) - 1))
                    model_state = model_states[idx]
                    if isinstance(model_state, TrainState):
                        self.train_state = model_state
                        return
                    if isinstance(model_state, dict) and "params" in model_state:
                        self.train_state = self.train_state.replace(params=model_state["params"])
                        return
        except Exception:
            # Fallback to template-based restore below.
            pass

        candidate_targets = [
            {
                "model": [self.train_state],
                "metrics": {"train_rewards": [np.nan]},
            },
            {
                "model": [self.train_state],
                "metrics": {"train_rewards": [np.nan], "eval_rewards": [np.nan]},
            },
            {
                "model": [self.train_state],
            },
        ]

        last_err = None
        restored = None
        for target in candidate_targets:
            try:
                restored = manager.restore(
                    int(step),
                    items=target,
                    restore_kwargs={"restore_args": orbax_utils.restore_args_from_target(target)},
                )
                break
            except Exception as err:  # noqa: BLE001
                last_err = err

        if restored is None:
            raise RuntimeError(
                "Failed to restore learned policy checkpoint with supported schemas"
            ) from last_err

        model_states = restored.get("model", None)
        if not isinstance(model_states, (list, tuple)) or len(model_states) == 0:
            raise RuntimeError("Restored checkpoint missing non-empty 'model' list")

        idx = max(0, min(self.model_index, len(model_states) - 1))
        self.train_state = model_states[idx]

    def act(self, obs_vec: jnp.ndarray, done: bool = False) -> int:
        if obs_vec.shape[-1] != self.obs_dim:
            raise ValueError(
                f"Policy obs dim mismatch: expected {self.obs_dim}, got {obs_vec.shape[-1]}"
            )

        dones = jnp.array([done], dtype=jnp.bool_)
        ac_in = (obs_vec.reshape(1, 1, -1), dones.reshape(1, 1))
        self.hidden, pi, _value = self._apply_jit(self.train_state.params, self.hidden, ac_in)

        if self.deterministic and hasattr(pi, "logits"):
            action = jnp.argmax(pi.logits, axis=-1)
        else:
            self.rng, sample_rng = jax.random.split(self.rng)
            action = pi.sample(seed=sample_rng)

        return int(jnp.asarray(action).reshape(-1)[0])

    def act_with_state(self, obs_vec: jnp.ndarray, hidden: Any, done: bool = False) -> tuple[int, Any]:
        """Stateless act: takes hidden state explicitly, returns (action, new_hidden)."""
        if obs_vec.shape[-1] != self.obs_dim:
            raise ValueError(
                f"Policy obs dim mismatch: expected {self.obs_dim}, got {obs_vec.shape[-1]}"
            )
        dones = jnp.array([done], dtype=jnp.bool_)
        ac_in = (obs_vec.reshape(1, 1, -1), dones.reshape(1, 1))
        new_hidden, pi, _value = self._apply_jit(self.train_state.params, hidden, ac_in)
        if self.deterministic and hasattr(pi, "logits"):
            action = jnp.argmax(pi.logits, axis=-1)
        else:
            self.rng, sample_rng = jax.random.split(self.rng)
            action = pi.sample(seed=sample_rng)
        return int(jnp.asarray(action).reshape(-1)[0]), new_hidden

    def fresh_hidden(self) -> Any:
        """Return a fresh zero-initialized hidden carry for one trajectory."""
        return ScannedRNN.initialize_carry(1, int(self.config["GRU_HIDDEN_DIM"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Learned policy world-model rollout")
    parser.add_argument("--ckpt_path", default=str(DEFAULT_LOBS5_CKPT), help="LOBS5 checkpoint path")
    parser.add_argument("--data_dir", default=str(DEFAULT_DATA), help="Processed .npy data directory")
    parser.add_argument("--lobs5_root", default=str(DEFAULT_LOBS5_ROOT), help="LOBS5 repository path")
    parser.add_argument("--sample_index", type=int, default=0, help="Dataset sample index")
    parser.add_argument("--checkpoint_step", type=int, default=None, help="LOBS5 checkpoint step")
    parser.add_argument("--n_cond_msgs", type=int, default=64, help="Conditioning messages")
    parser.add_argument("--sample_top_n", type=int, default=1, help="1=greedy decode")
    parser.add_argument("--n_steps", type=int, default=25, help="Rollout steps")
    parser.add_argument("--seed", type=int, default=42, help="PRNG seed")
    parser.add_argument("--test_split", type=float, default=1.0, help="Dataset split fraction")
    parser.add_argument("--start_date", default="2026-01-01", help="Inclusive start date filter (YYYY-MM-DD)")
    parser.add_argument("--end_date", default="2026-01-31", help="Inclusive end date filter (YYYY-MM-DD)")

    parser.add_argument("--policy_ckpt_dir", default=str(DEFAULT_MARL_CKPT), help="Learned policy checkpoint dir")
    parser.add_argument("--policy_config", default=str(DEFAULT_MARL_CONFIG), help="Policy YAML config")
    parser.add_argument("--policy_checkpoint_step", type=int, default=None, help="Policy checkpoint step")
    parser.add_argument("--policy_model_index", type=int, default=1, help="Model index inside checkpoint model list")
    parser.add_argument("--policy_obs_dim", type=int, default=12, help="Expected policy observation dim")
    parser.add_argument("--policy_action_dim", type=int, default=5, help="Discrete action count")
    parser.add_argument("--policy_deterministic", action="store_true", help="Use argmax action")
    parser.add_argument("--allow_obs_pad", action="store_true", help="Pad/truncate obs vector to policy_obs_dim")

    parser.add_argument("--gpu_id", default="0", help="Single-GPU pinning target")
    parser.add_argument(
        "--compile_cache_dir",
        default=str(REPO_ROOT / ".cache" / "jax_compilation"),
        help="JAX compilation cache directory",
    )
    parser.add_argument(
        "--output_root",
        default=str(REPO_ROOT / "outputs" / "learned_mm_worldmodel_rollout"),
        help="Output root for summary artifacts",
    )
    parser.add_argument("--run_name", default="", help="Optional run name suffix")
    parser.add_argument("--fast_startup", action="store_true", help="Lower startup memory reservation")
    parser.add_argument("--n_envs", type=int, default=1, help="Number of parallel environment trajectories (batch-size sweep)")
    parser.add_argument(
        "--jit_message_build",
        action="store_true",
        help="JIT-compile world-state + get_messages path to reduce message_build latency",
    )
    return parser.parse_args()


def _fit_obs_dim(obs: jnp.ndarray, expected_dim: int, allow_pad: bool) -> jnp.ndarray:
    got = int(obs.shape[-1])
    if got == expected_dim:
        return obs
    if not allow_pad:
        raise ValueError(
            f"Policy observation mismatch: built {got} features but policy expects {expected_dim}. "
            "Pass --allow_obs_pad for temporary padding/truncation during smoke testing."
        )
    if got < expected_dim:
        pad = jnp.zeros((expected_dim - got,), dtype=obs.dtype)
        return jnp.concatenate([obs, pad], axis=0)
    return obs[:expected_dim]


def _build_policy_obs(step_i: int, bid: int, ask: int, agent_state, world_time: jnp.ndarray) -> jnp.ndarray:
    spread = int(ask - bid)
    mid = _midprice_from_quotes(bid, ask)
    return jnp.asarray(
        [
            float(step_i),
            float(agent_state.inventory),
            float(agent_state.cash_balance),
            float(agent_state.total_PnL),
            float(bid),
            float(ask),
            float(spread),
            float(mid),
            float(world_time[0]),
            float(world_time[1]),
            0.0,
            0.0,
        ],
        dtype=jnp.float32,
    )


def _safe_percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, p))


def _gpu_memory_snapshot(gpu_id: str) -> dict[str, float] | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
                "-i",
                str(gpu_id),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        line = result.stdout.strip().splitlines()[0]
        used_str, total_str = [x.strip() for x in line.split(",")[:2]]
        used_mib = float(used_str)
        total_mib = float(total_str)
        util_pct = (used_mib / total_mib * 100.0) if total_mib > 0 else 0.0
        return {
            "used_mib": used_mib,
            "total_mib": total_mib,
            "utilization_pct": util_pct,
            "source": "nvidia-smi",
        }
    except Exception:
        return None


def main() -> int:
    run_t0 = time.perf_counter()
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    _configure_runtime(args)

    output_root = Path(args.output_root).expanduser().resolve()
    run_name = args.run_name.strip() or time.strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.ckpt_path).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve()
    lobs5_root = Path(args.lobs5_root).expanduser().resolve()
    policy_ckpt_dir = Path(args.policy_ckpt_dir).expanduser().resolve()
    policy_config = Path(args.policy_config).expanduser().resolve()

    _add_python_paths(lobs5_root)

    step = args.checkpoint_step if args.checkpoint_step is not None else _latest_checkpoint_step(ckpt_path)
    t_restore_t0 = time.perf_counter()
    params = _restore_params_only(ckpt_path, step)
    restore_sec = float(time.perf_counter() - t_restore_t0)
    ckpt_vocab_size = int(params["message_encoder"]["encoder"]["embedding"].shape[0])
    if ckpt_vocab_size >= 10000:
        _enable_legacy_token_mode_22()

    from lob.encoding import Message_Tokenizer, Vocab
    from lob.init_train import init_train_state
    from lob import inference_no_errcorr as inference
    from gymnax_exchange.jaxob.jaxob_config import (
        JAXLOB_Configuration,
        MarketMaking_EnvironmentConfig,
        World_EnvironmentConfig,
    )
    from gymnax_exchange.jaxen.StatesandParams import MMEnvParams, MMEnvState
    from gymnax_exchange.jaxen.mm_env import MarketMakingAgent

    model_args = _load_metadata_robust(ckpt_path)
    model_args = _ensure_model_args_defaults(model_args)
    model_args.num_devices = 1
    model_args.bsz = 1
    model_args.micro_bsz = 1
    model_args.global_bsz = 1
    if ckpt_vocab_size >= 10000:
        model_args.token_mode = 22

    vocab = Vocab()
    n_eval_messages = max(args.n_steps + 1, 2)
    eval_seq_len = (n_eval_messages - 1) * Message_Tokenizer.MSG_LEN

    init_state, model_cls = init_train_state(
        model_args,
        n_classes=ckpt_vocab_size,
        seq_len=eval_seq_len,
        book_dim=503,
        book_seq_len=eval_seq_len,
    )
    state = init_state.replace(params=params, step=step)
    model = model_cls(training=False, step_rescale=1.0)

    t_dataset_t0 = time.perf_counter()
    selected_data_dir, temp_data_ctx = _prepare_date_filtered_data_dir(data_dir, args.start_date, args.end_date)
    ds = inference.get_dataset(
        str(selected_data_dir),
        args.n_cond_msgs,
        n_eval_messages,
        test_split=args.test_split,
    )
    dataset_load_sec = float(time.perf_counter() - t_dataset_t0)
    if len(ds) == 0:
        raise RuntimeError("Dataset is empty")
    idx = max(0, min(args.sample_index, len(ds) - 1))

    n_envs = max(1, args.n_envs)
    sample_indices = [max(0, min(idx + i, len(ds) - 1)) for i in range(n_envs)]

    m_seq, _, b_seq_pv, msg_seq_raw, book_l2_init = ds[sample_indices]
    m_seq = jnp.array(m_seq)
    b_seq_pv = jnp.array(b_seq_pv)
    msg_seq_raw = jnp.array(msg_seq_raw)
    book_l2_init = jnp.array(book_l2_init)

    b_seq = inference.transform_L2_state_batch(b_seq_pv, 500, 100)
    m_seq_inp = m_seq[:, : args.n_cond_msgs * Message_Tokenizer.MSG_LEN + 1]  # (n_envs, seq_len)
    b_seq_inp = b_seq[:, : args.n_cond_msgs + 1]                              # (n_envs, n_cond+1, ...)
    m_seq_raw_inp = msg_seq_raw[:, : args.n_cond_msgs]                        # (n_envs, n_cond, ...)
    init_time_batched = b_seq_pv[:, 0, 1:3]                                   # (n_envs, 2)

    t_sim_init_t0 = time.perf_counter()
    sim = inference.OrderBook(cfg=JAXLOB_Configuration())
    sim_states_all = inference.get_sims_vmap(book_l2_init, m_seq_raw_inp, init_time_batched, sim)
    env_sim_states = [
        jax.tree_util.tree_map(lambda x, i=i: x[i], sim_states_all)
        for i in range(n_envs)
    ]
    sim_init_sec = float(time.perf_counter() - t_sim_init_t0)

    world_cfg = World_EnvironmentConfig(tick_size=100)
    mm_cfg = MarketMaking_EnvironmentConfig(action_space="bobStrategy", fixed_quant_value=10, bob_v0=10)
    mm_agent = MarketMakingAgent(cfg=mm_cfg, world_config=world_cfg)

    initial_agent_state = MMEnvState(
        posted_distance_bid=0,
        posted_distance_ask=0,
        inventory=0,
        total_PnL=0.0,
        cash_balance=0.0,
    )
    agent_params = MMEnvParams(
        trader_id=jnp.int32(-101),
        time_delay_obs_act=jnp.int32(0),
        normalize=jnp.bool_(True),
    )

    env_agent_states = [initial_agent_state for _ in range(n_envs)]
    env_world_times = [jnp.array(init_time_batched[i], dtype=jnp.int32) for i in range(n_envs)]

    message_build_fn = None
    message_build_jit_enabled = False
    message_build_jit_compile_sec = 0.0

    if args.jit_message_build:
        def _message_build_impl(action, sim_state, world_time, cur_agent_state, cur_agent_params):
            world_state = _build_world_state(sim, sim_state, world_time)
            return mm_agent.get_messages(action, world_state, cur_agent_state, cur_agent_params)

        message_build_fn = jax.jit(_message_build_impl)
        t_msg_jit_t0 = time.perf_counter()
        try:
            warm_action_msgs, _warm_cancel_msgs, _warm_extras = message_build_fn(
                jnp.int32(0),
                env_sim_states[0],
                env_world_times[0],
                env_agent_states[0],
                agent_params,
            )
            jax.block_until_ready(warm_action_msgs)
            message_build_jit_enabled = True
        except Exception as err:
            print(f"[WARN] message_build JIT compile failed; falling back to eager path: {err}")
            message_build_fn = None
            message_build_jit_enabled = False
        message_build_jit_compile_sec = float(time.perf_counter() - t_msg_jit_t0)

    t_hidden_init_t0 = time.perf_counter()
    init_hidden = model.initialize_carry(
        1,
        hidden_size=(model_args.ssm_size_base // pow(2, int(model_args.conj_sym))),
        n_message_layers=model_args.n_message_layers,
        n_book_pre_layers=model_args.n_book_pre_layers,
        n_book_post_layers=model_args.n_book_post_layers,
        n_fused_layers=model_args.n_layers,
        h_size_ema=model_args.ssm_size_base,
    )
    hidden_init_sec = float(time.perf_counter() - t_hidden_init_t0)

    t_policy_init_t0 = time.perf_counter()
    policy = LearnedPolicyAdapter(
        checkpoint_dir=policy_ckpt_dir,
        config_path=policy_config,
        obs_dim=args.policy_obs_dim,
        action_dim=args.policy_action_dim,
        seed=args.seed,
        checkpoint_step=args.policy_checkpoint_step,
        deterministic=args.policy_deterministic,
        model_index=args.policy_model_index,
    )
    policy_init_sec = float(time.perf_counter() - t_policy_init_t0)
    env_policy_hiddens = [policy.fresh_hidden() for _ in range(n_envs)]

    rng = jax.random.key(args.seed)
    step_latencies_ms: list[float] = []
    policy_latencies_ms: list[float] = []
    message_build_latencies_ms: list[float] = []
    sim_apply_latencies_ms: list[float] = []
    generate_latencies_ms: list[float] = []
    post_latencies_ms: list[float] = []
    action_hist: dict[int, int] = {}
    inventory_trace: list[float] = []
    pnl_trace: list[float] = []
    placed_order_msgs = 0
    generated_msgs_total = 0
    post_init_memory = _gpu_memory_snapshot(args.gpu_id)
    post_first_step_memory: dict[str, float] | None = None
    peak_memory_used_mib = float(post_init_memory["used_mib"]) if post_init_memory else 0.0
    total_steps_done = 0

    for step_i in range(args.n_steps):
        for env_i in range(n_envs):
            step_t0 = time.perf_counter()

            current_sim_state = env_sim_states[env_i]
            current_world_time = env_world_times[env_i]
            agent_state = env_agent_states[env_i]

            bid_before, ask_before = _best_quotes(sim, current_sim_state)
            obs_raw = _build_policy_obs(step_i, bid_before, ask_before, agent_state, current_world_time)
            obs = _fit_obs_dim(obs_raw, policy.obs_dim, args.allow_obs_pad)

            policy_t0 = time.perf_counter()
            action_int_val, new_hidden = policy.act_with_state(obs, env_policy_hiddens[env_i], done=False)
            action_this_step = jnp.int32(action_int_val)
            env_policy_hiddens[env_i] = new_hidden
            policy_ms = (time.perf_counter() - policy_t0) * 1000.0

            msg_build_t0 = time.perf_counter()
            if message_build_fn is not None:
                action_msgs, cancel_msgs, _extras = message_build_fn(
                    action_this_step,
                    current_sim_state,
                    current_world_time,
                    agent_state,
                    agent_params,
                )
            else:
                world_state = _build_world_state(sim, current_sim_state, current_world_time)
                action_msgs, cancel_msgs, _extras = mm_agent.get_messages(
                    action_this_step,
                    world_state,
                    agent_state,
                    agent_params,
                )

            num_action_msgs = int(action_msgs.shape[0])
            if num_action_msgs > 0:
                base_order_id = -100000 - ((step_i * n_envs + env_i) * num_action_msgs)
                action_order_ids = (base_order_id - jnp.arange(num_action_msgs, dtype=jnp.int32)).astype(jnp.int32)
                action_msgs = action_msgs.at[:, 4].set(action_order_ids)

            action_msgs = _sanitize_action_msgs(action_msgs)
            cancel_msgs = _sanitize_action_msgs(cancel_msgs)
            valid_place_msgs = (action_msgs[:, 0] == 1) & (action_msgs[:, 2] != 0)
            placed_order_msgs += int(jnp.sum(valid_place_msgs))
            combined_agent_msgs = jnp.concatenate([cancel_msgs, action_msgs], axis=0)
            msg_build_ms = (time.perf_counter() - msg_build_t0) * 1000.0

            sim_apply_t0 = time.perf_counter()
            sim_state_after_action = sim.process_orders_array(current_sim_state, combined_agent_msgs)
            bid_after_action, ask_after_action = _best_quotes(sim, sim_state_after_action)
            if bid_after_action <= 0 or ask_after_action <= 0:
                sim_state_after_action = current_sim_state
            sim_apply_ms = (time.perf_counter() - sim_apply_t0) * 1000.0

            rng, rng_gen = jax.random.split(rng)
            gen_t0 = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                msgs_decoded, _l2_states, _num_errors, _msg_tokens = inference.generate(
                    sim,
                    state,
                    model,
                    model_args.batchnorm,
                    vocab.ENCODING,
                    args.sample_top_n,
                    100,
                    m_seq_inp[env_i],
                    b_seq_inp[env_i],
                    1,
                    sim_state_after_action,
                    rng_gen,
                    init_hidden,
                    True,
                    jnp.asarray(current_world_time),
                    False,
                    None,
                )
            gen_ms = (time.perf_counter() - gen_t0) * 1000.0

            first_msg = msgs_decoded[0]
            post_t0 = time.perf_counter()
            gen_sim_msg = inference.msg_to_jnp(first_msg)
            sim_state_after_step = sim.process_order_array(sim_state_after_action, gen_sim_msg)
            bid_after_step, ask_after_step = _best_quotes(sim, sim_state_after_step)
            if bid_after_step <= 0 or ask_after_step <= 0:
                sim_state_after_step = sim_state_after_action
                bid_after_step, ask_after_step = _best_quotes(sim, sim_state_after_step)

            action_int = int(action_this_step)
            action_hist[action_int] = action_hist.get(action_int, 0) + 1
            generated_msgs_total += 1

            mid = _midprice_from_quotes(bid_after_step, ask_after_step)
            pnl_snapshot = _compute_agent_pnl_from_trades(
                sim_state_after_step.trades,
                trader_id=int(agent_params.trader_id),
                tick_size=int(world_cfg.tick_size),
                final_midprice=mid,
            )
            inventory_trace.append(float(pnl_snapshot["inventory"]))
            pnl_trace.append(float(pnl_snapshot["total_pnl"]))

            env_agent_states[env_i] = MMEnvState(
                posted_distance_bid=0,
                posted_distance_ask=0,
                inventory=int(round(float(pnl_snapshot["inventory"]))),
                total_PnL=float(pnl_snapshot["total_pnl"]),
                cash_balance=float(pnl_snapshot["cash_pnl"]),
            )

            print(
                f"step={step_i + 1}/{args.n_steps} env={env_i} action={action_int} "
                f"mid={mid:.1f} spread={ask_after_step - bid_after_step} "
                f"policy_ms={policy_ms:.2f} gen_ms={gen_ms:.2f}"
            )

            msg_time_s = int(first_msg[8])
            msg_time_ns = int(first_msg[9])
            if msg_time_s >= 0 and msg_time_ns >= 0:
                current_world_time = jnp.array([msg_time_s, msg_time_ns], dtype=jnp.int32)

            env_sim_states[env_i] = sim_state_after_step
            env_world_times[env_i] = current_world_time
            post_ms = (time.perf_counter() - post_t0) * 1000.0

            mem_now = _gpu_memory_snapshot(args.gpu_id)
            if mem_now is not None:
                peak_memory_used_mib = max(peak_memory_used_mib, float(mem_now["used_mib"]))
                if step_i == 0 and env_i == 0:
                    post_first_step_memory = mem_now

            step_ms = (time.perf_counter() - step_t0) * 1000.0
            step_latencies_ms.append(step_ms)
            policy_latencies_ms.append(policy_ms)
            message_build_latencies_ms.append(msg_build_ms)
            sim_apply_latencies_ms.append(sim_apply_ms)
            generate_latencies_ms.append(gen_ms)
            post_latencies_ms.append(post_ms)
            total_steps_done += 1

    rollout_sec = float(sum(step_latencies_ms) / 1000.0)
    total_sec = float(time.perf_counter() - run_t0)
    samples_per_sec = float(total_steps_done / rollout_sec) if rollout_sec > 0 else 0.0
    steps_per_sec = samples_per_sec
    generated_msgs_per_sec = float(generated_msgs_total / rollout_sec) if rollout_sec > 0 else 0.0

    # Aggregate trade stats across all environments
    agent_trade_count = 0
    final_pnl: dict[str, Any] = {}
    for _env_i in range(n_envs):
        final_sim_i = env_sim_states[_env_i]
        final_mid_i = _midprice_from_quotes(*_best_quotes(sim, final_sim_i))
        pnl_i = _compute_agent_pnl_from_trades(
            final_sim_i.trades,
            trader_id=int(agent_params.trader_id),
            tick_size=int(world_cfg.tick_size),
            final_midprice=final_mid_i,
        )
        agent_trade_count += int(pnl_i["agent_trade_count"])
        if _env_i == 0:
            final_pnl = pnl_i
    fill_rate_proxy = float(agent_trade_count / placed_order_msgs) if placed_order_msgs > 0 else 0.0

    steady_step_latencies_ms = step_latencies_ms[1:] if len(step_latencies_ms) > 1 else step_latencies_ms
    warmup = {
        "first_step_latency_ms": float(step_latencies_ms[0]) if step_latencies_ms else 0.0,
        "steady_state_step_latency_ms_p50": _safe_percentile(steady_step_latencies_ms, 50),
        "steady_state_step_latency_ms_p95": _safe_percentile(steady_step_latencies_ms, 95),
    }

    summary = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "gpu_id": str(args.gpu_id),
        "n_envs": n_envs,
        "checkpoint_path": str(ckpt_path),
        "policy_checkpoint_dir": str(policy_ckpt_dir),
        "policy_config": str(policy_config),
        "data_dir": str(data_dir),
        "dataset_effective_dir": str(selected_data_dir),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "sample_index": int(idx),
        "seed": int(args.seed),
        "n_steps": int(args.n_steps),
        "n_cond_msgs": int(args.n_cond_msgs),
        "policy_obs_dim": int(policy.obs_dim),
        "policy_action_dim": int(policy.action_dim),
        "action_histogram": {str(k): int(v) for k, v in sorted(action_hist.items())},
        "throughput": {
            "total_samples_processed": int(total_steps_done),
            "total_steps": int(total_steps_done),
            "generated_messages_total": int(generated_msgs_total),
            "samples_per_sec": samples_per_sec,
            "steps_per_sec": steps_per_sec,
            "generated_msgs_per_sec": generated_msgs_per_sec,
        },
        "timing_breakdown": {
            "restore_sec": restore_sec,
            "dataset_load_sec": dataset_load_sec,
            "sim_init_sec": sim_init_sec,
            "hidden_init_sec": hidden_init_sec,
            "policy_init_sec": policy_init_sec,
            "message_build_jit_enabled": bool(message_build_jit_enabled),
            "message_build_jit_compile_sec": float(message_build_jit_compile_sec),
            "step_latency_ms_p50": _safe_percentile(step_latencies_ms, 50),
            "step_latency_ms_p95": _safe_percentile(step_latencies_ms, 95),
            "policy_latency_ms_p50": _safe_percentile(policy_latencies_ms, 50),
            "policy_latency_ms_p95": _safe_percentile(policy_latencies_ms, 95),
            "message_build_latency_ms_p50": _safe_percentile(message_build_latencies_ms, 50),
            "message_build_latency_ms_p95": _safe_percentile(message_build_latencies_ms, 95),
            "sim_apply_latency_ms_p50": _safe_percentile(sim_apply_latencies_ms, 50),
            "sim_apply_latency_ms_p95": _safe_percentile(sim_apply_latencies_ms, 95),
            "generate_latency_ms_p50": _safe_percentile(generate_latencies_ms, 50),
            "generate_latency_ms_p95": _safe_percentile(generate_latencies_ms, 95),
            "postprocess_latency_ms_p50": _safe_percentile(post_latencies_ms, 50),
            "postprocess_latency_ms_p95": _safe_percentile(post_latencies_ms, 95),
            "warmup": warmup,
        },
        "policy_behavior": {
            "placed_order_msgs": int(placed_order_msgs),
            "agent_trade_count": int(agent_trade_count),
            "trade_incidence_nonzero": bool(agent_trade_count > 0),
            "fill_rate_proxy": fill_rate_proxy,
            "inventory_trace": [float(x) for x in inventory_trace],
            "pnl_trace": [float(x) for x in pnl_trace],
            "final_pnl": final_pnl,
        },
        "memory_telemetry": {
            "post_init": post_init_memory,
            "post_first_step": post_first_step_memory,
            "peak_used_mib": float(peak_memory_used_mib),
            "steady_state_peak_used_mib": float(peak_memory_used_mib),
        },
        "rollout_runtime_sec": rollout_sec,
        "total_runtime_sec": total_sec,
    }

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("=" * 80)
    print(f"Done. Summary written to: {summary_path}")
    print(
        f"Throughput={samples_per_sec:.2f} samples/s | "
        f"step p50={summary['timing_breakdown']['step_latency_ms_p50']:.2f}ms "
        f"p95={summary['timing_breakdown']['step_latency_ms_p95']:.2f}ms"
    )
    print("=" * 80)
    if temp_data_ctx is not None:
        temp_data_ctx.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
