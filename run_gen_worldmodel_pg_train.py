#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

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

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_LOBS5_ROOT = Path(os.environ.get("LOBS5_ROOT", "/home/s5e/satyamaga.s5e/LOBS5"))
DEFAULT_LOBS5_CKPT = Path(
    os.environ.get(
        "WORLD_MODEL_CKPT",
        "/lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_H1-scaling-law/checkpoints/j2514440_bkotgtm5_2514440",
    )
)
DEFAULT_DATA = Path(os.environ.get("LOB_PREPROC_DATA_DIR", "/lus/lfs1aip2/projects/s5e/lob_preproc/GOOG"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Policy-gradient training in generative world-model loop")
    p.add_argument("--ckpt_path", default=str(DEFAULT_LOBS5_CKPT))
    p.add_argument("--data_dir", default=str(DEFAULT_DATA))
    p.add_argument("--lobs5_root", default=str(DEFAULT_LOBS5_ROOT))
    p.add_argument("--checkpoint_step", type=int, default=None)
    p.add_argument("--sample_index", type=int, default=0)
    p.add_argument("--n_cond_msgs", type=int, default=8)
    p.add_argument("--n_steps", type=int, default=10)
    p.add_argument("--n_envs", type=int, default=4)
    p.add_argument("--n_updates", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_split", type=float, default=1.0)
    p.add_argument("--start_date", default="2026-01-01")
    p.add_argument("--end_date", default="2026-01-31")
    p.add_argument("--sample_top_n", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--entropy_coef", type=float, default=1e-3)
    p.add_argument("--baseline_momentum", type=float, default=0.9)
    p.add_argument("--gpu_id", default="0")
    p.add_argument("--fast_startup", action="store_true")
    p.add_argument("--jit_message_build", action="store_true")
    p.add_argument("--compile_cache_dir", default=str(REPO_ROOT / ".cache" / "jax_compilation"))
    p.add_argument("--output_root", default=str(REPO_ROOT / "outputs" / "gen_worldmodel_pg_train"))
    p.add_argument("--run_name", default="")
    return p.parse_args()


def _build_obs(step_i: int, bid: int, ask: int, agent_state, world_time: jnp.ndarray) -> jnp.ndarray:
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


def _init_policy_params(key: jax.Array, obs_dim: int, hidden_dim: int, n_actions: int) -> dict[str, jax.Array]:
    k1, k2 = jax.random.split(key)
    return {
        "w1": 0.02 * jax.random.normal(k1, (obs_dim, hidden_dim)),
        "b1": jnp.zeros((hidden_dim,), dtype=jnp.float32),
        "w2": 0.02 * jax.random.normal(k2, (hidden_dim, n_actions)),
        "b2": jnp.zeros((n_actions,), dtype=jnp.float32),
    }


def _policy_logits(params: dict[str, jax.Array], obs: jax.Array) -> jax.Array:
    h = jnp.tanh(obs @ params["w1"] + params["b1"])
    return h @ params["w2"] + params["b2"]


@jax.jit
def _policy_loss(
    params: dict[str, jax.Array],
    obs_batch: jax.Array,
    act_batch: jax.Array,
    adv_batch: jax.Array,
    entropy_coef: float,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    logits = _policy_logits(params, obs_batch)
    logp = jax.nn.log_softmax(logits, axis=-1)
    probs = jax.nn.softmax(logits, axis=-1)
    chosen_logp = jnp.take_along_axis(logp, act_batch[:, None], axis=1).squeeze(1)
    entropy = -jnp.sum(probs * logp, axis=1)
    loss = -jnp.mean(adv_batch * chosen_logp + entropy_coef * entropy)
    return loss, {"entropy": jnp.mean(entropy), "logp": jnp.mean(chosen_logp)}


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

    selected_data_dir, temp_data_ctx = _prepare_date_filtered_data_dir(data_dir, args.start_date, args.end_date)
    t_dataset_t0 = time.perf_counter()
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
    m_seq_inp = m_seq[:, : args.n_cond_msgs * Message_Tokenizer.MSG_LEN + 1]
    b_seq_inp = b_seq[:, : args.n_cond_msgs + 1]
    m_seq_raw_inp = msg_seq_raw[:, : args.n_cond_msgs]
    init_time_batched = b_seq_pv[:, 0, 1:3]

    t_sim_init_t0 = time.perf_counter()
    sim = inference.OrderBook(cfg=JAXLOB_Configuration())
    sim_states_all = inference.get_sims_vmap(book_l2_init, m_seq_raw_inp, init_time_batched, sim)
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
    agent_params = MMEnvParams(trader_id=jnp.int32(-101), time_delay_obs_act=jnp.int32(0), normalize=jnp.bool_(True))

    hidden_init = model.initialize_carry(
        1,
        hidden_size=(model_args.ssm_size_base // pow(2, int(model_args.conj_sym))),
        n_message_layers=model_args.n_message_layers,
        n_book_pre_layers=model_args.n_book_pre_layers,
        n_book_post_layers=model_args.n_book_post_layers,
        n_fused_layers=model_args.n_layers,
        h_size_ema=model_args.ssm_size_base,
    )

    message_build_fn = None
    if args.jit_message_build:
        def _message_build_impl(action, sim_state, world_time, cur_agent_state, cur_agent_params):
            world_state = _build_world_state(sim, sim_state, world_time)
            return mm_agent.get_messages(action, world_state, cur_agent_state, cur_agent_params)

        message_build_fn = jax.jit(_message_build_impl)
        try:
            warm = message_build_fn(jnp.int32(0), jax.tree_util.tree_map(lambda x: x[0], sim_states_all), jnp.array(init_time_batched[0], dtype=jnp.int32), initial_agent_state, agent_params)
            jax.block_until_ready(warm[0])
        except Exception:
            message_build_fn = None

    rng = jax.random.key(args.seed)
    policy_key, rng = jax.random.split(rng)
    obs_dim = 12
    n_actions = int(mm_cfg.n_actions)
    policy_params = _init_policy_params(policy_key, obs_dim=obs_dim, hidden_dim=64, n_actions=n_actions)
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(policy_params)

    baseline = 0.0
    update_logs: list[dict[str, Any]] = []

    for upd in range(args.n_updates):
        env_sim_states = [jax.tree_util.tree_map(lambda x, i=i: x[i], sim_states_all) for i in range(n_envs)]
        env_agent_states = [initial_agent_state for _ in range(n_envs)]
        env_world_times = [jnp.array(init_time_batched[i], dtype=jnp.int32) for i in range(n_envs)]

        obs_rows: list[jax.Array] = []
        act_rows: list[int] = []
        final_pnls: list[float] = []
        step_lat_ms: list[float] = []
        t_roll_t0 = time.perf_counter()

        for step_i in range(args.n_steps):
            for env_i in range(n_envs):
                t0 = time.perf_counter()
                cur_sim = env_sim_states[env_i]
                cur_time = env_world_times[env_i]
                agent_state = env_agent_states[env_i]

                bid, ask = _best_quotes(sim, cur_sim)
                obs = _build_obs(step_i, bid, ask, agent_state, cur_time)
                logits = _policy_logits(policy_params, obs)
                rng, rs = jax.random.split(rng)
                action = int(jax.random.categorical(rs, logits))
                action_j = jnp.int32(action)

                if message_build_fn is not None:
                    action_msgs, cancel_msgs, _ = message_build_fn(action_j, cur_sim, cur_time, agent_state, agent_params)
                else:
                    world_state = _build_world_state(sim, cur_sim, cur_time)
                    action_msgs, cancel_msgs, _ = mm_agent.get_messages(action_j, world_state, agent_state, agent_params)

                num_action_msgs = int(action_msgs.shape[0])
                if num_action_msgs > 0:
                    base_order_id = -100000 - ((step_i * n_envs + env_i) * num_action_msgs)
                    action_order_ids = (base_order_id - jnp.arange(num_action_msgs, dtype=jnp.int32)).astype(jnp.int32)
                    action_msgs = action_msgs.at[:, 4].set(action_order_ids)

                action_msgs = _sanitize_action_msgs(action_msgs)
                cancel_msgs = _sanitize_action_msgs(cancel_msgs)
                combined = jnp.concatenate([cancel_msgs, action_msgs], axis=0)
                sim_after_action = sim.process_orders_array(cur_sim, combined)
                bid2, ask2 = _best_quotes(sim, sim_after_action)
                if bid2 <= 0 or ask2 <= 0:
                    sim_after_action = cur_sim

                rng, rg = jax.random.split(rng)
                with contextlib.redirect_stdout(io.StringIO()):
                    msgs_decoded, _, _, _ = inference.generate(
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
                        sim_after_action,
                        rg,
                        hidden_init,
                        True,
                        jnp.asarray(cur_time),
                        False,
                        None,
                    )

                first_msg = msgs_decoded[0]
                gen_sim_msg = inference.msg_to_jnp(first_msg)
                sim_after_step = sim.process_order_array(sim_after_action, gen_sim_msg)
                bid3, ask3 = _best_quotes(sim, sim_after_step)
                if bid3 <= 0 or ask3 <= 0:
                    sim_after_step = sim_after_action
                    bid3, ask3 = _best_quotes(sim, sim_after_step)

                mid = _midprice_from_quotes(bid3, ask3)
                pnl = _compute_agent_pnl_from_trades(
                    sim_after_step.trades,
                    trader_id=int(agent_params.trader_id),
                    tick_size=int(world_cfg.tick_size),
                    final_midprice=mid,
                )
                env_agent_states[env_i] = MMEnvState(
                    posted_distance_bid=0,
                    posted_distance_ask=0,
                    inventory=int(round(float(pnl["inventory"]))),
                    total_PnL=float(pnl["total_pnl"]),
                    cash_balance=float(pnl["cash_pnl"]),
                )

                msg_time_s = int(first_msg[8])
                msg_time_ns = int(first_msg[9])
                if msg_time_s >= 0 and msg_time_ns >= 0:
                    cur_time = jnp.array([msg_time_s, msg_time_ns], dtype=jnp.int32)

                env_sim_states[env_i] = sim_after_step
                env_world_times[env_i] = cur_time

                obs_rows.append(obs)
                act_rows.append(action)
                step_lat_ms.append((time.perf_counter() - t0) * 1000.0)

        for env_i in range(n_envs):
            final_sim_i = env_sim_states[env_i]
            final_mid_i = _midprice_from_quotes(*_best_quotes(sim, final_sim_i))
            final_pnl = _compute_agent_pnl_from_trades(
                final_sim_i.trades,
                trader_id=int(agent_params.trader_id),
                tick_size=int(world_cfg.tick_size),
                final_midprice=final_mid_i,
            )
            final_pnls.append(float(final_pnl["total_pnl"]))

        rollout_sec = float(time.perf_counter() - t_roll_t0)
        obs_batch = jnp.stack(obs_rows, axis=0)
        act_batch = jnp.asarray(act_rows, dtype=jnp.int32)
        rewards_env = np.asarray(final_pnls, dtype=np.float32)
        avg_pnl = float(rewards_env.mean())
        baseline = args.baseline_momentum * baseline + (1.0 - args.baseline_momentum) * avg_pnl
        advantages = np.repeat(rewards_env - baseline, args.n_steps).astype(np.float32)
        adv_batch = jnp.asarray(advantages)

        (loss, aux), grads = jax.value_and_grad(_policy_loss, has_aux=True)(
            policy_params,
            obs_batch,
            act_batch,
            adv_batch,
            float(args.entropy_coef),
        )
        updates, opt_state = optimizer.update(grads, opt_state, policy_params)
        policy_params = optax.apply_updates(policy_params, updates)

        steps_done = n_envs * args.n_steps
        steps_per_sec = float(steps_done / rollout_sec) if rollout_sec > 0 else 0.0
        update_log = {
            "update": upd + 1,
            "avg_pnl": avg_pnl,
            "pnl_std": float(rewards_env.std()),
            "loss": float(loss),
            "entropy": float(aux["entropy"]),
            "steps_per_sec": steps_per_sec,
            "step_latency_ms_p50": float(np.percentile(step_lat_ms, 50)) if step_lat_ms else 0.0,
            "step_latency_ms_p95": float(np.percentile(step_lat_ms, 95)) if step_lat_ms else 0.0,
        }
        update_logs.append(update_log)
        print(
            f"update={upd + 1}/{args.n_updates} avg_pnl={avg_pnl:.4f} loss={float(loss):.5f} "
            f"steps_per_sec={steps_per_sec:.3f}"
        )

    avg_pnls = [x["avg_pnl"] for x in update_logs]
    summary = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "checkpoint_path": str(ckpt_path),
        "checkpoint_step": int(step),
        "data_dir": str(data_dir),
        "dataset_effective_dir": str(selected_data_dir),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "seed": int(args.seed),
        "n_envs": int(n_envs),
        "n_steps": int(args.n_steps),
        "n_updates": int(args.n_updates),
        "n_cond_msgs": int(args.n_cond_msgs),
        "action_dim": int(n_actions),
        "throughput": {
            "updates_mean_steps_per_sec": float(np.mean([u["steps_per_sec"] for u in update_logs])) if update_logs else 0.0,
            "updates_max_steps_per_sec": float(np.max([u["steps_per_sec"] for u in update_logs])) if update_logs else 0.0,
        },
        "pnl": {
            "updates_avg_pnl": avg_pnls,
            "mean_avg_pnl": float(np.mean(avg_pnls)) if avg_pnls else 0.0,
            "best_avg_pnl": float(np.max(avg_pnls)) if avg_pnls else 0.0,
            "final_avg_pnl": float(avg_pnls[-1]) if avg_pnls else 0.0,
        },
        "timing_breakdown": {
            "restore_sec": restore_sec,
            "dataset_load_sec": dataset_load_sec,
            "sim_init_sec": sim_init_sec,
            "total_runtime_sec": float(time.perf_counter() - run_t0),
        },
        "update_logs": update_logs,
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Done. Summary written to: {summary_path}")

    if temp_data_ctx is not None:
        temp_data_ctx.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
