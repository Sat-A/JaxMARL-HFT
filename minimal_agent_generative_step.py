#!/usr/bin/env python3
"""Minimal one-step test: agent action + generative world update.

Flow:
1) Load historical messages and initialize simulator state.
2) Initialize a simple market-making agent and produce one action.
3) Apply that action to the simulator.
4) Run one LOBS5 generative step from the post-action state.
5) Print clear status lines for quick validation.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
from dataclasses import dataclass
import io
import json
import os
from pathlib import Path
import time
from typing import Any

import jax
import jax.numpy as jnp

from run_one_step_inference import (
    _add_python_paths,
    _enable_legacy_token_mode_22,
    _ensure_model_args_defaults,
    _latest_checkpoint_step,
    _load_metadata_robust,
    _restore_params_only,
)


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_LOBS5_ROOT = Path("/homes/80/satyam/LOBS5")
DEFAULT_CKPT = Path(
    "/scratch/local/homes/groups/finance/data/checkpoints/lobs5_v2/twilight-sound-77_s42sujip"
)
DEFAULT_DATA = Path("/scratch/local/homes/groups/finance/data/processed_data/GOOG/2022")


@dataclass
class RunArtifacts:
    run_dir: Path
    action_csv: Path
    generated_csv: Path
    summary_json: Path
    report_txt: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal one-step agent + generative world test")
    parser.add_argument("--ckpt_path", default=str(DEFAULT_CKPT), help="LOBS5 checkpoint path")
    parser.add_argument("--data_dir", default=str(DEFAULT_DATA), help="Processed .npy data directory")
    parser.add_argument("--lobs5_root", default=str(DEFAULT_LOBS5_ROOT), help="LOBS5 repository path")
    parser.add_argument("--sample_index", type=int, default=0, help="Dataset sample index")
    parser.add_argument("--checkpoint_step", type=int, default=None, help="Checkpoint step (default latest)")
    parser.add_argument("--n_cond_msgs", type=int, default=64, help="Conditioning messages")
    parser.add_argument("--sample_top_n", type=int, default=1, help="Sampling mode (1=greedy)")
    parser.add_argument("--agent_action", type=int, default=0, help="MM action id for one-step action")
    parser.add_argument("--seed", type=int, default=42, help="PRNG seed")
    parser.add_argument("--test_split", type=float, default=1.0, help="Dataset split fraction")
    parser.add_argument(
        "--compile_cache_dir",
        default=str(REPO_ROOT / ".cache" / "jax_compilation"),
        help="JAX compilation cache directory",
    )
    parser.add_argument(
        "--output_root",
        default=str(REPO_ROOT / "outputs" / "minimal_agent_generative_step"),
        help="Directory root for verification outputs",
    )
    parser.add_argument("--run_name", default="", help="Optional run name suffix")
    parser.add_argument(
        "--show_model_stdout",
        action="store_true",
        help="Show raw stdout emitted by inference.generate (can be noisy)",
    )
    parser.add_argument("--fast_startup", action="store_true", help="Lower startup overhead")
    return parser.parse_args()


def _build_world_state(sim, sim_state, world_time):
    from gymnax_exchange.jaxen.StatesandParams import WorldState

    best_ask, best_bid = sim.get_best_bid_and_ask_inclQuants(sim_state)
    best_asks = jnp.expand_dims(best_ask, axis=0)
    best_bids = jnp.expand_dims(best_bid, axis=0)
    mid_price = jnp.float32((best_bid[0] + best_ask[0]) / 2)

    return WorldState(
        ask_raw_orders=sim_state.asks,
        bid_raw_orders=sim_state.bids,
        trades=sim_state.trades,
        init_time=world_time,
        window_index=0,
        max_steps_in_episode=1,
        start_index=0,
        step_counter=0,
        best_bids=best_bids,
        best_asks=best_asks,
        time=world_time,
        order_id_counter=-200,
        mid_price=mid_price,
        delta_time=0.0,
    )


def _best_quotes(sim, sim_state):
    best_ask, best_bid = sim.get_best_bid_and_ask_inclQuants(sim_state)
    return int(best_bid[0]), int(best_ask[0])


def _write_csv_rows(path: Path, rows, header):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def _configure_runtime(args: argparse.Namespace) -> None:
    if args.fast_startup:
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.50")
    else:
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(Path(args.compile_cache_dir).expanduser().resolve()))
    os.environ.setdefault("PYTHONNOUSERSITE", "1")


def _prepare_run_artifacts(args: argparse.Namespace) -> RunArtifacts:
    output_root = Path(args.output_root).expanduser().resolve()
    run_name = args.run_name.strip() if args.run_name.strip() else time.strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"{run_name}_action{int(args.agent_action)}_sample{int(args.sample_index)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return RunArtifacts(
        run_dir=run_dir,
        action_csv=run_dir / "agent_action_messages.csv",
        generated_csv=run_dir / "generated_message.csv",
        summary_json=run_dir / "verification_summary.json",
        report_txt=run_dir / "verification_report.txt",
    )


def _print_status_block(title: str, lines: list[str]) -> None:
    print("=" * 56)
    print(title)
    for line in lines:
        print(line)
    print("=" * 56)


def main() -> int:
    args = parse_args()
    _configure_runtime(args)
    artifacts = _prepare_run_artifacts(args)

    ckpt_path = Path(args.ckpt_path).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve()
    lobs5_root = Path(args.lobs5_root).expanduser().resolve()

    _add_python_paths(lobs5_root)

    step = args.checkpoint_step if args.checkpoint_step is not None else _latest_checkpoint_step(ckpt_path)
    params = _restore_params_only(ckpt_path, step)
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
    n_gen_msgs = 1
    n_eval_messages = n_gen_msgs + 1
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

    ds = inference.get_dataset(
        str(data_dir),
        args.n_cond_msgs,
        n_eval_messages,
        test_split=args.test_split,
    )
    if len(ds) == 0:
        raise RuntimeError("Dataset is empty")
    idx = max(0, min(args.sample_index, len(ds) - 1))

    m_seq, _, b_seq_pv, msg_seq_raw, book_l2_init = ds[[idx]]
    m_seq = jnp.array(m_seq)
    b_seq_pv = jnp.array(b_seq_pv)
    msg_seq_raw = jnp.array(msg_seq_raw)
    book_l2_init = jnp.array(book_l2_init)

    b_seq = inference.transform_L2_state_batch(b_seq_pv, 500, 100)
    m_seq_inp = m_seq[:, : args.n_cond_msgs * Message_Tokenizer.MSG_LEN + 1]
    b_seq_inp = b_seq[:, : args.n_cond_msgs + 1]
    m_seq_raw_inp = msg_seq_raw[:, : args.n_cond_msgs]
    init_time_batched = b_seq_pv[:, 0, 1:3]

    print("Historical messages loaded")

    sim = inference.OrderBook(cfg=JAXLOB_Configuration())
    sim_states = inference.get_sims_vmap(book_l2_init, m_seq_raw_inp, init_time_batched, sim)
    sim_state = jax.tree_util.tree_map(lambda x: x[0], sim_states)

    world_cfg = World_EnvironmentConfig(tick_size=100)
    mm_cfg = MarketMaking_EnvironmentConfig(action_space="simple", fixed_quant_value=10)
    mm_agent = MarketMakingAgent(cfg=mm_cfg, world_config=world_cfg)

    agent_state = MMEnvState(
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
    world_time = jnp.array(init_time_batched[0], dtype=jnp.int32)
    world_state = _build_world_state(sim, sim_state, world_time)
    print("Agent initialised")

    action = jnp.int32(args.agent_action)
    action_msgs, _cancel_msgs, _extras = mm_agent.get_messages(action, world_state, agent_state, agent_params)
    action_msgs = action_msgs.at[:, 4].set(jnp.array([-9001, -9002], dtype=jnp.int32))
    print(f"Agent action: {int(action)}")

    bid_before, ask_before = _best_quotes(sim, sim_state)
    sim_state_after_action = sim.process_orders_array(sim_state, action_msgs)
    bid_after_action, ask_after_action = _best_quotes(sim, sim_state_after_action)

    init_hidden = model.initialize_carry(
        1,
        hidden_size=(model_args.ssm_size_base // pow(2, int(model_args.conj_sym))),
        n_message_layers=model_args.n_message_layers,
        n_book_pre_layers=model_args.n_book_pre_layers,
        n_book_post_layers=model_args.n_book_post_layers,
        n_fused_layers=model_args.n_layers,
        h_size_ema=model_args.ssm_size_base,
    )

    rng = jax.random.key(args.seed)
    rng, rng_gen = jax.random.split(rng)
    model_stdout = io.StringIO()
    with contextlib.redirect_stdout(model_stdout):
        msgs_decoded, _l2_states, _num_errors, _msg_tokens = inference.generate(
            sim,
            state,
            model,
            model_args.batchnorm,
            vocab.ENCODING,
            args.sample_top_n,
            100,
            m_seq_inp[0],
            b_seq_inp[0],
            1,
            sim_state_after_action,
            rng_gen,
            init_hidden,
            True,
            jnp.asarray(world_time),
            False,
            None,
        )
    raw_generate_stdout = model_stdout.getvalue().strip()
    if raw_generate_stdout and args.show_model_stdout:
        print(raw_generate_stdout)

    bid_after_gen, ask_after_gen = _best_quotes(sim, sim_state_after_action)
    changed = (bid_before != bid_after_action) or (ask_before != ask_after_action)
    if changed:
        print(
            "Update in the orderbook if any: "
            f"best_bid {bid_before} -> {bid_after_action}, best_ask {ask_before} -> {ask_after_action}"
        )
    else:
        print("Update in the orderbook if any: no best-quote change from agent action")

    first_msg = msgs_decoded[0]
    evt = int(first_msg[1])
    px = int(first_msg[4])
    qty = int(first_msg[5])
    print(
        "Step taken by inference model: "
        f"event_type={evt}, price={px}, size={qty}, "
        f"best_bid={bid_after_gen}, best_ask={ask_after_gen}"
    )

    action_msgs_np = jnp.asarray(action_msgs).tolist()
    generated_msg_np = jnp.asarray(first_msg).tolist()

    _write_csv_rows(
        artifacts.action_csv,
        action_msgs_np,
        ["type", "side", "qty", "price", "order_id", "trader_id", "time_s", "time_ns"],
    )
    _write_csv_rows(
        artifacts.generated_csv,
        [generated_msg_np],
        [
            "col0",
            "event_type",
            "direction",
            "col3",
            "price",
            "size",
            "delta_t_s",
            "delta_t_ns",
            "time_s",
            "time_ns",
            "price_ref",
            "size_ref",
            "time_s_ref",
            "time_ns_ref",
        ],
    )

    summary = {
        "run_name": artifacts.run_dir.name,
        "run_dir": str(artifacts.run_dir),
        "checkpoint_path": str(ckpt_path),
        "checkpoint_step": int(step),
        "data_dir": str(data_dir),
        "sample_index": int(idx),
        "n_cond_msgs": int(args.n_cond_msgs),
        "sample_top_n": int(args.sample_top_n),
        "seed": int(args.seed),
        "agent_action": int(action),
        "best_bid_before": int(bid_before),
        "best_ask_before": int(ask_before),
        "best_bid_after_action": int(bid_after_action),
        "best_ask_after_action": int(ask_after_action),
        "best_bid_after_inference": int(bid_after_gen),
        "best_ask_after_inference": int(ask_after_gen),
        "orderbook_changed_after_action": bool(changed),
        "generated_event_type": int(evt),
        "generated_price": int(px),
        "generated_size": int(qty),
        "artifact_files": {
            "action_messages_csv": str(artifacts.action_csv),
            "generated_message_csv": str(artifacts.generated_csv),
            "summary_json": str(artifacts.summary_json),
            "report_txt": str(artifacts.report_txt),
        },
        "captured_generate_stdout": raw_generate_stdout,
    }
    with open(artifacts.summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    report_lines = [
        "Minimal Agent + Generative Step Verification",
        f"Run directory: {artifacts.run_dir}",
        f"Checkpoint: {ckpt_path} (step {step})",
        f"Data dir: {data_dir}",
        f"Sample index: {idx}",
        f"Agent action: {int(action)}",
        f"Best quote before action: bid={bid_before}, ask={ask_before}",
        f"Best quote after action:  bid={bid_after_action}, ask={ask_after_action}",
        f"Best quote after model:   bid={bid_after_gen}, ask={ask_after_gen}",
        f"Orderbook changed by action: {changed}",
        f"Generated message summary: event_type={evt}, price={px}, size={qty}",
        f"Action CSV: {artifacts.action_csv}",
        f"Generated CSV: {artifacts.generated_csv}",
        f"Summary JSON: {artifacts.summary_json}",
    ]
    artifacts.report_txt.write_text("\n".join(report_lines) + "\n")

    _print_status_block(
        "Verification Complete",
        [
            f"Run dir: {artifacts.run_dir}",
            f"Orderbook changed by action: {changed}",
            f"Generated event: type={evt}, price={px}, size={qty}",
            f"Saved summary: {artifacts.summary_json}",
            f"Saved report:  {artifacts.report_txt}",
        ],
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
