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
import math
import os
from pathlib import Path
import time
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

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
    step_trace_csv: Path
    summary_json: Path
    report_txt: Path
    midprice_plot_png: Path
    action_midprice_plot_png: Path


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
    parser.add_argument("--n_steps", type=int, default=5, help="Number of rollout steps with fixed action")
    parser.add_argument(
        "--action_policy",
        choices=["fixed", "random"],
        default="fixed",
        help="Agent action policy across rollout steps",
    )
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


def _midprice_from_quotes(bid: int, ask: int) -> float:
    return (float(bid) + float(ask)) / 2.0


def _price_to_dollars(price_int: float) -> float:
    # LOBSTER-style prices are integerized (1/10000 dollar units).
    return float(price_int) / 10000.0


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
        step_trace_csv=run_dir / "step_trace.csv",
        summary_json=run_dir / "verification_summary.json",
        report_txt=run_dir / "verification_report.txt",
        midprice_plot_png=run_dir / "midprice_trajectory.png",
        action_midprice_plot_png=run_dir / "action_midprice_trajectory.png",
    )


def _print_status_block(title: str, lines: list[str]) -> None:
    print("=" * 56)
    print(title)
    for line in lines:
        print(line)
    print("=" * 56)


def _plot_midprice(
    historical_midprices: list[float],
    generated_midprices: list[float],
    output_path: Path,
) -> None:
    hist_x = list(range(len(historical_midprices)))
    gen_x = list(range(len(historical_midprices), len(historical_midprices) + len(generated_midprices)))

    plt.figure(figsize=(11, 5))
    hist_usd = [_price_to_dollars(x) for x in historical_midprices]
    gen_usd = [_price_to_dollars(x) for x in generated_midprices]
    plt.plot(hist_x, hist_usd, marker="o", linewidth=2, label="Historical (conditioning)")
    plt.plot(gen_x, gen_usd, marker="s", linewidth=2, label="Generated (post-action steps)")
    if historical_midprices:
        plt.axvline(x=len(historical_midprices) - 0.5, linestyle="--", linewidth=1, color="gray")
    plt.title("Midprice Trajectory: Historical vs Generated")
    plt.xlabel("Step index")
    plt.ylabel("Midprice (USD)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_action_with_midprice(
    historical_midprices: list[float],
    generated_midprices: list[float],
    actions: list[int],
    output_path: Path,
) -> None:
    hist_len = len(historical_midprices)
    hist_x = list(range(hist_len))
    gen_x = list(range(hist_len, hist_len + len(generated_midprices)))

    hist_usd = [_price_to_dollars(x) for x in historical_midprices]
    gen_usd = [_price_to_dollars(x) for x in generated_midprices]

    fig, ax_price = plt.subplots(figsize=(11, 6))
    ax_price.plot(hist_x, hist_usd, marker="o", linewidth=2, label="Historical midprice")
    ax_price.plot(gen_x, gen_usd, marker="s", linewidth=2, label="Generated midprice")
    if historical_midprices:
        ax_price.axvline(x=hist_len - 0.5, linestyle="--", linewidth=1, color="gray")
    ax_price.set_xlabel("Step index")
    ax_price.set_ylabel("Midprice (USD)")
    ax_price.grid(alpha=0.3)

    ax_action = ax_price.twinx()
    ax_action.step(gen_x, actions, where="mid", linewidth=1.8, color="tab:red", label="Action ID")
    ax_action.scatter(gen_x, actions, color="tab:red", s=25)
    ax_action.set_ylabel("Action ID")

    lines1, labels1 = ax_price.get_legend_handles_labels()
    lines2, labels2 = ax_action.get_legend_handles_labels()
    ax_price.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.title("Midprice Trajectory With Fixed-Policy Actions")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def _compute_agent_pnl_from_trades(
    trades: jnp.ndarray,
    trader_id: int,
    tick_size: int,
    final_midprice: float,
) -> dict[str, float]:
    # Keep only valid (non-empty) trade rows.
    valid = trades[trades[:, 0] >= 0]
    if valid.shape[0] == 0:
        return {
            "income": 0.0,
            "outgoing": 0.0,
            "cash_pnl": 0.0,
            "inventory": 0.0,
            "inventory_mark_to_market": 0.0,
            "total_pnl": 0.0,
            "buy_qty": 0.0,
            "sell_qty": 0.0,
            "agent_trade_count": 0.0,
        }

    pass_tid = valid[:, 6]
    aggr_tid = valid[:, 7]
    qty = valid[:, 1]
    price = valid[:, 0].astype(jnp.float32)

    # Same buy/sell ownership logic as MM reward code.
    is_buy = ((qty >= 0) & (pass_tid == trader_id)) | ((qty < 0) & (aggr_tid == trader_id))
    is_sell = ((qty < 0) & (pass_tid == trader_id)) | ((qty >= 0) & (aggr_tid == trader_id))

    abs_qty = jnp.abs(qty).astype(jnp.float32)
    buy_qty = jnp.where(is_buy, abs_qty, 0.0).sum()
    sell_qty = jnp.where(is_sell, abs_qty, 0.0).sum()

    # Cash flow in "tick-size normalized" units, consistent with mm_env reward logic.
    income = jnp.where(is_sell, (price / tick_size) * abs_qty, 0.0).sum()
    outgoing = jnp.where(is_buy, (price / tick_size) * abs_qty, 0.0).sum()
    cash_pnl = income - outgoing
    inventory = buy_qty - sell_qty
    inventory_mtm = inventory * (jnp.float32(final_midprice) / tick_size)
    total_pnl = cash_pnl + inventory_mtm

    involved = (pass_tid == trader_id) | (aggr_tid == trader_id)
    agent_trade_count = involved.astype(jnp.int32).sum()

    return {
        "income": float(income),
        "outgoing": float(outgoing),
        "cash_pnl": float(cash_pnl),
        "inventory": float(inventory),
        "inventory_mark_to_market": float(inventory_mtm),
        "total_pnl": float(total_pnl),
        "buy_qty": float(buy_qty),
        "sell_qty": float(sell_qty),
        "agent_trade_count": float(agent_trade_count),
    }


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
    n_eval_messages = max(n_gen_msgs + 1, args.n_steps + 1)
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

    # Build historical midprice series from the conditioning replay only.
    historical_midprices: list[float] = []
    historical_state = sim.reset(book_l2_init[0], world_time)
    for i in range(args.n_cond_msgs):
        sim_msg_hist = inference.msg_to_jnp(m_seq_raw_inp[0][i])
        historical_state = sim.process_order_array(historical_state, sim_msg_hist)
        h_bid, h_ask = _best_quotes(sim, historical_state)
        if h_bid > 0 and h_ask > 0:
            historical_midprices.append(_midprice_from_quotes(h_bid, h_ask))

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
    fixed_action = jnp.int32(args.agent_action)
    if args.action_policy == "fixed":
        print(f"Agent action policy (fixed): {int(fixed_action)} for {int(args.n_steps)} steps")
    else:
        print(f"Agent action policy (random): sampled each step for {int(args.n_steps)} steps")

    current_sim_state = sim_state
    current_world_time = world_time

    per_step_records: list[dict[str, Any]] = []
    action_message_rows: list[list[Any]] = []
    generated_message_rows: list[list[Any]] = []
    generated_midprices: list[float] = []
    action_trace: list[int] = []
    generate_stdout_fragments: list[str] = []

    for step_i in range(args.n_steps):
        if args.action_policy == "fixed":
            action_this_step = fixed_action
        else:
            rng, rng_action = jax.random.split(rng)
            action_this_step = jax.random.randint(
                rng_action,
                shape=(),
                minval=0,
                maxval=int(mm_cfg.n_actions),
                dtype=jnp.int32,
            )

        step_world_state = _build_world_state(sim, current_sim_state, current_world_time)

        action_msgs, _cancel_msgs, _extras = mm_agent.get_messages(
            action_this_step,
            step_world_state,
            agent_state,
            agent_params,
        )
        # Use deterministic synthetic order ids per step for trace readability.
        action_msgs = action_msgs.at[:, 4].set(
            jnp.array([-9000 - (2 * step_i + 1), -9000 - (2 * step_i + 2)], dtype=jnp.int32)
        )

        bid_before, ask_before = _best_quotes(sim, current_sim_state)
        sim_state_after_action = sim.process_orders_array(current_sim_state, action_msgs)
        bid_after_action, ask_after_action = _best_quotes(sim, sim_state_after_action)
        changed = (bid_before != bid_after_action) or (ask_before != ask_after_action)

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
                jnp.asarray(current_world_time),
                False,
                None,
            )
        raw_generate_stdout = model_stdout.getvalue().strip()
        if raw_generate_stdout:
            generate_stdout_fragments.append(raw_generate_stdout)
            if args.show_model_stdout:
                print(raw_generate_stdout)

        first_msg = msgs_decoded[0]
        gen_sim_msg = inference.msg_to_jnp(first_msg)
        sim_state_after_step = sim.process_order_array(sim_state_after_action, gen_sim_msg)
        bid_after_step, ask_after_step = _best_quotes(sim, sim_state_after_step)
        mid_after_step = _midprice_from_quotes(bid_after_step, ask_after_step)
        spread_after_step = int(ask_after_step - bid_after_step)

        evt = int(first_msg[1])
        px = int(first_msg[4])
        qty = int(first_msg[5])

        if step_i == 0:
            print(f"Agent action: {int(action_this_step)}")
            if changed:
                print(
                    "Update in the orderbook if any: "
                    f"best_bid {bid_before} -> {bid_after_action}, "
                    f"best_ask {ask_before} -> {ask_after_action}"
                )
            else:
                print("Update in the orderbook if any: no best-quote change from agent action")
            print(
                "Step taken by inference model: "
                f"event_type={evt}, price={px}, size={qty}, "
                f"best_bid={bid_after_step}, best_ask={ask_after_step}"
            )

        print(
            f"Rollout step {step_i + 1}/{args.n_steps}: "
            f"action={int(action_this_step)}, midprice={mid_after_step:.1f}, spread={spread_after_step}"
        )

        step_record = {
            "step_index": int(step_i + 1),
            "action_taken": int(action_this_step),
            "best_bid_before_action": int(bid_before),
            "best_ask_before_action": int(ask_before),
            "best_bid_after_action": int(bid_after_action),
            "best_ask_after_action": int(ask_after_action),
            "generated_event_type": int(evt),
            "generated_price": int(px),
            "generated_size": int(qty),
            "best_bid_after_step": int(bid_after_step),
            "best_ask_after_step": int(ask_after_step),
            "midprice_after_step": float(mid_after_step),
            "spread_after_step": int(spread_after_step),
            "orderbook_changed_after_action": bool(changed),
        }
        per_step_records.append(step_record)
        action_trace.append(int(action_this_step))
        generated_midprices.append(float(mid_after_step))

        action_msgs_np = jnp.asarray(action_msgs).tolist()
        for row in action_msgs_np:
            action_message_rows.append([int(step_i + 1)] + row)

        generated_message_rows.append([int(step_i + 1)] + jnp.asarray(first_msg).tolist())

        # Update state and time for next step.
        current_sim_state = sim_state_after_step
        msg_time_s = int(first_msg[8])
        msg_time_ns = int(first_msg[9])
        if msg_time_s >= 0 and msg_time_ns >= 0:
            current_world_time = jnp.array([msg_time_s, msg_time_ns], dtype=jnp.int32)

    _write_csv_rows(
        artifacts.action_csv,
        action_message_rows,
        ["step_index", "type", "side", "qty", "price", "order_id", "trader_id", "time_s", "time_ns"],
    )
    _write_csv_rows(
        artifacts.generated_csv,
        generated_message_rows,
        [
            "step_index",
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
    _write_csv_rows(
        artifacts.step_trace_csv,
        [
            [
                r["step_index"],
                r["action_taken"],
                r["best_bid_before_action"],
                r["best_ask_before_action"],
                r["best_bid_after_action"],
                r["best_ask_after_action"],
                r["generated_event_type"],
                r["generated_price"],
                r["generated_size"],
                r["best_bid_after_step"],
                r["best_ask_after_step"],
                r["midprice_after_step"],
                r["spread_after_step"],
                r["orderbook_changed_after_action"],
            ]
            for r in per_step_records
        ],
        [
            "step_index",
            "action_taken",
            "best_bid_before_action",
            "best_ask_before_action",
            "best_bid_after_action",
            "best_ask_after_action",
            "generated_event_type",
            "generated_price",
            "generated_size",
            "best_bid_after_step",
            "best_ask_after_step",
            "midprice_after_step",
            "spread_after_step",
            "orderbook_changed_after_action",
        ],
    )

    if historical_midprices and generated_midprices:
        _plot_midprice(historical_midprices, generated_midprices, artifacts.midprice_plot_png)
        _plot_action_with_midprice(
            historical_midprices,
            generated_midprices,
            action_trace,
            artifacts.action_midprice_plot_png,
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
        "agent_action": int(fixed_action),
        "policy_type": "fixed_constant_action" if args.action_policy == "fixed" else "random_uniform_action",
        "action_policy": args.action_policy,
        "action_space_n": int(mm_cfg.n_actions),
        "n_steps": int(args.n_steps),
        "action_trace": action_trace,
        "historical_midprice_points": int(len(historical_midprices)),
        "generated_midprice_points": int(len(generated_midprices)),
        "final_midprice": float(generated_midprices[-1]) if generated_midprices else math.nan,
        "final_midprice_usd": _price_to_dollars(generated_midprices[-1]) if generated_midprices else math.nan,
        "final_spread": int(per_step_records[-1]["spread_after_step"]) if per_step_records else -1,
        "price_unit": "integer_price_units_1e-4_USD",
        "step_records": per_step_records,
        "artifact_files": {
            "action_messages_csv": str(artifacts.action_csv),
            "generated_message_csv": str(artifacts.generated_csv),
            "step_trace_csv": str(artifacts.step_trace_csv),
            "summary_json": str(artifacts.summary_json),
            "report_txt": str(artifacts.report_txt),
            "midprice_plot_png": str(artifacts.midprice_plot_png),
            "action_midprice_plot_png": str(artifacts.action_midprice_plot_png),
        },
        "captured_generate_stdout": "\n".join(generate_stdout_fragments),
    }

    # End-of-run PnL snapshot from executed trades.
    pnl = _compute_agent_pnl_from_trades(
        current_sim_state.trades,
        trader_id=int(agent_params.trader_id),
        tick_size=int(world_cfg.tick_size),
        final_midprice=summary["final_midprice"],
    )
    summary["agent_pnl"] = pnl

    with open(artifacts.summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    report_lines = [
        "Minimal Agent + Generative Step Verification",
        f"Run directory: {artifacts.run_dir}",
        f"Checkpoint: {ckpt_path} (step {step})",
        f"Data dir: {data_dir}",
        f"Sample index: {idx}",
        f"Policy: {'fixed constant action ' + str(int(fixed_action)) if args.action_policy == 'fixed' else 'random uniform over action ids'}",
        f"Rollout steps: {int(args.n_steps)}",
        f"Action trace: {action_trace}",
        f"Final midprice (raw units): {summary['final_midprice']}",
        f"Final midprice (USD): {summary['final_midprice_usd']:.4f}",
        f"Final spread: {summary['final_spread']}",
        f"Agent total PnL (tick-normalized): {summary['agent_pnl']['total_pnl']:.4f}",
        f"Agent cash PnL: {summary['agent_pnl']['cash_pnl']:.4f}",
        f"Agent inventory MTM: {summary['agent_pnl']['inventory_mark_to_market']:.4f}",
        f"Agent trade count: {int(summary['agent_pnl']['agent_trade_count'])}",
        "Price units: integer units where 10000 = $1.00",
        f"Historical points: {len(historical_midprices)}",
        f"Generated points: {len(generated_midprices)}",
        f"Action CSV: {artifacts.action_csv}",
        f"Generated CSV: {artifacts.generated_csv}",
        f"Step trace CSV: {artifacts.step_trace_csv}",
        f"Midprice plot: {artifacts.midprice_plot_png}",
        f"Action+midprice plot: {artifacts.action_midprice_plot_png}",
        f"Summary JSON: {artifacts.summary_json}",
    ]
    artifacts.report_txt.write_text("\n".join(report_lines) + "\n")

    _print_status_block(
        "Verification Complete",
        [
            f"Run dir: {artifacts.run_dir}",
            f"Action trace: {action_trace}",
            f"Final midprice: {summary['final_midprice']} ({summary['final_midprice_usd']:.4f} USD)",
            f"Agent total PnL: {summary['agent_pnl']['total_pnl']:.4f}",
            f"Saved summary: {artifacts.summary_json}",
            f"Saved report:  {artifacts.report_txt}",
            f"Saved plot:    {artifacts.midprice_plot_png}",
            f"Saved plot 2:  {artifacts.action_midprice_plot_png}",
        ],
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
