#!/usr/bin/env python3
"""Run minimal one-message inference from a LOBS5 checkpoint and save CSV output.

This script lives in JaxMARL-HFT but reuses LOBS5 inference modules from:
    /homes/80/satyam/LOBS5
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
import time
from glob import glob
from pathlib import Path

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_LOBS5_ROOT = Path("/homes/80/satyam/LOBS5")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "one_step_runs"


def _add_python_paths(lobs5_root: Path) -> None:
    for p in [lobs5_root, lobs5_root / "AlphaTrade"]:
        p_str = str(p)
        if p_str not in sys.path:
            sys.path.insert(0, p_str)


def _latest_checkpoint_step(ckpt_path: Path) -> int:
    numeric_dirs = []
    for p in ckpt_path.iterdir():
        if p.is_dir() and p.name.isdigit():
            numeric_dirs.append(int(p.name))
    if not numeric_dirs:
        raise FileNotFoundError(f"No numeric checkpoint step directories found in {ckpt_path}")
    return max(numeric_dirs)


def _restore_params_only(ckpt_path: Path, step: int):
    state_dir = ckpt_path / str(step) / "state"
    if not state_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint state directory not found: {state_dir}")
    restored = ocp.PyTreeCheckpointer().restore(str(state_dir))
    if not isinstance(restored, dict) or "params" not in restored:
        raise RuntimeError(f"Unexpected checkpoint state format in {state_dir}")
    return restored["params"]


def _enable_legacy_token_mode_22() -> None:
    """Patch lob.encoding in-process to legacy 22-token behavior.

    This mirrors the intent of pipeline legacy compatibility for older checkpoints
    trained with vocab size 12012 (base-10000 size encoding).
    """
    enc = importlib.import_module("lob.encoding")

    def _legacy_vocab_init(self) -> None:
        self.counter = 4
        self.ENCODING = {}
        self.DECODING = {}
        self.DECODING_GLOBAL = {}
        self.TOKEN_DELIM_IDX = {}

        self._add_field("time", range(1000), [3, 6, 9, 12])
        self._add_field("event_type", range(1, 5), None)
        self._add_field("size", range(10000), [])
        self._add_field("price", range(1000), [1])
        self._add_field("sign", [-1, 1], None)
        self._add_field("direction", [0, 1], None)

    enc.Vocab.__init__ = _legacy_vocab_init

    np = __import__("numpy")
    enc.Message_Tokenizer.TOK_LENS = np.array((1, 1, 2, 1, 1, 3, 2, 3, 2, 1, 2, 3))
    enc.Message_Tokenizer.TOK_DELIM = np.cumsum(enc.Message_Tokenizer.TOK_LENS[:-1])
    enc.Message_Tokenizer.MSG_LEN = int(np.sum(enc.Message_Tokenizer.TOK_LENS))
    enc.Message_Tokenizer.NEW_MSG_LEN = int(
        enc.Message_Tokenizer.MSG_LEN
        - sum(
            int(enc.Message_Tokenizer.TOK_LENS[i])
            for i, f in enumerate(enc.Message_Tokenizer.FIELDS)
            if f.endswith("_ref")
        )
    )
    enc.Message_Tokenizer.TIME_START_I = 9
    enc.Message_Tokenizer.TIME_END_I = 13
    enc.Message_Tokenizer.FIELD_ENC_TYPES = {
        "event_type": "event_type",
        "direction": "direction",
        "price": "price",
        "size": "size",
        "delta_t_s": "time",
        "delta_t_ns": "time",
        "time_s": "time",
        "time_ns": "time",
        "price_ref": "price",
        "size_ref": "size",
        "time_s_ref": "time",
        "time_ns_ref": "time",
    }

    @jax.jit
    def _encode_msg_22(msg, encoding):
        event_type = enc.encode(msg[1], *encoding["event_type"])
        direction = enc.encode(msg[2], *encoding["direction"])
        price = enc.split_field(msg[4], 1, 3, True)
        price_sign = enc.encode(price[0], *encoding["sign"])
        price = enc.encode(price[1], *encoding["price"])
        size_enc = enc.encode(msg[5], *encoding["size"])
        time_comb = enc.encode_time(
            time_s=msg[8],
            time_ns=msg[9],
            encoding=encoding,
            delta_t_s=msg[6],
            delta_t_ns=msg[7],
        )
        price_ref = enc.split_field(msg[10], 1, 3, True)
        price_ref_sign = enc.encode(price_ref[0], *encoding["sign"])
        price_ref = enc.encode(price_ref[1], *encoding["price"])
        size_ref_enc = enc.encode(msg[11], *encoding["size"])
        time_ref_comb = enc.encode_time(time_s=msg[12], time_ns=msg[13], encoding=encoding)
        return jnp.hstack(
            [
                event_type,
                direction,
                price_sign,
                price,
                size_enc,
                time_comb,
                price_ref_sign,
                price_ref,
                size_ref_enc,
                time_ref_comb,
            ]
        )

    @jax.jit
    def _decode_msg_22(msg_enc, encoding):
        event_type = enc.decode(msg_enc[0], *encoding["event_type"])
        direction = enc.decode(msg_enc[1], *encoding["direction"])
        price_sign = enc.decode(msg_enc[2], *encoding["sign"])
        price = enc.decode(msg_enc[3], *encoding["price"])
        price = enc.combine_field(price, 3, price_sign)
        size = enc.decode(msg_enc[4], *encoding["size"])
        delta_t_s, delta_t_ns, time_s, time_ns = enc.decode_time(msg_enc[5:14], encoding)
        price_ref_sign = enc.decode(msg_enc[14], *encoding["sign"])
        price_ref = enc.decode(msg_enc[15], *encoding["price"])
        price_ref = enc.combine_field(price_ref, 3, price_ref_sign)
        size_ref = enc.decode(msg_enc[16], *encoding["size"])
        time_s_ref, time_ns_ref = enc.decode_time(msg_enc[17:22], encoding)
        return jnp.hstack(
            [
                enc.NA_VAL,
                event_type,
                direction,
                enc.NA_VAL,
                price,
                size,
                delta_t_s,
                delta_t_ns,
                time_s,
                time_ns,
                price_ref,
                size_ref,
                time_s_ref,
                time_ns_ref,
            ]
        )

    enc.encode_msg = _encode_msg_22
    enc.decode_msg = _decode_msg_22
    enc.encode_msgs = jax.jit(jax.vmap(_encode_msg_22, in_axes=(0, None)))
    enc.decode_msgs = jax.jit(jax.vmap(_decode_msg_22, in_axes=(0, None)), backend="cpu")


def _load_metadata_robust(ckpt_path: Path):
    from argparse import Namespace

    root_meta = ckpt_path / "metadata" / "_ROOT_METADATA"
    raw = root_meta.read_text()

    try:
        metadata = json.loads(raw)
    except json.JSONDecodeError:
        depth = 0
        end = 0
        for i, c in enumerate(raw):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            if depth == 0 and i > 0:
                end = i + 1
                break
        if end <= 0:
            raise
        metadata = json.loads(raw[:end])

    data = metadata.get("custom", metadata.get("custom_metadata", metadata))
    if isinstance(data, dict) and "config" in data and isinstance(data["config"], dict):
        data = data["config"]
    return Namespace(**data)


def _ensure_model_args_defaults(ns):
    # Legacy checkpoints may miss newer config keys expected by init_train_state.
    if not hasattr(ns, "bsz"):
        ns.bsz = 1
    if not hasattr(ns, "num_devices"):
        ns.num_devices = 1
    if not hasattr(ns, "micro_bsz"):
        ns.micro_bsz = max(1, int(ns.bsz))
    if not hasattr(ns, "global_bsz"):
        ns.global_bsz = max(1, int(ns.bsz) * int(ns.num_devices))
    if not hasattr(ns, "merging"):
        ns.merging = "padded"
    if not hasattr(ns, "batchnorm"):
        ns.batchnorm = False
    if not hasattr(ns, "token_mode"):
        ns.token_mode = 22
    return ns


def _default_run_name() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-message LOBS5 inference and CSV export")
    parser.add_argument("--ckpt_path", required=True, help="Checkpoint dir path")
    parser.add_argument("--data_dir", required=True, help="Input data dir with *message*.npy and *book*.npy")
    parser.add_argument("--output_file", default="", help="Output CSV path (default: <output_root>/<run_name>/verification.csv)")

    parser.add_argument("--stock", default="GOOG", help="Stock symbol label for output")
    parser.add_argument("--checkpoint_step", type=int, default=None, help="Checkpoint step (default: latest)")
    parser.add_argument("--lobs5_root", default=str(DEFAULT_LOBS5_ROOT), help="Path to LOBS5 repository")
    parser.add_argument("--run_name", default="", help="Run folder name (default: timestamp YYYYmmdd_HHMMSS)")
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT), help="Centralized root for one-step outputs")

    parser.add_argument("--n_cond_msgs", type=int, default=500, help="Number of conditioning messages")
    parser.add_argument("--n_gen_msgs", type=int, default=1, help="Number of messages (steps) to generate")
    parser.add_argument("--sample_top_n", type=int, default=-1, help="Sampling mode (-1 full distribution, 1 greedy)")
    parser.add_argument("--sample_index", type=int, default=0, help="Dataset index to use for deterministic single sample")
    parser.add_argument(
        "--sample_indices",
        default="",
        help="Comma-separated dataset indices to run in one process (compile amortization)",
    )
    parser.add_argument("--test_split", type=float, default=1.0, help="Fraction of files to include from tail")
    parser.add_argument("--seed", type=int, default=42, help="JAX random seed")
    parser.add_argument("--n_samples", type=int, default=1, help="Number of samples to run in this invocation")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for parallel generation in one invocation")
    parser.add_argument(
        "--compile_cache_dir",
        default=str(REPO_ROOT / ".cache" / "jax_compilation"),
        help="JAX compilation cache directory (persisted across runs)",
    )
    parser.add_argument(
        "--fast_startup",
        action="store_true",
        help="Prefer lower startup overhead (disables big upfront GPU preallocation)",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.fast_startup:
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.50")
    else:
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(Path(args.compile_cache_dir).expanduser().resolve()))
    os.environ.setdefault("PYTHONNOUSERSITE", "1")

    t0 = time.time()
    timing = {}

    ckpt_path = Path(args.ckpt_path).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve()
    lobs5_root = Path(args.lobs5_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    run_name = args.run_name.strip() if args.run_name.strip() else _default_run_name()
    run_root = output_root / run_name
    run_dir = run_root / "artifacts"
    output_file = (
        Path(args.output_file).expanduser().resolve()
        if args.output_file.strip()
        else run_root / "verification.csv"
    )

    if not ckpt_path.exists() or not ckpt_path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_path}")
    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not lobs5_root.exists() or not lobs5_root.is_dir():
        raise FileNotFoundError(f"LOBS5 root not found: {lobs5_root}")

    msg_files = glob(str(data_dir / "*message*.npy"))
    book_files = glob(str(data_dir / "*book*.npy")) + glob(str(data_dir / "*orderbook*.npy"))
    if not msg_files:
        raise FileNotFoundError(f"No *message*.npy files found in {data_dir}")
    if not book_files:
        raise FileNotFoundError(f"No *book*.npy or *orderbook*.npy files found in {data_dir}")

    t1 = time.time()
    step = args.checkpoint_step if args.checkpoint_step is not None else _latest_checkpoint_step(ckpt_path)
    params = _restore_params_only(ckpt_path, step)
    ckpt_vocab_size = int(params["message_encoder"]["encoder"]["embedding"].shape[0])
    timing["checkpoint_restore_seconds"] = time.time() - t1

    t2 = time.time()
    _add_python_paths(lobs5_root)

    if ckpt_vocab_size >= 10000:
        _enable_legacy_token_mode_22()

    from lob.encoding import Message_Tokenizer, Vocab
    from lob.init_train import init_train_state
    from lob import inference_no_errcorr as inference
    timing["imports_and_compat_seconds"] = time.time() - t2

    t3 = time.time()
    model_args = _load_metadata_robust(ckpt_path)
    model_args = _ensure_model_args_defaults(model_args)
    model_args.num_devices = 1
    model_args.bsz = 1
    model_args.micro_bsz = 1
    model_args.global_bsz = 1
    if ckpt_vocab_size >= 10000:
        model_args.token_mode = 22

    n_gen_msgs = max(1, int(args.n_gen_msgs))
    n_eval_messages = n_gen_msgs + 1
    cond_seq_len = args.n_cond_msgs * Message_Tokenizer.MSG_LEN
    eval_seq_len = (n_eval_messages - 1) * Message_Tokenizer.MSG_LEN

    vocab = Vocab()
    n_classes = ckpt_vocab_size
    book_dim = 503

    init_state, model_cls = init_train_state(
        model_args,
        n_classes=n_classes,
        seq_len=eval_seq_len,
        book_dim=book_dim,
        book_seq_len=eval_seq_len,
    )

    state = init_state.replace(params=params, step=step)
    model = model_cls(training=False, step_rescale=1.0)
    timing["model_init_seconds"] = time.time() - t3

    t4 = time.time()
    ds = inference.get_dataset(
        str(data_dir),
        args.n_cond_msgs,
        n_eval_messages,
        test_split=args.test_split,
    )
    if len(ds) == 0:
        raise RuntimeError("Dataset is empty after applying test_split")

    requested_indices = []
    if args.sample_indices.strip():
        for tok in args.sample_indices.split(","):
            tok = tok.strip()
            if not tok:
                continue
            requested_indices.append(int(tok))

    if requested_indices:
        selected_indices = [max(0, min(i, len(ds) - 1)) for i in requested_indices]
    else:
        selected_indices = [max(0, min(args.sample_index, len(ds) - 1))]

    # If user did not override n_samples/batch_size, auto-fit to selected indices.
    if args.sample_indices.strip() and args.n_samples == 1 and args.batch_size == 1:
        args.n_samples = len(selected_indices)
        args.batch_size = len(selected_indices)

    if args.n_samples > len(selected_indices):
        raise ValueError(
            f"n_samples ({args.n_samples}) cannot exceed selected dataset length ({len(selected_indices)})"
        )

    if args.n_samples % args.batch_size != 0:
        raise ValueError("n_samples must be divisible by batch_size")

    class _SingleSampleDataset:
        def __init__(self, base_ds, idx_list):
            self.base_ds = base_ds
            self.idx_list = list(idx_list)

        def __len__(self):
            return len(self.idx_list)

        def _map_index(self, i: int) -> int:
            return self.idx_list[max(0, min(i, len(self.idx_list) - 1))]

        def __getitem__(self, query_idx):
            if isinstance(query_idx, list):
                mapped = [self._map_index(i) for i in query_idx]
                return self.base_ds[mapped]
            if isinstance(query_idx, int):
                return self.base_ds[[self._map_index(query_idx)]]
            return self.base_ds[[self._map_index(0)]]

        def get_date(self, _):
            return self.base_ds.get_date(self._map_index(0))

    ds_one = _SingleSampleDataset(ds, selected_indices)
    timing["dataset_prepare_seconds"] = time.time() - t4

    run_dir.mkdir(parents=True, exist_ok=True)

    rng = jax.random.key(args.seed)

    start = time.time()
    inference.sample_new(
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        ds=ds_one,
        rng=rng,
        seq_len_cond=cond_seq_len,
        n_cond_msgs=args.n_cond_msgs,
        n_gen_msgs=n_gen_msgs,
        train_state=state,
        model=model,
        batchnorm=model_args.batchnorm,
        encoder=vocab.ENCODING,
        stock_symbol=args.stock,
        save_folder=str(run_dir),
        sample_top_n=args.sample_top_n,
        args=model_args,
        conditional=True,
        overfit_debug=False,
    )
    runtime_s = time.time() - start
    timing["sample_new_seconds"] = runtime_s

    t5 = time.time()
    gen_files = sorted(glob(str(run_dir / "data_gen" / "*message*.csv")))
    if not gen_files:
        raise RuntimeError(f"No generated message CSV files found in {run_dir / 'data_gen'}")

    def _sample_index_from_filename(p: str) -> int:
        m = re.search(r"real_id_(\d+)", Path(p).name)
        return int(m.group(1)) if m else -1

    frames = []
    for gf in gen_files:
        df = pd.read_csv(
            gf,
            header=None,
            names=["time", "event_type", "order_id", "size", "price", "direction"],
        )
        if not df.empty:
            df.insert(0, "inferred_sample_index", _sample_index_from_filename(gf))
            frames.append(df)

    gen_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if gen_df.empty:
        out_df = pd.DataFrame(
            [
                {
                    "checkpoint_path": str(ckpt_path),
                    "checkpoint_step": int(step),
                    "stock": args.stock,
                    "sample_index": int(selected_indices[0]),
                    "seed": int(args.seed),
                    "runtime_seconds": float(runtime_s),
                    "source_generated_file": str(gen_files[0]),
                    "generation_status": "empty_generation_csv",
                    "generated_rows": 0,
                }
            ]
        )
    else:
        out_df = gen_df.copy()
        out_df.insert(0, "checkpoint_path", str(ckpt_path))
        out_df.insert(1, "checkpoint_step", int(step))
        out_df.insert(2, "stock", args.stock)
        out_df.insert(3, "sample_index", int(selected_indices[0]))
        out_df.insert(4, "seed", int(args.seed))
        out_df.insert(5, "runtime_seconds", float(runtime_s))
        out_df.insert(6, "source_generated_file", str(gen_files[0]))
        out_df.insert(7, "generation_status", "ok")
        out_df.insert(8, "generated_rows", len(gen_df))
    timing["csv_postprocess_seconds"] = time.time() - t5

    output_file.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_file, index=False)

    total_runtime = time.time() - t0
    summary = {
        "run_name": run_name,
        "run_root": str(run_root),
        "artifact_dir": str(run_dir),
        "verification_csv": str(output_file),
        "checkpoint_path": str(ckpt_path),
        "checkpoint_step": int(step),
        "data_dir": str(data_dir),
        "stock": args.stock,
        "sample_index": int(selected_indices[0]),
        "sample_indices": [int(i) for i in selected_indices],
        "seed": int(args.seed),
        "runtime_seconds": float(runtime_s),
        "total_runtime_seconds": float(total_runtime),
        "timing_breakdown": timing,
        "n_samples": int(args.n_samples),
        "batch_size": int(args.batch_size),
        "n_cond_msgs": int(args.n_cond_msgs),
        "n_gen_msgs": int(n_gen_msgs),
        "sample_top_n": int(args.sample_top_n),
        "jax_backend": jax.default_backend(),
        "jax_visible_devices": int(jax.local_device_count()),
        "compile_cache_dir": os.environ.get("JAX_COMPILATION_CACHE_DIR", ""),
        "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", ""),
        "xla_mem_fraction": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", ""),
        "generation_status": str(out_df.iloc[0]["generation_status"]),
        "generated_rows": int(out_df.iloc[0]["generated_rows"]),
    }
    with open(run_root / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("==============================================")
    print("One-step inference complete")
    print(f"Checkpoint: {ckpt_path} (step {step})")
    print(f"Data dir:    {data_dir}")
    print(f"Run name:    {run_name}")
    print(f"Run root:    {run_root}")
    print(f"Run dir:     {run_dir}")
    print(f"Output CSV:  {output_file}")
    print(f"Rows:        {len(out_df)}")
    print(f"Sample time: {runtime_s:.3f}s")
    print(f"Total time:  {total_runtime:.3f}s")
    print("==============================================")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
