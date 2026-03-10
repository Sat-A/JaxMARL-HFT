#!/usr/bin/env python3
"""Run multiple one-step inference jobs in parallel, one process per GPU.

This improves throughput (many one-step jobs at once). It does not reduce
latency for a single sample because each process still compiles once.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parallel one-step inference across GPUs")
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--stock", default="GOOG")
    p.add_argument("--checkpoint_step", type=int, default=None)
    p.add_argument("--lobs5_root", default="/homes/80/satyam/LOBS5")
    p.add_argument("--output_root", default=str(REPO_ROOT / "outputs" / "one_step_runs"))
    p.add_argument("--run_name_prefix", default="multigpu")
    p.add_argument("--sample_indices", default="0,1,2,3", help="Comma-separated dataset indices")
    p.add_argument("--gpu_ids", default="0,1,2,3", help="Comma-separated GPU IDs")
    p.add_argument("--n_cond_msgs", type=int, default=500)
    p.add_argument("--sample_top_n", type=int, default=1)
    p.add_argument("--test_split", type=float, default=1.0)
    p.add_argument("--seed_base", type=int, default=42)
    p.add_argument("--fast_startup", action="store_true")
    p.add_argument("--compile_cache_dir", default=str(REPO_ROOT / ".cache" / "jax_compilation"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    sample_indices = [int(x.strip()) for x in args.sample_indices.split(",") if x.strip()]
    gpu_ids = [x.strip() for x in args.gpu_ids.split(",") if x.strip()]

    if not sample_indices:
        raise ValueError("No sample indices provided")
    if not gpu_ids:
        raise ValueError("No GPU IDs provided")

    procs = []
    start = time.time()

    for rank, sample_idx in enumerate(sample_indices):
        gpu_id = gpu_ids[rank % len(gpu_ids)]
        run_name = f"{args.run_name_prefix}_{time.strftime('%Y%m%d_%H%M%S')}_gpu{gpu_id}_idx{sample_idx}"

        cmd = [
            sys.executable,
            str(REPO_ROOT / "run_one_step_inference.py"),
            "--ckpt_path",
            args.ckpt_path,
            "--data_dir",
            args.data_dir,
            "--stock",
            args.stock,
            "--lobs5_root",
            args.lobs5_root,
            "--output_root",
            args.output_root,
            "--run_name",
            run_name,
            "--sample_index",
            str(sample_idx),
            "--n_cond_msgs",
            str(args.n_cond_msgs),
            "--sample_top_n",
            str(args.sample_top_n),
            "--test_split",
            str(args.test_split),
            "--seed",
            str(args.seed_base + rank),
            "--n_samples",
            "1",
            "--batch_size",
            "1",
            "--compile_cache_dir",
            args.compile_cache_dir,
        ]
        if args.checkpoint_step is not None:
            cmd.extend(["--checkpoint_step", str(args.checkpoint_step)])
        if args.fast_startup:
            cmd.append("--fast_startup")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        log_path = Path(args.output_root) / f"{run_name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_f = open(log_path, "w")
        proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, stdout=log_f, stderr=subprocess.STDOUT)
        procs.append((proc, log_f, gpu_id, sample_idx, log_path))

    failed = 0
    for proc, log_f, gpu_id, sample_idx, log_path in procs:
        code = proc.wait()
        log_f.close()
        if code != 0:
            failed += 1
            print(f"FAILED gpu={gpu_id} sample_index={sample_idx} log={log_path}")
        else:
            print(f"OK gpu={gpu_id} sample_index={sample_idx} log={log_path}")

    elapsed = time.time() - start
    print(f"Completed {len(procs)} jobs in {elapsed:.2f}s; failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
