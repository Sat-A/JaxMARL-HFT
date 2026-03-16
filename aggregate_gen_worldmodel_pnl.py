#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import statistics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate PnL and throughput from generative training summaries")
    p.add_argument("--glob", dest="glob_pattern", required=True, help="Glob for summary.json files")
    p.add_argument("--output", required=True, help="Output JSON path")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    files = sorted(glob.glob(args.glob_pattern))
    if not files:
        raise RuntimeError(f"No summary files found for pattern: {args.glob_pattern}")

    rows = []
    pnls = []
    tputs = []
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        pnl = float(d.get("pnl", {}).get("final_avg_pnl", d.get("pnl", {}).get("mean_avg_pnl", 0.0)))
        tput = float(d.get("throughput", {}).get("updates_mean_steps_per_sec", 0.0))
        rows.append({
            "run_name": d.get("run_name", Path(f).parent.name),
            "path": f,
            "seed": d.get("seed"),
            "n_envs": d.get("n_envs"),
            "final_avg_pnl": pnl,
            "mean_steps_per_sec": tput,
        })
        pnls.append(pnl)
        tputs.append(tput)

    out = {
        "n_runs": len(rows),
        "runs": rows,
        "pnl": {
            "mean": statistics.fmean(pnls),
            "median": statistics.median(pnls),
            "std": statistics.pstdev(pnls) if len(pnls) > 1 else 0.0,
            "min": min(pnls),
            "max": max(pnls),
        },
        "throughput": {
            "mean_steps_per_sec": statistics.fmean(tputs),
            "median_steps_per_sec": statistics.median(tputs),
            "max_steps_per_sec": max(tputs),
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote aggregate: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
