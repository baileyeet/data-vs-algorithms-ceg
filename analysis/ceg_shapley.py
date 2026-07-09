"""Compute-equivalent gain + Shapley decomposition for one model size.

Implements requirements #6 and #7:
  - For each arm, find the GPU-hours at which neutral-corpus BPB first crosses
    the reference threshold, interpolating BPB vs log(compute) between the two
    straddling checkpoints (never snapping to a checkpoint).
  - Shapley decomposition in LOG-compute space:
      data = 0.5*[(logC(A0D0)-logC(A0D1)) + (logC(A1D0)-logC(A1D1))]
      algo = 0.5*[(logC(A0D0)-logC(A1D0)) + (logC(A0D1)-logC(A1D1))]
    These sum to logC(A0D0)-logC(A1D1) by construction. Exponentiate for the
    "Nx reduction" framing and report N*M as the sanity check.

Usage:
  python analysis/ceg_shapley.py --size small \
    --a0d0 runs/small_a0d0 --a0d1 runs/small_a0d1 \
    --a1d0 runs/small_a1d0 --a1d1 runs/small_a1d1 \
    [--threshold 1.05]        # default: final neutral_bpb of A0D0

Each run dir needs the metrics.csv written by the trainers.
"""

import argparse
import csv
import json
import math
from pathlib import Path


def load_curve(run_dir):
    rows = list(csv.DictReader(open(Path(run_dir) / "metrics.csv")))
    curve = [(float(r["gpu_hours"]), float(r["neutral_bpb"])) for r in rows
             if float(r["gpu_hours"]) > 0]
    if not curve:
        raise SystemExit(f"{run_dir}: no checkpoints with nonzero timed compute")
    return curve


def hours_to_threshold(curve, thr):
    """First crossing of thr, interpolated linearly in (log hours, bpb)."""
    prev = None
    for h, b in curve:
        if b <= thr:
            if prev is None:
                return h, "crossed_at_first_checkpoint"
            h0, b0 = prev
            frac = (b0 - thr) / (b0 - b)
            return math.exp(math.log(h0) + frac * (math.log(h) - math.log(h0))), "interpolated"
        prev = (h, b)
    return None, "never_crossed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True)
    for arm in ["a0d0", "a0d1", "a1d0", "a1d1"]:
        ap.add_argument(f"--{arm}", required=True)
    ap.add_argument("--threshold", type=float, default=None,
                    help="reference neutral BPB; default = A0D0's final value")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    curves = {arm: load_curve(getattr(args, arm)) for arm in ["a0d0", "a0d1", "a1d0", "a1d1"]}
    thr = args.threshold if args.threshold is not None else curves["a0d0"][-1][1]

    C, flags = {}, {}
    for arm, curve in curves.items():
        C[arm], flags[arm] = hours_to_threshold(curve, thr)
        if C[arm] is None:
            raise SystemExit(f"{arm} never reached threshold {thr:.4f} "
                             f"(best {min(b for _, b in curve):.4f}) — no CEG defined")
        if flags[arm] == "crossed_at_first_checkpoint":
            print(f"WARNING: {arm} already below threshold at its first checkpoint — "
                  f"compute-to-threshold is an upper bound; densify early checkpoints")

    logC = {k: math.log(v) for k, v in C.items()}
    data_log = 0.5 * ((logC["a0d0"] - logC["a0d1"]) + (logC["a1d0"] - logC["a1d1"]))
    algo_log = 0.5 * ((logC["a0d0"] - logC["a1d0"]) + (logC["a0d1"] - logC["a1d1"]))
    total_log = logC["a0d0"] - logC["a1d1"]

    result = {
        "size": args.size,
        "threshold_neutral_bpb": thr,
        "gpu_hours_to_threshold": C,
        "crossing_type": flags,
        "shapley_log": {"data": data_log, "algorithm": algo_log, "total": total_log},
        "multipliers": {"data": math.exp(data_log), "algorithm": math.exp(algo_log),
                        "total": math.exp(total_log)},
        "sanity_product_equals_total": math.isclose(
            math.exp(data_log) * math.exp(algo_log), math.exp(total_log), rel_tol=1e-9),
    }
    print(json.dumps(result, indent=2))
    print(f"\ndata contributes a {math.exp(data_log):.2f}x compute reduction, "
          f"algorithm a {math.exp(algo_log):.2f}x; product {math.exp(data_log)*math.exp(algo_log):.2f}x "
          f"vs observed total {math.exp(total_log):.2f}x")
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
