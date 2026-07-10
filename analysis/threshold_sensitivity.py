"""Threshold-sensitivity sweep: CEG Shapley multipliers as a function of the
reference BPB threshold, not just at the single A0D0-endpoint anchor.

The headline numbers use one threshold (A0D0's final neutral BPB, per size);
CEG multipliers are generally threshold-dependent, so this sweep is the
robustness companion figure. The sweep range is the BPB interval every arm's
curve actually spans (from just below the worst arm's best value down to the
shallowest arm's floor), sampled log-uniformly in compute via the same
interpolation as the headline number.

Usage:
  python analysis/threshold_sensitivity.py --size small \
    --a0d0 runs/small_a0d0 --a0d1 runs/small_a0d1 \
    --a1d0 runs/small_a1d0 --a1d1 runs/small_a1d1 \
    --out-csv analysis/out/small_sensitivity.csv [--n-points 40]
"""

import argparse
import csv
import math
from pathlib import Path

from ceg_shapley import hours_to_threshold, load_curve

ARMS = ["a0d0", "a0d1", "a1d0", "a1d1"]


def sweep(curves, n_points):
    # thresholds every arm can reach AND every arm has already passed its first
    # checkpoint for (so every crossing is a true interpolation, not an upper bound)
    lo = max(min(b for _, b in c) for c in curves.values())          # deepest shared BPB
    hi = min(c[0][1] for c in curves.values())                        # first-checkpoint BPB
    if not lo < hi:
        raise SystemExit(f"no shared threshold range: lo={lo:.4f} >= hi={hi:.4f} "
                         f"(densify early checkpoints)")
    eps = (hi - lo) * 1e-3
    rows = []
    for i in range(n_points):
        thr = hi - eps - (hi - lo - 2 * eps) * i / (n_points - 1)
        C = {}
        ok = True
        for arm, curve in curves.items():
            C[arm], flag = hours_to_threshold(curve, thr)
            if C[arm] is None or flag != "interpolated":
                ok = False
                break
        if not ok:
            continue
        lc = {k: math.log(v) for k, v in C.items()}
        data = 0.5 * ((lc["a0d0"] - lc["a0d1"]) + (lc["a1d0"] - lc["a1d1"]))
        algo = 0.5 * ((lc["a0d0"] - lc["a1d0"]) + (lc["a0d1"] - lc["a1d1"]))
        rows.append({"threshold_bpb": thr,
                     **{f"hours_{k}": v for k, v in C.items()},
                     "data_multiplier": math.exp(data),
                     "algorithm_multiplier": math.exp(algo),
                     "total_multiplier": math.exp(data + algo)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True)
    for arm in ARMS:
        ap.add_argument(f"--{arm}", required=True)
    ap.add_argument("--n-points", type=int, default=40)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    curves = {arm: load_curve(getattr(args, arm)) for arm in ARMS}
    rows = sweep(curves, args.n_points)
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} thresholds swept "
          f"[{rows[-1]['threshold_bpb']:.4f}, {rows[0]['threshold_bpb']:.4f}] -> {out}")
    print(f"data multiplier range: {min(r['data_multiplier'] for r in rows):.2f}x "
          f"- {max(r['data_multiplier'] for r in rows):.2f}x; "
          f"algorithm: {min(r['algorithm_multiplier'] for r in rows):.2f}x "
          f"- {max(r['algorithm_multiplier'] for r in rows):.2f}x")


if __name__ == "__main__":
    main()
