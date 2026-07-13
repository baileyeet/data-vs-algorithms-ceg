"""Matched-compute comparison: interpolated neutral BPB for every arm at
shared fixed compute points — the budget-honest replacement for the endpoint
comparison lost to the A1 native-budget redesign.

Only defined on the compute window where ALL arms' curves overlap:
[max(first-checkpoint compute), min(final compute)]. Interpolation is the
same convention as the threshold crossing (linear in BPB vs log-compute).

Usage:
  python analysis/matched_compute.py --a0d0 runs/small_a0d0 ... \
    [--points 0.02,0.05,0.1,0.17] --out-csv results/small/matched_compute.csv
Default points: 4 log-spaced values inside the shared window.
"""

import argparse
import csv
import math
from pathlib import Path

from ceg_shapley import load_curve

ARMS = ["a0d0", "a0d1", "a1d0", "a1d1"]


def bpb_at(curve, hours):
    """interpolate BPB at a given compute (linear in bpb vs log-hours)"""
    if hours < curve[0][0] or hours > curve[-1][0]:
        return None
    for (h0, b0), (h1, b1) in zip(curve, curve[1:]):
        if h0 <= hours <= h1:
            if h0 == h1:
                return b1
            f = (math.log(hours) - math.log(h0)) / (math.log(h1) - math.log(h0))
            return b0 + f * (b1 - b0)
    return None


def main():
    ap = argparse.ArgumentParser()
    for arm in ARMS:
        ap.add_argument(f"--{arm}", required=True)
    ap.add_argument("--points", default=None,
                    help="comma-separated GPU-hours; default 4 log-spaced in shared window")
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    curves = {arm: load_curve(getattr(args, arm)) for arm in ARMS}
    lo = max(c[0][0] for c in curves.values())
    hi = min(c[-1][0] for c in curves.values())
    if not lo < hi:
        raise SystemExit(f"no shared compute window: lo={lo} hi={hi}")
    if args.points:
        pts = [float(p) for p in args.points.split(",")]
    else:
        pts = [math.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * i / 3)
               for i in range(4)]
    rows = []
    for p in pts:
        row = {"gpu_hours": round(p, 4)}
        for arm in ARMS:
            v = bpb_at(curves[arm], p)
            row[f"bpb_{arm}"] = round(v, 4) if v is not None else ""
        rows.append(row)
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"shared window: [{lo:.4f}, {hi:.4f}] GPU-hours")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
