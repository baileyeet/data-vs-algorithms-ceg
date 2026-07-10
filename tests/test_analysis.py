"""Validate the CEG/Shapley analysis chain end-to-end.

1. Synthetic fixture: four arms with BPB curves linear in log(GPU-hours),
   constructed so compute-to-threshold is exactly C = 100/25/10/2.5 h
   -> ground truth: data 4x, algorithm 10x, total 40x. The analysis must
   recover these via interpolation (threshold placed BETWEEN checkpoints).
2. Real-data fixture: the toy A0D0 run's actual metrics.csv, with the other
   three arms as copies whose gpu_hours are scaled by known factors -> checks
   CSV parsing + interpolation on real training curves.
"""

import csv
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "tests" / "fixtures"
THR = 1.0


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "tokens", "timed_hours", "gpu_hours", "train_loss_ema",
                    "neutral_bpb", "ownval_bpb", "lr", "wallclock_s"])
        for i, (h, b) in enumerate(rows):
            w.writerow([i + 1, (i + 1) * 1000, h, h, 5.0, b, b - 0.2, 1e-4, h * 3600])


def synth_arm(c_star, n=9):
    """BPB linear in log(h): bpb = THR - s*(log h - log c_star); checkpoints at
    log-spaced h from c_star/8 to 8*c_star so the crossing sits between rows."""
    s = 0.35
    return [(h, THR - s * (math.log(h) - math.log(c_star)))
            for h in [c_star * 2 ** (k / 2) for k in range(-6, 7)]]


def run_analysis(dirs, threshold, out_json):
    cmd = [sys.executable, str(ROOT / "analysis" / "ceg_shapley.py"), "--size", "test",
           "--threshold", str(threshold), "--out-json", str(out_json)]
    for arm, d in dirs.items():
        cmd += [f"--{arm}", str(d)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr + r.stdout
    return json.loads(Path(out_json).read_text())


def main():
    # --- 1. synthetic ground truth ---
    C = {"a0d0": 100.0, "a0d1": 25.0, "a1d0": 10.0, "a1d1": 2.5}
    dirs = {}
    for arm, c in C.items():
        d = OUT / "synth" / arm
        write_csv(d / "metrics.csv", synth_arm(c))
        dirs[arm] = d
    res = run_analysis(dirs, THR, OUT / "synth" / "result.json")
    m = res["multipliers"]
    for k, expected in [("data", 4.0), ("algorithm", 10.0), ("total", 40.0)]:
        assert abs(m[k] - expected) / expected < 1e-6, f"{k}: {m[k]} != {expected}"
    for arm, c in C.items():
        got = res["gpu_hours_to_threshold"][arm]
        assert abs(got - c) / c < 1e-6, f"{arm}: {got} != {c}"
        assert res["crossing_type"][arm] == "interpolated"
    assert res["sanity_product_equals_total"]
    print(f"synthetic fixture: data {m['data']:.3f}x, algo {m['algorithm']:.3f}x, "
          f"total {m['total']:.3f}x — exact recovery OK")

    # --- 2. real toy curve, scaled ---
    toy = ROOT / "runs" / "toy_a0d0" / "metrics.csv"
    if not toy.exists():
        print("toy run metrics not found; skipping real-data fixture")
        return
    rows = [(float(r["gpu_hours"]), float(r["neutral_bpb"]))
            for r in csv.DictReader(open(toy)) if float(r["gpu_hours"]) > 0]
    scale = {"a0d0": 1.0, "a0d1": 0.5, "a1d0": 0.25, "a1d1": 0.125}
    dirs = {}
    for arm, s in scale.items():
        d = OUT / "real" / arm
        write_csv(d / "metrics.csv", [(h * s, b) for h, b in rows])
        dirs[arm] = d
    # threshold between the two real checkpoints' BPB values
    thr = (rows[-1][1] + rows[-2][1]) / 2
    res = run_analysis(dirs, thr, OUT / "real" / "result.json")
    m = res["multipliers"]
    # identical curves scaled in h -> data 2x, algo 4x, total 8x exactly
    for k, expected in [("data", 2.0), ("algorithm", 4.0), ("total", 8.0)]:
        assert abs(m[k] - expected) / expected < 1e-9, f"{k}: {m[k]} != {expected}"
    print(f"real-toy-curve fixture (thr={thr:.4f}): data {m['data']:.3f}x, "
          f"algo {m['algorithm']:.3f}x, total {m['total']:.3f}x — OK")
    print("ALL ANALYSIS TESTS PASSED")


if __name__ == "__main__":
    main()
