"""Canonical, deterministic reference-threshold definition (methodology).

THE threshold for a tier is the mean neutral-corpus BPB over the checkpoints in
the final `tail_frac` of A0D0's TRAINING, measured by token fraction (not
checkpoint count, not step index — token fraction is the unambiguous, schedule-
independent anchor). One function, applied bit-identically to every tier/arm;
no per-arm logic.

A checkpoint is "in the final tail" iff  tokens >= (1 - tail_frac) * total_tokens
with total_tokens = the last row's token count. The number of checkpoints this
captures may vary by ±1 across arms depending on where the 0.90-boundary
checkpoint rounds — that is expected and harmless (it shifts the mean < 1e-3);
what matters is that the RULE is identical everywhere.
"""

import csv
from pathlib import Path

TAIL_FRAC = 0.10


def final_tail_threshold(metrics_csv, tail_frac: float = TAIL_FRAC,
                         col: str = "neutral_bpb"):
    """Mean `col` over A0D0 checkpoints in the final `tail_frac` of training
    (by token fraction). Returns (threshold, n_points, steps_used)."""
    rows = list(csv.DictReader(open(Path(metrics_csv))))
    if not rows:
        raise ValueError(f"{metrics_csv}: empty")
    total_tokens = int(rows[-1]["tokens"])
    cutoff = (1.0 - tail_frac) * total_tokens
    tail = [(int(r["step"]), float(r[col])) for r in rows
            if int(r["tokens"]) >= cutoff]
    if not tail:
        raise ValueError(f"{metrics_csv}: no checkpoints in final {tail_frac}")
    thr = sum(b for _, b in tail) / len(tail)
    return thr, len(tail), [s for s, _ in tail]


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        thr, n, steps = final_tail_threshold(p)
        print(f"{p}: threshold={thr:.6f}  (n={n} tail checkpoints, steps {steps})")
