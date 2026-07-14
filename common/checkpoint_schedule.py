"""Log-spaced (denser-early) checkpoint schedule — methodology requirement #6.

Checkpoints at token counts t_i = t0 * (T/t0)^(i/(N-1)), i.e. geometrically
spaced from an early first checkpoint t0 (default T/1000) to the full budget T.
This keeps checkpoint granularity from dominating the measured
compute-to-threshold for arms that cross the reference BPB very early (A1D1).
"""

import math


def checkpoint_tokens(total_tokens: int, n_checkpoints: int = 25,
                      first_frac: float = 0.001, tail_points: int = 5,
                      tail_frac: float = 0.10) -> list[int]:
    """Log-spaced (denser-early) points plus a linearly-spaced tail.

    The tail points (default 5 across the final 10% of training) exist because
    the reference threshold is defined on the threshold arm's end-of-training
    plateau: pure log spacing is sparsest exactly there, which made the
    plateau estimate (and hence the headline CEG numbers) fragile at 124M.
    Applied to every arm — any arm could be the flat one at another size.
    """
    assert n_checkpoints >= 2
    t0 = max(1, int(total_tokens * first_frac))
    pts = [
        int(round(t0 * (total_tokens / t0) ** (i / (n_checkpoints - 1))))
        for i in range(n_checkpoints)
    ]
    # linear tail: tail_points evenly spaced over [1-tail_frac, 1.0)
    for i in range(tail_points):
        frac = 1 - tail_frac + tail_frac * i / tail_points
        pts.append(int(round(total_tokens * frac)))
    pts.sort()
    pts[-1] = total_tokens
    # dedupe while preserving order (can collide after rounding on tiny runs)
    out = []
    for p in pts:
        if not out or p > out[-1]:
            out.append(p)
    return out


def checkpoint_steps(total_tokens: int, tokens_per_step: int,
                     n_checkpoints: int = 25, first_frac: float = 0.001) -> list[int]:
    total_steps = math.ceil(total_tokens / tokens_per_step)
    steps = sorted({
        min(total_steps, max(1, round(t / tokens_per_step)))
        for t in checkpoint_tokens(total_tokens, n_checkpoints, first_frac)
    })
    if steps[-1] != total_steps:
        steps.append(total_steps)
    return steps
