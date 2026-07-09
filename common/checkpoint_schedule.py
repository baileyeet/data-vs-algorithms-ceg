"""Log-spaced (denser-early) checkpoint schedule — methodology requirement #6.

Checkpoints at token counts t_i = t0 * (T/t0)^(i/(N-1)), i.e. geometrically
spaced from an early first checkpoint t0 (default T/1000) to the full budget T.
This keeps checkpoint granularity from dominating the measured
compute-to-threshold for arms that cross the reference BPB very early (A1D1).
"""

import math


def checkpoint_tokens(total_tokens: int, n_checkpoints: int = 25,
                      first_frac: float = 0.001) -> list[int]:
    assert n_checkpoints >= 2
    t0 = max(1, int(total_tokens * first_frac))
    pts = [
        int(round(t0 * (total_tokens / t0) ** (i / (n_checkpoints - 1))))
        for i in range(n_checkpoints)
    ]
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
