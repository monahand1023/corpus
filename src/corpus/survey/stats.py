"""Tiny, dependency-free statistics helpers for survey estimators.

No `scipy`/`statsmodels` — corpus's base install stays minimal, and a single
closed-form interval doesn't justify pulling in a stats library.
"""

from __future__ import annotations

import math

# z for a 95% two-sided normal confidence interval.
Z_95 = 1.959963984540054


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion — a defensible
    "sampled M of N, X matched" confidence interval that, unlike the naive
    normal-approximation interval, stays inside `[0, 1]` and doesn't
    degenerate to a zero-width interval at `successes in (0, n)`. Standard
    choice for small-to-moderate sample sizes, which is exactly the regime
    `corpus-survey overlap` samples in.

    `n` must be > 0 (callers check this — there is no meaningful interval
    for an empty sample).
    """
    if n <= 0:
        raise ValueError("wilson_interval requires n > 0")
    p_hat = successes / n
    denom = 1 + z**2 / n
    center = p_hat + z**2 / (2 * n)
    spread = z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
    lower = (center - spread) / denom
    upper = (center + spread) / denom
    return max(0.0, lower), min(1.0, upper)
