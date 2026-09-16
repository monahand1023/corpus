"""How much does this eval move when NOTHING changes?

WHY THIS EXISTS. A hosted embedding provider does not return a bit-identical
vector for a fixed query. Measured on a live live archive: four
embeddings of one query differed, and ranks 2 and 3 swapped. Across five full
runs of the same 31-query gold set, against an archive nothing had written to:

    recall@5   0.935  0.935  0.935  0.935  0.935     spread 0.000
    MRR        0.844  0.871  0.866  0.860  0.866     spread 0.027
    nDCG@5     0.806  0.822  0.818  0.813  0.818     spread 0.016

Set membership is stable; ORDER is not -- which is exactly the split between
recall@k and the rank-weighted metrics. Every number reported to three
decimals had a third decimal that meant nothing.

WHAT IT COSTS TO IGNORE. A gate set at a single measurement sits at an unknown
distance from the instrument's own noise. One archive's CI floor for MRR sits
0.029 below its measured value -- barely one 0.027 noise-width, measured on
comparable material. Close enough that a run with no change in it could fail,
be read as a regression, and send someone looking for a cause that was never
there. The reverse is worse: a real regression smaller than the noise,
dismissed as one.

THE RULE. A gate is judged against the WORST observed run, never the average.
A gate that passes on the mean and fails one run in three is a flaky gate, not
a passing one. And with a single run there IS no measured noise: this reports
that as unmeasured rather than as zero, because "I did not look" and "I looked
and it was stable" are different facts.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from corpus.eval.metrics import MetricSummary

METRICS = ("recall_at_k", "mrr", "ndcg_at_k")

# How many noise-widths of headroom a gate needs before it is called stable.
#
# One is not enough. A floor exactly one spread below the worst observed run
# survives the range already seen and nothing more -- the next run only has to
# be slightly worse than any so far. Measured shapes this separates: an
# archive with MRR 0.479 against a floor of 0.450 has 0.029 of headroom
# against a 0.027 noise band, which reads as comfortable and is not.
NOISE_SAFETY_FACTOR = 2.0

# Metrics whose per-query score is 0 or 1, so the macro-average over n queries
# can only take the values k/n. Their RESOLUTION is 1/n, and a gate closer
# than that to the measured value has no effective margin: one query flipping
# is the smallest thing that can happen, and it fails. Measured on a live
# archive: recall@8 = 0.875 against a floor of 0.850, which reads as 0.025 of
# headroom and is really zero. Run-to-run spread cannot reveal this -- that
# archive's recall was identical in all five runs.
_BINARY_METRICS = frozenset({"recall_at_k"})


@dataclass(frozen=True)
class MetricSpread:
    """One metric's values across repeated runs of the same gold set."""

    metric: str
    values: list[float] = field(default_factory=list)

    @property
    def measured(self) -> bool:
        """Whether variance was actually observable.

        One run yields a spread of 0.0 arithmetically while proving nothing.
        Callers must not read that as stability.
        """
        return len(self.values) > 1

    @property
    def minimum(self) -> float:
        return min(self.values) if self.values else 0.0

    @property
    def maximum(self) -> float:
        return max(self.values) if self.values else 0.0

    @property
    def mean(self) -> float:
        return sum(self.values) / len(self.values) if self.values else 0.0

    @property
    def spread(self) -> float:
        return self.maximum - self.minimum

    def describe(self) -> str:
        if not self.measured:
            return f"{self.mean:.3f} (single run -- noise unmeasured)"
        return (
            f"{self.mean:.3f}  min {self.minimum:.3f}  max {self.maximum:.3f}  "
            f"spread {self.spread:.3f}"
        )


def spread_report(runs: Sequence[MetricSummary]) -> dict[str, MetricSpread]:
    """Per-metric spread across repeated runs of one gold set."""
    return {
        metric: MetricSpread(
            metric=metric, values=[getattr(r, metric) for r in runs]
        )
        for metric in METRICS
    }


@dataclass(frozen=True)
class GateVerdict:
    metric: str
    floor: float
    worst: float
    passed: bool
    # The floor is close enough to what the instrument does on its own that
    # this result is not reproducible. True for a gate that fails only some
    # runs, AND for one that passes with less headroom than known noise.
    # The same data produces different answers near this floor: run-to-run
    # noise can cross it, so the result is not reproducible.
    flaky: bool
    # The floor is closer than the smallest change the metric can express, so
    # ANY single query regressing crosses it. Distinct from flaky, and often
    # deliberate -- reported so it is a choice rather than a surprise.
    tight: bool
    noise_measured: bool
    # Observed run-to-run movement of this metric.
    noise: float
    # The smallest change this metric can express at all: 1/n for a metric
    # whose per-query score is 0 or 1, 0.0 for the continuous ones.
    resolution: float = 0.0
    # How much headroom the floor needs before it is called stable, from
    # whichever of the two above demands more.
    required_headroom: float = 0.0

    def describe(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        line = (
            f"[gate] {self.metric} = {self.worst:.3f}  floor {self.floor:.3f}  "
            f"{status}"
        )
        headroom = self.worst - self.floor
        if self.flaky:
            line += (
                f"  FLAKY: headroom {headroom:+.3f}, but this eval moved "
                f"{self.noise:.3f} between identical runs"
            )
        if self.tight:
            line += (
                f"  TIGHT: headroom {headroom:+.3f} is under one query "
                f"({self.resolution:.3f}), so any single regression fails"
            )
        if not self.flaky and not self.tight and not self.noise_measured:
            line += "  (noise unmeasured -- pass --repeat N)"
        return line


def gate_verdict(
    *,
    floor: float,
    spread: MetricSpread,
    reference_spread: float | None = None,
    n_queries: int | None = None,
) -> GateVerdict:
    """Judge one metric's floor against the runs actually observed.

    Judged on the WORST run, not the mean. `reference_spread` supplies a noise
    figure measured elsewhere -- another archive, an earlier session -- so a
    single-run eval can still be told that its headroom is smaller than noise
    someone has already measured on comparable material.

    `n_queries` lets a binary metric be judged against its own RESOLUTION as
    well as its noise: recall over 8 queries cannot move by less than 0.125,
    so a floor 0.025 below it has no margin at all. See `_BINARY_METRICS`.
    """
    noise = spread.spread if spread.measured else (reference_spread or 0.0)
    noise_measured = spread.measured or reference_spread is not None
    resolution = (
        1.0 / n_queries
        if n_queries and n_queries > 0 and spread.metric in _BINARY_METRICS
        else 0.0
    )
    worst = spread.minimum
    passed = worst >= floor

    # Two independent reasons a floor can be too close to the measurement:
    # the instrument moves on its own (noise, doubled -- see
    # NOISE_SAFETY_FACTOR), and the metric cannot move by less than one query
    # (resolution). Whichever demands more headroom wins.
    required = max(NOISE_SAFETY_FACTOR * noise, resolution)

    headroom = worst - floor
    if not passed:
        # A failure is flaky only if some run cleared the floor. If every run
        # was below it, that is a regression and must not be excused as noise.
        flaky = spread.maximum >= floor
        tight = False
    else:
        flaky = noise > 0.0 and headroom < NOISE_SAFETY_FACTOR * noise
        tight = resolution > 0.0 and headroom < resolution

    return GateVerdict(
        metric=spread.metric,
        floor=floor,
        worst=worst,
        passed=passed,
        flaky=flaky,
        tight=tight,
        noise_measured=noise_measured,
        noise=noise,
        resolution=resolution,
        required_headroom=required,
    )


__all__ = ["METRICS", "NOISE_SAFETY_FACTOR", "GateVerdict", "MetricSpread", "gate_verdict", "spread_report"]
