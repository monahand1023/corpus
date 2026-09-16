"""A gate set inside the instrument's noise is a gate that flakes.

MEASURED, NOT ASSUMED. Embedding a fixed query through a hosted provider
does not return a bit-identical vector every time. On a live 45,594-chunk
archive, four embeddings of one query differed, and ranks 2 and 3 swapped.
Across five full runs of the same 31-query gold set, against an archive
nothing had written to:

    recall@5   0.935  0.935  0.935  0.935  0.935     spread 0.000
    MRR        0.844  0.871  0.866  0.860  0.866     spread 0.027
    nDCG@5     0.806  0.822  0.818  0.813  0.818     spread 0.016

Set membership is stable; ORDER is not, which is exactly the split between
recall and the rank-weighted metrics. One archive's CI gate sat 6.4% above
its measured MRR -- inside a 3.1% noise band on a related archive, close
enough that a run with no change in it could fail and be read as a
regression.

So the eval reports its own noise floor, and the gate is judged against the
WORST run rather than the average: a gate that passes on the mean and fails
one run in three is a flaky gate, not a passing one.
"""

from __future__ import annotations

import pytest

from corpus.eval.metrics import MetricSummary
from corpus.eval.noise import MetricSpread, gate_verdict, spread_report


def _summary(recall: float, mrr: float, ndcg: float) -> MetricSummary:
    return MetricSummary(recall_at_k=recall, mrr=mrr, ndcg_at_k=ndcg, n=31)


REAL_RUNS = [
    _summary(0.935, 0.844, 0.806),
    _summary(0.935, 0.871, 0.822),
    _summary(0.935, 0.866, 0.818),
    _summary(0.935, 0.860, 0.813),
    _summary(0.935, 0.866, 0.818),
]


# --- measuring the spread -----------------------------------------------------


def test_a_single_run_has_no_measurable_noise():
    """One run cannot tell you anything about variance, and must not be
    reported as if it proved stability."""
    report = spread_report([_summary(0.9, 0.8, 0.7)])
    assert report["mrr"].spread == 0.0
    assert report["mrr"].measured is False


def test_repeated_runs_report_the_observed_spread():
    report = spread_report(REAL_RUNS)
    assert report["mrr"].measured is True
    assert report["mrr"].spread == pytest.approx(0.027, abs=1e-9)
    assert report["mrr"].minimum == 0.844
    assert report["mrr"].maximum == 0.871


def test_a_stable_metric_reports_zero_spread():
    """recall@k was identical in every run. Reporting it as noisy would
    bury the metrics that actually move."""
    assert spread_report(REAL_RUNS)["recall_at_k"].spread == 0.0


def test_the_spread_covers_every_metric_not_just_the_first():
    report = spread_report(REAL_RUNS)
    assert set(report) == {"recall_at_k", "mrr", "ndcg_at_k"}


# --- judging a gate against it ------------------------------------------------


def test_a_gate_inside_the_noise_band_is_reported_as_flaky():
    """0.844 is the worst observed run. A floor of 0.855 passes on the mean
    and fails two runs in five -- which is a flaky gate, not a passing one."""
    spread = spread_report(REAL_RUNS)["mrr"]
    verdict = gate_verdict(floor=0.855, spread=spread)
    assert verdict.passed is False
    assert verdict.flaky is True
    assert verdict.tight is False, "an order-noise problem is not a resolution one"


def test_a_gate_below_every_observed_run_passes_cleanly():
    spread = spread_report(REAL_RUNS)["mrr"]
    verdict = gate_verdict(floor=0.60, spread=spread)
    assert verdict.passed is True
    assert verdict.flaky is False


def test_a_gate_above_every_observed_run_fails_and_is_not_called_flaky():
    """A real regression must not be excused as noise."""
    spread = spread_report(REAL_RUNS)["mrr"]
    verdict = gate_verdict(floor=0.95, spread=spread)
    assert verdict.passed is False
    assert verdict.flaky is False


def test_a_gate_judged_on_one_run_says_the_noise_is_unmeasured():
    """Without --repeat there is no noise floor, and claiming one would be
    the same defect as any other check that did not run."""
    spread = spread_report([_summary(0.9, 0.8, 0.7)])["mrr"]
    verdict = gate_verdict(floor=0.79, spread=spread)
    assert verdict.passed is True
    assert verdict.noise_measured is False


def test_a_gate_within_a_hair_of_the_worst_run_is_called_out():
    """A live archive's real shape: MRR 0.479 measured, floor 0.450. That is
    0.029 of headroom against a 0.027 noise band measured on comparable
    material -- barely one noise-width, which survives the range already seen
    and nothing more. See NOISE_SAFETY_FACTOR."""
    spread = MetricSpread(metric="mrr", values=[0.479])
    verdict = gate_verdict(floor=0.45, spread=spread, reference_spread=0.027)
    assert verdict.passed is True
    assert verdict.flaky is True, "headroom smaller than known noise went unflagged"


# --- the metric's own resolution ----------------------------------------------


def test_a_sub_resolution_gate_is_tight_but_not_flaky():
    """Two different problems, and conflating them makes the check cry wolf.

    recall@k is a mean of 0/1 scores, so over 8 queries it can only take
    values k/8. A live archive gates it at 0.850 against a measured 0.875:
    0.025 of headroom on a metric whose smallest possible step is 0.125, so
    ANY single query regressing fails it. That is TIGHT -- which may be
    exactly what its owner wants.

    It is not FLAKY. Flaky means the same data produces different answers,
    and this archive's recall was identical in all five runs. Labelling a
    deliberate tight gate as flaky is a false alarm, and a check that cries
    wolf gets ignored."""
    spread = MetricSpread(metric="recall_at_k", values=[0.875] * 5)
    verdict = gate_verdict(floor=0.850, spread=spread, n_queries=8)
    assert verdict.passed is True
    assert verdict.tight is True, "a sub-resolution gate looked comfortable"
    assert verdict.flaky is False, "a stable metric was called flaky"
    assert verdict.resolution == pytest.approx(0.125)


def test_a_recall_gate_with_room_for_a_whole_query_is_not_flagged():
    spread = MetricSpread(metric="recall_at_k", values=[0.875] * 5)
    v = gate_verdict(floor=0.50, spread=spread, n_queries=8)
    assert v.tight is False and v.flaky is False


def test_resolution_does_not_apply_to_the_continuous_metrics():
    """MRR and nDCG move in fractions of a query's score, so 1/n is not a
    floor on what they can do."""
    spread = MetricSpread(metric="mrr", values=[0.479] * 5)
    verdict = gate_verdict(floor=0.450, spread=spread, n_queries=8)
    assert verdict.resolution == 0.0


def test_resolution_is_ignored_when_the_query_count_is_unknown():
    spread = MetricSpread(metric="recall_at_k", values=[0.875] * 5)
    assert gate_verdict(floor=0.850, spread=spread).flaky is False


def test_exactly_one_query_of_headroom_is_not_reported_as_tight():
    """A floor set exactly one query below the measurement is the intended
    shape, and floating point called it TIGHT.

    16/24 = 0.6666666666666666, floor 0.625, headroom 0.04166666666666663,
    resolution 1/24 = 0.041666666666666664. The headroom is smaller by
    3.5e-17 -- an artefact of binary floats, not a property of the gate -- so
    a bare `<` flagged a correctly-set floor on a real archive.
    """
    spread = MetricSpread(metric="recall_at_k", values=[16 / 24] * 3)
    verdict = gate_verdict(floor=0.625, spread=spread, n_queries=24)
    assert verdict.passed is True
    assert verdict.tight is False, "a float epsilon was reported as a tight gate"


def test_genuinely_less_than_one_query_of_headroom_is_still_tight():
    """The negative control: the tolerance must not swallow a real case."""
    spread = MetricSpread(metric="recall_at_k", values=[16 / 24] * 3)
    verdict = gate_verdict(floor=0.65, spread=spread, n_queries=24)
    assert verdict.tight is True
