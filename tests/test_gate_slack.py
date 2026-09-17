"""A floor far below the measurement cannot fail, so it is not a gate.

`gate_verdict` already reports a floor that is too CLOSE -- `tight` when it
sits inside one query's resolution, `flaky` when it sits inside the noise.
Nothing reported the opposite, and the opposite is just as useless:

    a document consumer, 24 queries, measured over three runs
      recall@5   worst 0.750   floor 0.500   headroom 0.250 = SIX queries
      MRR        worst 0.580   floor 0.400   headroom 0.180
      nDCG@5     worst 0.599   floor 0.450   headroom 0.149

Six queries of the gold set could regress before that gate said a word. The
floors were not wrong when they were set -- they were set against an
8-query, transcripts-only version of the same file, and the gold set grew
underneath them. Nothing connects a threshold to the gold set it was
measured from, so a stale floor looks exactly like a comfortable one.

Reported in QUERIES rather than as a ratio, because that is the unit the
reader has to act in: "six queries could regress" is a decision, "headroom
0.250" is a number.
"""

from __future__ import annotations

import pytest

from corpus.eval.noise import MetricSpread, gate_verdict


def test_a_floor_six_queries_below_the_measurement_is_reported_as_slack():
    v = gate_verdict(
        floor=0.500,
        spread=MetricSpread("recall_at_k", [0.750, 0.792, 0.750]),
        n_queries=24,
    )
    assert v.passed
    assert v.slack_queries is not None
    assert round(v.slack_queries) == 6
    assert v.slack, "a gate six queries loose was not flagged"
    assert "6" in v.describe()


def test_a_floor_one_query_below_is_not_slack():
    """The intended shape. One query below the measurement is the tightest
    gate that is not `tight`, and it must not be flagged as loose too."""
    v = gate_verdict(
        floor=0.750 - 1 / 24,
        spread=MetricSpread("recall_at_k", [0.750, 0.792, 0.750]),
        n_queries=24,
    )
    assert v.passed
    assert not v.slack, v.describe()


def test_slack_is_not_computed_for_a_metric_with_no_query_resolution():
    """MRR is continuous: 'how many queries of headroom' is not meaningful
    for it, and inventing a number would be worse than saying nothing."""
    v = gate_verdict(
        floor=0.400,
        spread=MetricSpread("mrr", [0.580, 0.585, 0.580]),
        n_queries=24,
    )
    assert v.slack_queries is None
    assert not v.slack


def test_a_failing_gate_is_never_also_reported_as_loose():
    v = gate_verdict(
        floor=0.900,
        spread=MetricSpread("recall_at_k", [0.750, 0.792, 0.750]),
        n_queries=24,
    )
    assert not v.passed
    assert not v.slack


def test_without_a_query_count_slack_is_unknown_not_zero():
    v = gate_verdict(
        floor=0.500, spread=MetricSpread("recall_at_k", [0.750, 0.750])
    )
    assert v.slack_queries is None
    assert not v.slack


def test_noise_measured_elsewhere_is_not_discarded_by_a_quiet_run():
    """Three identical runs are three samples, not a proof of stability.

    `gate_verdict` read `reference_spread` only when the local spread was
    UNMEASURED -- a single run. With two or more runs it took the local
    figure even when that was 0.000 and the reference was larger, which is
    the OPTIMISTIC direction for a safety margin and the wrong one.

    Watched on one archive in a single day: recall@5 moved 0.042 between two
    identical runs in the morning, and three runs that evening moved 0.000.
    Nothing about the metric changed -- boundary-adjacency is a property of
    the archive's state, and the state had been re-ingested in between. A
    floor set from the quiet measurement is a floor that fails the next time
    a hit lands on the k boundary.

    So the margin uses whichever noise figure is LARGER. The local run can
    only ever widen it, never narrow what someone measured before.
    """
    from corpus.eval.noise import MetricSpread, gate_verdict

    quiet = MetricSpread("recall_at_k", [0.792, 0.792, 0.792])
    v = gate_verdict(
        floor=0.750, spread=quiet, reference_spread=0.042, n_queries=24
    )

    assert v.noise == 0.042, (
        "a quiet local run discarded a larger noise figure measured elsewhere"
    )
    assert v.flaky, (
        "a floor one query below the measurement is not safe against noise "
        "of a full query"
    )


def test_a_larger_local_spread_still_wins():
    """The reference must not narrow a margin either -- it is a floor on the
    estimate, not a replacement for it."""
    from corpus.eval.noise import MetricSpread, gate_verdict

    noisy = MetricSpread("mrr", [0.50, 0.58, 0.52])
    v = gate_verdict(floor=0.40, spread=noisy, reference_spread=0.005)

    assert v.noise == pytest.approx(0.08)


def test_no_reference_behaves_exactly_as_before():
    from corpus.eval.noise import MetricSpread, gate_verdict

    v = gate_verdict(floor=0.70, spread=MetricSpread("mrr", [0.75, 0.78]))
    assert v.noise == pytest.approx(0.03)
    assert v.noise_measured is True


def test_a_cross_archive_noise_figure_must_be_in_the_right_units():
    """recall@k moves in units of 1/n, so an absolute figure does not travel.

    A boundary flip moves recall by exactly one query. In absolute recall
    that is 1/n, which differs per archive -- so carrying 0.042 (one query
    at n=24) onto an n=41 gold set overstates the noise by nearly 2x.

    It is not a rounding error. At n=41 the flaky rule then demands
    2 x 0.042 = 0.084 of headroom, which is 3.4 queries, while the slack
    rule flags anything over 3 queries as LOOSE. The two guards become
    mutually unsatisfiable and NO floor is clean -- which is what happened
    when a real archive's floors were being set.

    Expressed per-archive as 1/n, both rules are satisfiable together.
    """
    from corpus.eval.noise import MetricSpread, gate_verdict

    quiet = MetricSpread("recall_at_k", [0.897, 0.897, 0.897])

    # Borrowed absolute figure: nothing works.
    borrowed = [
        gate_verdict(floor=f, spread=quiet, reference_spread=0.042, n_queries=41)
        for f in (0.87, 0.85, 0.84, 0.82, 0.80)
    ]
    assert all(v.flaky or v.slack for v in borrowed), (
        "expected the borrowed figure to leave no clean floor"
    )

    # One query at THIS archive's size: a clean floor exists.
    per_archive = gate_verdict(
        floor=0.84, spread=quiet, reference_spread=1 / 41, n_queries=41
    )
    assert per_archive.passed
    assert not per_archive.flaky and not per_archive.slack and not per_archive.tight
