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
