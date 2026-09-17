"""A query a coin flip passes is a query that measures nothing.

recall@k asks whether any acceptable answer landed in the top k. When a
query accepts 14% of an archive's 196,575 documents, five random results
contain one of them 53% of the time -- so the query scores ~0.5 for a
retriever that does not work at all, and its contribution to the gate is
mostly noise wearing the shape of a result.

This is the same defect as every other one in this codebase: a check that
cannot fail is indistinguishable from a check that passes. It just arrives
through the gold set instead of through the code.

Measured on a real archive, which is why the threshold is not stricter:
of 24 queries, 23 sat at or below 0.094 and one sat at 0.535. The set as a
whole scored 0.050 for a random retriever against a floor of 0.708 -- the
gate was overwhelmingly earned, and exactly one query was not pulling its
weight. A check that flagged the other 23 would be wrong and would get
switched off.
"""

from __future__ import annotations

import pytest

from corpus.eval.triviality import (
    DEFAULT_TRIVIAL_ABOVE,
    random_hit_rate,
    triviality_report,
)


def test_a_query_accepting_most_of_the_archive_is_flagged():
    report = triviality_report(
        {"my wife's phone photos": 27_946, "a bridge over a river": 7},
        documents=196_575,
        top_k=5,
    )
    assert report.trivial == ["my wife's phone photos"]
    assert not report.is_clean
    assert "27,946" in report.describe()


def test_a_normal_gold_set_is_not_flagged():
    """23 of 24 real queries, verbatim. If this trips, the check is useless:
    every archive with broad topical queries would fail it and it would be
    turned off, taking the real signal with it."""
    sizes = {
        f"q{i}": n for i, n in enumerate(
            [3855, 3710, 2360, 1919, 1458, 1455, 1301, 1202, 1187, 1016,
             941, 929, 923, 908, 654, 610, 607, 597, 368, 231, 200, 38, 7]
        )
    }
    report = triviality_report(sizes, documents=196_575, top_k=5)
    assert report.is_clean, report.describe()


def test_the_set_wide_floor_is_what_a_random_retriever_would_score():
    """The number that says whether a gate is earned at all. A floor of
    0.708 over a set a coin flip scores 0.700 on is not a gate."""
    report = triviality_report(
        {"a": 27_946, "b": 27_946}, documents=196_575, top_k=5
    )
    assert report.random_recall == pytest.approx(0.535, abs=0.01)


def test_an_empty_archive_does_not_divide_by_zero():
    """Zero documents means "I cannot judge this", not "everything is fine"."""
    report = triviality_report({"a": 5}, documents=0, top_k=5)
    assert report.random_recall == 0.0
    assert report.is_clean
    assert "cannot" in report.describe().lower()


def test_a_negative_query_is_not_counted_as_trivial():
    """A query with no acceptable answer is a NEGATIVE control -- it exists
    to check that nothing is returned. Scoring it for triviality would flag
    the one query shape that is deliberately unanswerable."""
    report = triviality_report({"nothing matches this": 0}, documents=1000, top_k=5)
    assert report.is_clean
    assert report.random_recall == 0.0


def test_more_keys_than_documents_is_reported_not_silently_clamped():
    """A gold set naming more answers than the archive holds is a bug in the
    gold set. Clamping to 1.0 would hide it behind a plausible number."""
    with pytest.raises(ValueError, match="more acceptable answers"):
        triviality_report({"a": 2000}, documents=1000, top_k=5)


def test_the_hit_rate_matches_the_closed_form():
    # 1 - (1 - K/N)^k
    assert random_hit_rate(keys=1, documents=2, top_k=1) == pytest.approx(0.5)
    assert random_hit_rate(keys=0, documents=100, top_k=5) == 0.0
    assert random_hit_rate(keys=100, documents=100, top_k=1) == pytest.approx(1.0)


def test_the_default_threshold_is_a_coin_flip():
    """Stated once, here, because a threshold nobody can justify is the thing
    this project keeps finding at the bottom of a broken check."""
    assert DEFAULT_TRIVIAL_ABOVE == 0.5


def test_the_eval_reports_the_random_baseline_beside_the_score(tmp_path, capsys):
    """A number with nothing to compare it to is the reason a 0.708 floor
    over a set chance scores 0.700 on looks like a passing gate.

    This is the part that makes the check real: it runs on every eval,
    beside the metric it qualifies, rather than being a library nobody
    calls.
    """
    from corpus.cli.eval import _report_triviality

    _report_triviality(
        key_counts={"broad": 27_946, "narrow": 7},
        documents=196_575,
        top_k=5,
    )
    out = capsys.readouterr().out

    assert "random retriever" in out
    assert "0.268" in out or "0.267" in out, out  # (0.535 + 0.000) / 2
    assert "broad" in out, "the offending query was not named"


def test_the_eval_says_nothing_when_the_gold_set_is_sound(tmp_path, capsys):
    """Silence on the clean path. A line printed every run stops being read,
    and this one has to be read on the run where it matters."""
    from corpus.cli.eval import _report_triviality

    _report_triviality(key_counts={"a": 7, "b": 38}, documents=196_575, top_k=5)
    out = capsys.readouterr().out

    assert out.strip() == "", f"printed on a clean gold set:\n{out}"
