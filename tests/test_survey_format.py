"""Human-readable size and count formatting, used in every survey report.

Small, but it is the layer a person reads before deciding to spend hours of
GPU or real API tokens on an ingest. A wrong unit boundary understates a
corpus by three orders of magnitude.
"""

from __future__ import annotations

import pytest

from corpus.survey.format import human_count, human_size


@pytest.mark.parametrize(
    "num,expected_unit",
    [
        (0, "B"), (1, "B"), (1023, "B"),
        (1024, "KB"), (1024 * 1023, "KB"),
        (1024 ** 2, "MB"),
        (1024 ** 3, "GB"),
        (1024 ** 4, "TB"),
    ],
)
def test_size_crosses_units_at_the_right_boundaries(num: int, expected_unit: str) -> None:
    assert human_size(num).endswith(expected_unit)


def test_size_never_renders_a_bare_number() -> None:
    for n in (0, 999, 10**9):
        assert any(c.isalpha() for c in human_size(n)), human_size(n)


def test_a_very_large_size_does_not_fall_off_the_unit_table() -> None:
    """A petabyte-scale figure must still render, not raise or come back
    empty -- an archive on a NAS can reach it."""
    out = human_size(1024 ** 6)
    assert out and any(c.isalpha() for c in out)


@pytest.mark.parametrize("n,expected", [(0, "0"), (999, "999"), (1000, "1,000"),
                                        (1234567, "1,234,567")])
def test_counts_are_thousands_separated(n: int, expected: str) -> None:
    assert human_count(n) == expected


def test_negative_values_do_not_crash() -> None:
    # Not expected in practice, but a formatter that raises takes down the
    # whole report it was summarising.
    assert human_count(-5)
    assert human_size(-1)
