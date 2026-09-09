"""Tests for `corpus.survey.stats.wilson_interval`."""

from __future__ import annotations

import pytest

from corpus.survey.stats import wilson_interval


def test_zero_n_raises() -> None:
    with pytest.raises(ValueError, match="n > 0"):
        wilson_interval(0, 0)


def test_interval_is_within_unit_range() -> None:
    lo, hi = wilson_interval(5, 10)
    assert 0.0 <= lo <= hi <= 1.0


def test_interval_centered_near_observed_proportion_for_large_n() -> None:
    lo, hi = wilson_interval(500, 1000)
    assert lo < 0.5 < hi
    assert hi - lo < 0.1  # tight interval at n=1000


def test_interval_widens_for_small_samples() -> None:
    lo_small, hi_small = wilson_interval(2, 4)
    lo_large, hi_large = wilson_interval(200, 400)
    assert (hi_small - lo_small) > (hi_large - lo_large)


def test_zero_successes_lower_bound_is_zero_but_upper_is_not() -> None:
    lo, hi = wilson_interval(0, 20)
    assert lo == 0.0
    assert hi > 0.0


def test_all_successes_upper_bound_is_at_most_one() -> None:
    lo, hi = wilson_interval(20, 20)
    assert hi <= 1.0
    assert lo < 1.0
