"""How close is a threshold to the data it must not cross?

Nothing measured this, and it cost twice in two days:

  * the speech-rate ceiling sat 0.6% above the fastest real speech, having
    been set when the archive's maximum was 41% lower
  * the loop ceiling sat 0.3% above the highest real score and was deleting
    family recordings -- a birthday song survived by 0.012 while another was
    deleted

Both were found by hand, by accident. There are eleven tunable thresholds.

A margin is not a pass/fail: a threshold legitimately sits near data it
should nearly touch. It is a number a person has to look at, which is why
this reports rather than raises.
"""

from __future__ import annotations

import pytest

from corpus.verify import Margin, margin


def test_a_comfortable_margin_is_not_tight() -> None:
    m = margin(threshold=40.0, observed_max=24.85, label="chars/sec")
    assert isinstance(m, Margin)
    assert round(m.percent, 1) == 61.0
    assert m.tight is False


def test_a_threshold_almost_touching_real_data_is_tight() -> None:
    m = margin(threshold=0.6, observed_max=0.5984, label="looping share")
    assert m.percent < 1
    assert m.tight is True


def test_a_threshold_ALREADY_CROSSED_is_tight_and_negative() -> None:
    """Real data above the threshold means it is actively rejecting real material."""
    m = margin(threshold=0.6, observed_max=0.75, label="looping share")
    assert m.percent < 0
    assert m.tight is True


def test_the_description_leads_with_the_number_a_person_must_judge() -> None:
    text = margin(threshold=0.6, observed_max=0.5984, label="looping share").describe()
    assert "0.3%" in text
    assert "looping share" in text


def test_observing_nothing_yields_no_opinion() -> None:
    # Same rule as everywhere else: examining nothing is not a verdict.
    m = margin(threshold=1.0, observed_max=None, label="x")
    assert m.percent is None and m.tight is False
    assert "nothing" in m.describe().lower()


@pytest.mark.parametrize("pct,expected", [(10.0, True), (24.9, True), (25.1, False)])
def test_the_tightness_boundary_is_explicit(pct: float, expected: bool) -> None:
    observed = 100.0
    m = margin(threshold=observed * (1 + pct / 100), observed_max=observed, label="x")
    assert m.tight is expected
