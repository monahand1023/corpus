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


# --- "real data" is an assumption the check cannot make ------------------------


def test_the_margin_never_calls_what_it_measured_real_data():
    """It measured what SURVIVED the threshold, which is not the same thing.

    Found live. `corpus-doctor` reported:

        looping share: 0.7% headroom (threshold 0.85, real data reaches
        0.844444)

    The 0.844444 was five copies of a Whisper decode loop -- the same
    Japanese sentence thirteen times over a video of a child in a playroom.
    Junk the ceiling was too permissive to catch, measured as if it were the
    material the ceiling exists to protect.

    The reading that invites -- "only 0.7% of headroom, raise the ceiling" --
    is exactly backwards. A threshold that is too permissive will ALWAYS look
    like it has a tight margin, because its own failures are in the sample.

    The check cannot tell real from junk. So it must not claim to.
    """
    m = margin(threshold=0.85, observed_max=0.844444, label="looping share")
    text = m.describe()
    assert "real data" not in text
    assert "0.85" in text and "0.844" in text


def test_the_margin_says_what_it_actually_measured():
    m = margin(threshold=40.0, observed_max=24.85, label="max chars/sec")
    assert "kept" in m.describe().lower() or "survived" in m.describe().lower()
