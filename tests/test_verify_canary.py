"""A detector must prove it fires before its "clean" verdict is believed.

The failure this exists for is in this codebase's own history: a review of an
early caption phrase list found **22 of 45 entries could never match
anything** -- and the tests passed, because they asserted against the list
rather than against strings a model really produces. A scanner built on a
detector that cannot fire reports a clean index forever.

The same shape bit the privacy audit during the 2026-09-16 incident: a grep
exited 2 (an error, from a shell alias mangling its arguments) and the exit
status was read as "no matches", three times in a row. A planted positive
would have caught it the first time.
"""

from __future__ import annotations

import pytest

from corpus.verify import DetectorBroken, self_check


def test_a_detector_that_fires_on_the_planted_positive_passes() -> None:
    self_check(lambda t: "needle" in t, positive="a needle here", label="needle finder")


def test_a_detector_that_cannot_fire_is_rejected() -> None:
    with pytest.raises(DetectorBroken) as caught:
        self_check(lambda t: False, positive="a needle here", label="needle finder")
    assert "needle finder" in str(caught.value)


def test_the_error_says_the_scan_is_invalid_not_that_the_data_is_clean() -> None:
    # The whole point: distinguish "found nothing" from "cannot find anything".
    with pytest.raises(DetectorBroken) as caught:
        self_check(lambda t: False, positive="x", label="phrase list")
    message = str(caught.value).lower()
    assert "cannot" in message or "did not fire" in message
    assert "clean" not in message, "must not imply a clean result"


def test_a_detector_that_raises_is_also_broken() -> None:
    # A scanner whose detector throws is not a scanner that found nothing.
    def explodes(_text: str) -> bool:
        raise RuntimeError("boom")

    with pytest.raises(DetectorBroken):
        self_check(explodes, positive="x", label="exploding detector")


def test_the_real_boilerplate_detector_passes_its_own_canary() -> None:
    # Guards the actual regression: someone edits the phrase list, or the
    # normalisation, in a way that makes it unable to match anything.
    from corpus.transcripts.quality import subtitle_boilerplate

    self_check(subtitle_boilerplate, positive="Thanks for watching!", label="boilerplate")


def test_the_real_tail_stripper_passes_its_own_canary() -> None:
    from corpus.transcripts.quality import strip_caption_tail

    self_check(
        lambda t: strip_caption_tail(t) != t,
        positive="and then we drove home. Thanks for watching",
        label="caption tail",
    )
