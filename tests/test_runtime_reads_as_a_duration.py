"""A runtime of "~0.0 h" tells you nothing, and reads like "nothing to do".

Hours are the right unit for the number this command usually prints. They are
the wrong unit once the work shrinks, which is exactly what happens after the
skip set is subtracted: a real archive went from "~2.9 h" to "~0.0 h" for 75
files. Both are true; only one is useful, and "0.0" invites the reader to
conclude the run is unnecessary rather than quick.
"""

from __future__ import annotations

import pytest

from corpus.cli.transcribe import _format_duration


@pytest.mark.parametrize(
    ("hours", "expected"),
    [
        (14.4, "~14.4 h"),
        (2.9, "~2.9 h"),
        (1.0, "~1.0 h"),
        (0.6, "~36 min"),
        (0.04, "~2 min"),
        (0.0055, "~20 s"),
        (0.0, "~0 s"),
    ],
)
def test_a_duration_is_shown_in_a_unit_that_carries_information(
    hours: float, expected: str
) -> None:
    assert _format_duration(hours) == expected


def test_no_duration_ever_renders_as_zero_of_a_big_unit() -> None:
    """The actual defect: a small non-zero duration must not print as 0."""
    for hours in (0.9, 0.5, 0.1, 0.02, 0.001):
        rendered = _format_duration(hours)
        assert not rendered.startswith("~0 h"), rendered
        assert not rendered.startswith("~0.0"), rendered
