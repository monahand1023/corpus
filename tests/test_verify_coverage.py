"""A check that examined nothing must not report success.

Every defect found on 2026-09-15/16 was one shape: a check that did not run,
or ran against the wrong surface, is indistinguishable from a check that
passed. `repeat_share` sat at a 0.9 threshold whose real-data maximum was
0.250; `corpus-doctor` printed "a skipped check is not a passing one" and then
skipped its query-log check by default; `corpus-smoke` reported "1/2 servers
healthy" on a machine with four archives configured; a privacy grep exited 2
(error) three times and was read as "no matches".

`Coverage` is the shared answer: a check reports WHAT IT EXAMINED, and
examining zero is never a pass.
"""

from __future__ import annotations

import pytest

from corpus.verify import Coverage


def test_examining_nothing_is_vacuous() -> None:
    assert Coverage(examined=0, unit="chunks").vacuous is True


def test_examining_something_is_not_vacuous() -> None:
    assert Coverage(examined=1, unit="chunks").vacuous is False


def test_a_vacuous_coverage_says_so_in_words() -> None:
    # The message is the point: "0 chunks" reads like a pass to a human
    # skimming output, which is exactly how these went unnoticed.
    text = Coverage(examined=0, unit="chunks").describe()
    assert "nothing" in text.lower()
    assert "chunks" in text


def test_a_real_coverage_reports_the_count_and_unit() -> None:
    assert Coverage(examined=1405053, unit="chunks").describe() == "1,405,053 chunks"


def test_negative_coverage_is_a_programming_error() -> None:
    with pytest.raises(ValueError):
        Coverage(examined=-1, unit="chunks")


def test_coverage_is_falsy_when_vacuous_so_it_guards_naturally() -> None:
    # Lets call sites write `if not coverage: ...` rather than remembering
    # to compare a count against zero, which is the comparison people forget.
    assert not Coverage(examined=0, unit="servers")
    assert Coverage(examined=3, unit="servers")
