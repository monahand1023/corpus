""""audited and reversed against real evidence" -- but not the evidence needed.

`demote_transcript` keeps the rejected text so "a rule that turns out to be
too aggressive can be audited and reversed against real evidence instead of
a re-run", and `--refilter` tells the operator the same: "their text is
retained there, so this is auditable."

The text alone cannot reproduce the decision. The pipeline filters WINDOWS
first, and the speech-rate rule divides by the WINDOW's duration. All that
survives a demotion is the joined text and the FILE's duration -- so
re-judging a stored rejection treats a burst as if it were spread across
the whole recording, which is precisely the defect that put decode loops
into a live index this morning.

Measured on that archive: of 293 text-bearing rejections, the 5 that
re-judge as "keep" are the caption sign-offs and the three decode loops --
including the ones demoted hours earlier. A feature built on this data
would restore exactly what was just removed, and would look correct doing
it.

So the geometry is stored with the rejection. It costs one column and
nothing at decode time, and it makes the auditability the code already
advertises actually true.
"""

from __future__ import annotations

import json

import pytest

from corpus.transcripts import store
from corpus.transcripts.pipeline import Settings, filter_windows


def test_a_demotion_keeps_the_window_geometry(tmp_path):
    db = tmp_path / "t.db"
    with store.open_store(db) as conn:
        store.demote_transcript(
            conn,
            "/w/a.m4a",
            duration_s=64.0,
            policy="p",
            reason="empty",
            rejected_text="ta lom" + ",0" * 100,
            segments=json.dumps([{"start": 23.8, "end": 25.5, "text": "ta lom" + ",0" * 100}]),
        )
        row = conn.execute(
            "SELECT segments FROM no_text WHERE path = '/w/a.m4a'"
        ).fetchone()

    assert row is not None and row[0], "the windows were not kept with the rejection"
    parsed = json.loads(row[0])
    # approx: 25.5 - 23.8 is 1.7000000000000028 in binary floats.
    assert parsed[0]["end"] - parsed[0]["start"] == pytest.approx(1.7)


def test_the_stored_rejection_re_judges_the_same_way(tmp_path):
    """The point of keeping it: the decision must reproduce.

    Re-judged from the stored geometry this stays rejected. Re-judged from
    the text plus the FILE duration -- all that used to be kept -- it comes
    back as material to restore, at 3.7 chars/s instead of 138.
    """
    burst = "ta lom, e o lomi" + ",0" * 100
    settings = Settings()

    from_geometry, dropped = filter_windows([(burst, 1.7, False)], settings)
    assert from_geometry == [], f"should still be rejected: {dropped}"

    from_text_only, _ = filter_windows([(burst, 64.0, False)], settings)
    assert from_text_only, (
        "this is the wrong answer the old evidence produced; if it now "
        "matches, the test no longer demonstrates why geometry is kept"
    )
