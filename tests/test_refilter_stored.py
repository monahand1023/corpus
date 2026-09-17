"""Re-apply text filters to STORED windows, without re-decoding audio.

A re-transcribe of one live archive was measured at 61.7 GPU-hours, from
its own recorded timings. Every setting that changed was a TEXT FILTER --
what counts as a loop, what counts as boilerplate -- and the per-window
text is already in the sidecar's `segments` column. Re-decoding 380 hours
of audio to re-run a regex is work nobody needs to do.

WHAT MAKES IT EQUIVALENT, and the guard that keeps it honest. Stored
segments are the windows that survived the PREVIOUS filter, so re-applying
a STRICTER one gives exactly what a re-decode would. It is only valid while
the decode itself is unchanged: `window_s`, `overlap_s`, the VAD threshold
and the model decide which audio becomes which window, and a re-filter
cannot re-derive those. A row whose stored model differs is skipped and
counted, never silently restamped with a policy that was not applied.

`continues_previous` is not stored, and it is load-bearing on rejoin. It is
reconstructed from the geometry: windows within one speech region overlap
(`step = window_s - overlap_s`), so `start < previous end` means a mid-speech
cut whose duplicated text is an artefact.
"""

from __future__ import annotations

import json
import sqlite3

from corpus.transcripts import store
from corpus.transcripts.pipeline import Settings
from corpus.transcripts.run import refilter_stored

LOOP = "そのため、" * 30
REAL_A = "We walked down to the harbour and the boats were all out."
REAL_B = "Then we found a bakery and sat outside for an hour."
# An UNTIDY repeat, which is the point: it scores 0.6471, above the 0.6 window
# ceiling and below the 0.85 transcript one -- precisely the band the fallback
# exists for. A tidy `"Happy Birthday to you. " * 8` scores 0.8667 and would be
# rejected on its own merits, proving nothing. Real family recordings repeat
# like this, imperfectly; that is why a synthetic-looking fixture cannot stand
# in for one here.
SONG = (
    "a birthday line, then another "
    "Happy Birthday to you. Happy Birthday to you."
)


def _sidecar(tmp_path, rows, *, model="whisper-large-v3"):
    """rows: (path, [(start, end, text)], duration_s)."""
    db = tmp_path / "t.db"
    conn = store.connect(db)
    for path, windows, duration in rows:
        segs = json.dumps(
            [{"start": s, "end": e, "text": t, "lang": "en"} for s, e, t in windows]
        )
        text = " ".join(t for _s, _e, t in windows)
        conn.execute(
            "INSERT INTO transcripts (path, duration_s, dropped_windows, text,"
            " languages, segments, model, transcribed_at, elapsed_s, policy)"
            " VALUES (?, ?, 0, ?, ?, ?, ?, '2026-01-01', 1.0, 'old-policy')",
            (path, duration, text, json.dumps(["en"]), segs, model),
        )
    conn.commit()
    return conn


def _text(conn: sqlite3.Connection, path: str) -> str | None:
    row = conn.execute(
        "SELECT text FROM transcripts WHERE path = ?", (path,)
    ).fetchone()
    return row[0] if row else None


# --- what it changes ----------------------------------------------------------


def test_a_loop_between_real_speech_is_removed_without_touching_the_audio(tmp_path):
    conn = _sidecar(
        tmp_path,
        [("/a.mov", [(0, 30, REAL_A), (28, 58, LOOP), (56, 86, REAL_B)], 86.0)],
    )
    stats = refilter_stored(
        conn, policy="new", settings=Settings(), model_name="whisper-large-v3"
    )
    assert stats.windows_dropped == 1
    text = _text(conn, "/a.mov")
    assert text is not None and LOOP[:12] not in text
    assert "harbour" in text and "bakery" in text


def test_a_single_window_song_is_kept(tmp_path):
    """The whole point of the fallback, exercised through the re-filter."""
    conn = _sidecar(tmp_path, [("/b.mov", [(0, 30, SONG)], 30.0)])
    refilter_stored(
        conn, policy="new", settings=Settings(), model_name="whisper-large-v3"
    )
    assert _text(conn, "/b.mov") is not None, "a family recording was deleted"


def test_a_transcript_left_empty_is_demoted_not_left_as_an_empty_string(tmp_path):
    conn = _sidecar(
        tmp_path, [("/c.mov", [(0, 20, "Thank you."), (18, 38, "Thank you.")], 38.0)]
    )
    stats = refilter_stored(
        conn, policy="new", settings=Settings(), model_name="whisper-large-v3"
    )
    assert stats.demoted == 1
    assert _text(conn, "/c.mov") is None
    assert conn.execute("SELECT count(*) FROM no_text").fetchone()[0] == 1


def test_an_untouched_transcript_is_restamped_so_it_is_not_redone(tmp_path):
    conn = _sidecar(tmp_path, [("/d.mov", [(0, 30, REAL_A)], 30.0)])
    refilter_stored(
        conn, policy="new", settings=Settings(), model_name="whisper-large-v3"
    )
    row = conn.execute("SELECT policy FROM transcripts WHERE path='/d.mov'").fetchone()
    assert row[0] == "new"


# --- the guard ----------------------------------------------------------------


def test_a_different_model_is_skipped_rather_than_restamped(tmp_path):
    """The stored windows came from a different decode. Re-filtering them and
    stamping the current policy would claim a run that never happened."""
    conn = _sidecar(tmp_path, [("/e.mov", [(0, 30, LOOP)], 30.0)], model="whisper-tiny")
    stats = refilter_stored(
        conn, policy="new", settings=Settings(), model_name="whisper-large-v3"
    )
    assert stats.skipped_other_model == 1
    assert stats.rejudged == 0
    row = conn.execute("SELECT policy FROM transcripts WHERE path='/e.mov'").fetchone()
    assert row[0] == "old-policy", "a row from another decode was restamped"


def test_a_transcript_with_no_stored_segments_is_skipped(tmp_path):
    """Nothing to re-filter. Restamping would assert the new rules were
    applied to windows that are not there."""
    conn = _sidecar(tmp_path, [("/f.mov", [], 10.0)])
    stats = refilter_stored(
        conn, policy="new", settings=Settings(), model_name="whisper-large-v3"
    )
    assert stats.skipped_no_segments == 1
    row = conn.execute("SELECT policy FROM transcripts WHERE path='/f.mov'").fetchone()
    assert row[0] == "old-policy"


def test_rows_already_on_the_current_policy_ARE_re_filtered(tmp_path):
    """This asserted the opposite, and the opposite was a bug.

    Skipping rows stamped with the current policy is an optimisation that
    assumes the stamp captures the filter's BEHAVIOUR. It does not: the stamp
    is derived from settings, so a fix inside `filter_windows` changes what
    survives while every threshold, and the hash, stay identical.

    That is the exact case this command exists for -- apply a filter change
    without re-decoding -- and the optimisation made it blind to it. Watched
    on a live archive: a decode-loop fix shipped, a 7-hour re-transcribe ran
    with the old code, and `--refilter` reported "0 re-filtered" while three
    recordings still carried a window of 138 chars/s.

    The old test carried no reasoning, unlike its neighbours here, which is
    the tell: it was describing the implementation rather than a property
    anyone wanted.
    """
    conn = _sidecar(tmp_path, [("/g.mov", [(0, 30, LOOP)], 30.0)])
    conn.execute("UPDATE transcripts SET policy = 'new'")
    conn.commit()
    stats = refilter_stored(
        conn, policy="new", settings=Settings(), model_name="whisper-large-v3"
    )
    assert stats.rejudged == 1


def test_overlapping_windows_are_rejoined_without_duplicating_the_seam(tmp_path):
    """`continues_previous` is reconstructed from the geometry. Getting it
    wrong doubles the overlapping words in every rejoined transcript."""
    conn = _sidecar(
        tmp_path,
        [("/h.mov", [(0, 30, "the boats were all out"),
                     (28, 58, "were all out and we walked on")], 58.0)],
    )
    refilter_stored(
        conn, policy="new", settings=Settings(), model_name="whisper-large-v3"
    )
    text = _text(conn, "/h.mov")
    assert text is not None
    assert text.count("were all out") == 1, f"seam duplicated: {text!r}"


def test_non_overlapping_windows_are_joined_with_a_space(tmp_path):
    """A boundary that fell on real silence is not a seam, and trimming it
    would delete real repetition."""
    conn = _sidecar(
        tmp_path, [("/i.mov", [(0, 30, "one two three"), (60, 90, "one two three")], 90.0)]
    )
    refilter_stored(
        conn, policy="new", settings=Settings(), model_name="whisper-large-v3"
    )
    text = _text(conn, "/i.mov")
    assert text is not None and text.count("one two three") == 2


def test_it_costs_no_model_time(tmp_path):
    """The claim the whole feature rests on. No backend is passed, and none
    can be reached."""
    import inspect

    assert "backend" not in inspect.signature(refilter_stored).parameters


def test_the_summary_reports_what_was_not_done(tmp_path):
    conn = _sidecar(
        tmp_path,
        [("/j.mov", [(0, 30, REAL_A)], 30.0)],
        model="whisper-tiny",
    )
    stats = refilter_stored(
        conn, policy="new", settings=Settings(), model_name="whisper-large-v3"
    )
    assert "skipped" in stats.describe().lower()
    assert "1" in stats.describe()


def test_a_row_whose_windows_are_wider_than_window_s_is_skipped(tmp_path):
    """`window_s` decides which audio becomes which window, so a stored
    window wider than it did not come from this geometry and cannot be
    re-filtered as if it had.

    Found on a live sidecar: 104 rows held ONE "window" spanning the whole
    file, up to 2,251 seconds. They were RESTORATION records -- text
    recovered from `no_text` and written back with a synthetic window --
    which `elapsed_s IS NULL` and `lang IS NULL` also show. Treating those as
    per-window data would have re-judged a whole file as a single window.

    Skipped and counted, per row: one unusable row must not abandon the
    other 7,229, and must not be quietly restamped either.
    """
    conn = _sidecar(
        tmp_path,
        [("/k.mov", [(0, 255, REAL_A)], 255.0), ("/l.mov", [(0, 30, REAL_B)], 30.0)],
    )
    stats = refilter_stored(
        conn, policy="new", settings=Settings(), model_name="whisper-large-v3"
    )
    assert stats.skipped_wide_windows == 1
    assert stats.rejudged == 1
    assert (
        conn.execute("SELECT policy FROM transcripts WHERE path='/k.mov'").fetchone()[0]
        == "old-policy"
    ), "an un-refilterable row was restamped"
    assert "skipped" in stats.describe().lower()


# --- a partial re-judge must not claim to be a full one -----------------------


def test_rejudge_does_not_restamp_a_row_whose_windows_it_cannot_refilter(tmp_path):
    """`rejudge_stored` applies the WHOLE-TRANSCRIPT rules to stored text.
    It cannot apply the per-window ones, because a restoration record has no
    real windows -- one synthetic span covering the file.

    Restamping such a row says "this faced the current rules". Half of them
    did. Measured live: 104 rows were restamped that way and every one still
    held loop text at 0.80-0.82, which the per-window rule would have
    stripped -- and being stamped current, they would never be redone.

    Left stale instead, so a real re-transcribe picks them up.
    """
    from corpus.transcripts.run import rejudge_stored

    conn = _sidecar(tmp_path, [("/wide.mov", [(0, 255, REAL_A)], 255.0)])
    rejudged, _demoted = rejudge_stored(
        conn, policy="new", settings=Settings(), model_name="whisper-large-v3"
    )
    row = conn.execute(
        "SELECT policy FROM transcripts WHERE path='/wide.mov'"
    ).fetchone()
    assert row[0] == "old-policy", "a partially-judged row was stamped current"
    assert rejudged == 0


def test_rejudge_still_restamps_a_row_it_can_fully_judge(tmp_path):
    from corpus.transcripts.run import rejudge_stored

    conn = _sidecar(tmp_path, [("/ok.mov", [(0, 30, REAL_A)], 30.0)])
    rejudged, _ = rejudge_stored(
        conn, policy="new", settings=Settings(), model_name="whisper-large-v3"
    )
    assert rejudged == 1
    row = conn.execute("SELECT policy FROM transcripts WHERE path='/ok.mov'").fetchone()
    assert row[0] == "new"


def test_a_row_already_at_the_current_policy_is_still_re_filtered(tmp_path):
    """The policy hash is derived from SETTINGS, not from code.

    `refilter_stored` scoped itself to `stale_transcripts(policy=...)` --
    rows whose stamp differs from the current policy. That assumes the stamp
    captures the filter's behaviour. It does not: a fix to `filter_windows`
    changes what survives while every threshold, and therefore the hash,
    stays identical.

    Watched on a live archive. A decode-loop fix shipped, a 7-hour
    re-transcribe ran with the OLD code, and `--refilter` -- the command that
    exists to apply a filter change without re-decoding -- reported

        0 re-filtered, 0 shortened, 0 windows dropped, 0 demoted

    while three recordings still carried a window of 138 chars/s. The rows
    were stamped current, so the one tool for the job could not see them.

    Re-judging is text-only and idempotent: a row that does not change is not
    rewritten, so examining everything costs a pass over strings.
    """
    loop = "ta lom, e o lomi, a stl, e4,8,1" + ",0" * 101
    conn = _sidecar(tmp_path, [("/w/a.m4a", [(23.8, 25.5, loop)], 64.0)])
    # Stamp it with the CURRENT policy, exactly as a fresh re-transcribe does.
    conn.execute("UPDATE transcripts SET policy = 'current'")
    conn.commit()

    stats = refilter_stored(
        conn, policy="current", settings=Settings(), model_name="whisper-large-v3"
    )

    assert stats.rejudged == 1, (
        f"a row at the current policy was never examined: {stats}"
    )
    assert _text(conn, "/w/a.m4a") in (None, ""), (
        "the impossible-rate window survived a re-filter"
    )


def test_re_filtering_twice_changes_nothing_the_second_time(tmp_path):
    """Examining every row is only acceptable because it is idempotent."""
    conn = _sidecar(
        tmp_path, [("/w/b.m4a", [(0.0, 30.0, "A perfectly ordinary sentence.")], 30.0)]
    )
    conn.execute("UPDATE transcripts SET policy = 'current'")
    conn.commit()

    first = refilter_stored(
        conn, policy="current", settings=Settings(), model_name="whisper-large-v3"
    )
    before = _text(conn, "/w/b.m4a")
    second = refilter_stored(
        conn, policy="current", settings=Settings(), model_name="whisper-large-v3"
    )

    assert _text(conn, "/w/b.m4a") == before
    assert second.windows_dropped == first.windows_dropped == 0
