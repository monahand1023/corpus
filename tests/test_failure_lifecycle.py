"""A `failures` row must not outlive the failure.

Two defects, both found on a live archive and both of the same shape --
a record that says something which stopped being true.

STALE FAILURE ROWS. `failures` is deliberately excluded from
`already_done`, so a failed file is retried on the next run. Nothing ever
DELETED the row when that retry succeeded. A live sidecar held 94 failure
rows for files that had since been given a proper verdict: the table read
as 94 broken files and there were none. `corpus-doctor` and any operator
reading it are told about work that is already done.

DISPLAYED AS AN ERROR. A file with no audio track is recorded correctly as
`no_text`, and the progress line still printed `ERR` because the callback
was handed `None` -- the same value used for a real crash. 94 of them in a
row look exactly like a run falling over.
"""

from __future__ import annotations

from corpus.transcripts import store
from corpus.transcripts.pipeline import Settings
from corpus.transcripts.run import transcribe_directory


class _Backend:
    model_name = "test-model"


def _outcome(*, text: str | None = None, reason: str = ""):
    o = type("O", (), {})()
    o.duration_s = 1.0
    o.dropped = []
    o.elapsed_s = 0.1
    o.empty_reason = reason
    o.rejected_text = ""
    o.produced_text = text is not None
    if text is None:
        o.transcript = None
    else:
        o.transcript = store.Transcript(
            path="x", text=text, windows=[], duration_s=1.0, model="test-model"
        )
    return o


def test_a_failure_row_is_cleared_when_the_file_later_succeeds(tmp_path):
    media = tmp_path / "flaky.mov"
    media.write_bytes(b"x")
    db = tmp_path / "t.db"
    attempts = {"n": 0}

    def flaky(path, backend, *, settings=None):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise OSError("transient read error")
        o = _outcome(text="it worked the second time")
        o.transcript.path = str(path)
        return o

    transcribe_directory(tmp_path, db, _Backend(), settings=Settings(), transcribe=flaky)
    with store.open_store(db, read_only=True) as conn:
        assert conn.execute("SELECT count(*) FROM failures").fetchone()[0] == 1

    transcribe_directory(tmp_path, db, _Backend(), settings=Settings(), transcribe=flaky)
    with store.open_store(db, read_only=True) as conn:
        assert conn.execute("SELECT count(*) FROM failures").fetchone()[0] == 0, (
            "the failure row outlived the failure"
        )


def test_a_failure_row_is_cleared_when_the_file_gets_a_no_text_verdict(tmp_path):
    """A settled "nothing here" is an answer too. Leaving the failure row
    beside it says the file is still broken."""
    media = tmp_path / "silent.mov"
    media.write_bytes(b"x")
    db = tmp_path / "t.db"
    attempts = {"n": 0}

    def flaky(path, backend, *, settings=None):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise OSError("transient read error")
        return _outcome(reason="silence")

    transcribe_directory(tmp_path, db, _Backend(), settings=Settings(), transcribe=flaky)
    transcribe_directory(tmp_path, db, _Backend(), settings=Settings(), transcribe=flaky)
    with store.open_store(db, read_only=True) as conn:
        assert conn.execute("SELECT count(*) FROM failures").fetchone()[0] == 0


def test_a_file_that_still_fails_keeps_its_row(tmp_path):
    """The negative control. Clearing on every attempt would empty the table
    and make a genuinely broken file invisible."""
    (tmp_path / "broken.mov").write_bytes(b"x")
    db = tmp_path / "t.db"

    def always(path, backend, *, settings=None):
        raise OSError("still broken")

    for _ in range(2):
        transcribe_directory(
            tmp_path, db, _Backend(), settings=Settings(), transcribe=always
        )
    with store.open_store(db, read_only=True) as conn:
        assert conn.execute("SELECT count(*) FROM failures").fetchone()[0] == 1


# --- the progress line ---------------------------------------------------------


def test_a_file_with_no_audio_track_is_not_displayed_as_an_error(tmp_path):
    """It is recorded correctly as no_text, and the progress line still said
    ERR because the callback was handed None -- the same value a real crash
    passes. Ninety-four in a row look like a run falling over."""
    from corpus.transcripts.audio import NoAudioStreamError

    (tmp_path / "silent.mp4").write_bytes(b"x")
    db = tmp_path / "t.db"
    marks: list[object] = []

    def boom(path, backend, *, settings=None):
        raise NoAudioStreamError("no audio stream")

    transcribe_directory(
        tmp_path, db, _Backend(), settings=Settings(), transcribe=boom,
        on_progress=lambda i, n, p, outcome: marks.append(outcome),
    )
    assert marks, "progress was never reported"
    assert marks[0] is not None, "a settled verdict was reported as a crash"
    assert getattr(marks[0], "produced_text", None) is False


def test_a_real_crash_is_still_displayed_as_an_error(tmp_path):
    (tmp_path / "broken.mov").write_bytes(b"x")
    db = tmp_path / "t.db"
    marks: list[object] = []

    def boom(path, backend, *, settings=None):
        raise OSError("genuinely broken")

    transcribe_directory(
        tmp_path, db, _Backend(), settings=Settings(), transcribe=boom,
        on_progress=lambda i, n, p, outcome: marks.append(outcome),
    )
    assert marks == [None], "a crash stopped being visible as one"


# --- rows that will never be revisited -----------------------------------------


def test_failure_rows_for_already_settled_files_are_swept_on_startup(tmp_path):
    """Clearing on write is not enough. A file that failed once and was later
    given a verdict is SETTLED -- `already_done` skips it, so it is never
    processed again and its stale failure row is never reached.

    That is exactly the live state: 94 rows for files that had since been
    recorded as `no_audio_stream`, which the run will now never revisit. The
    repair has to be a sweep, not a side effect of processing.
    """
    db = tmp_path / "t.db"
    policy = store.policy_fingerprint(Settings().as_policy("test-model"))
    with store.open_store(db) as conn:
        store.save_failure(conn, "/settled-no-text.mov", "old crash")
        store.save_failure(conn, "/settled-transcript.mov", "old crash")
        store.save_failure(conn, "/still-broken.mov", "genuinely broken")
        store.save_no_text(
            conn, "/settled-no-text.mov", duration_s=1.0,
            reason="no_audio_stream", policy=policy,
        )
        store.save_transcript(
            conn,
            store.Transcript(
                path="/settled-transcript.mov", text="real speech", windows=[],
                duration_s=1.0, model="test-model", policy=policy,
            ),
        )
        assert conn.execute("SELECT count(*) FROM failures").fetchone()[0] == 3

        swept = store.clear_settled_failures(conn, policy=policy)
        assert swept == 2
        rows = [r[0] for r in conn.execute("SELECT path FROM failures")]
        assert rows == ["/still-broken.mov"], (
            "a genuinely broken file lost its row, or a settled one kept it"
        )


def test_the_sweep_runs_as_part_of_a_normal_pass(tmp_path):
    """It has to be automatic. A repair someone must remember is a repair
    that does not happen."""
    (tmp_path / "x.mov").write_bytes(b"x")
    db = tmp_path / "t.db"
    policy = store.policy_fingerprint(Settings().as_policy("test-model"))
    with store.open_store(db) as conn:
        store.save_failure(conn, "/gone.mov", "old crash")
        store.save_no_text(
            conn, "/gone.mov", duration_s=1.0, reason="silence", policy=policy
        )

    transcribe_directory(
        tmp_path, db, _Backend(), settings=Settings(),
        transcribe=lambda p, b, *, settings=None: _outcome(reason="silence"),
    )
    with store.open_store(db, read_only=True) as conn:
        assert conn.execute("SELECT count(*) FROM failures").fetchone()[0] == 0
