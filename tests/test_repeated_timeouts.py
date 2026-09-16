"""A file that has never once worked should stop costing time every run.

A timeout is recorded as a FAILURE rather than a settled verdict, and
failures are deliberately retried: a hang is usually about the run --
thermal state, memory pressure, a transient decode loop -- not the file.

That reasoning holds for the first hang and stops holding for the third. One
recording on a live archive hung three passes running: 45 minutes before the
deadline existed, then 162 seconds, then 162 seconds again. Retrying it
forever buys nothing and costs the deadline every time.

So `failures` counts attempts, and a file that has hit its deadline
`MAX_TIMEOUT_ATTEMPTS` times is recorded as `no_text` with a reason that says
exactly that. It carries a policy fingerprint like any other verdict, so
changing a rule brings it back -- this is "we asked three times", not "this
file is worthless".

Only TIMEOUTS count. A file that failed three times for three different
reasons has not told us anything consistent.
"""

from __future__ import annotations

from pathlib import Path

from corpus.transcripts import store
from corpus.transcripts.pipeline import Settings
from corpus.transcripts.run import MAX_TIMEOUT_ATTEMPTS, transcribe_directory


class _Backend:
    model_name = "test-model"


def _hangs(path, backend, *, settings=None):
    import time

    time.sleep(30)


def test_attempts_are_counted(tmp_path):
    db = tmp_path / "t.db"
    with store.open_store(db) as conn:
        store.save_failure(conn, "/a.mov", "timed out after 10s")
        store.save_failure(conn, "/a.mov", "timed out after 10s")
        assert store.failure_attempts(conn, "/a.mov") == 2


def test_a_different_error_resets_the_count(tmp_path):
    """Three different failures are not three of the same failure."""
    db = tmp_path / "t.db"
    with store.open_store(db) as conn:
        store.save_failure(conn, "/a.mov", "timed out after 10s")
        store.save_failure(conn, "/a.mov", "moov atom not found")
        assert store.failure_attempts(conn, "/a.mov") == 1


def test_a_file_that_keeps_timing_out_is_eventually_settled(tmp_path):
    (tmp_path / "hangs.mov").write_bytes(b"x")
    db = tmp_path / "t.db"

    for _ in range(MAX_TIMEOUT_ATTEMPTS):
        transcribe_directory(
            tmp_path, db, _Backend(), settings=Settings(),
            transcribe=_hangs, file_timeout_s=0.2,
        )

    with store.open_store(db, read_only=True) as conn:
        row = conn.execute("SELECT reason FROM no_text").fetchone()
        assert row is not None, "still being retried after every attempt hung"
        assert row[0] == "repeatedly_timed_out"


def test_it_is_then_skipped_rather_than_retried(tmp_path):
    (tmp_path / "hangs.mov").write_bytes(b"x")
    db = tmp_path / "t.db"
    for _ in range(MAX_TIMEOUT_ATTEMPTS):
        transcribe_directory(
            tmp_path, db, _Backend(), settings=Settings(),
            transcribe=_hangs, file_timeout_s=0.2,
        )

    attempted: list[Path] = []

    def watch(path, backend, *, settings=None):
        attempted.append(Path(path))
        raise AssertionError("should not have been attempted")

    transcribe_directory(
        tmp_path, db, _Backend(), settings=Settings(),
        transcribe=watch, file_timeout_s=0.2,
    )
    assert attempted == [], "a settled file was decoded again"


def test_one_or_two_hangs_are_still_retried(tmp_path):
    """The negative control. Writing a file off on its first bad night is
    exactly the mistake this is trying not to make."""
    (tmp_path / "hangs.mov").write_bytes(b"x")
    db = tmp_path / "t.db"
    transcribe_directory(
        tmp_path, db, _Backend(), settings=Settings(),
        transcribe=_hangs, file_timeout_s=0.2,
    )
    with store.open_store(db, read_only=True) as conn:
        assert conn.execute("SELECT count(*) FROM no_text").fetchone()[0] == 0
        policy = store.policy_fingerprint(Settings().as_policy("test-model"))
        assert str(tmp_path / "hangs.mov") not in store.already_done(
            conn, policy=policy
        )


def test_a_rule_change_brings_it_back(tmp_path):
    """"We asked three times", not "this file is worthless". The verdict
    carries a policy fingerprint like any other."""
    (tmp_path / "hangs.mov").write_bytes(b"x")
    db = tmp_path / "t.db"
    for _ in range(MAX_TIMEOUT_ATTEMPTS):
        transcribe_directory(
            tmp_path, db, _Backend(), settings=Settings(),
            transcribe=_hangs, file_timeout_s=0.2,
        )
    with store.open_store(db, read_only=True) as conn:
        changed = store.policy_fingerprint(
            Settings(max_looping_share=0.5).as_policy("test-model")
        )
        assert str(tmp_path / "hangs.mov") not in store.already_done(
            conn, policy=changed
        )
