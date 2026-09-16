"""`corpus-doctor` never looked at the run's own record of what went wrong.

The sidecar's `failures` table is where a transcription pass writes every
file it could not finish -- and the doctor, whose entire job is "are the
instruments trustworthy?", read `no_text`, `dropped_windows`, `transcripts`
and the query log, and not this one.

What that hid, all of it live on one archive:

  - 94 failure rows for files that had since been given a proper verdict,
    left behind because nothing cleared them
  - a file that timed out three passes running while the run reported success
  - timeouts at all: a hang is recorded and then never surfaced anywhere a
    person looks

The point is not that these are catastrophic. It is that the pass RECORDED
each one and the tool that exists to read the record did not read it.
"""

from __future__ import annotations

from corpus.transcripts import store


def _sidecar(tmp_path, failures=(), no_text=(), policy="p"):
    db = tmp_path / "t.db"
    with store.open_store(db) as conn:
        for path, error in failures:
            store.save_failure(conn, path, error)
        for path, reason in no_text:
            store.save_no_text(
                conn, path, duration_s=1.0, reason=reason, policy=policy
            )
    return db


def test_a_clean_sidecar_reports_ok(tmp_path, capsys):
    """The positive control. Without it every assertion below could pass on a
    check that never runs."""
    from corpus.cli.doctor import _check_transcribe_health

    db = _sidecar(tmp_path, no_text=[("/a.mov", "silence")])
    assert _check_transcribe_health(str(db), policy="p") is True
    out = capsys.readouterr().out
    assert "ok" in out


def test_timeouts_are_surfaced(tmp_path, capsys):
    from corpus.cli.doctor import _check_transcribe_health

    db = _sidecar(
        tmp_path,
        failures=[("/slow.mov", "timed out after 162s"),
                  ("/broken.mov", "OSError: moov atom not found")],
    )
    _check_transcribe_health(str(db), policy="p")
    out = capsys.readouterr().out
    assert "timed out" in out.lower() or "timeout" in out.lower()
    assert "slow.mov" in out


def test_files_written_off_after_repeated_timeouts_are_named(tmp_path, capsys):
    """A `repeatedly_timed_out` verdict is the run saying it gave up. That is
    exactly the thing a person should see, and it was recorded where nobody
    looked."""
    from corpus.cli.doctor import _check_transcribe_health

    db = _sidecar(tmp_path, no_text=[("/hangs.mov", "repeatedly_timed_out")])
    _check_transcribe_health(str(db), policy="p")
    out = capsys.readouterr().out
    assert "hangs.mov" in out
    assert "gave up" in out.lower() or "repeatedly" in out.lower()


def test_stale_failure_rows_are_reported_as_stale_not_as_failures(tmp_path, capsys):
    """94 of these read as 94 broken files, and there were none."""
    from corpus.cli.doctor import _check_transcribe_health

    db = tmp_path / "t.db"
    with store.open_store(db) as conn:
        store.save_failure(conn, "/settled.mov", "old crash")
        store.save_no_text(
            conn, "/settled.mov", duration_s=1.0, reason="silence", policy="p"
        )
    _check_transcribe_health(str(db), policy="p")
    out = capsys.readouterr().out
    assert "stale" in out.lower()
    assert "1" in out


def test_a_missing_sidecar_is_skipped_not_reported_clean(tmp_path, capsys):
    """"I could not look" and "I looked and it was fine" are the two facts
    this command exists to keep apart."""
    from corpus.cli.doctor import _check_transcribe_health

    assert _check_transcribe_health(None, policy="p") is True
    assert "SKIPPED" in capsys.readouterr().out


def test_it_works_on_a_sidecar_that_predates_the_attempts_column(tmp_path, capsys):
    """The doctor is a READ-ONLY diagnostic, and column migrations only run
    on a read-write open. So a sidecar written by an older version has no
    `attempts` column, the query raised OperationalError, and the check
    reported SKIPPED -- on the live archive, on its first run.

    A diagnostic that requires a migration to have happened cannot diagnose
    the state it most needs to: the one before anyone upgraded.
    """
    import sqlite3

    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE failures (path TEXT PRIMARY KEY, error TEXT NOT NULL,
                               failed_at TEXT NOT NULL);
        CREATE TABLE transcripts (path TEXT PRIMARY KEY, policy TEXT);
        CREATE TABLE no_text (path TEXT PRIMARY KEY, reason TEXT, policy TEXT);
        """
    )
    conn.execute(
        "INSERT INTO failures VALUES ('/slow.mov', 'timed out after 90s', 'now')"
    )
    conn.commit()
    conn.close()

    from corpus.cli.doctor import _check_transcribe_health

    assert _check_transcribe_health(str(db), policy="p") is True
    out = capsys.readouterr().out
    assert "SKIPPED" not in out, "the check refused to run on an old sidecar"
    assert "slow.mov" in out
