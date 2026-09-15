"""Tests for the transcription sidecar database.

Every property here exists because a long run needed it. The expensive ones to
get wrong are resumption (a restart that re-does finished work) and policy
fingerprinting (a stored verdict outliving the rule that produced it).
"""

from __future__ import annotations

import contextlib
import sqlite3
from pathlib import Path

from corpus.transcripts.store import (
    Transcript,
    Window,
    already_done,
    connect,
    counts,
    migrate,
    open_store,
    policy_fingerprint,
    save_dropped_windows,
    save_failure,
    save_no_text,
    save_transcript,
)

POLICY = policy_fingerprint({"model": "m", "threshold": 0.5})


def _transcript(path: str = "/a.mov") -> Transcript:
    return Transcript(
        path=path,
        text="we went to the park and fed the ducks",
        windows=[
            Window(0.0, 30.0, "we went to the park", "en"),
            Window(30.0, 60.0, "and fed the ducks", "en"),
        ],
        duration_s=60.0,
        model="test-model",
        policy=POLICY,
    )


# --- round-tripping --------------------------------------------------------


def test_a_transcript_survives_a_round_trip(tmp_path: Path) -> None:
    db = tmp_path / "t.db"
    with open_store(db) as conn:
        save_transcript(conn, _transcript())
    with open_store(db, read_only=True) as conn:
        row = conn.execute("SELECT * FROM transcripts").fetchone()
    assert row["path"] == "/a.mov"
    assert row["text"].startswith("we went")
    assert row["model"] == "test-model"


def test_per_window_language_and_timing_are_preserved(tmp_path: Path) -> None:
    # Window boundaries carry the language and the timestamp, which is what
    # lets a search hit say WHERE in a two-hour recording it came from and
    # keeps a Japanese passage from straddling an English one.
    import json

    db = tmp_path / "t.db"
    with open_store(db) as conn:
        save_transcript(conn, _transcript())
        row = conn.execute("SELECT segments, languages FROM transcripts").fetchone()
    segments = json.loads(row["segments"])
    assert [s["start"] for s in segments] == [0.0, 30.0]
    assert json.loads(row["languages"]) == ["en", "en"]


def test_re_saving_the_same_path_replaces_rather_than_duplicates(
    tmp_path: Path,
) -> None:
    db = tmp_path / "t.db"
    with open_store(db) as conn:
        save_transcript(conn, _transcript())
        save_transcript(conn, _transcript())
        assert counts(conn)["transcripts"] == 1


# --- resumption: the property that makes a long run restartable -------------


def test_a_finished_file_is_not_done_twice(tmp_path: Path) -> None:
    with open_store(tmp_path / "t.db") as conn:
        save_transcript(conn, _transcript("/done.mov"))
        assert "/done.mov" in already_done(conn, policy=POLICY)


def test_a_file_with_no_speech_is_remembered_so_it_is_not_retried(
    tmp_path: Path,
) -> None:
    # Without this row the file is simply ABSENT from transcripts, so the next
    # run re-decodes and re-transcribes it in full to reach the same answer.
    # One interrupted pass re-paid that for 889 already-examined silent clips.
    with open_store(tmp_path / "t.db") as conn:
        save_no_text(conn, "/silent.mov", duration_s=12.0, policy=POLICY,
                     reason="no_speech_detected")
        assert "/silent.mov" in already_done(conn, policy=POLICY)


def test_a_no_text_verdict_expires_when_the_rules_change(tmp_path: Path) -> None:
    # The whole point of the fingerprint. A verdict reached under one set of
    # thresholds must not silently survive a change to them.
    with open_store(tmp_path / "t.db") as conn:
        save_no_text(conn, "/silent.mov", duration_s=12.0, policy=POLICY)
        stricter = policy_fingerprint({"model": "m", "threshold": 0.9})
        assert "/silent.mov" not in already_done(conn, policy=stricter)


def test_a_failure_is_retried_rather_than_treated_as_settled(tmp_path: Path) -> None:
    # A failure usually means a broken decode or a transient resource problem,
    # not a settled verdict about the audio. Skipping it forever would bury a
    # file that a rerun might handle fine.
    with open_store(tmp_path / "t.db") as conn:
        save_failure(conn, "/broken.mov", "ffmpeg exited 1")
        assert "/broken.mov" not in already_done(conn, policy=POLICY)


# --- the audit trail -------------------------------------------------------


def test_discarded_windows_keep_the_evidence_they_were_judged_on(
    tmp_path: Path,
) -> None:
    # Without these rows the filter is unauditable: thousands of destructive
    # decisions and only a COUNT of them, so precision cannot be measured at
    # all.
    with open_store(tmp_path / "t.db") as conn:
        save_dropped_windows(
            conn,
            "/a.mov",
            [{"window_start": 30.0, "no_speech": 0.78, "avg_logprob": -0.24,
              "text": "Thank you."}],
            policy=POLICY,
        )
        row = conn.execute("SELECT * FROM dropped_windows").fetchone()
    assert row["no_speech"] == 0.78
    assert row["avg_logprob"] == -0.24
    assert row["text"] == "Thank you."


def test_windows_are_kept_for_a_file_that_produced_no_transcript(
    tmp_path: Path,
) -> None:
    # Keyed on path, not on a transcript row, precisely so a file whose windows
    # were ALL discarded still leaves evidence. Those are the most interesting
    # cases to audit and would otherwise be the ones that vanish.
    with open_store(tmp_path / "t.db") as conn:
        save_dropped_windows(
            conn, "/all-junk.mov",
            [{"window_start": 0.0, "no_speech": 0.9, "avg_logprob": -0.1,
              "text": "Thanks for watching"}],
            policy=POLICY,
        )
        assert counts(conn)["transcripts"] == 0
        assert counts(conn)["dropped_windows"] == 1


# --- policy fingerprinting -------------------------------------------------


def test_the_same_settings_always_fingerprint_the_same(tmp_path: Path) -> None:
    a = policy_fingerprint({"model": "m", "langs": {"en", "ja"}})
    b = policy_fingerprint({"langs": {"ja", "en"}, "model": "m"})
    assert a == b, "key order and set order must not change the fingerprint"


def test_changing_any_setting_changes_the_fingerprint() -> None:
    base = {"model": "m", "vad_speech_prob": 0.2, "max_chars_per_second": 25.0}
    for key, value in (
        ("model", "other"),
        ("vad_speech_prob", 0.3),
        ("max_chars_per_second", 30.0),
    ):
        assert policy_fingerprint({**base, key: value}) != policy_fingerprint(base), (
            f"{key} must participate -- a setting that can reject audio and "
            "does not change the fingerprint leaves stale verdicts in place"
        )


def test_a_set_valued_setting_is_order_independent() -> None:
    # Phrase lists and language sets are the settings most likely to be edited,
    # and an unstable hash would retry the whole archive on every reorder.
    a = policy_fingerprint({"phrases": {"a", "b", "c"}})
    b = policy_fingerprint({"phrases": {"c", "a", "b"}})
    assert a == b


# --- schema and migration --------------------------------------------------


def test_opening_an_existing_database_twice_is_safe(tmp_path: Path) -> None:
    db = tmp_path / "t.db"
    with open_store(db) as conn:
        save_transcript(conn, _transcript())
    with open_store(db) as conn:
        assert counts(conn)["transcripts"] == 1


def test_a_database_from_an_earlier_version_gains_the_new_columns(
    tmp_path: Path,
) -> None:
    # CREATE TABLE IF NOT EXISTS leaves an existing table alone, so a database
    # made before a column was added keeps the old shape unless migrated.
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        "CREATE TABLE transcripts (path TEXT PRIMARY KEY, duration_s REAL,"
        " dropped_windows INTEGER NOT NULL DEFAULT 0, text TEXT NOT NULL,"
        " languages TEXT NOT NULL, segments TEXT NOT NULL, model TEXT NOT NULL,"
        " transcribed_at TEXT NOT NULL, elapsed_s REAL);"
        "CREATE TABLE no_text (path TEXT PRIMARY KEY, duration_s REAL NOT NULL,"
        " policy TEXT NOT NULL, checked_at TEXT NOT NULL);"
    )
    conn.commit()
    applied = migrate(conn)
    conn.close()

    assert "transcripts.policy" in applied
    assert "no_text.reason" in applied
    with open_store(db) as fresh:
        save_transcript(fresh, _transcript())
        assert counts(fresh)["transcripts"] == 1


def test_a_read_only_open_does_not_create_the_file(tmp_path: Path) -> None:
    # A reader must never bring a database into existence: an empty one looks
    # exactly like an archive with nothing in it, and the ingester would treat
    # every indexed transcript as an orphan.
    missing = tmp_path / "nope.db"
    with contextlib.suppress(sqlite3.Error):
        connect(missing, read_only=True)
    assert not missing.exists()
