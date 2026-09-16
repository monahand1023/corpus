"""A NaN from the model must not take the run with it.

WHAT HAPPENED. Whisper returns `avg_logprob` per window, and on degenerate
audio it can be NaN. **SQLite has no NaN: inserting one stores NULL.** The
column is NOT NULL, so the insert raised `IntegrityError`, and because the
store write sits OUTSIDE the per-file exception handling, ONE bad window
aborted the whole pass. A live re-transcribe died at file 59 of 2,454 with

    sqlite3.IntegrityError: NOT NULL constraint failed: dropped_windows.avg_logprob

Two separate defects, and the second is the worse one: a run that dies on
file 59 loses nothing already written, but it stops, and the only sign is a
traceback in a log nobody is watching.
"""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path

from corpus.transcripts import store
from corpus.transcripts.pipeline import DroppedWindow, Outcome, Settings
from corpus.transcripts.run import transcribe_directory


class _Backend:
    model_name = "test-model"


def test_sqlite_really_does_turn_nan_into_null():
    """The premise, pinned. If this ever stops being true the coercion below
    becomes dead weight and someone should know why it was added."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (x REAL)")
    conn.execute("INSERT INTO t VALUES (?)", (float("nan"),))
    assert conn.execute("SELECT x FROM t").fetchone()[0] is None


def test_a_nan_logprob_is_stored_not_rejected(tmp_path):
    db = tmp_path / "t.db"
    with store.open_store(db) as conn:
        store.save_dropped_windows(
            conn,
            "/a.mov",
            [{
                "window_start": 0.0,
                "no_speech": float("nan"),
                "avg_logprob": float("nan"),
                "text": "junk",
                "reason": "looping_repetition",
            }],
            policy="p",
        )
        row = conn.execute(
            "SELECT no_speech, avg_logprob, reason FROM dropped_windows"
        ).fetchone()
        assert row is not None, "the window was not recorded at all"
        assert row[2] == "looping_repetition", "the evidence lost its reason"
        assert not math.isnan(row[0]) and not math.isnan(row[1])


def test_an_infinite_logprob_is_stored_too(tmp_path):
    """-inf is what a logprob degenerates to before it becomes NaN. SQLite
    accepts it, so it would not crash -- it would sit in the table as a value
    no comparison behaves sensibly against."""
    db = tmp_path / "t.db"
    with store.open_store(db) as conn:
        store.save_dropped_windows(
            conn, "/b.mov",
            [{"window_start": 0.0, "no_speech": 0.5,
              "avg_logprob": float("-inf"), "text": "x", "reason": "r"}],
            policy="p",
        )
        value = conn.execute("SELECT avg_logprob FROM dropped_windows").fetchone()[0]
        assert math.isfinite(value)


def test_a_real_value_is_untouched(tmp_path):
    """The negative control: coercion must not flatten the evidence this
    table exists to keep."""
    db = tmp_path / "t.db"
    with store.open_store(db) as conn:
        store.save_dropped_windows(
            conn, "/c.mov",
            [{"window_start": 1.5, "no_speech": 0.805,
              "avg_logprob": -0.200, "text": "x", "reason": "r"}],
            policy="p",
        )
        row = conn.execute(
            "SELECT window_start, no_speech, avg_logprob FROM dropped_windows"
        ).fetchone()
        assert tuple(row) == (1.5, 0.805, -0.200)


# --- the run must survive one bad file ----------------------------------------


def test_a_storage_error_on_one_file_does_not_end_the_run(tmp_path, monkeypatch):
    """The defect that actually cost the pass. The store write sits outside
    the per-file exception handling, so an IntegrityError on file 59 aborted
    the remaining 2,395."""
    for name in ("a.mov", "poison.mov", "c.mov"):
        (tmp_path / name).write_bytes(b"x")
    db = tmp_path / "t.db"
    seen: list[str] = []

    def fake(path, backend, *, settings=None):
        seen.append(Path(path).name)
        o = Outcome(path=str(path), empty_reason="silence")
        o.duration_s = 1.0
        o.dropped = [
            DroppedWindow(
                window_start=0.0, no_speech=0.1, avg_logprob=-0.2,
                text="x", reason="r",
            )
        ]
        return o

    # The store itself raises for one file. Injected rather than crafted from
    # a poisonous VALUE, because which values are unstorable is exactly what
    # keeps changing -- NaN was one nobody had thought of. What must hold is
    # that ANY failure here costs one file.
    import corpus.transcripts.run as run_mod

    real_save = run_mod.store.save_dropped_windows

    def flaky_save(conn, path, dropped, *, policy):
        if "poison" in str(path):
            raise sqlite3.IntegrityError("NOT NULL constraint failed: x.y")
        return real_save(conn, path, dropped, policy=policy)

    monkeypatch.setattr(run_mod.store, "save_dropped_windows", flaky_save)
    stats = transcribe_directory(
        tmp_path, db, _Backend(), settings=Settings(), transcribe=fake
    )
    assert len(seen) == 3, f"the run stopped at the poisoned file: {seen}"
    assert stats.failed >= 1
    with store.open_store(db, read_only=True) as conn:
        row = conn.execute(
            "SELECT error FROM failures WHERE path LIKE '%poison.mov'"
        ).fetchone()
        assert row is not None, "the storage error was not recorded against the file"
