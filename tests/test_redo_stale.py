"""Redo what a policy change invalidated, without rediscovering the world.

`transcribe_directory` walks a tree and transcribes what it finds. After a
threshold change that is the wrong operation on an established archive: the
files to redo are ALREADY KNOWN — they are the rows the new policy
invalidated — and re-walking rediscovers everything the archive deliberately
excluded.

Measured on a live archive: its media roots hold ~58,000 photo-library
videos, 81% of them the sub-4-second clip Apple stores beside each Live
Photo. Its own filters exclude those, plus karaoke backing tracks and a
15-second floor. A blind re-walk would have queued ~46,800 near-empty clips
for ~33 hours of room tone, and none of those exclusions live in corpus.

The sidecar's contents already encode every one of those decisions. So
`stale_paths` reads the work list from the store rather than the filesystem.
"""

from __future__ import annotations

from pathlib import Path

from corpus.transcripts import store
from corpus.transcripts.run import stale_paths


def _sidecar(tmp_path, transcripts, no_texts):
    db = tmp_path / "t.db"
    conn = store.connect(db)
    for path, policy in transcripts:
        conn.execute(
            "INSERT INTO transcripts (path, duration_s, dropped_windows, text,"
            " languages, segments, model, transcribed_at, elapsed_s, policy)"
            " VALUES (?, 10.0, 0, 'hello', '[]', '[]', 'm', '2026-01-01', 1.0, ?)",
            (path, policy),
        )
    for path, policy in no_texts:
        conn.execute(
            "INSERT INTO no_text (path, duration_s, reason, policy, checked_at)"
            " VALUES (?, 10.0, 'no_speech_detected', ?, '2026-01-01')",
            (path, policy),
        )
    conn.commit()
    return conn


def test_only_rows_the_new_policy_invalidated_are_returned(tmp_path):
    conn = _sidecar(
        tmp_path,
        [("/a.mov", "current"), ("/b.mov", "old")],
        [("/c.mov", "current"), ("/d.mov", "old")],
    )
    assert stale_paths(conn, policy="current") == [Path("/b.mov"), Path("/d.mov")]


def test_rejected_files_are_included_not_just_transcripts(tmp_path):
    """A `no_text` verdict is a verdict. Redoing only the transcripts leaves
    every rejection frozen under rules that no longer apply."""
    conn = _sidecar(tmp_path, [], [("/only-rejected.mov", "old")])
    assert stale_paths(conn, policy="current") == [Path("/only-rejected.mov")]


def test_a_path_recorded_in_both_tables_appears_once(tmp_path):
    conn = _sidecar(tmp_path, [("/x.mov", "old")], [("/x.mov", "old")])
    assert stale_paths(conn, policy="current") == [Path("/x.mov")]


def test_an_archive_already_on_the_current_policy_has_nothing_to_redo(tmp_path):
    conn = _sidecar(tmp_path, [("/a.mov", "cur")], [("/b.mov", "cur")])
    assert stale_paths(conn, policy="cur") == []


def test_files_missing_from_disk_are_dropped_with_a_count(tmp_path):
    """An unmounted volume must not become thousands of recorded failures.
    The archive spans external drives; a path that is not there right now is
    not a broken file."""
    from corpus.transcripts.run import stale_paths_present

    present = tmp_path / "here.mov"
    present.write_bytes(b"x")
    conn = _sidecar(tmp_path, [(str(present), "old"), ("/gone/away.mov", "old")], [])
    paths, missing = stale_paths_present(conn, policy="current")
    assert paths == [present]
    assert missing == 1


def test_the_work_list_comes_from_the_store_not_the_filesystem(tmp_path):
    """The point of the whole function: a file on disk that the archive
    never chose to transcribe stays unchosen."""
    stranger = tmp_path / "never-indexed.mov"
    stranger.write_bytes(b"x")
    conn = _sidecar(tmp_path, [("/a.mov", "old")], [])
    assert stranger not in stale_paths(conn, policy="current")


def test_since_limits_the_redo_to_rows_written_on_or_after_a_date(tmp_path):
    """A windowing change invalidates every row's policy, but only the rows a
    given pipeline wrote carry its defect. Redoing an archive's older rows
    too would triple the GPU time for no change in what they say."""
    conn = _sidecar(tmp_path, [("/old.mov", "old")], [("/old-empty.mov", "old")])
    conn.execute(
        "INSERT INTO transcripts (path, duration_s, dropped_windows, text, languages,"
        " segments, model, transcribed_at, elapsed_s, policy)"
        " VALUES ('/new.mov', 10.0, 0, 'hi', '[]', '[]', 'm', '2026-09-20T00:00:00+00:00', 1.0, 'old')"
    )
    conn.execute(
        "UPDATE transcripts SET transcribed_at = '2026-09-01T00:00:00+00:00' WHERE path = '/old.mov'"
    )
    assert stale_paths(conn, policy="current", since="2026-09-14") == [Path("/new.mov")]
    assert len(stale_paths(conn, policy="current")) == 3
