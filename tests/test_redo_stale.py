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
    paths, missing, _ = stale_paths_present(conn, policy="current")
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


# --- a decode change must re-decode, not only re-judge ------------------------------


def test_a_transcript_restamped_by_a_rejudge_is_still_stale_if_its_decode_differs(tmp_path):
    """A re-judge re-reads stored TEXT and restamps the policy. That is valid
    only while the decode (model, windows, languages) is unchanged; after a
    windowing change it marked 4,393 transcripts current without re-decoding
    any of them, so --redo-stale then skipped the very files it was run for."""
    conn = _sidecar(tmp_path, [("/restamped.mov", "current"), ("/fine.mov", "current")], [])
    conn.execute("UPDATE transcripts SET decode_policy = 'decode-now' WHERE path = '/fine.mov'")
    conn.commit()
    assert stale_paths(conn, policy="current", decode_policy="decode-now") == [Path("/restamped.mov")]


def test_redo_retranscribes_files_that_already_have_a_transcript(tmp_path):
    from corpus.transcripts.pipeline import Settings
    from corpus.transcripts.run import transcribe_directory

    media = tmp_path / "a.mov"
    media.write_bytes(b"x")
    db = tmp_path / "t.db"
    conn = store.connect(db)
    conn.execute(
        "INSERT INTO transcripts (path, duration_s, dropped_windows, text, languages,"
        " segments, model, transcribed_at, elapsed_s, policy)"
        " VALUES (?, 10.0, 0, 'old text', '[]', '[]', 'm', '2026-09-20', 1.0, 'x')",
        (str(media),),
    )
    conn.commit()
    conn.close()
    calls = []

    def fake(path, backend, *, settings=None):
        calls.append(path)
        outcome = type("O", (), {})()
        outcome.duration_s, outcome.dropped, outcome.elapsed_s = 10.0, [], 0.1
        outcome.transcript = store.Transcript(
            path=str(path), text="new text",
            windows=[store.Window(0.0, 10.0, "new text", "en")], duration_s=10.0, model="m",
        )
        outcome.empty_reason, outcome.rejected_text = None, ""
        return outcome

    class _Backend:
        model_name = "m"

    transcribe_directory(tmp_path, db, _Backend(), settings=Settings(), transcribe=fake, only=[media])
    assert calls == [], "without redo, a file with a transcript is skipped as before"

    transcribe_directory(
        tmp_path, db, _Backend(), settings=Settings(), transcribe=fake, only=[media], redo=True
    )
    assert calls == [media]
    with store.open_store(db, read_only=True) as c:
        row = c.execute("SELECT text, decode_policy FROM transcripts").fetchone()
    expected = store.policy_fingerprint(Settings().as_decode_policy("m"))
    assert (row["text"], row["decode_policy"]) == ("new text", expected)


def test_a_demoted_transcript_keeps_the_decode_it_came_from(tmp_path):
    conn = _sidecar(tmp_path, [("/a.mov", "old")], [])
    conn.execute("UPDATE transcripts SET decode_policy = 'decode-then' WHERE path = '/a.mov'")
    store.demote_transcript(conn, "/a.mov", duration_s=10.0, policy="new", reason="r", rejected_text="t")
    assert conn.execute("SELECT decode_policy FROM no_text WHERE path = '/a.mov'").fetchone()[0] == "decode-then"


def test_a_sidecar_from_before_the_decode_column_is_all_stale_not_empty(tmp_path):
    """Opened read-only, an older sidecar is not migrated, so the column is
    absent. The query error was swallowed and --redo-stale --dry-run
    reported '0 files' for an archive that needed 6,000 redone."""
    import sqlite3

    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE transcripts (path TEXT PRIMARY KEY, duration_s REAL, text TEXT,"
        " languages TEXT, segments TEXT, model TEXT, policy TEXT, transcribed_at TEXT)"
    )
    conn.execute(
        "INSERT INTO transcripts VALUES ('/a.mov', 1.0, 't', '[]', '[]', 'm', 'current', '2026-09-20')"
    )
    conn.commit()
    conn.close()
    with store.open_store(db, read_only=True) as ro:
        assert stale_paths(ro, policy="current", decode_policy="decode-now") == [Path("/a.mov")]


# --- one verdict per file ---------------------------------------------------------


def test_a_no_speech_verdict_replaces_an_existing_transcript(tmp_path):
    """A redo that ended in 'no speech' wrote the verdict and left the old
    transcript beside it -- still indexed, because the connector reads the
    transcripts table. 244 files after one run, 55 of them loops."""
    conn = _sidecar(tmp_path, [("/a.mov", "old")], [])
    store.save_no_text(conn, "/a.mov", duration_s=10.0, policy="new", reason="looping_repetition")
    assert conn.execute("SELECT count(*) FROM transcripts WHERE path = '/a.mov'").fetchone()[0] == 0


def test_a_transcript_replaces_an_older_rejection(tmp_path):
    conn = _sidecar(tmp_path, [], [("/a.mov", "old")])
    store.save_transcript(conn, store.Transcript(path="/a.mov", text="hi", model="m", policy="new"))
    assert conn.execute("SELECT count(*) FROM no_text WHERE path = '/a.mov'").fetchone()[0] == 0


def test_opening_the_sidecar_resolves_existing_double_verdicts_by_the_newer(tmp_path):
    db = tmp_path / "t.db"
    conn = _sidecar(tmp_path, [("/stale-transcript.mov", "p"), ("/fresh-transcript.mov", "p")],
                    [("/stale-transcript.mov", "p"), ("/fresh-transcript.mov", "p")])
    conn.execute("UPDATE transcripts SET transcribed_at = '2026-09-01'")
    conn.execute("UPDATE no_text SET checked_at = '2026-09-25' WHERE path = '/stale-transcript.mov'")
    conn.execute("UPDATE no_text SET checked_at = '2026-08-01' WHERE path = '/fresh-transcript.mov'")
    conn.commit()
    conn.close()

    with store.open_store(db) as c:
        assert [r[0] for r in c.execute("SELECT path FROM transcripts")] == ["/fresh-transcript.mov"]
        assert [r[0] for r in c.execute("SELECT path FROM no_text")] == ["/stale-transcript.mov"]


def test_the_redo_list_honours_the_archives_media_scope(tmp_path):
    """The list comes from the sidecar, not a walk, so the archive's own
    media policy never saw it: a redo transcribed 103 photo-library renders
    the archive excludes and would never index."""
    from corpus.transcripts.run import media_scope, stale_paths_present

    keep = tmp_path / "keep.mov"
    render = tmp_path / "render.mov"
    keep.write_bytes(b"x")
    render.write_bytes(b"x")
    conn = _sidecar(tmp_path, [(str(keep), "old"), (str(render), "old")], [])
    with media_scope(lambda p: "render" not in p.name):
        present, missing, out_of_scope = stale_paths_present(conn, policy="current")
    assert present == [keep] and missing == 0 and out_of_scope == 1
