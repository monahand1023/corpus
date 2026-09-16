"""Tests for the run loop: what gets visited, skipped, and written down.

Resumption is the property under test. A real archive is hours to days of
compute and WILL be interrupted, so a run that cannot resume is a run that
never finishes.
"""

from __future__ import annotations

from pathlib import Path

from corpus.transcripts import store
from corpus.transcripts.pipeline import DroppedWindow, Outcome, Settings
from corpus.transcripts.run import find_media, transcribe_directory
from corpus.transcripts.store import Transcript, Window


class Backend:
    model_name = "fake-v1"


def _media(tmp_path: Path, *names: str) -> Path:
    root = tmp_path / "media"
    root.mkdir(exist_ok=True)
    for name in names:
        (root / name).write_bytes(b"not really audio")
    return root


def _ok(path, backend, *, settings=None):
    return Outcome(
        path=str(path),
        transcript=Transcript(
            path=str(path), text="we fed the ducks",
            windows=[Window(0.0, 30.0, "we fed the ducks", "en")],
            duration_s=30.0, model=backend.model_name,
        ),
        duration_s=30.0,
    )


def _empty(path, backend, *, settings=None):
    return Outcome(
        path=str(path), empty_reason="silence", duration_s=12.0,
        dropped=[DroppedWindow(0.0, 0.78, -0.24, "Thank you.", "caption_boilerplate")],
    )


# --- what gets visited -----------------------------------------------------


def test_only_media_files_are_visited(tmp_path: Path) -> None:
    root = _media(tmp_path, "a.mov", "b.m4a", "notes.md", "sheet.csv", "c.mp4")
    assert [p.name for p in find_media(root)] == ["a.mov", "b.m4a", "c.mp4"]


def test_files_are_visited_in_a_stable_order(tmp_path: Path) -> None:
    # An interrupted run should resume in the same sequence rather than in
    # whatever order the filesystem returns, so progress stays legible.
    root = _media(tmp_path, "c.mov", "a.mov", "b.mov")
    assert [p.name for p in find_media(root)] == ["a.mov", "b.mov", "c.mov"]
    assert [p.name for p in find_media(root)] == ["a.mov", "b.mov", "c.mov"]


def test_excludes_are_honoured(tmp_path: Path) -> None:
    root = _media(tmp_path, "keep.mov", "skip.mov")
    names = [p.name for p in find_media(root, excludes=("*skip*",))]
    assert names == ["keep.mov"]


# --- writing outcomes down -------------------------------------------------


def test_a_transcript_is_stored_with_the_runs_policy(tmp_path: Path) -> None:
    root = _media(tmp_path, "a.mov")
    db = tmp_path / "t.db"
    stats = transcribe_directory(root, db, Backend(), transcribe=_ok)

    assert stats.transcribed == 1
    with store.open_store(db, read_only=True) as conn:
        row = conn.execute("SELECT policy, text FROM transcripts").fetchone()
    assert row["policy"], "a stored transcript must record the rules it was kept under"
    assert "ducks" in row["text"]


def test_a_file_with_no_usable_text_is_recorded_rather_than_forgotten(
    tmp_path: Path,
) -> None:
    # THE resumption property. Without this row the file is simply absent from
    # `transcripts`, so the next run re-decodes and re-transcribes it to reach
    # the same answer -- 889 silent clips were re-paid for exactly this way.
    root = _media(tmp_path, "silent.mov")
    db = tmp_path / "t.db"
    stats = transcribe_directory(root, db, Backend(), transcribe=_empty)

    assert stats.empty == 1
    with store.open_store(db, read_only=True) as conn:
        assert store.counts(conn)["no_text"] == 1


def test_discarded_windows_are_kept_even_when_nothing_survived(
    tmp_path: Path,
) -> None:
    root = _media(tmp_path, "junk.mov")
    db = tmp_path / "t.db"
    transcribe_directory(root, db, Backend(), transcribe=_empty)

    with store.open_store(db, read_only=True) as conn:
        counts = store.counts(conn)
    assert counts["transcripts"] == 0
    assert counts["dropped_windows"] == 1, (
        "a file whose windows were ALL discarded is the most interesting case "
        "to audit and would otherwise leave no trace"
    )


# --- resumption ------------------------------------------------------------


def test_a_second_run_skips_everything_already_answered(tmp_path: Path) -> None:
    root = _media(tmp_path, "a.mov", "b.mov")
    db = tmp_path / "t.db"
    transcribe_directory(root, db, Backend(), transcribe=_ok)

    calls: list[str] = []

    def counting(path, backend, *, settings=None):
        calls.append(str(path))
        return _ok(path, backend)

    again = transcribe_directory(root, db, Backend(), transcribe=counting)
    assert calls == [], "nothing should be re-transcribed"
    assert again.skipped_done == 2


def test_an_empty_verdict_is_retried_when_the_rules_change(tmp_path: Path) -> None:
    root = _media(tmp_path, "silent.mov")
    db = tmp_path / "t.db"
    transcribe_directory(root, db, Backend(), transcribe=_empty)

    stricter = Settings(max_chars_per_second=10.0)
    again = transcribe_directory(
        root, db, Backend(), settings=stricter, transcribe=_empty
    )
    assert again.skipped_done == 0, (
        "a verdict must not outlive the rules that produced it"
    )


def test_a_failure_is_retried_on_the_next_run(tmp_path: Path) -> None:
    root = _media(tmp_path, "broken.mov")
    db = tmp_path / "t.db"

    def boom(path, backend, *, settings=None):
        raise RuntimeError("ffmpeg exited 1")

    first = transcribe_directory(root, db, Backend(), transcribe=boom)
    assert first.failed == 1
    assert first.errors[0][1].startswith("RuntimeError")

    second = transcribe_directory(root, db, Backend(), transcribe=_ok)
    assert second.skipped_done == 0, (
        "a failure is usually a broken decode, not a settled verdict"
    )
    assert second.transcribed == 1


def test_one_bad_file_does_not_end_the_run(tmp_path: Path) -> None:
    root = _media(tmp_path, "a.mov", "broken.mov", "c.mov")
    db = tmp_path / "t.db"

    def sometimes(path, backend, *, settings=None):
        if "broken" in str(path):
            raise RuntimeError("bad file")
        return _ok(path, backend)

    stats = transcribe_directory(root, db, Backend(), transcribe=sometimes)
    assert stats.transcribed == 2
    assert stats.failed == 1


# --- reporting -------------------------------------------------------------


def test_progress_is_reported_for_every_file_including_failures(
    tmp_path: Path,
) -> None:
    root = _media(tmp_path, "a.mov", "broken.mov")
    seen: list[tuple[int, int, str]] = []

    def sometimes(path, backend, *, settings=None):
        if "broken" in str(path):
            raise RuntimeError("bad file")
        return _ok(path, backend)

    transcribe_directory(
        root, tmp_path / "t.db", Backend(),
        transcribe=sometimes,
        on_progress=lambda i, total, p, o: seen.append((i, total, p.name)),
    )
    assert [s[2] for s in seen] == ["a.mov", "broken.mov"]
    assert all(s[1] == 2 for s in seen)


def test_a_limit_caps_the_run_without_losing_the_rest(tmp_path: Path) -> None:
    root = _media(tmp_path, "a.mov", "b.mov", "c.mov")
    db = tmp_path / "t.db"
    first = transcribe_directory(root, db, Backend(), limit=1, transcribe=_ok)
    assert first.transcribed == 1

    second = transcribe_directory(root, db, Backend(), transcribe=_ok)
    assert second.skipped_done == 1
    assert second.transcribed == 2


def test_a_transcript_kept_under_old_rules_faces_the_new_ones(tmp_path) -> None:
    """The other half of the policy fingerprint's promise.

    A rule change already invalidated stored "no speech" verdicts. A stored
    TRANSCRIPT was equally a verdict -- "this text is real" -- and was
    inherited forever, so a filter written to catch something never reached
    the material already indexed under the older rules.
    """
    root = _media(tmp_path, "loop.m4a")
    db = tmp_path / "t.db"
    looped = "I am going to draw a small map. " * 9

    # Accepted under rules that did not look for loops.
    loose = Settings(max_looping_share=1.1)
    transcribe_directory(
        root, db, Backend(), settings=loose,
        transcribe=lambda p, b, *, settings=None: Outcome(
            path=str(p), duration_s=30.0,
            transcript=Transcript(
                path=str(p), text=looped,
                windows=[Window(0.0, 30.0, looped, "en")],
                duration_s=30.0, model=b.model_name),
        ),
    )
    with store.open_store(db) as conn:
        assert store.counts(conn)["transcripts"] == 1

    # Now the rules look for loops. No model runs: the text is already stored.
    calls: list[str] = []

    def must_not_run(path, backend, *, settings=None):
        calls.append(str(path))
        raise AssertionError("re-judging must not re-transcribe")

    stats = transcribe_directory(
        root, db, Backend(), settings=Settings(), transcribe=must_not_run
    )
    assert calls == []
    assert stats.demoted == 1
    with store.open_store(db) as conn:
        assert store.counts(conn)["transcripts"] == 0
        row = conn.execute(
            "SELECT reason, rejected_text FROM no_text WHERE path = ?",
            (str(root / "loop.m4a"),),
        ).fetchone()
    assert row[0] == "looping_repetition"
    # The text is kept, so a rule that proves too aggressive can be reversed
    # against real evidence rather than a re-run.
    assert row[1] == looped


def test_a_transcript_that_still_passes_is_restamped_not_redone(tmp_path) -> None:
    root = _media(tmp_path, "talk.m4a")
    db = tmp_path / "t.db"
    transcribe_directory(root, db, Backend(), settings=Settings(), transcribe=_ok)

    stats = transcribe_directory(
        root, db, Backend(), settings=Settings(max_looping_share=0.55),
        transcribe=lambda *a, **k: (_ for _ in ()).throw(AssertionError("no re-run")),
    )
    assert stats.rejudged == 1 and stats.demoted == 0
    with store.open_store(db) as conn:
        assert store.counts(conn)["transcripts"] == 1
    # And now it is current, so a third run has nothing to do.
    stats2 = transcribe_directory(
        root, db, Backend(), settings=Settings(max_looping_share=0.55),
        transcribe=lambda *a, **k: (_ for _ in ()).throw(AssertionError("no re-run")),
    )
    assert stats2.rejudged == 0 and stats2.skipped_done == 1


def test_a_run_persists_WHICH_filter_rejected_each_window(tmp_path) -> None:
    """The reason was computed per window and then dropped on the way to disk.

    `run.py` built the row from window_start/no_speech/avg_logprob/text and
    omitted `reason`, so the store could not answer "is any of these filters
    dead?" -- which is the question a count of discards cannot answer.
    """
    root = _media(tmp_path, "clip.m4a")
    db = tmp_path / "t.db"

    def with_drops(path, backend, *, settings=None):
        return Outcome(
            path=str(path), duration_s=60.0,
            transcript=Transcript(
                path=str(path), text="we fed the ducks by the pond",
                windows=[Window(0.0, 30.0, "we fed the ducks by the pond", "en")],
                duration_s=60.0, model=backend.model_name),
            dropped=[
                DroppedWindow(30.0, 0.8, -0.2, "Thank you.", "caption_boilerplate"),
                DroppedWindow(60.0, 0.7, -0.3, "rice " * 40, "looping_repetition"),
            ],
        )

    transcribe_directory(root, db, Backend(), transcribe=with_drops)

    with store.open_store(db) as conn:
        activity = store.filter_activity(conn)
    assert activity.get("caption_boilerplate") == 1
    assert activity.get("looping_repetition") == 1
