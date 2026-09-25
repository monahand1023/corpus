"""Tests for `corpus-transcribe`, the command that spends the hours.

The dry run is what a user is supposed to consult BEFORE committing to a run
measured in hours, so the numbers it prints are the ones under test here. A
dry run that understates the remaining work is worse than no dry run: it is
wrong in the direction that gets trusted.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from corpus.cli.transcribe import main_argv
from corpus.transcripts import store
from corpus.transcripts.pipeline import Settings
from corpus.transcripts.store import Transcript, Window


@pytest.fixture(autouse=True)
def _fake_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """The dry run scopes "already done" to a policy that names the backend's
    model, and the rows below are seeded under `fake-v1`. Without pinning it,
    these tests passed only where the real backend failed to load, and failed
    on an Apple-Silicon machine with every extra installed."""

    class _Backend:
        model_name = "fake-v1"

    monkeypatch.setattr(
        "corpus.transcripts.backends.default_backend", lambda: _Backend()
    )


def _media(tmp_path: Path, *names: str) -> Path:
    root = tmp_path / "media"
    root.mkdir(exist_ok=True)
    for name in names:
        (root / name).write_bytes(b"not really audio")
    return root


def _seed(db: Path, root: Path, *, transcribed: str, silent: str) -> None:
    """One file that produced text and one that was judged to hold none."""
    policy = store.policy_fingerprint(Settings().as_policy("fake-v1"))
    with store.open_store(db) as conn:
        transcript = Transcript(
            path=str(root / transcribed),
            text="we fed the ducks",
            windows=[Window(0.0, 30.0, "we fed the ducks", "en")],
            duration_s=30.0,
            model="fake-v1",
        )
        transcript.policy = policy
        store.save_transcript(conn, transcript)
        store.save_no_text(
            conn, str(root / silent), duration_s=12.0, policy=policy, reason="silence"
        )


def test_dry_run_counts_files_with_no_speech_as_already_done(
    tmp_path, capsys
) -> None:
    # The bug this pins: the count read `transcripts` only, so a folder whose
    # files were ALL examined still reported most of them as outstanding. On a
    # real archive most files produce no text -- those rows exist precisely so
    # a second run does not pay for them again, and a dry run that ignores
    # them tells you to expect hours of work that will not happen.
    root = _media(tmp_path, "talk.m4a", "silence.mov")
    db = tmp_path / "transcripts.db"
    _seed(db, root, transcribed="talk.m4a", silent="silence.mov")

    main_argv([str(root), "--db", str(db), "--dry-run"])
    out = capsys.readouterr().out

    assert "2 of these" in out, out


def test_dry_run_ignores_rows_left_by_some_other_folder(tmp_path, capsys) -> None:
    # Sidecars are shared across runs, so "rows in this database" and "work
    # this run skips" are different sets.
    root = _media(tmp_path, "talk.m4a")
    elsewhere = tmp_path / "other"
    elsewhere.mkdir()
    db = tmp_path / "transcripts.db"
    _seed(db, elsewhere, transcribed="unrelated.m4a", silent="also-unrelated.mov")

    main_argv([str(root), "--db", str(db), "--dry-run"])
    out = capsys.readouterr().out

    assert "already done" not in out, out


def test_dry_run_says_so_when_transcription_cannot_run(tmp_path, capsys) -> None:
    """The cheapest moment to learn the run is impossible, whatever the reason.

    There are TWO reasons, and this test originally knew only one. Written on
    Apple Silicon, it asserted the message names the `transcribe-mlx` extra --
    true when the extra is merely missing. On Linux CI the backend is
    unavailable for a different reason (not Apple Silicon at all), and that
    message names the protocol to implement instead. The test failed on every
    CI platform while passing locally.

    What actually matters is invariant across both: it says so up front, it
    exits non-zero, and it names something the reader can act on.
    """
    root = _media(tmp_path, "talk.m4a")
    code = main_argv([str(root), "--db", str(tmp_path / "t.db"), "--dry-run"])
    out = capsys.readouterr().out

    if "cannot run yet" not in out:  # a machine that really can transcribe
        assert code == 0
        return

    assert code == 2, "a run that cannot happen must not report success"
    actionable = ("transcribe-mlx" in out) or ("TranscriberBackend" in out)
    assert actionable, f"must name the extra OR how to supply a backend: {out!r}"


def test_an_empty_folder_is_reported_rather_than_started(tmp_path, capsys) -> None:
    root = tmp_path / "empty"
    root.mkdir()
    code = main_argv([str(root), "--db", str(tmp_path / "t.db"), "--dry-run"])
    assert code == 1
    assert "Nothing to transcribe" in capsys.readouterr().out


def test_a_path_that_is_not_a_directory_fails_before_anything_else(
    tmp_path, capsys
) -> None:
    lone = tmp_path / "one.m4a"
    lone.write_bytes(b"not really audio")
    assert main_argv([str(lone), "--dry-run"]) == 1
    assert "not a directory" in capsys.readouterr().err


@pytest.mark.parametrize("flag", ["--limit", "--rate"])
def test_numeric_flags_are_parsed_not_swallowed(tmp_path, flag) -> None:
    root = _media(tmp_path, "talk.m4a")
    # Exercises the parser rather than the run: a typo'd dest here would
    # silently ignore the flag, which on --limit means transcribing an entire
    # archive when a 20-file sample was asked for.
    code = main_argv([str(root), "--db", str(tmp_path / "t.db"), "--dry-run", flag, "5"])
    assert code in (0, 2)


def test_dry_run_reports_what_is_done_before_it_prices_the_run(tmp_path, capsys) -> None:
    """Order is part of the warning, not presentation.

    The estimate used to print ABOVE the skip count and to cover every file
    that cleared the duration floor, already-done ones included. On a real
    archive that read:

        estimated runtime : ~14.4 h at 15x realtime
        already done      : 5,407 of these (skipped; ...)

    -- where 363 files actually needed transcribing. Someone reads the first
    line and abandons a twenty-minute job. The skip set has to be resolved
    first so the price can be about the work that is left.
    """
    root = _media(tmp_path, "talk.m4a", "silence.mov", "fresh.m4a")
    db = tmp_path / "transcripts.db"
    _seed(db, root, transcribed="talk.m4a", silent="silence.mov")

    main_argv([str(root), "--db", str(db), "--dry-run"])
    out = capsys.readouterr().out

    done_at = out.find("already done")
    assert done_at != -1, out
    priced = [i for i in (out.find("estimated runtime"), out.find("duration ")) if i != -1]
    assert priced, f"nothing priced the run at all:\n{out}"
    assert done_at < min(priced), (
        "the run was priced before the skip set was known:\n" + out
    )


def test_transcribed_since_scopes_redo_stale(tmp_path, monkeypatch, capsys) -> None:
    import corpus.transcripts.run as run_mod

    seen = {}

    def fake_present(conn, *, policy, since=None, decode_policy=None):
        seen["since"] = since
        return [], 0, 0

    monkeypatch.setattr(run_mod, "stale_paths_present", fake_present)
    db = tmp_path / "t.db"
    store.open_store(db).__enter__()
    main_argv([str(tmp_path), "--db", str(db), "--redo-stale", "--transcribed-since", "2026-09-14"])
    assert seen["since"] == "2026-09-14"


def test_transcribed_since_without_redo_stale_is_refused(tmp_path, capsys) -> None:
    with pytest.raises(SystemExit):
        main_argv([str(tmp_path), "--transcribed-since", "2026-09-14"])
    assert "only applies with --redo-stale" in capsys.readouterr().err


def test_a_redo_stale_dry_run_prices_the_redo_not_a_walk(tmp_path, capsys) -> None:
    """--dry-run ignored --redo-stale and walked the given path instead, so
    the one warning before a multi-hour redo reported 'nothing to transcribe'."""
    db = tmp_path / "t.db"
    media = tmp_path / "media"
    media.mkdir()
    for name, dur, at in (("a.mov", 3600.0, "2026-09-20"), ("b.mov", 1800.0, "2026-09-01")):
        (media / name).write_bytes(b"x")
        with store.open_store(db) as conn:
            conn.execute(
                "INSERT INTO transcripts (path, duration_s, dropped_windows, text, languages,"
                " segments, model, transcribed_at, elapsed_s, policy)"
                " VALUES (?, ?, 0, 'hi', '[]', '[]', 'fake-v1', ?, 1.0, 'old')",
                (str(media / name), dur, at),
            )
            conn.commit()

    main_argv([".", "--db", str(db), "--redo-stale", "--dry-run", "--rate", "10"])
    out = capsys.readouterr().out
    assert "2 file(s) invalidated" in out, out
    assert "1.5 h" in out, out

    main_argv([".", "--db", str(db), "--redo-stale", "--transcribed-since", "2026-09-14", "--dry-run"])
    out = capsys.readouterr().out
    assert "1 file(s) invalidated" in out and "1.0 h" in out, out


def test_redo_stale_asks_the_run_to_redo_not_skip(tmp_path, monkeypatch) -> None:
    import corpus.cli.transcribe as mod
    import corpus.transcripts.run as run_mod

    captured = {}

    def fake_directory(*args, **kwargs):
        captured.update(kwargs)
        return run_mod.RunStats()

    monkeypatch.setattr(mod, "transcribe_directory", fake_directory)
    monkeypatch.setattr(run_mod, "stale_paths_present", lambda conn, **k: ([tmp_path / "a.mov"], 0, 0))
    monkeypatch.setattr("corpus.transcripts.audio.ffmpeg_available", lambda: True)
    db = tmp_path / "t.db"
    store.open_store(db).__enter__()
    mod.main_argv([".", "--db", str(db), "--redo-stale"])
    assert captured.get("redo") is True
