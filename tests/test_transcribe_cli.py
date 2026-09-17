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
