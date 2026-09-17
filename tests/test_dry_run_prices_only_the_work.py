"""The dry run must price what it will DO, not what it will look at.

`--dry-run` is the only warning before hours of local compute, and its
estimate was the audio of every file that cleared the duration floor --
printed BEFORE the already-done set was consulted, and never reduced by it.

Measured on a real archive: one library reported "estimated runtime ~14.4 h"
on the line above "already done: 5,407 of these (skipped)". 5,770 files
cleared the floor; 363 actually needed transcribing. The warning overstated
the work by roughly fifteen times.

The existing code already guards the opposite direction -- its comment says
"understating it is the direction that costs someone an unexpected night of
compute" -- and it is right that understating is worse. That is why the
subtraction happens ONLY when the skip set is exact. Without a model there is
no policy, the done set is an upper bound, and subtracting it would understate
the run. Overstating is not free either: it talks you out of a twenty-minute
job.
"""

from __future__ import annotations

from pathlib import Path

from corpus.cli.transcribe import _runtime_hours


def _files(n: int) -> list[Path]:
    return [Path(f"/clips/{i}.mov") for i in range(n)]


def test_hours_exclude_files_already_done() -> None:
    files = _files(4)
    durations = {f: 3600.0 for f in files}
    done = {str(files[0]), str(files[1])}

    hours, count = _runtime_hours(files, durations, done, done_is_exact=True)

    assert (hours, count) == (2.0, 2)


def test_an_inexact_skip_set_is_not_subtracted() -> None:
    """An upper-bound done set would understate the run -- the one direction
    that costs someone an unexpected night of compute."""
    files = _files(4)
    durations = {f: 3600.0 for f in files}
    done = {str(f) for f in files}

    hours, count = _runtime_hours(files, durations, done, done_is_exact=False)

    assert (hours, count) == (4.0, 4)


def test_a_single_unreadable_duration_withholds_the_number() -> None:
    """A partial sum looks like a total and is quietly wrong; the existing
    code refuses one, and that must survive the subtraction."""
    files = _files(3)
    durations: dict[Path, float | None] = {files[0]: 60.0, files[1]: None, files[2]: 60.0}

    hours, count = _runtime_hours(files, durations, set(), done_is_exact=True)

    assert hours is None
    assert count == 3


def test_everything_already_done_is_zero_not_the_whole_archive() -> None:
    files = _files(3)
    durations = {f: 7200.0 for f in files}
    done = {str(f) for f in files}

    hours, count = _runtime_hours(files, durations, done, done_is_exact=True)

    assert (hours, count) == (0.0, 0)


def test_no_files_reports_nothing_rather_than_zero_hours() -> None:
    assert _runtime_hours([], {}, set(), done_is_exact=True) == (None, 0)


def test_an_unreadable_duration_falls_back_to_the_sampled_estimate(
    tmp_path, capsys, monkeypatch
) -> None:
    """Regression: exact pricing must DEGRADE to the old estimate, not replace it.

    When every surviving file is probed but one duration cannot be read,
    `_runtime_hours` correctly refuses to sum a partial total. The first
    version of this change then skipped the sampled fallback entirely and
    printed "duration: unknown (ffprobe unavailable...)" -- on a machine where
    ffprobe was present and working. A real library reported its skip count
    and then no runtime at all, which is worse than the overstated number this
    change set out to fix.
    """
    from corpus.cli import transcribe as cli
    from corpus.survey import media as media_mod

    root = tmp_path / "media"
    root.mkdir()
    for name in ("a.m4a", "b.mov"):
        (root / name).write_bytes(b"not really audio")

    # Imported INSIDE _dry_run, so the patch has to land on the source module.
    monkeypatch.setattr(
        media_mod, "probe_all", lambda paths, probe, **kw: dict.fromkeys(paths)
    )
    class _Sampled:
        estimated_total_hours = 3.0

    monkeypatch.setattr(media_mod, "run_media_survey", lambda *a, **k: _Sampled())

    cli.main_argv([str(root), "--db", str(tmp_path / "none.db"), "--dry-run", "--min-seconds", "1"])
    out = capsys.readouterr().out

    assert "ffprobe unavailable" not in out, out
    assert "estimated runtime" in out, out
