"""Tests for `corpus.survey.media` — audio/video duration survey.

`_probe_duration_seconds` is monkeypatched throughout rather than shelling
out to a real `ffprobe`: this keeps the suite hermetic (no ffmpeg dependency
in CI) while still exercising the sampling, extrapolation, and
graceful-degradation logic, which is what this module actually owns —
`ffprobe` invocation itself is a thin, separately-reasoned-about subprocess
call.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import corpus.survey.media as media_mod
from corpus.survey.media import run_media_survey


def _touch(root: Path, rel: str, content: str = "x") -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return p


def test_ffprobe_unavailable_degrades_to_counts_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(media_mod.shutil, "which", lambda _: None)
    _touch(tmp_path, "call.mp3", "x" * 100)

    result = run_media_survey(tmp_path)

    assert result.ffprobe_available is False
    assert result.total_files == 1
    [t] = result.types
    assert t.mean_duration_seconds is None
    assert result.estimated_total_hours is None


def test_non_media_files_are_ignored(tmp_path: Path) -> None:
    _touch(tmp_path, "notes.txt")
    _touch(tmp_path, "photo.jpg")

    result = run_media_survey(tmp_path, ffprobe_path="/usr/bin/true")

    assert result.total_files == 0


def test_sampled_duration_extrapolates_to_full_population(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for i in range(10):
        _touch(tmp_path, f"clip{i}.mp3")

    monkeypatch.setattr(media_mod, "_probe_duration_seconds", lambda path, bin_, timeout: 60.0)

    result = run_media_survey(tmp_path, ffprobe_path="/fake/ffprobe", sample_size_per_type=4, seed=1)

    [t] = result.types
    assert t.count == 10
    assert t.sample_size == 4
    assert t.probed_ok == 4
    assert t.mean_duration_seconds == 60.0
    # Extrapolated: 10 files * 60s = 600s = 10 minutes = 1/6 hour.
    assert t.estimated_total_seconds == 600.0
    assert result.estimated_total_hours == pytest.approx(600.0 / 3600.0)


def test_probe_failures_are_counted_not_included_in_mean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _touch(tmp_path, "a.mp3")
    _touch(tmp_path, "b.mp3")

    def fake_probe(path: Path, bin_: str, timeout: float) -> float | None:
        return None if path.name == "b.mp3" else 30.0

    monkeypatch.setattr(media_mod, "_probe_duration_seconds", fake_probe)

    result = run_media_survey(tmp_path, ffprobe_path="/fake/ffprobe")

    [t] = result.types
    assert t.probed_ok == 1
    assert t.probe_failed == 1
    assert t.mean_duration_seconds == 30.0


def test_all_probes_failing_yields_no_total_estimate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _touch(tmp_path, "corrupt.mp3")
    monkeypatch.setattr(media_mod, "_probe_duration_seconds", lambda path, bin_, timeout: None)

    result = run_media_survey(tmp_path, ffprobe_path="/fake/ffprobe")

    assert result.estimated_total_hours is None


def test_stratified_by_extension_not_pooled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _touch(tmp_path, "voice.mp3")
    _touch(tmp_path, "movie.mp4")

    def fake_probe(path: Path, bin_: str, timeout: float) -> float | None:
        return 30.0 if path.suffix == ".mp3" else 7200.0

    monkeypatch.setattr(media_mod, "_probe_duration_seconds", fake_probe)

    result = run_media_survey(tmp_path, ffprobe_path="/fake/ffprobe")

    by_ext = {t.extension: t for t in result.types}
    assert by_ext[".mp3"].mean_duration_seconds == 30.0
    assert by_ext[".mp4"].mean_duration_seconds == 7200.0


def test_rate_projects_processing_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _touch(tmp_path, "a.mp3")
    monkeypatch.setattr(media_mod, "_probe_duration_seconds", lambda path, bin_, timeout: 3600.0)

    result = run_media_survey(tmp_path, ffprobe_path="/fake/ffprobe", rate=15.0)

    assert result.estimated_total_hours == pytest.approx(1.0)
    assert result.estimated_processing_hours == pytest.approx(1.0 / 15.0)


def test_no_rate_means_no_processing_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _touch(tmp_path, "a.mp3")
    monkeypatch.setattr(media_mod, "_probe_duration_seconds", lambda path, bin_, timeout: 60.0)

    result = run_media_survey(tmp_path, ffprobe_path="/fake/ffprobe")

    assert result.estimated_processing_hours is None


def test_real_ffprobe_subprocess_call_returns_none_on_missing_binary(tmp_path: Path) -> None:
    # Exercises the real (non-monkeypatched) subprocess path against a
    # deliberately bogus binary, confirming it degrades to None rather than
    # raising — this is the one test in the module that doesn't stub
    # `_probe_duration_seconds` itself.
    target = _touch(tmp_path, "a.mp3")
    result = media_mod._probe_duration_seconds(target, "/definitely/not/a/real/binary", 5.0)
    assert result is None
