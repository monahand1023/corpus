"""Audacity projects are transcribed like any other recording.

An `.aup3` is a SQLite database, so ffmpeg and ffprobe cannot read it, and
the media walk did not look for it. What was said in one was unsearchable.
Now the walk finds it, its duration comes from its own layout (so its
deadline scales like any file's), and `decode` mixes it and hands the
samples to ffmpeg to resample.
"""

from __future__ import annotations

import contextlib
import shutil
import sqlite3
from pathlib import Path

import pytest

from tests.aup3_fixture import ClipSpec, TrackSpec, make_project, tone


def _stereo(path: Path, seconds: float = 1.0, rate: int = 8000) -> Path:
    n = round(seconds * rate)
    return make_project(path, [
        TrackSpec([ClipSpec(tone(n, 0.2))], rate=rate, channel=0, linked=3),
        TrackSpec([ClipSpec(tone(n, 0.4))], rate=rate, channel=1, linked=0),
    ])


def test_the_media_walk_finds_audacity_projects(tmp_path: Path) -> None:
    from corpus.transcripts.run import find_media

    _stereo(tmp_path / "session.aup3")
    assert [p.name for p in find_media(tmp_path)] == ["session.aup3"]


def test_a_projects_duration_comes_from_its_layout_not_ffprobe(tmp_path: Path) -> None:
    """ffprobe cannot read a project; a None here would give it the flat base
    deadline, which is the defect fixed for new files, reintroduced."""
    from corpus.survey.media import _probe_duration_seconds

    path = _stereo(tmp_path / "session.aup3", seconds=2.5)
    assert _probe_duration_seconds(path, "/nonexistent/ffprobe", 5.0) == pytest.approx(2.5)


def test_a_project_without_a_layout_has_no_duration(tmp_path: Path) -> None:
    from corpus.survey.media import _probe_duration_seconds

    path = _stereo(tmp_path / "session.aup3")
    conn = sqlite3.connect(path)
    conn.execute("DELETE FROM project")
    conn.commit()
    conn.close()
    assert _probe_duration_seconds(path, "/nonexistent/ffprobe", 5.0) is None


def test_a_project_without_a_layout_is_refused_not_guessed(tmp_path: Path) -> None:
    """Guessing a rate or channel layout produces audio that transcribes to
    nonsense -- or the left channel's words followed by the right's."""
    pytest.importorskip("numpy")
    from corpus.transcripts.audio import AudioUnavailableError, decode

    path = _stereo(tmp_path / "session.aup3")
    conn = sqlite3.connect(path)
    conn.execute("DELETE FROM project")
    conn.commit()
    conn.close()
    with pytest.raises(AudioUnavailableError, match="layout"):
        decode(path)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_a_project_decodes_to_16k_mono_of_its_real_length(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    from corpus.transcripts.audio import SAMPLE_RATE, decode

    path = _stereo(tmp_path / "session.aup3", seconds=2.0, rate=48000)
    samples = decode(path)

    assert samples.dtype == np.float32
    # Two seconds, not four: the pair is mixed, not played one after the other.
    assert abs(len(samples) - 2 * SAMPLE_RATE) <= 32
    assert np.allclose(samples[1000:-1000], 0.3, atol=0.01)


def test_the_source_project_is_never_modified(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    from corpus.transcripts.audio import AudioUnavailableError, decode

    path = _stereo(tmp_path / "session.aup3")
    before = (path.stat().st_mtime_ns, path.read_bytes())
    with contextlib.suppress(AudioUnavailableError):  # no ffmpeg: the read still happened
        decode(path)
    assert (path.stat().st_mtime_ns, path.read_bytes()) == before


def test_a_filename_with_url_characters_opens(tmp_path: Path) -> None:
    """The read-only open is a URI; unquoted, a `#` ends the path there."""
    from corpus.survey.media import _probe_duration_seconds

    path = _stereo(tmp_path / "take #2?.aup3", seconds=1.5)
    assert _probe_duration_seconds(path, "/nonexistent/ffprobe", 5.0) == pytest.approx(1.5)
