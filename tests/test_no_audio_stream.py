"""A video with no audio track is a settled verdict, not a failure.

`already_done` deliberately excludes `failures`, because a failure "usually
means a broken decode or a transient resource problem rather than a settled
verdict about the audio". That reasoning is right and it does not cover the
commonest case in a photo library: a video FILE WITH NO AUDIO STREAM AT ALL.

ffmpeg reports it as `Output file does not contain any stream`, corpus
recorded it as a failure, and a failure is retried on every future run
forever. Measured on a live re-transcribe: 67 of the first 665 files -- 10%
-- were exactly this. They would be re-decoded on every run from now on, and
each one is an error the operator has to triage past to find a real one.

A file having no audio track is a property of the file. It belongs in
`no_text`, which carries a policy fingerprint, so it is skipped until a rule
that could change the answer actually changes.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from corpus.transcripts.audio import AudioUnavailableError, NoAudioStreamError, decode


def _fake_ffmpeg(monkeypatch, *, stderr: str, returncode: int = 1) -> None:
    def fake_run(*_a, **_k):
        return subprocess.CompletedProcess(
            args=[], returncode=returncode, stdout=b"", stderr=stderr.encode()
        )

    monkeypatch.setattr(subprocess, "run", fake_run)


def test_no_audio_stream_raises_its_own_error(tmp_path, monkeypatch):
    media = tmp_path / "silent.mp4"
    media.write_bytes(b"x")
    _fake_ffmpeg(
        monkeypatch,
        stderr="[out#0/f32le @ 0x1] Output file does not contain any stream\n"
               "Error opening output file -.",
    )
    with pytest.raises(NoAudioStreamError):
        decode(media)


def test_it_is_still_an_audio_unavailable_error(tmp_path, monkeypatch):
    """Existing callers that catch the broad type keep working."""
    media = tmp_path / "silent.mp4"
    media.write_bytes(b"x")
    _fake_ffmpeg(monkeypatch, stderr="Output file does not contain any stream")
    with pytest.raises(AudioUnavailableError):
        decode(media)


def test_a_genuine_decode_failure_is_not_reclassified(tmp_path, monkeypatch):
    """The negative control. A corrupt file IS worth retrying -- misfiling it
    as a settled verdict would stop corpus ever looking at it again."""
    media = tmp_path / "broken.mp4"
    media.write_bytes(b"x")
    _fake_ffmpeg(monkeypatch, stderr="moov atom not found")
    with pytest.raises(AudioUnavailableError) as exc:
        decode(media)
    assert not isinstance(exc.value, NoAudioStreamError)


def test_a_timeout_is_not_reclassified(tmp_path, monkeypatch):
    media = tmp_path / "huge.mp4"
    media.write_bytes(b"x")

    def fake_run(*_a, **_k):
        raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=1.0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(AudioUnavailableError) as exc:
        decode(media)
    assert not isinstance(exc.value, NoAudioStreamError)


# --- what the run does with it -----------------------------------------------


def test_a_stream_less_file_is_recorded_as_no_text_not_a_failure(tmp_path):
    from corpus.transcripts import store
    from corpus.transcripts.pipeline import Settings
    from corpus.transcripts.run import transcribe_directory

    media = tmp_path / "clip.mp4"
    media.write_bytes(b"x")
    db = tmp_path / "t.db"

    def boom(path, backend, *, settings=None):
        raise NoAudioStreamError(f"{path} has no audio stream")

    class _Backend:
        model_name = "test-model"

    stats = transcribe_directory(
        tmp_path, db, _Backend(), settings=Settings(), transcribe=boom
    )
    assert stats.failed == 0, "a settled verdict was recorded as a failure"
    with store.open_store(db, read_only=True) as conn:
        assert conn.execute("SELECT count(*) FROM failures").fetchone()[0] == 0
        row = conn.execute("SELECT reason FROM no_text").fetchone()
        assert row is not None and row[0] == "no_audio_stream"


def test_it_is_skipped_on_the_next_run(tmp_path):
    """The whole point: 10% of a photo library re-decoded on every run is a
    cost, and an error nobody can act on is noise that hides real ones."""
    from corpus.transcripts.pipeline import Settings
    from corpus.transcripts.run import transcribe_directory

    media = tmp_path / "clip.mp4"
    media.write_bytes(b"x")
    db = tmp_path / "t.db"
    calls: list[Path] = []

    def boom(path, backend, *, settings=None):
        calls.append(path)
        raise NoAudioStreamError(f"{path} has no audio stream")

    class _Backend:
        model_name = "test-model"

    for _ in range(2):
        transcribe_directory(
            tmp_path, db, _Backend(), settings=Settings(), transcribe=boom
        )
    assert len(calls) == 1, f"re-decoded a file with no audio track: {calls}"


def test_a_real_failure_is_still_retried(tmp_path):
    """The negative control for resumption."""
    from corpus.transcripts.pipeline import Settings
    from corpus.transcripts.run import transcribe_directory

    (tmp_path / "clip.mp4").write_bytes(b"x")
    db = tmp_path / "t.db"
    calls: list[Path] = []

    def boom(path, backend, *, settings=None):
        calls.append(path)
        raise AudioUnavailableError("moov atom not found")

    class _Backend:
        model_name = "test-model"

    for _ in range(2):
        transcribe_directory(
            tmp_path, db, _Backend(), settings=Settings(), transcribe=boom
        )
    assert len(calls) == 2, "a retryable failure stopped being retried"
