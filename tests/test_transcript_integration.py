"""End-to-end transcription with REAL audio decoding and REAL speech detection.

Skipped unless ffmpeg and the `transcribe` extra are both present, so a base
install still has a green suite. The model is still a fake -- what is being
exercised here is everything AROUND it, which is where the decisions are.

This is the test that demonstrates the anti-hallucination strategy rather than
describing it:

* silence never reaches the model at all
* audio that is quiet but not silent IS transcribed, rather than discarded on
  a detector's word
* a confident hallucination is caught by judging the TEXT
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from corpus.transcripts.backends import WindowResult
from corpus.transcripts.pipeline import transcribe_file


def _have_extra() -> bool:
    import importlib.util

    return (
        shutil.which("ffmpeg") is not None
        and importlib.util.find_spec("silero_vad") is not None
        and importlib.util.find_spec("torch") is not None
    )


pytestmark = pytest.mark.skipif(
    not _have_extra(),
    reason="needs ffmpeg and the 'transcribe' extra (silero-vad, torch)",
)


def _wav(path, spec: str) -> str:
    """Synthesise a wav with ffmpeg, so no audio fixture is committed."""
    out = str(path)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", spec,
         "-ar", "16000", "-ac", "1", out, "-y"],
        check=True, capture_output=True,
    )
    return out


class HallucinatingModel:
    """Returns what a real model actually returns when handed silence.

    Not an invented failure: measured on generated silence, Whisper produced
    "Thank you." at no_speech=0.782 and avg_logprob=-0.24. It is CONFIDENTLY
    wrong, which is why its own confidence cannot be the filter.
    """

    model_name = "fake-whisper"

    def __init__(self) -> None:
        self.calls = 0

    def transcribe_window(self, samples):
        self.calls += 1
        return WindowResult(text="Thank you.", language="en",
                            no_speech=0.782, avg_logprob=-0.24)


def test_silence_never_reaches_the_model(tmp_path) -> None:
    # The cheapest possible win: no GPU time spent, and no opportunity for the
    # model to invent anything.
    path = _wav(tmp_path / "silence.wav", "anullsrc=r=16000:cl=mono:d=6")
    model = HallucinatingModel()

    outcome = transcribe_file(path, model)

    assert model.calls == 0, "silence must not be transcribed at all"
    assert not outcome.produced_text
    assert outcome.empty_reason == "silence"
    assert 5.5 < outcome.duration_s < 6.5


def test_quiet_audio_is_transcribed_and_its_invention_caught(tmp_path) -> None:
    # A tone is not speech, so the detector finds no regions -- but the file is
    # not silent either, and discarding on that basis is what loses quiet,
    # distant and reverberant recordings. So it IS transcribed, and the
    # resulting text is what gets judged.
    path = _wav(
        tmp_path / "tone.wav",
        "sine=frequency=200:duration=3",
    )
    model = HallucinatingModel()

    outcome = transcribe_file(path, model)

    assert model.calls >= 1, "quiet audio must still be transcribed"
    assert not outcome.produced_text, "the invented sign-off must not survive"
    assert outcome.dropped, "and the discard must leave evidence"
    assert outcome.dropped[0].no_speech == 0.782, (
        "the model was confident and was discarded anyway -- confidence is "
        "recorded as evidence, never used as the verdict"
    )


def test_real_speech_shaped_audio_produces_a_transcript(tmp_path) -> None:
    class RealSpeech:
        model_name = "fake-whisper"

        def transcribe_window(self, samples):
            return WindowResult(
                text="we walked down to the water and talked about the trip",
                language="en", no_speech=0.02, avg_logprob=-0.3,
            )

    path = _wav(tmp_path / "tone.wav", "sine=frequency=200:duration=3")
    outcome = transcribe_file(path, RealSpeech())

    assert outcome.produced_text
    assert outcome.transcript is not None
    assert "walked down to the water" in outcome.transcript.text
    assert outcome.transcript.windows, "windows carry the timing and language"
