"""Getting audio out of a media file, and finding the speech in it.

Two jobs with one thing in common: both are about NOT handing silence to a
speech-to-text model.

A model trained on audio paired with scraped subtitles learned that silence
maps to caption boilerplate, and reproduces it whenever handed audio without
speech. That is not a decoding bug and no threshold on the OUTPUT fixes it,
because the model is confidently wrong -- measured on generated silence,
"Thank you." came back at no_speech=0.782 with avg_logprob=-0.24. The only
reliable defence is to not hand it the silence.

WHAT VOICE-ACTIVITY DETECTION IS FOR HERE, AND WHAT IT IS NOT. It decides
where to SPEND transcription time. It must never be the thing that decides
whether a recording is worth keeping. Measured on one archive, a hallucinated
sign-off peaked at 0.145 speech probability and a genuine recording of a
parent calling a child's name peaked at 0.144 -- and auditing nine detector
rejections by hand, four were real family recordings that were quiet, distant
or reverberant. A detector trained on clean speech does not hear those. Judge
the resulting TEXT instead; `corpus.transcripts.quality` is that judgement.

Everything here imports its heavy dependencies lazily, so importing this
module costs nothing on a machine that only ever reads an existing sidecar.
"""

from __future__ import annotations

import re
import shutil
import sqlite3
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

# What speech models and the detector both expect.
SAMPLE_RATE = 16_000

# The detector's fixed frame size at 16 kHz. Not tunable: the shipped model
# was traced for exactly this.
VAD_FRAME = 512

# Above this a frame counts as speech. LOW on purpose -- the cost of including
# a quiet frame is a little wasted compute, and the cost of excluding one is
# losing a recording. See the module docstring for the 0.145/0.144 measurement.
VAD_SPEECH_PROB = 0.2

# A region shorter than this is a click or a breath, not speech.
VAD_MIN_SPEECH_S = 0.25
# Gaps shorter than this are pauses inside one utterance, not boundaries.
VAD_MAX_GAP_S = 0.6
# Padding either side, so a region does not start on the second phoneme.
VAD_PAD_S = 0.3


class AudioUnavailableError(RuntimeError):
    """Audio could not be decoded, or the tools to decode it are missing."""


class NoAudioStreamError(AudioUnavailableError):
    """The file has no audio track at all -- a settled fact about the file.

    Separate from its parent because the RUN treats the two differently, and
    must: an ordinary decode failure is retried on every future run, since it
    usually means a corrupt read or a transient resource problem. "This video
    has no audio" is not transient, and retrying it forever costs an ffmpeg
    invocation per run and puts an error in front of the operator that they
    can never act on.

    Measured on a live re-transcribe of a photo library: 67 of the first 665
    files -- 10% -- were exactly this, mostly the silent clip a phone records
    beside a still. A subclass, so any caller catching the broad type is
    unaffected.
    """


# ffmpeg's wording when the input has no audio track. Matched on the message
# rather than on the exit code, which is the same for every decode problem.
_NO_STREAM_RE = re.compile(
    r"does not contain any stream|Output file .* does not contain", re.I
)


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def decode(path: Path | str, *, timeout_s: float = 1800.0) -> np.ndarray:
    """Whole file to 16 kHz mono float32, in ONE ffmpeg call.

    `-vn` comes BEFORE `-i`, which is load-bearing rather than stylistic: it
    drops the video stream at the decoder instead of at the output. On one
    archive that single ordering was 77% of per-file time, because every frame
    of a two-hour video was being decoded and then discarded.

    One call, not one per window. A per-window extract re-invoked ffmpeg and
    re-seeked the source for every window, so a forty-minute video with sixty
    windows spawned sixty-one processes to read one file. Samples go straight
    to the model, so no temporary wav is written at all.

    The whole file is held in memory -- four bytes per sample, so about 58 MB
    per hour. One file is decoded at a time, so that is the peak.
    """
    import numpy as np

    if Path(path).suffix.lower() == ".aup3":
        return _decode_aup3(Path(path), timeout_s=timeout_s)
    if not ffmpeg_available():
        raise AudioUnavailableError(
            "ffmpeg is not on PATH. Transcription decodes audio with it; "
            "install it (macOS: `brew install ffmpeg`) and retry."
        )
    try:
        proc = subprocess.run(
            [
                "ffmpeg", "-v", "error",
                "-vn",  # BEFORE -i: drop video at the decoder, not the output
                "-i", str(path),
                "-f", "f32le", "-acodec", "pcm_f32le",
                "-ac", "1", "-ar", str(SAMPLE_RATE), "-",
            ],
            capture_output=True, timeout=timeout_s, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioUnavailableError(f"ffmpeg timed out after {timeout_s}s") from exc
    if proc.returncode != 0 or not proc.stdout:
        detail = proc.stderr.decode(errors="replace").strip()[:200]
        if _NO_STREAM_RE.search(detail):
            raise NoAudioStreamError(f"{path} has no audio stream: {detail}")
        raise AudioUnavailableError(f"ffmpeg could not decode {path}: {detail}")
    return np.frombuffer(proc.stdout, dtype=np.float32)


def _decode_aup3(path: Path, *, timeout_s: float) -> np.ndarray:
    """An Audacity project, mixed to mono at its own rate, then resampled to
    16 kHz by ffmpeg reading the samples from a pipe.

    A project whose layout cannot be read is refused: guessing its rate or
    channel layout produces audio that transcribes to nonsense.
    """
    import numpy as np

    from corpus.connectors.aup3_layout import LayoutError, mix_to_mono, read_layout
    from corpus.util.sqlite_ro import connect_ro

    try:
        conn = connect_ro(path, immutable=True)
        try:
            layout = read_layout(conn)
            if layout is None:
                raise LayoutError("the project has no saved layout")
            samples, rate = mix_to_mono(conn, layout)
        finally:
            conn.close()
    except (LayoutError, sqlite3.Error) as e:
        raise AudioUnavailableError(f"cannot read Audacity project layout of {path}: {e}") from e
    if not samples.size:
        raise NoAudioStreamError(f"{path} has no audible audio")
    if not ffmpeg_available():
        raise AudioUnavailableError(
            "ffmpeg is not on PATH. Transcription resamples audio with it; "
            "install it (macOS: `brew install ffmpeg`) and retry."
        )
    try:
        proc = subprocess.run(
            [
                "ffmpeg", "-v", "error",
                "-f", "f32le", "-ar", str(rate), "-ac", "1", "-i", "pipe:0",
                "-f", "f32le", "-acodec", "pcm_f32le",
                "-ac", "1", "-ar", str(SAMPLE_RATE), "-",
            ],
            input=samples.astype("<f4").tobytes(),
            capture_output=True, timeout=timeout_s, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioUnavailableError(f"ffmpeg timed out after {timeout_s}s") from exc
    if proc.returncode != 0 or not proc.stdout:
        detail = proc.stderr.decode(errors="replace").strip()[:200]
        raise AudioUnavailableError(f"ffmpeg could not resample {path}: {detail}")
    return np.frombuffer(proc.stdout, dtype=np.float32)


def _load_detector() -> Any:
    """Load the Silero voice-activity model, without its package's imports.

    `silero_vad.__init__` imports torchaudio for file loading this never uses
    -- ffmpeg already hands over samples. `find_spec` LOCATES the package
    without executing that `__init__`, so the dependency is avoided rather
    than merely unused.
    """
    import importlib.util

    try:
        import torch
    except ImportError as exc:
        raise AudioUnavailableError(
            "voice-activity detection needs the 'transcribe' extra, which is "
            "not installed. Add it with: uv add 'corpus-rag[transcribe]'"
        ) from exc

    spec = importlib.util.find_spec("silero_vad")
    if spec is None or not spec.submodule_search_locations:
        raise AudioUnavailableError(
            "silero-vad is not installed. It ships with the 'transcribe' extra."
        )
    model_path = Path(next(iter(spec.submodule_search_locations)))
    jit = model_path / "data" / "silero_vad.jit"
    if not jit.is_file():
        raise AudioUnavailableError(f"silero-vad model not found at {jit}")
    # torch.jit.load is untyped upstream; the return is opaque by design
    # anyway -- callers only ever call it and reset_states().
    model = torch.jit.load(str(jit))  # type: ignore[no-untyped-call]
    model.eval()
    return model


_DETECTOR: Any = None


def _detector() -> Any:
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = _load_detector()
    return _DETECTOR


def _frame_probabilities(audio: np.ndarray) -> list[float]:
    import torch

    model = _detector()
    model.reset_states()
    probs: list[float] = []
    with torch.no_grad():
        for start in range(0, len(audio) - VAD_FRAME + 1, VAD_FRAME):
            # .copy() because frombuffer returns a read-only view and torch
            # warns on wrapping one. 512 floats, so the copy is free.
            frame = torch.from_numpy(audio[start:start + VAD_FRAME].copy())
            probs.append(float(model(frame.unsqueeze(0), SAMPLE_RATE).item()))
    return probs


def peak_speech_probability(audio: np.ndarray) -> float:
    """Highest speech probability anywhere in `audio`.

    Used to tell TRUE SILENCE from merely-quiet speech, which is the one
    distinction a detector can still be trusted to make on difficult material.
    """
    probs = _frame_probabilities(audio)
    return max(probs) if probs else 0.0


def speech_regions(audio: np.ndarray) -> list[tuple[float, float]]:
    """Spans of `audio` containing speech, in seconds.

    Returns an empty list when nothing clears the threshold. That is NOT the
    same as "this file has no speech" -- see `fixed_windows`, which is the
    fallback for exactly this case, because a detector missing quiet audio is
    a measured failure mode rather than a hypothetical one.
    """
    probs = _frame_probabilities(audio)
    if not probs:
        return []

    per_frame = VAD_FRAME / SAMPLE_RATE
    regions: list[list[float]] = []
    for index, prob in enumerate(probs):
        if prob < VAD_SPEECH_PROB:
            continue
        start, end = index * per_frame, (index + 1) * per_frame
        if regions and start - regions[-1][1] <= VAD_MAX_GAP_S:
            regions[-1][1] = end
        else:
            regions.append([start, end])

    total = len(audio) / SAMPLE_RATE
    return [
        (max(0.0, start - VAD_PAD_S), min(total, end + VAD_PAD_S))
        for start, end in regions
        if end - start >= VAD_MIN_SPEECH_S
    ]


def duration_seconds(audio: np.ndarray) -> float:
    return len(audio) / SAMPLE_RATE
