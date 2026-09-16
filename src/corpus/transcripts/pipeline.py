"""Transcribing one file: decode, segment, transcribe, judge, store.

This is where the pieces meet, and it holds no model and no audio library of
its own -- it is handed a backend and a decoder, which is what makes the whole
thing testable without a GPU.

THE ORDER MATTERS AND IS NOT OBVIOUS.

1. Decode once, to samples.
2. Ask whether there is ANY speech. True silence is the one case where a file
   is refused without a model running, and its threshold is far below the one
   that marks speech, because this verdict is the destructive one.
3. Segment into windows, skipping the silence between speech regions. If the
   detector found nothing but the audio is not silent, fall back to covering
   the file in fixed windows -- a detector that misses quiet, distant or
   reverberant recordings is a measured failure mode, not a hypothetical one,
   and discarding a file on its word loses exactly the recordings worth most.
4. Transcribe each window, keeping the model's own confidence as EVIDENCE.
5. Judge each window's TEXT. This is the step that catches invention, because
   the model's confidence cannot: on generated silence it produced "Thank
   you." at no_speech=0.782.
6. Rejoin, trimming only the seams that were cuts mid-speech.
7. Judge the whole transcript, and store the outcome -- including the negative
   ones, because a file judged empty must not be re-transcribed on the next
   run, and every discarded window is kept so the filter can be audited.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from corpus.transcripts import quality
from corpus.transcripts.segment import (
    OVERLAP_S,
    WINDOW_S,
    Window,
    file_timeout,
    fixed_windows,
    join_windows,
    windows_for_regions,
)
from corpus.transcripts.store import Transcript
from corpus.transcripts.store import Window as StoredWindow

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

    from corpus.transcripts.backends import TranscriberBackend


@dataclass(frozen=True)
class Settings:
    """Everything that can change what this run decides.

    ALL of it belongs in the policy fingerprint. That completeness was once
    false: the voice-activity thresholds were left out even though they
    produce the most destructive verdict available -- "no speech", reached
    without the model running at all -- so tightening them left stale verdicts
    in place and those files were never retried.
    """

    expected_languages: frozenset[str] = frozenset()
    window_s: float = WINDOW_S
    overlap_s: float = OVERLAP_S
    silence_max_prob: float = 0.05
    max_repeat_share: float = 0.9
    max_chars_per_second: float = quality.DEFAULT_MAX_CHARS_PER_SECOND
    unspoken_max_chars: int = quality.DEFAULT_UNSPOKEN_MAX_CHARS

    def as_policy(self, model_name: str) -> dict[str, Any]:
        return {
            "model": model_name,
            "window_s": self.window_s,
            "overlap_s": self.overlap_s,
            "silence_max_prob": self.silence_max_prob,
            "max_repeat_share": self.max_repeat_share,
            "max_chars_per_second": self.max_chars_per_second,
            "unspoken_max_chars": self.unspoken_max_chars,
            "expected_languages": sorted(self.expected_languages),
            # The phrase lists decide rejections too, so editing one has to
            # invalidate stored verdicts.
            "boilerplate": sorted(quality.SUBTITLE_BOILERPLATE),
            "credit_prefixes": sorted(quality.SUBTITLE_CREDIT_PREFIXES),
        }


@dataclass
class DroppedWindow:
    window_start: float
    no_speech: float
    avg_logprob: float
    text: str
    reason: str


@dataclass
class Outcome:
    """What happened to one file. Exactly one of these fields is meaningful."""

    path: str
    transcript: Transcript | None = None
    empty_reason: str | None = None
    rejected_text: str = ""
    dropped: list[DroppedWindow] = field(default_factory=list)
    duration_s: float = 0.0
    elapsed_s: float = 0.0

    @property
    def produced_text(self) -> bool:
        return self.transcript is not None


def _window_is_junk(text: str, window: Window, settings: Settings) -> str | None:
    """Why this window's text should be discarded, or None to keep it.

    Per-WINDOW, not per-file. A recording can be entirely real and still
    contain one window where the model emitted nothing but a sign-off, and
    that window would otherwise become an indexed, searchable chunk of its
    own.
    """
    stripped = text.strip()
    if not stripped:
        return "empty"
    if quality.subtitle_boilerplate(stripped):
        return "caption_boilerplate"
    if quality.repeat_share(stripped) >= settings.max_repeat_share:
        return "degenerate_repetition"
    if quality.impossible_speech_rate(
        stripped, window.duration, ceiling=settings.max_chars_per_second
    ):
        return "impossible_speech_rate"
    return None


def transcribe_file(
    path: Path | str,
    backend: TranscriberBackend,
    *,
    settings: Settings | None = None,
    decode: Callable[[Path | str], np.ndarray] | None = None,
    speech_regions: Callable[[np.ndarray], Sequence[tuple[float, float]]] | None = None,
    peak_speech_probability: Callable[[np.ndarray], float] | None = None,
    duration_seconds: Callable[[np.ndarray], float] | None = None,
) -> Outcome:
    """Transcribe one file and judge the result.

    The audio functions are injected rather than imported at module scope so
    this is exercisable without ffmpeg, torch or a model. They default to the
    real ones.
    """
    settings = settings or Settings()
    if decode is None or speech_regions is None or peak_speech_probability is None:
        from corpus.transcripts import audio as audio_module

        decode = decode or audio_module.decode
        speech_regions = speech_regions or audio_module.speech_regions
        peak_speech_probability = (
            peak_speech_probability or audio_module.peak_speech_probability
        )
        duration_seconds = duration_seconds or audio_module.duration_seconds
    if duration_seconds is None:
        from corpus.transcripts import audio as audio_module

        duration_seconds = audio_module.duration_seconds

    started = time.monotonic()
    samples = decode(path)
    duration = duration_seconds(samples)
    outcome = Outcome(path=str(path), duration_s=duration)

    # True silence: the one verdict reached without the model running. Its
    # threshold sits far below the one that marks speech precisely because
    # this is the destructive direction.
    if peak_speech_probability(samples) < settings.silence_max_prob:
        outcome.empty_reason = "silence"
        outcome.elapsed_s = time.monotonic() - started
        return outcome

    regions = list(speech_regions(samples))
    if regions:
        plan = list(
            windows_for_regions(
                regions, window_s=settings.window_s, overlap_s=settings.overlap_s
            )
        )
    else:
        # The detector found nothing but the audio is not silent. Transcribe
        # the whole file and let the text be judged, rather than discarding a
        # recording on the word of a detector that demonstrably misses the
        # quietest and most valuable material.
        plan = list(
            fixed_windows(
                duration, window_s=settings.window_s, overlap_s=settings.overlap_s
            )
        )

    kept: list[tuple[str, bool]] = []
    stored: list[StoredWindow] = []
    languages: list[str] = []
    for window in plan:
        begin = int(window.start * 16_000)
        finish = int(window.end * 16_000)
        result = backend.transcribe_window(samples[begin:finish])
        reason = _window_is_junk(result.text, window, settings)
        if reason:
            outcome.dropped.append(
                DroppedWindow(
                    window_start=window.start,
                    no_speech=result.no_speech if result.no_speech is not None else -1.0,
                    avg_logprob=(
                        result.avg_logprob if result.avg_logprob is not None else -1.0
                    ),
                    text=result.text,
                    reason=reason,
                )
            )
            continue
        kept.append((result.text.strip(), window.continues_previous))
        stored.append(
            StoredWindow(window.start, window.end, result.text.strip(), result.language)
        )
        if result.language:
            languages.append(result.language)

    text = join_windows(kept)
    outcome.elapsed_s = time.monotonic() - started

    verdict = quality.judge_transcript(
        text,
        duration_s=duration,
        languages=languages,
        expected_languages=settings.expected_languages or None,
        max_repeat_share=settings.max_repeat_share,
        max_chars_per_second=settings.max_chars_per_second,
        unspoken_max_chars=settings.unspoken_max_chars,
    )
    if not text or not verdict.keep:
        outcome.empty_reason = verdict.reason or "no_text"
        outcome.rejected_text = text
        return outcome

    outcome.transcript = Transcript(
        path=str(path),
        text=text,
        windows=stored,
        duration_s=duration,
        dropped_windows=len(outcome.dropped),
        model=backend.model_name,
        elapsed_s=outcome.elapsed_s,
    )
    return outcome


__all__ = [
    "DroppedWindow",
    "Outcome",
    "Settings",
    "file_timeout",
    "transcribe_file",
]
