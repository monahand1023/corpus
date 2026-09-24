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
    fixed_windows,
    join_windows,
    pack_regions,
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
    # Merge nearby speech regions into windows of up to `window_s` (see
    # `segment.pack_regions`). Off reproduces one window per region.
    pack_regions: bool = True
    # Below this peak speech probability, for the WHOLE file, the audio is
    # treated as true silence and skipped. Deliberately far below the speech
    # threshold: the one verdict that discards a file without a model ever
    # running, so it only fires when there is nothing at all.
    silence_max_prob: float = 0.05
    max_repeat_share: float = 0.9
    max_looping_share: float = quality.DEFAULT_MAX_LOOPING_SHARE
    # Stricter, and deliberately so -- see DEFAULT_MAX_WINDOW_LOOPING_SHARE.
    max_window_looping_share: float = quality.DEFAULT_MAX_WINDOW_LOOPING_SHARE
    max_chars_per_second: float = quality.DEFAULT_MAX_CHARS_PER_SECOND
    unspoken_max_chars: int = quality.DEFAULT_UNSPOKEN_MAX_CHARS

    def judge(
        self, text: str, *, duration_s: float, languages: list[str] | None
    ) -> quality.TranscriptVerdict:
        """`quality.judge_transcript` under this run's thresholds."""
        return quality.judge_transcript(
            text,
            duration_s=duration_s,
            languages=languages,
            expected_languages=self.expected_languages or None,
            max_repeat_share=self.max_repeat_share,
            max_looping_share=self.max_looping_share,
            max_chars_per_second=self.max_chars_per_second,
            unspoken_max_chars=self.unspoken_max_chars,
        )

    def as_policy(self, model_name: str) -> dict[str, Any]:
        return {
            "model": model_name,
            "window_s": self.window_s,
            "overlap_s": self.overlap_s,
            "pack_regions": self.pack_regions,
            "silence_max_prob": self.silence_max_prob,
            "max_repeat_share": self.max_repeat_share,
            "max_looping_share": self.max_looping_share,
            "max_window_looping_share": self.max_window_looping_share,
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


def _window_is_junk(text: str, duration_s: float, settings: Settings) -> str | None:
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
    if quality.looping_share(stripped) >= settings.max_window_looping_share:
        return "looping_repetition"
    if quality.impossible_speech_rate(
        stripped, duration_s, ceiling=settings.max_chars_per_second
    ):
        return "impossible_speech_rate"
    return None


# Per-window reasons that mean "this text repeats", as opposed to "there is
# nothing here". Only these are reinstated when dropping would empty a
# transcript -- see `filter_windows`.
_REPETITION_REASONS = frozenset({"looping_repetition", "degenerate_repetition"})


def _is_physically_impossible(text: str, duration_s: float, settings: Settings) -> bool:
    """Could this much text have come out of this much audio at all?

    Split out of `_window_is_junk` because REINSTATEMENT NEEDS IT SEPARATELY.
    That function returns at the first rule that matches, so a window caught
    by a repetition rule never reaches the rate check below it -- and the
    fallback then put the window back, treating a check that never ran as a
    check that passed.

    Found live: a 64-second video whose whole transcript was 235 characters
    from a 1.7-second window -- 138 chars/s against a ceiling of 40. Dropped
    as `looping_repetition` at 0.843, reinstated, then measured by the
    whole-transcript judge against the FILE's 64 seconds (3.7 chars/s) and
    kept. All three defences declined and the loop was indexed.

    The repetition thresholds are a judgement call about taste -- a child
    chanting one word is repetitive and real, which is why the fallback
    exists. This is not a judgement call. No throat produces this.
    """
    return quality.impossible_speech_rate(
        text.strip(), duration_s, ceiling=settings.max_chars_per_second
    )


def filter_windows(
    windows: Sequence[tuple[str, float, bool]], settings: Settings
) -> tuple[list[tuple[str, bool]], list[tuple[int, str]]]:
    """Split windows into kept `(text, continues_previous)` and dropped
    `(index, reason)`.

    Indices, not text: two windows of one recording can hold identical text
    (that is what a loop IS), so matching them back up by value mislabels
    which one was dropped.

    THE FALLBACK, and why it is not optional. The case for a STRICT per-window
    threshold is that dropping a window costs one chunk while the rest of the
    recording survives. On a short recording there is no rest: the file is one
    window, so dropping it IS deleting the recording -- the exact outcome the
    permissive whole-transcript ceiling exists to prevent, arrived at through
    the side door.

    Measured on a full transcript archive before this existed: 176 recordings
    would have been emptied, 174 of them single-window: a birthday line sung
    over and over, an exclamation repeated three times, a toddler naming a
    parent. The same recordings a 0.6 whole-transcript ceiling had deleted
    earlier the same day.

    So when dropping would leave NOTHING, repetition-dropped windows are
    reinstated and the permissive `judge_transcript` decides the file's fate.
    Boilerplate is not reinstated: "Thank you. Thank you." is the model
    filling silence, not a song, and putting it back would return an empty
    result to search.
    """
    verdicts = [
        _window_is_junk(text, duration, settings)
        for text, duration, _ in windows
    ]
    if not any(v is None for v in verdicts) and any(
        v in _REPETITION_REASONS for v in verdicts
    ):
        # Nothing would survive and at least one drop was repetition:
        # reinstate those and let the permissive whole-transcript rule judge.
        #
        # Except where the rate check -- which sits BELOW the repetition rules
        # and so never ran on these windows -- says the text could not have
        # been spoken in the time available. Reinstating there does not hand
        # the decision to a more permissive rule, it hands it to a rule
        # measuring against the whole FILE's duration, where a burst of
        # invented text averages down into silence and disappears.
        verdicts = [
            (
                "impossible_speech_rate"
                if _is_physically_impossible(text, duration, settings)
                else None
            )
            if verdict in _REPETITION_REASONS
            else verdict
            for verdict, (text, duration, _) in zip(verdicts, windows, strict=True)
        ]

    kept: list[tuple[str, bool]] = []
    dropped: list[tuple[int, str]] = []
    for i, ((text, _d, continues), reason) in enumerate(
        zip(windows, verdicts, strict=True)
    ):
        if reason is None:
            kept.append((text.strip(), continues))
        else:
            dropped.append((i, reason))
    return kept, dropped


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
    if regions and settings.pack_regions:
        regions = pack_regions(regions, window_s=settings.window_s)
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

    # Transcribe every window first, then filter as a set. Filtering inside
    # the loop cannot see whether anything else survives, and that is exactly
    # the judgement `filter_windows` has to make: dropping the ONLY window is
    # deleting the recording, not trimming it.
    results = []
    # A backend written before `languages` existed still runs; the language
    # filter in `judge` then remains the only check, as it was.
    constrain = bool(settings.expected_languages) and _accepts_languages(backend)
    for window in plan:
        begin = int(window.start * 16_000)
        finish = int(window.end * 16_000)
        clip = samples[begin:finish]
        result = (
            backend.transcribe_window(clip, languages=settings.expected_languages)
            if constrain
            else backend.transcribe_window(clip)
        )
        results.append((window, result))

    kept, dropped_pairs = filter_windows(
        [(r.text, w.duration, w.continues_previous) for w, r in results], settings
    )
    dropped_reason = dict(dropped_pairs)

    stored: list[StoredWindow] = []
    languages: list[str] = []
    for i, (window, result) in enumerate(results):
        reason = dropped_reason.get(i)
        if reason is None:
            stored.append(
                StoredWindow(
                    window.start, window.end, result.text.strip(), result.language
                )
            )
            if result.language:
                languages.append(result.language)
            continue
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

    text = join_windows(kept)
    outcome.elapsed_s = time.monotonic() - started

    verdict = settings.judge(text, duration_s=duration, languages=languages)
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
    "transcribe_file",
]


def _accepts_languages(backend: TranscriberBackend) -> bool:
    import inspect

    try:
        params = inspect.signature(backend.transcribe_window).parameters
    except (TypeError, ValueError):
        return False
    return "languages" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
