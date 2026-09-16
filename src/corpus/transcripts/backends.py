"""The interface a speech-to-text model has to satisfy, and the ones shipped.

WHY THERE IS AN INTERFACE AT ALL. The implementation this project was built
against is `mlx-whisper`, which runs only on Apple Silicon. A public package
cannot make a single vendor's hardware a requirement of its headline feature,
so the pipeline talks to a protocol and the Apple-specific part sits behind
it. A `faster-whisper`, `whisper.cpp` or API-backed implementation slots in
without the pipeline knowing.

WHAT A BACKEND IS RESPONSIBLE FOR. Exactly one thing: given samples for ONE
window, return what was said and how confident the model was. It does not
decide where windows fall, whether a window is worth transcribing, whether the
text is real, or what gets stored -- all of that is the pipeline's, and
keeping it there is what makes the quality rules apply identically whichever
model produced the text.

WHAT `no_speech` AND `avg_logprob` ARE FOR, AND WHAT THEY ARE NOT. They are
recorded as EVIDENCE, kept in `dropped_windows` so a discard can be audited
later. They are deliberately NOT the thing that decides whether text is
invented: measured on generated silence, a model returned "Thank you." at
no_speech=0.782 and avg_logprob=-0.24. It is confidently wrong, so its own
confidence cannot separate invention from speech. `corpus.transcripts.quality`
judges the text instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np


@dataclass(frozen=True)
class WindowResult:
    """What one window produced.

    `no_speech` and `avg_logprob` may be None: not every model exposes them,
    and the pipeline treats their absence as "no evidence recorded" rather
    than as a verdict either way.
    """

    text: str
    language: str | None = None
    no_speech: float | None = None
    avg_logprob: float | None = None


@runtime_checkable
class TranscriberBackend(Protocol):
    """One window of samples in, text out."""

    @property
    def model_name(self) -> str:
        """Identifies the model in stored transcripts and policy fingerprints.

        It belongs in the fingerprint because changing the model changes what
        counts as "nothing usable", and a verdict must not outlive the model
        that produced it.
        """
        ...

    def transcribe_window(self, samples: np.ndarray) -> WindowResult:
        """Transcribe ONE window of 16 kHz mono float32 samples.

        Windowing is the caller's job, and deliberately so: a model detects
        language once -- usually on the first thirty seconds -- and applies
        that verdict to everything it is handed. On a recording that starts in
        English and continues in Japanese, everything after the switch comes
        back transcribed as English. One call per window forces a fresh
        decision each time.
        """
        ...


class BackendUnavailableError(RuntimeError):
    """The requested backend is not installed, or cannot run here."""


class MlxWhisperBackend:
    """Apple-Silicon Whisper via `mlx-whisper`.

    Requires the `transcribe-mlx` extra AND Apple Silicon. Constructing it is
    cheap; the model materialises on the first window, so a run that never
    reaches one pays nothing.
    """

    DEFAULT_MODEL = "mlx-community/whisper-large-v3-mlx"

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    def transcribe_window(self, samples: np.ndarray) -> WindowResult:
        try:
            import mlx_whisper
        except ImportError as exc:
            raise BackendUnavailableError(
                "mlx-whisper is not installed. It ships with the "
                "'transcribe-mlx' extra: uv add 'corpus-rag[transcribe-mlx]'. "
                "That extra runs on Apple Silicon only -- on other hardware, "
                "supply your own backend satisfying TranscriberBackend."
            ) from exc

        out: dict[str, Any] = mlx_whisper.transcribe(
            samples,
            path_or_hf_repo=self._model,
            # Sampling makes an invented sign-off MORE fluent, not less, and
            # the whole downstream filter is built on the text being the
            # model's best single guess rather than one sample from it.
            temperature=0.0,
            # Asked for because they are stored as evidence behind a discard,
            # not because they decide anything -- see the module docstring.
            condition_on_previous_text=False,
        )
        segments = out.get("segments") or []
        return WindowResult(
            text=(out.get("text") or "").strip(),
            language=out.get("language"),
            no_speech=_mean(segments, "no_speech_prob"),
            avg_logprob=_mean(segments, "avg_logprob"),
        )


def _mean(segments: list[dict[str, Any]], key: str) -> float | None:
    values = [s[key] for s in segments if isinstance(s.get(key), (int, float))]
    return sum(values) / len(values) if values else None


def default_backend() -> TranscriberBackend:
    """The best backend available here, or an error naming what to install.

    There is only one shipped implementation today. This function is the place
    a second one gets chosen from, so callers never name a backend directly
    and adding one does not change them.
    """
    import platform

    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return MlxWhisperBackend()
    raise BackendUnavailableError(
        "No transcription backend is available for this platform. The shipped "
        f"one (mlx-whisper) requires Apple Silicon; this is "
        f"{platform.system()}/{platform.machine()}. Supply your own object "
        "satisfying corpus.transcripts.backends.TranscriberBackend -- it needs "
        "only `model_name` and `transcribe_window(samples) -> WindowResult`."
    )
