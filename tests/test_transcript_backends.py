"""Tests for the transcription backend seam.

The whole reason this seam exists is that the shipped implementation runs on
Apple Silicon only. So the test that matters most is that a backend can be
supplied WITHOUT any of that -- if these tests needed a GPU, the abstraction
would have failed at its one job.
"""

from __future__ import annotations

import platform

import pytest

from corpus.transcripts.backends import (
    BackendUnavailableError,
    MlxWhisperBackend,
    TranscriberBackend,
    WindowResult,
    default_backend,
)


class FakeBackend:
    """A backend in 6 lines, with no model anywhere."""

    model_name = "fake-v1"

    def transcribe_window(self, samples) -> WindowResult:
        return WindowResult(text="we fed the ducks", language="en",
                            no_speech=0.01, avg_logprob=-0.2)


def test_a_plain_object_satisfies_the_protocol() -> None:
    # If satisfying this needed a base class or an install, every alternative
    # implementation would inherit the dependency the protocol exists to avoid.
    assert isinstance(FakeBackend(), TranscriberBackend)


def test_a_backend_returns_text_and_the_evidence_behind_it() -> None:
    result = FakeBackend().transcribe_window(None)
    assert result.text == "we fed the ducks"
    assert result.language == "en"
    assert result.no_speech == 0.01


def test_confidence_fields_are_optional() -> None:
    # Not every model exposes them, and their absence must read as "no evidence
    # recorded" rather than as a verdict. They are stored to make a discard
    # auditable -- they never decide whether text is invented, because a model
    # returned "Thank you." on pure silence at no_speech=0.782.
    result = WindowResult(text="something said")
    assert result.no_speech is None
    assert result.avg_logprob is None
    assert result.language is None


# --- the shipped backend ---------------------------------------------------


def test_constructing_the_mlx_backend_loads_no_model() -> None:
    # Construction has to be free: a run that never reaches a window, or that
    # resumes and finds everything already done, must not pay for a 3 GB load.
    backend = MlxWhisperBackend()
    assert backend.model_name.startswith("mlx-community/")


def test_the_model_name_is_exposed_because_it_belongs_in_the_fingerprint() -> None:
    # Changing the model changes what counts as "nothing usable", so a stored
    # verdict must not outlive the model that produced it.
    assert MlxWhisperBackend("some/other-model").model_name == "some/other-model"


@pytest.mark.skipif(
    platform.system() == "Darwin" and platform.machine() == "arm64",
    reason="Apple Silicon: the shipped backend is genuinely available here",
)
def test_an_unsupported_platform_names_what_to_do_about_it() -> None:
    with pytest.raises(BackendUnavailableError) as caught:
        default_backend()
    message = str(caught.value)
    assert "TranscriberBackend" in message, "must say how to supply your own"
    assert platform.machine() in message, "must say what it detected"


APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"


def _hide_mlx(monkeypatch) -> None:
    """Make `import mlx_whisper` fail, whether or not it is installed here.

    The extra is optional, so a test that merely relies on it being absent
    passes for the wrong reason on a machine that has it.
    """
    import builtins

    real_import = builtins.__import__

    def without_mlx(name, *args, **kwargs):
        if name == "mlx_whisper":
            raise ImportError("No module named 'mlx_whisper'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_mlx)


@pytest.mark.skipif(not APPLE_SILICON, reason="not Apple Silicon")
def test_apple_silicon_gets_the_shipped_backend(monkeypatch) -> None:
    # preflight is stubbed rather than satisfied: what is under test is the
    # SELECTION, and requiring the extra here would make the suite need Apple
    # Silicon AND an optional install to check which branch was taken.
    monkeypatch.setattr(MlxWhisperBackend, "preflight", lambda self: None)
    assert isinstance(default_backend(), MlxWhisperBackend)


@pytest.mark.skipif(not APPLE_SILICON, reason="not Apple Silicon")
def test_a_missing_extra_is_caught_before_the_run_not_once_per_file(
    monkeypatch,
) -> None:
    # The regression this pins: default_backend() checked the PLATFORM only,
    # so on Apple Silicon without the extra it handed back a backend that
    # could not work. The failure then surfaced on the first window of every
    # file -- one archive recorded the same ImportError thousands of times
    # before anyone learned which package to install.
    _hide_mlx(monkeypatch)
    with pytest.raises(BackendUnavailableError) as caught:
        default_backend()
    assert "corpus-rag[transcribe-mlx]" in str(caught.value), "must name the extra"


def test_a_missing_extra_names_the_extra_rather_than_raising_importerror(
    monkeypatch,
) -> None:
    # The same failure shape the reranker uses: five frames of
    # ModuleNotFoundError tells a user nothing about which extra to install.
    _hide_mlx(monkeypatch)
    with pytest.raises(BackendUnavailableError) as caught:
        MlxWhisperBackend().transcribe_window(None)
    assert "corpus-rag[transcribe-mlx]" in str(caught.value)
    assert "Apple Silicon" in str(caught.value)


# --- restricting the language to the ones actually spoken ----------------------------


class _ScriptedMlx(MlxWhisperBackend):
    """The MLX backend with the model calls scripted, so the language logic
    is tested without a model or Apple Silicon."""

    def __init__(self, detected: str, probs: dict[str, float]) -> None:
        super().__init__()
        self.detected, self.probs, self.decodes = detected, probs, []

    def preflight(self) -> None:
        pass

    def _decode(self, samples, language=None):
        self.decodes.append(language)
        lang = language or self.detected
        return {"text": f"text in {lang}", "language": lang, "segments": []}

    def _language_probs(self, samples):
        return self.probs


def test_a_language_outside_the_allowed_set_is_decoded_again_in_the_likeliest_allowed_one() -> None:
    """Short or quiet windows get labelled as languages nobody in the archive
    speaks (Nynorsk, Javanese, Hawaiian...) and their text is then garbage in
    that language. Forcing the likeliest allowed language keeps the speech."""
    backend = _ScriptedMlx("nn", {"nn": 0.5, "en": 0.1, "ja": 0.3, "es": 0.1})
    result = backend.transcribe_window(None, languages=frozenset({"en", "ja", "es"}))
    assert result.language == "ja"
    assert result.text == "text in ja"
    assert backend.decodes == [None, "ja"]


def test_an_allowed_language_costs_one_decode() -> None:
    backend = _ScriptedMlx("es", {"es": 0.9})
    result = backend.transcribe_window(None, languages=frozenset({"en", "es"}))
    assert result.language == "es" and backend.decodes == [None]


def test_without_a_language_set_nothing_is_constrained() -> None:
    backend = _ScriptedMlx("nn", {"nn": 0.9})
    assert backend.transcribe_window(None).language == "nn"
    assert backend.decodes == [None]


def test_the_pipeline_passes_the_run_languages_to_the_backend(tmp_path) -> None:
    import numpy as np

    from corpus.transcripts.pipeline import Settings, transcribe_file

    seen = []

    class _Backend:
        model_name = "fake"

        def transcribe_window(self, samples, languages=None):
            seen.append(languages)
            return WindowResult(text="we fed the ducks by the pond", language="en")

    transcribe_file(
        tmp_path / "a.wav", _Backend(), settings=Settings(expected_languages=frozenset({"en", "ja"})),
        decode=lambda p: np.full(16000 * 5, 0.1, dtype=np.float32),
        speech_regions=lambda s: [(0.0, 5.0)],
        peak_speech_probability=lambda s: 0.9,
    )
    assert seen == [frozenset({"en", "ja"})]
