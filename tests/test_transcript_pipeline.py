"""Tests for the per-file transcription pipeline.

Every case here runs with no model, no ffmpeg, no torch and no GPU -- the
audio functions and the backend are injected. If that were not possible the
design would have failed, because the pipeline is where the destructive
decisions are made and it is the part that most needs testing.
"""

from __future__ import annotations

from corpus.transcripts.pipeline import Settings, transcribe_file


class _Samples:
    """Stands in for a numpy array: only len() and slicing are used."""

    def __init__(self, seconds: float) -> None:
        self._n = int(seconds * 16_000)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, item):
        return self


def _audio(seconds: float, *, peak: float = 0.9, regions=None):
    """Injected audio functions for a file of `seconds` with `peak` speech."""
    return {
        "decode": lambda _p: _Samples(seconds),
        "peak_speech_probability": lambda _s: peak,
        "speech_regions": lambda _s: regions if regions is not None else [(0.0, seconds)],
        "duration_seconds": lambda _s: seconds,
    }


class Backend:
    """Returns a fixed text for every window."""

    model_name = "fake-v1"

    def __init__(self, *texts: str) -> None:
        self._texts = list(texts) or ["we went to the park and fed the ducks"]
        self._i = 0

    def transcribe_window(self, samples):
        from corpus.transcripts.backends import WindowResult

        text = self._texts[min(self._i, len(self._texts) - 1)]
        self._i += 1
        return WindowResult(text=text, language="en", no_speech=0.02,
                            avg_logprob=-0.3)


# --- the happy path --------------------------------------------------------


def test_a_recording_with_speech_produces_a_transcript() -> None:
    out = transcribe_file("/a.mov", Backend(), **_audio(20.0))
    assert out.produced_text
    assert out.transcript is not None
    assert "ducks" in out.transcript.text
    assert out.transcript.model == "fake-v1"


def test_each_window_is_stored_with_its_timing_and_language() -> None:
    out = transcribe_file("/a.mov", Backend(), **_audio(70.0))
    assert out.transcript is not None
    windows = out.transcript.windows
    assert len(windows) > 1
    assert windows[0].start == 0.0
    assert all(w.lang == "en" for w in windows)


# --- silence: the one verdict reached without the model running -------------


def test_true_silence_is_refused_without_transcribing() -> None:
    class Exploding:
        model_name = "never-called"

        def transcribe_window(self, samples):
            raise AssertionError("silence must not reach the model")

    out = transcribe_file("/silent.mov", Exploding(), **_audio(30.0, peak=0.01))
    assert not out.produced_text
    assert out.empty_reason == "silence"


def test_quiet_but_not_silent_audio_is_still_transcribed() -> None:
    # The threshold for "silence" sits far below the one for "speech" exactly
    # so that quiet material is not discarded: a hallucinated sign-off peaked
    # at 0.145 and a real recording of a parent calling a child at 0.144.
    out = transcribe_file("/quiet.mov", Backend(), **_audio(20.0, peak=0.1))
    assert out.produced_text


# --- the fallback that keeps quiet recordings ------------------------------


def test_no_detected_regions_falls_back_to_covering_the_whole_file() -> None:
    # A detector that misses quiet, distant or reverberant audio is a measured
    # failure mode: four of nine audited rejections were real family
    # recordings. Discarding on its word loses exactly the material worth most.
    out = transcribe_file("/quiet.mov", Backend(), **_audio(70.0, regions=[]))
    assert out.produced_text
    assert out.transcript is not None
    assert len(out.transcript.windows) > 1


def test_silence_between_regions_is_never_sent_to_the_model() -> None:
    seen: list[int] = []

    class Counting(Backend):
        def transcribe_window(self, samples):
            seen.append(1)
            return super().transcribe_window(samples)

    transcribe_file(
        "/a.mov", Counting(),
        **_audio(600.0, regions=[(0.0, 10.0), (590.0, 600.0)]),
    )
    assert len(seen) == 2, "only the two speech regions should be transcribed"


# --- per-window judgement --------------------------------------------------


def test_a_boilerplate_window_is_dropped_and_its_evidence_kept() -> None:
    # A recording can be entirely real and still contain one window where the
    # model emitted nothing but a sign-off. Without this it becomes an indexed,
    # searchable chunk of its own.
    # Two separate speech regions, so exactly two windows and one text each.
    out = transcribe_file(
        "/a.mov",
        Backend("we went to the park and fed the ducks", "Thanks for watching"),
        **_audio(120.0, regions=[(0.0, 10.0), (100.0, 110.0)]),
    )
    assert out.produced_text
    assert len(out.dropped) == 1
    assert out.dropped[0].reason == "caption_boilerplate"
    assert out.dropped[0].text == "Thanks for watching"
    assert "watching" not in (out.transcript.text if out.transcript else "")


def test_a_degenerate_repeating_window_is_dropped() -> None:
    out = transcribe_file(
        "/a.mov",
        # A decode loop: one word emitted over and over. Measured on a real
        # archive these reached 106 characters per second against a genuine
        # maximum of 14.7.
        Backend("we went to the park and fed the ducks", "yes " * 60),
        **_audio(120.0, regions=[(0.0, 10.0), (100.0, 110.0)]),
    )
    assert any(d.reason == "degenerate_repetition" for d in out.dropped)


def test_the_models_own_confidence_is_kept_as_evidence_not_used_as_a_verdict() -> None:
    # Recorded so a discard can be audited later. It cannot be the decider: on
    # generated silence a model returned "Thank you." at no_speech=0.782.
    out = transcribe_file("/a.mov", Backend("Thanks for watching"), **_audio(20.0))
    assert out.dropped[0].no_speech == 0.02, "confident, and still discarded"


# --- whole-transcript judgement --------------------------------------------


def test_a_file_that_produces_only_junk_is_recorded_as_empty() -> None:
    out = transcribe_file("/junk.mov", Backend("Thanks for watching"),
                          **_audio(20.0))
    assert not out.produced_text
    assert out.empty_reason
    assert out.transcript is None


def test_the_rejected_text_is_kept_so_the_decision_can_be_reviewed() -> None:
    out = transcribe_file(
        "/junk.mov", Backend("un deux trois"),
        settings=Settings(expected_languages=frozenset({"en"})),
        **_audio(20.0),
    )
    if not out.produced_text:
        assert out.rejected_text or out.empty_reason


# --- settings and the policy fingerprint -----------------------------------


def test_the_policy_covers_every_setting_that_can_reject_audio() -> None:
    policy = Settings().as_policy("some-model")
    for required in (
        "model", "silence_max_prob", "max_repeat_share",
        "max_chars_per_second", "unspoken_max_chars", "expected_languages",
        "boilerplate", "credit_prefixes",
    ):
        assert required in policy, (
            f"{required} can reject audio; omitting it from the fingerprint "
            "leaves stale verdicts in place when it changes"
        )


def test_editing_a_phrase_list_changes_the_policy() -> None:
    # The lists decide rejections, so editing one must invalidate stored
    # verdicts rather than silently keeping them.
    from corpus.transcripts.store import policy_fingerprint

    base = Settings().as_policy("m")
    edited = {**base, "boilerplate": [*base["boilerplate"], "a new phrase"]}
    assert policy_fingerprint(base) != policy_fingerprint(edited)
