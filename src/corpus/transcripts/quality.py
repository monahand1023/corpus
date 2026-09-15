"""Deciding whether a machine transcript contains anything real.

Three independent signals, each derived from a failure observed on a real
real archive rather than from theory. They are cheap, need no
model, and work on the text a transcriber already produced -- so they apply
to any speech-to-text output, not just one engine's.

WHY NOT VOICE-ACTIVITY DETECTION. The obvious approach is to run a VAD and
discard anything it reports as speechless. Measured, that is unsafe as a
QUALITY gate: on the archive above, a hallucinated subtitle sign-off peaked at
0.145 speech probability while a genuine recording of someone calling out a
name peaked at 0.144. Auditing nine VAD rejections by hand, four were real
recordings -- quiet, distant or reverberant audio that a model trained on
clean speech does not hear. Use VAD to skip near-silence, which saves the cost
of transcribing it, and judge the resulting text with these signals.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

__all__ = [
    "DEFAULT_MAX_CHARS_PER_SECOND",
    "DEFAULT_UNSPOKEN_MAX_CHARS",
    "SUBTITLE_BOILERPLATE",
    "TranscriptVerdict",
    "impossible_speech_rate",
    "judge_transcript",
    "only_unspoken_languages",
    "repeat_share",
    "subtitle_boilerplate",
]


# Caption sign-offs the model reproduces over silence, in the languages it
# tends to drift to. A hit is near-proof of invention rather than translation:
# these are the literal strings scraped subtitle files end with.
SUBTITLE_BOILERPLATE: frozenset[str] = frozenset({
    # English
    "thank you", "thanks", "thank you very much", "thank you for watching",
    "thanks for watching", "please subscribe", "subscribe", "bye", "bye bye",
    "subtitles by the amara org community", "amara org community",
    # Scandinavian -- the tell that started this: hundreds of clips came back
    # Norwegian on an archive with no Norwegian speakers.
    "takk for at du sa med", "takk for at du så med", "takk for ating med",
    "takk for at du så på", "tusen takk for at du så med", "teksting av",
    "undertekster av", "tekstning af",
    # Continental European
    "ondertiteling door", "untertitel von", "untertitelung im auftrag des zdf",
    "sous-titres realises par", "sous-titrage société radio-canada",
    "subtitulos realizados por", "subtitulos por la comunidad de amara org",
    "legendas pela comunidade amara org", "legendado pela comunidade amara org",
    "sottotitoli e revisione a cura di", "sottotitoli creati dalla comunità amara org",
    "napisy stworzone przez", "napisy od", "субтитры создавал", "субтитры сделал",
    # Asian
    "ご視聴ありがとうございました", "ご視聴ありがとうございます", "チャンネル登録お願いします",
    "字幕製作", "感谢观看", "시청해주셔서 감사합니다", "구독과 좋아요 부탁드립니다",
    "terima kasih telah menonton", "terima kasih", "altyazı m k", "abone ol",
})

# Real speech tops out far below this. On the reference archive the fastest
# genuine content was 14.7 characters per second (a recorded talk) and the
# 99th percentile was 17.7, while decode loops reached 106 -- over a thousand
# characters from an eleven-second clip. Scripts that pack more meaning into
# fewer characters, like Japanese, produce LOWER rates, so one ceiling serves
# every language.
DEFAULT_MAX_CHARS_PER_SECOND = 25.0

# The language signal is trusted only for text short enough to be a sign-off.
# Transcribers mislabel real speech too: two minutes of English conversation
# came back tagged Hawaiian and Portuguese.
DEFAULT_UNSPOKEN_MAX_CHARS = 150

_WHITESPACE = re.compile(r"\s+")
# Scripts written without spaces between words need character n-grams; split on
# whitespace and a wall of identical syllables is a single token scoring zero
# repetition.
_UNSPACED_SCRIPT = re.compile(r"[぀-ヿ一-鿿ក-៿฀-๿]")


def _normalise(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip().lower().rstrip(".!?。！？")


def subtitle_boilerplate(text: str, *, phrases: frozenset[str] | None = None) -> bool:
    """True when the WHOLE transcript is one known caption sign-off."""
    return _normalise(text) in (phrases if phrases is not None else SUBTITLE_BOILERPLATE)


def repeat_share(text: str) -> float:
    """Largest share of `text` occupied by one repeated unit, 0.0-1.0."""
    if _UNSPACED_SCRIPT.search(text):
        units = [text[i:i + 6] for i in range(max(1, len(text) - 5))]
    else:
        words = text.lower().split()
        units = [" ".join(words[i:i + 3]) for i in range(max(1, len(words) - 2))]
    if not units:
        return 0.0
    return Counter(units).most_common(1)[0][1] / len(units)


def only_unspoken_languages(
    detected: list[str] | set[str] | None, expected: set[str] | frozenset[str]
) -> bool:
    """True when EVERY detected language is one the archive's people do not speak.

    Every one, never merely one of them. Transcribers routinely tag a single
    real recording with several languages -- one genuine hour-long talk came
    back tagged Welsh, English and Romanian -- so an "any" test deletes real
    material. `expected` is the caller's to supply; an archive's languages are
    a property of its people, not of this library.
    """
    found = {lg for lg in (detected or []) if lg and lg != "?"}
    return bool(found) and not (found & set(expected))


def impossible_speech_rate(
    text: str, duration_s: float, *, ceiling: float = DEFAULT_MAX_CHARS_PER_SECOND
) -> bool:
    """True when more text was produced than the audio could physically contain.

    The clearest signal of a decode loop, and it needs no language knowledge:
    a transcriber stuck repeating a phrase emits text at a rate no speaker
    reaches.
    """
    return duration_s > 1.0 and len(text) / duration_s > ceiling


@dataclass(frozen=True)
class TranscriptVerdict:
    """Why a transcript was rejected, or that it was kept."""

    keep: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.keep


def judge_transcript(
    text: str,
    *,
    duration_s: float = 0.0,
    languages: list[str] | None = None,
    expected_languages: set[str] | frozenset[str] | None = None,
    max_repeat_share: float = 0.9,
    max_chars_per_second: float = DEFAULT_MAX_CHARS_PER_SECOND,
    unspoken_max_chars: int = DEFAULT_UNSPOKEN_MAX_CHARS,
) -> TranscriptVerdict:
    """Apply every signal. Returns why it was rejected, or keep=True.

    `max_repeat_share` is deliberately high. Lower thresholds delete real
    recordings: at 0.6 a reference archive lost a clip of a child repeating one
    word, which is noise to a filter and the reason its owner keeps the
    archive. Only genuinely degenerate output sits above 0.9.
    """
    if not text.strip():
        return TranscriptVerdict(False, "empty")
    if subtitle_boilerplate(text):
        return TranscriptVerdict(False, "subtitle_boilerplate")
    if len(text) > 40 and repeat_share(text) >= max_repeat_share:
        return TranscriptVerdict(False, "degenerate_repetition")
    if impossible_speech_rate(text, duration_s, ceiling=max_chars_per_second):
        return TranscriptVerdict(False, "impossible_speech_rate")
    if (
        expected_languages
        and len(text) < unspoken_max_chars
        and only_unspoken_languages(languages, expected_languages)
    ):
        return TranscriptVerdict(False, "only_unspoken_languages")
    return TranscriptVerdict(True)
