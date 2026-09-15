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
import unicodedata
from collections import Counter
from dataclasses import dataclass

__all__ = [
    "DEFAULT_MAX_CHARS_PER_SECOND",
    "DEFAULT_UNSPOKEN_MAX_CHARS",
    "SUBTITLE_BOILERPLATE",
    "SUBTITLE_CREDIT_PREFIXES",
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
# Caption sign-offs the model reproduces over silence. Split in two because
# they fail differently.
#
# EXACT: short, complete phrases. Matched against the WHOLE transcript, never
# a substring -- real talks say "thank you" too.
SUBTITLE_BOILERPLATE: frozenset[str] = frozenset({
    "thank you", "thanks", "thank you very much", "thank you for watching",
    "thanks for watching", "please subscribe", "subscribe", "bye", "bye bye",
    "takk for at du sa med", "takk for ating med", "takk for at du sa pa",
    "tusen takk for at du sa med", "tekstning af",
    "terima kasih telah menonton", "terima kasih",
    "感谢观看", "字幕製作", "ご視聴ありがとうございました", "ご視聴ありがとうございます",
    "チャンネル登録お願いします", "시청해주셔서 감사합니다", "구독과 좋아요 부탁드립니다",
    "abone ol",
})

# PREFIXES: credit lines that end in a name, studio or community, so the full
# string is never known in advance -- "Субтитры создавал DimaTorzok",
# "Untertitel von Stefan K.", "Subtitles by the Amara.org community".
#
# Whole-string equality missed every one of these. Of forty-five entries in
# the first version of this list, twenty-two could not fire at all: prefixes
# awaiting a name, and entries written with accents or dots that normalisation
# never removed. Eight of ten realistic outputs slipped through.
#
# Each prefix is long and distinctive enough that a real sentence does not
# begin with it.
SUBTITLE_CREDIT_PREFIXES: tuple[str, ...] = (
    "subtitles by", "subtitling by", "captions by", "transcribed by",
    "transcription by", "amara org", "subtitles by the amara",
    "undertekster av", "teksting av", "ondertiteling door",
    "untertitel von", "untertitelung im auftrag",
    "sous titres realises par", "sous titrage", "soustitres par",
    "subtitulos realizados por", "subtitulos por", "subtitulado por",
    "legendas pela", "legendado pela", "legendas por",
    "sottotitoli e revisione a cura di", "sottotitoli creati",
    "napisy stworzone przez", "napisy od", "tlumaczenie",
    "субтитры создавал", "субтитры сделал", "субтитры делал",
    "перевод субтитров", "редактор субтитров",
    "altyazi", "字幕由", "字幕製作", "翻译由",
)

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
_PUNCT = re.compile(r"[.,!?;:'\"()\[\]<>/\\|`~@#$%^&*_+=-]|[。！？、，]")
# What may follow a credit prefix: a name, a studio, "the Amara.org
# community". Anything longer, or anything in a script the prefix's language
# does not use, is real speech that happens to sit behind an invented credit.
_CREDIT_TAIL_MAX_CHARS = 30
# Scripts written without spaces between words need character n-grams; split on
# whitespace and a wall of identical syllables is a single token scoring zero
# repetition.
_UNSPACED_SCRIPT = re.compile(r"[぀-ヿ一-鿿ក-៿฀-๿]")


def _is_latin_letter(ch: str) -> bool:
    try:
        return "LATIN" in unicodedata.name(ch)
    except ValueError:
        return False


def _normalise(text: str) -> str:
    """Lowercase, collapse whitespace, and drop what varies between renderings.

    Accents and interior punctuation are removed because these lines arrive
    inconsistently: "Amara.org" against "amara org", "réalisés" against
    "realises". Comparing raw strings meant entries in the list could never
    match the output they were written for.
    """
    out = []
    for ch in text:
        # Fold accents on LATIN letters only. Decomposing everything and
        # dropping combining marks destroys Japanese dakuten -- ご becomes こ --
        # so every Japanese phrase in the list stopped matching.
        if "a" <= ch.lower() <= "z" or _is_latin_letter(ch):
            d = unicodedata.normalize("NFKD", ch)
            out.append("".join(c for c in d if not unicodedata.combining(c)))
        else:
            out.append(ch)
    folded = _PUNCT.sub(" ", "".join(out).lower())
    # Turkish dotless i does not decompose to ASCII i.
    folded = folded.replace("\u0131", "i").replace("\u0130", "i")
    return _WHITESPACE.sub(" ", folded).strip()


def subtitle_boilerplate(
    text: str,
    *,
    phrases: frozenset[str] | None = None,
    prefixes: tuple[str, ...] | None = None,
) -> bool:
    """True when the WHOLE transcript is a caption sign-off.

    Exact against `phrases`, or beginning with one of `prefixes` -- credit
    lines end in a name that cannot be enumerated. Prefix matching is confined
    to short text so a real recording that happens to open with a similar
    phrase is not discarded.
    """
    norm = _normalise(text)
    if not norm:
        return False
    if norm in (phrases if phrases is not None else SUBTITLE_BOILERPLATE):
        return True
    heads = prefixes if prefixes is not None else SUBTITLE_CREDIT_PREFIXES
    for head in heads:
        if not norm.startswith(head):
            continue
        # What follows a credit prefix is a name, nothing more. Requiring that
        # keeps real speech that merely begins with an invented credit:
        # "Субтитры сделал DimaTorzok これは本物の音声です。" is a hallucinated
        # prefix in front of a genuine recording their family, and discarding the
        # whole transcript loses the recording to save the noise.
        tail = norm[len(head):].strip()
        if len(tail) <= _CREDIT_TAIL_MAX_CHARS and not _UNSPACED_SCRIPT.search(tail):
            return True
    return False


# Fewest units needed before a repetition ratio means anything. With one unit
# the ratio is 1.0 by construction -- "Happy birthday everyone" scored a perfect
# 1.00 and any long-enough three-word transcript was rejected as degenerate.
_MIN_UNITS_FOR_RATIO = 4


def repeat_share(text: str) -> float:
    """Largest share of `text` occupied by one repeated unit, 0.0-1.0.

    Returns 0.0 when the text is too short to measure. A ratio over one or two
    units says nothing about repetition, and reporting 1.0 there turns the
    shortest real utterances into false positives.
    """
    units: list[str] = []
    if not _UNSPACED_SCRIPT.search(text):
        words = text.lower().split()
        units = [" ".join(words[i:i + 3]) for i in range(len(words) - 2)]
    if len(units) < _MIN_UNITS_FOR_RATIO:
        # Character n-grams, for scripts without word spaces and for a single
        # enormous "word" -- "Mmmmmmmmmm..." is one token, so word units
        # cannot see it, and it is exactly the degenerate output to catch.
        lowered = text.lower()
        units = [lowered[i:i + 6] for i in range(len(lowered) - 5)]
    if len(units) < _MIN_UNITS_FOR_RATIO:
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
