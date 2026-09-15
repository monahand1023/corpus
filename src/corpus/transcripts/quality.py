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
from functools import cache

__all__ = [
    "CAPTION_SIGNOFF_TAILS",
    "DEFAULT_MAX_CHARS_PER_SECOND",
    "DEFAULT_UNSPOKEN_MAX_CHARS",
    "SUBTITLE_BOILERPLATE",
    "SUBTITLE_CREDIT_PREFIXES",
    "TranscriptVerdict",
    "impossible_speech_rate",
    "judge_transcript",
    "only_unspoken_languages",
    "repeat_share",
    "strip_caption_tail",
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
    # Sign-offs in more languages. A whole transcript that is only "gracias"
    # or "merci" is a thirty-second clip that produced one word -- invention,
    # not a person thanking someone.
    "danke", "danke fur zuschauen", "vielen dank", "gracias", "gracias por ver",
    "gracias por ver el video", "merci", "merci d avoir regarde", "obrigado",
    "obrigada", "obrigado por assistir", "grazie", "grazie per la visione",
    "dank je wel", "bedankt voor het kijken", "spasibo", "спасибо за просмотр",
    "subtitulos", "gracias por su atencion",
    # Spanish-language YouTube calls to action. Found glued to real family
    # speech -- "Cierra la puerta por favor ¡Suscríbete al canal".
    "suscribete",
    "suscribete al canal",
    "suscribete y activa la campanita",
    "dale like y suscribete",
    "suscribete al canal y activa las notificaciones",
    "si te gusta el video dale like y suscribete al canal y activa las notificaciones",
    # Non-speech annotations the model emits instead of text, and bare
    # interjections that carry nothing findable on their own.
    "applause", "music", "silence", "foreign", "laughter", "inaudible",
    "background noise", "no audio", "blank audio",
    "hmm", "mm", "mmm", "uh", "um", "oh", "ah", "eh", "ok", "okay",
    "yeah", "yes", "no", "so", "the", "you", "oh my god",
    # CJK sign-offs, filler and non-speech annotations.
    "ありがとうございました", "ありがとうございます", "おやすみなさい", "音楽", "拍手",
    "字幕", "字幕由amara org社群提供", "谢谢", "谢谢观看", "請不吝點贊",
    "请不吝点赞 订阅 转发 打赏支持明镜与点点栏目", "terima kasih banyak", "спасибо", "감사합니다", "구독과 좋아요",
    "매주 일요일 업로드됩니다",
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

# Sign-offs safe to cut off the END of a window that still holds real speech.
#
# This is a deliberately SMALLER set than SUBTITLE_BOILERPLATE, and the two do
# different jobs. SUBTITLE_BOILERPLATE answers "is this whole transcript a
# caption artefact?", where "Thank you." as the entire text of a 30-minute
# recording is obviously invented. That reasoning does not survive being moved
# to a single window: "Thank you." as one 30-second window inside a longer
# recording is somebody talking, and dropping it deletes real content.
#
# So membership here has one test -- could a person plausibly have SAID this?
# Excluded for failing it:
#   * short generic politeness ("ok", "thank you", "gracias", "ありがとうございます")
#   * "thank you very much", which is real speech. Stripping it anywhere it
#     appeared turned a recorded meeting's "thank you very much for coming
#     today" into "for coming today".
# Included: broadcast/caption register ("thank you for VIEWING", "please
# subscribe"), and sign-offs in languages the archive does not contain.
#
# Anything from this set is removed ONLY when it sits at the very end of the
# text. Matching it anywhere is what destroyed the meeting above, and an
# unanchored match also chewed into a real word -- "Takk for ating medieting."
# became "ieting."
CAPTION_SIGNOFF_TAILS: frozenset[str] = frozenset({
    # Caption register: about watching a video, not about the conversation.
    "ご視聴ありがとうございました",
    "ご視聴ありがとうございます",
    "チャンネル登録お願いします",
    "字幕由amara org社群提供",
    "구독과 좋아요 부탁드립니다",
    "매주 일요일 업로드됩니다",
    "请不吝点赞 订阅 转发 打赏支持明镜与点点栏目",
    "please subscribe",
    "thanks for watching",
    "thank you for watching",
    "gracias por ver",
    "gracias por ver el video",
    "gracias por su atencion",
    "спасибо за просмотр",
    "suscribete",
    "suscribete al canal",
    "suscribete y activa la campanita",
    "dale like y suscribete",
    "suscribete al canal y activa las notificaciones",
    "si te gusta el video dale like y suscribete al canal y activa las notificaciones",
    # A transcriber's own annotation, never a spoken line.
    "background noise",
    # Sign-offs in languages a given archive may not contain at all. Harmless
    # to carry everywhere: each is a fixed caption phrase, not conversation.
    "takk for ating med",
    "takk for at du sa pa",
    "takk for at du sa med",
    "tusen takk for at du sa med",
    "danke fur zuschauen",
    "vielen dank",
    "bedankt voor het kijken",
    "merci d avoir regarde",
    "obrigado por assistir",
    "grazie per la visione",
    "terima kasih telah menonton",
    "terima kasih banyak",
})

_WHITESPACE = re.compile(r"\s+")
# Includes the characters transcribers actually emit: U+2019 is Whisper's
# default apostrophe, and Spanish opens with ¡ and ¿.
_PUNCT = re.compile(
    r"[.,!?;:'\"()\[\]<>/\\|`~@#$%^&*_+=-]"
    r"|[\u2018\u2019\u201c\u201d\u2026\u00a1\u00bf]"
    r"|[。！？、，「」・～]"
)
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
    # NFC first. Folding examines each character, so a standalone combining
    # mark -- which is what NFD produces, and what macOS emits routinely --
    # slipped through untouched and "vídeo" could not match "video".
    text = unicodedata.normalize("NFC", text)
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
        # keeps real speech that merely begins with an invented credit: a
        # fabricated credit line often precedes genuine audio, and matching
        # the prefix alone discards the recording to remove the noise.
        tail = norm[len(head):].strip()
        if len(tail) > _CREDIT_TAIL_MAX_CHARS:
            continue
        # The script guard exists to spare real speech that follows a
        # fabricated Latin-script credit. A CJK credit's tail is ALWAYS CJK,
        # so applying it there made those prefixes unreachable -- the same
        # "cannot fire" defect this list has had twice. Judge the tail against
        # the script the PREFIX is written in.
        if _UNSPACED_SCRIPT.search(head) or not _UNSPACED_SCRIPT.search(tail):
            return True
    return False


# Combining marks, which NFD splits off from their base letter.
# Combining marks NFD splits off from their base character. The second range
# is the Japanese voiced/semi-voiced sound marks: NFD turns ご into こ + U+3099,
# and omitting them here made every CJK phrase unmatchable -- the same class of
# silent failure as NFKD collapsing ご onto こ.
_COMBINING = "̀-゙ͯ-゚"
# Punctuation that OPENS a clause, so it belongs to the sign-off being cut
# rather than to the speech before it. Without this, cutting the sign-off off
# "...por favor ¡Suscríbete al canal" left a lone "¡" behind.
_OPENING_PUNCT = "¡¿"
# How much of the end to screen for a possible sign-off. Comfortably longer
# than the longest phrase (79 characters) and than the longest credit prefix
# plus the name tail it permits, with room for the punctuation and accents
# that normalisation strips from the stored forms but not from real text.
_SCREEN_WINDOW_CHARS = 512
_MARK_RANGE = frozenset(
    [chr(c) for c in range(0x0300, 0x0370)] + [chr(0x3099), chr(0x309A)]
)
_IS_MARK = _MARK_RANGE.__contains__


def _accent_tolerant(token: str) -> str:
    """A token pattern that also matches its accented spellings.

    The phrase lists are stored normalised, which folds accents away, but the
    text being cleaned keeps them: the stored "suscribete" has to match a real
    "Suscríbete". Case-insensitivity does NOT do this -- `í` and `i` are
    different characters -- and that gap is the same "cannot fire" defect that
    left 22 of 45 entries in these lists unreachable. Matching against
    NFD-decomposed text, where an accent is a separate combining mark, makes
    the mark optional after every letter.
    """
    decomposed = unicodedata.normalize("NFD", token)
    return "".join(re.escape(ch) + f"[{_COMBINING}]*" for ch in decomposed)


@cache
def _tail_pattern(phrase: str) -> re.Pattern[str]:
    """Match a normalised phrase against NFD text, anchored to the end.

    `[\\s\\W]*` between tokens absorbs the punctuation and spacing that
    normalisation would have removed, and the trailing `$` is the whole safety
    property.
    """
    body = r"[\s\W]*".join(_accent_tolerant(token) for token in phrase.split())
    lead = rf"[\s]*[{_OPENING_PUNCT}]*[\s]*"
    return re.compile(lead + body + r"[\s\W]*$", re.IGNORECASE)


@cache
def _credit_pattern(prefix: str) -> re.Pattern[str]:
    """Find a credit prefix anywhere; the caller decides if it ends the text."""
    body = r"[\s\W]*".join(_accent_tolerant(token) for token in prefix.split())
    return re.compile(body, re.IGNORECASE)


def _strip_credit_line(text: str, prefixes: tuple[str, ...]) -> str:
    """Cut a trailing credit line -- prefix plus the name it ends in.

    These cannot go in a phrase list: "Субтитры создавал DimaTorzok" ends in a
    name that is not knowable in advance. What IS knowable is that only a name
    follows, so a prefix counts as a credit line when what trails it is short.
    Anything longer is real speech that happens to sit behind a fabricated
    credit, and cutting there would delete the recording to remove the noise.
    """
    best = None
    for prefix in prefixes:
        for match in _credit_pattern(prefix).finditer(text):
            tail = text[match.end():].strip()
            if len(_normalise(tail)) > _CREDIT_TAIL_MAX_CHARS:
                continue
            # Judge the tail against the script the PREFIX is written in: a CJK
            # credit's name is always CJK, so applying the Latin guard there
            # would make those prefixes unreachable.
            if _UNSPACED_SCRIPT.search(prefix) or not _UNSPACED_SCRIPT.search(tail):
                best = match.start() if best is None else min(best, match.start())
    return text if best is None else text[:best].strip()


def _without_marks(decomposed: str) -> str:
    """Drop combining marks from NFD text. Used on BOTH sides of the screen."""
    return "".join(ch for ch in decomposed if ch not in _MARK_RANGE)


def _fold_for_screen(text: str) -> str:
    return _without_marks(unicodedata.normalize("NFD", text)).casefold()


@cache
def _screen_anchors(phrases: frozenset[str], prefixes: tuple[str, ...]) -> tuple[str, ...]:
    """The longest word of each phrase, folded, as a cheap literal to look for.

    A regex alternation over ~70 phrases was the whole cost of screening: an
    accent-tolerant one puts a character class after every character and
    backtracks across all of them at every position, and even a literal one
    scales with the window. A plain `in` test uses CPython's substring search,
    which is roughly two orders of magnitude faster here.

    Taking the LONGEST word keeps the anchor selective -- "suscribete" rather
    than "y", "watching" rather than "for" -- and an anchor can only ever make
    the screen accept text the full pass then declines to change. It cannot
    make it reject text the full pass would have changed, which is the only
    property a screen has to have.
    """
    anchors = set()
    for phrase in (*phrases, *prefixes):
        folded = _fold_for_screen(phrase)
        words = folded.split() or [folded]
        anchors.add(max(words, key=len))
    return tuple(anchors)


def strip_caption_tail(
    text: str,
    *,
    tails: frozenset[str] | None = None,
    credit_prefixes: tuple[str, ...] | None = None,
) -> str:
    """Remove a caption sign-off from the end of `text`, leaving the rest.

    For text where a sign-off is glued to real speech -- the common shape, and
    one `subtitle_boilerplate` deliberately will not touch, because discarding
    a whole transcript for containing a sign-off measurably deletes real
    recordings. Returns `text` unchanged when nothing matches, and may return
    "" when the text was nothing but a sign-off.

    Removal is anchored to the end for a reason worth keeping: matching these
    phrases anywhere they appear cut "thank you very much for coming today"
    down to "for coming today", and truncated "Takk for ating medieting." to
    "ieting." by chewing into the following word.
    """
    if not text.strip():
        return text
    phrases = tails if tails is not None else CAPTION_SIGNOFF_TAILS
    heads = (
        credit_prefixes if credit_prefixes is not None else SUBTITLE_CREDIT_PREFIXES
    )
    # NFD so an accent is a separate combining mark the patterns can treat
    # as optional; recomposed to NFC on the way out so callers never see
    # decomposed text.
    out = unicodedata.normalize("NFD", text)
    # Nothing to remove: return the input UNCHANGED rather than a normalised,
    # whitespace-collapsed copy. Callers compare the result against the input
    # to decide whether a sign-off was present, so returning a cosmetically
    # different string for clean text would report contamination that is not
    # there.
    # Only the END of the text can matter, so only the end is screened.
    # Both rules are tail-bounded: a sign-off has to finish the text, and a
    # credit line counts only when at most _CREDIT_TAIL_MAX_CHARS of name
    # follow it. The longest phrase and the longest prefix plus that tail both
    # fit inside the window with room to spare, so nothing the full pass would
    # act on can start before it. Without this, screening a 2 GB index means
    # scanning every character of every chunk to decide it ends in "ducks".
    # Normalising only the window matters as much as searching only the
    # window: NFD over a whole 5,000-character chunk cost more than the search
    # it was preparing for.
    window = _fold_for_screen(text[-_SCREEN_WINDOW_CHARS:])
    anchors = _screen_anchors(frozenset(phrases), tuple(heads))
    if not any(anchor in window for anchor in anchors):
        return text
    out = unicodedata.normalize("NFD", text)
    # Longest first: "gracias por ver el video" must win over "gracias por ver",
    # which would otherwise leave "el video" stranded.
    for phrase in sorted(phrases, key=len, reverse=True):
        stripped = _tail_pattern(phrase).sub("", out)
        if stripped != out:
            out = stripped
    out = _strip_credit_line(out, heads)
    out = _WHITESPACE.sub(" ", out).strip()
    out = unicodedata.normalize("NFC", out)
    # Spanish opens with punctuation, so cutting "gracias por ver el video"
    # off "¡Gracias por ver el video!" leaves a lone "¡". Nothing that
    # normalises away to nothing is speech.
    return "" if out and not _normalise(out) else out


# Fewest units needed before a repetition ratio means anything. With one unit
# the ratio is 1.0 by construction, so any three-word transcript scored a
# perfect 1.00 and a long-enough one was rejected as degenerate.
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
    found = {lg.lower() for lg in (detected or []) if lg and lg != "?"}
    allowed = {lg.lower() for lg in expected}
    return bool(found) and not (found & allowed)


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
    duration_s: float,
    languages: list[str] | None = None,
    expected_languages: set[str] | frozenset[str] | None = None,
    max_repeat_share: float = 0.9,
    max_chars_per_second: float = DEFAULT_MAX_CHARS_PER_SECOND,
    unspoken_max_chars: int = DEFAULT_UNSPOKEN_MAX_CHARS,
) -> TranscriptVerdict:
    """Apply every signal. Returns why it was rejected, or keep=True.

    `duration_s` is required rather than defaulting to zero: a default of 0.0
    fails the rate check's own guard, so callers that omitted it silently got
    one fewer signal than they thought.

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
