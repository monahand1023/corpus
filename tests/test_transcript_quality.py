"""Signals separating invented transcript text from real speech.

Fixtures are shapes taken from a real real archive. Each rule
exists because a specific transcript looked wrong and the reason was chased
down; the tests keep those reasons enforced.
"""

from __future__ import annotations

import pytest

from corpus.transcripts import (
    TranscriptVerdict,
    impossible_speech_rate,
    judge_transcript,
    only_unspoken_languages,
    repeat_share,
    subtitle_boilerplate,
)

SPOKEN = {"en", "ja", "es"}


# --- caption boilerplate --------------------------------------------------

@pytest.mark.parametrize("text", [
    "Thanks for watching!", "Takk for at du så med.", "Terima kasih telah menonton",
    "ご視聴ありがとうございました", "Субтитры сделал",
])
def test_caption_signoffs_are_recognised(text: str) -> None:
    assert subtitle_boilerplate(text) is True


def test_a_signoff_inside_real_prose_is_not_a_match() -> None:
    """Only the WHOLE transcript counts. Real talks say "thank you" too."""
    assert subtitle_boilerplate(
        "Thanks for watching the demo, now let me show you the roadmap.") is False


# --- languages nobody speaks ----------------------------------------------

def test_a_wholly_foreign_transcript_is_flagged() -> None:
    assert only_unspoken_languages(["nn"], SPOKEN) is True
    assert only_unspoken_languages(["jw", "vi"], SPOKEN) is True


@pytest.mark.parametrize("langs", [["en"], ["ja"], ["en", "nn"], ["cy", "en", "ro"]])
def test_one_spoken_language_protects_the_transcript(langs: list[str]) -> None:
    """A genuine hour-long talk was tagged cy,en,ro.

    An "any foreign tag" test would delete it; requiring ALL foreign does not.
    """
    assert only_unspoken_languages(langs, SPOKEN) is False


def test_absent_or_unknown_language_is_not_grounds_for_rejection() -> None:
    for langs in ([], None, ["?"]):
        assert only_unspoken_languages(langs, SPOKEN) is False


def test_expected_languages_are_the_callers_to_supply() -> None:
    """An archive's languages belong to its people, not to this library."""
    assert only_unspoken_languages(["nn"], {"nn"}) is False


# --- impossible speech rate -----------------------------------------------

# Varied prose, so only the RATE can flag it -- a wall of one character is
# degenerate repetition as well, and would not prove this rule fires.
_FAST_BUT_VARIED = ("the quarterly numbers came in ahead of plan and the team "
                    "has already started work on the next phase of migration "
                    "which we expect to finish before the end of the year ") * 4


def test_a_decode_loop_exceeds_any_real_speech_rate() -> None:
    """Over a thousand characters from an eleven-second clip."""
    assert len(_FAST_BUT_VARIED) > 600
    assert repeat_share(_FAST_BUT_VARIED) < 0.9, "must not be flagged as repetition"
    assert impossible_speech_rate(_FAST_BUT_VARIED, 10.7) is True


@pytest.mark.parametrize("n,secs", [(1000, 68.0), (500, 36.0), (200, 60.0)])
def test_real_speech_rates_survive(n: int, secs: float) -> None:
    assert impossible_speech_rate("x" * n, secs) is False


def test_rate_is_not_judged_on_a_clip_too_short_to_measure() -> None:
    assert impossible_speech_rate("hello there", 0.5) is False


# --- repetition -----------------------------------------------------------

def test_degenerate_repetition_scores_high() -> None:
    assert repeat_share("Mm mm mm mm mm mm mm mm mm mm mm mm") > 0.9


def test_repetition_is_measurable_without_word_spaces() -> None:
    """Khmer, Japanese and Thai have no spaces; whitespace splitting scores 0."""
    wall = "បបបបបបបបបបបបបបបបបបបបបបបបបបបបបប"
    assert len(wall.split()) == 1
    assert repeat_share(wall) > 0.9


def test_varied_prose_scores_low() -> None:
    assert repeat_share(
        "The quarterly numbers came in ahead of plan and the team has started "
        "work on the next phase of the migration.") < 0.2


# --- the combined judgement ------------------------------------------------

def test_real_speech_is_kept() -> None:
    v = judge_transcript(
        "Welcome everyone. Thanks very much for coming today. My name is Tim "
        "and I will walk through the quarterly results before questions.",
        duration_s=68.0, languages=["cy", "en", "ro"], expected_languages=SPOKEN)
    assert v.keep is True and bool(v) is True


@pytest.mark.parametrize("text,secs,langs,reason", [
    ("Takk for at du så med.", 31.0, ["nn"], "subtitle_boilerplate"),
    ("Ae! Ae! Ae! Ae! Ae! Ae! Ae! Ae!", 16.0, ["haw"], "only_unspoken_languages"),
    (_FAST_BUT_VARIED, 10.7, ["jw"], "impossible_speech_rate"),
    ("", 10.0, ["en"], "empty"),
])
def test_each_signal_reports_its_own_reason(text, secs, langs, reason) -> None:
    v = judge_transcript(text, duration_s=secs, languages=langs,
                         expected_languages=SPOKEN)
    assert v.keep is False and v.reason == reason


def test_a_long_foreign_tagged_transcript_survives_the_language_rule() -> None:
    """Two minutes of English conversation came back tagged Hawaiian.

    Length is what keeps the language signal to the sign-offs it is for.
    """
    v = judge_transcript(
        "I'm going to put on some diamonds. Uh, the lollipops? Yeah. " * 6,
        duration_s=124.8, languages=["haw", "pt"], expected_languages=SPOKEN)
    assert v.keep is True


def test_no_expected_languages_disables_the_language_rule() -> None:
    v = judge_transcript("Takk for noe helt annet her", duration_s=30.0,
                         languages=["nn"])
    assert v.keep is True


def test_the_repetition_ceiling_stays_permissive() -> None:
    """Lower thresholds delete real recordings.

    At 0.6 a reference archive lost a clip of a child repeating one word --
    noise to a filter, and the reason its owner keeps the archive.
    """
    v = judge_transcript("Daddy Daddy Daddy Daddy no no no no yes ok",
                         duration_s=20.0, expected_languages=SPOKEN)
    assert v.keep is True


def test_the_verdict_is_truthy_for_keep() -> None:
    assert bool(TranscriptVerdict(True)) is True
    assert bool(TranscriptVerdict(False, "x")) is False


# --- credit lines end in a name, and nothing else --------------------------

@pytest.mark.parametrize("text", [
    "Субтитры создавал DimaTorzok",
    "Untertitel von Stefan K.",
    "Teksting av Nicolai Winther",
    "Subtitles by the Amara.org community",
    "Sous-titres réalisés par la communauté d'Amara.org",
    "Altyazı M.K.",
])
def test_a_credit_line_is_caught_whatever_name_it_ends_with(text: str) -> None:
    """Whole-string equality could not match these.

    A credit ends in a name that cannot be enumerated, so 22 of the first 45
    entries in this list were unreachable and 8 of 10 realistic outputs went
    straight through.
    """
    assert subtitle_boilerplate(text) is True


@pytest.mark.parametrize("text", [
    "Субтитры сделал DimaTorzok 今日はいい天気ですね。",
    "Субтитры сделал DimaTorzok これは本物の音声です。聞こえますか?",
    "Altyazı M.K. Terima kasih telah menonton! 本日はご参加ありがとうございます。",
])
def test_real_speech_behind_an_invented_credit_survives(text: str) -> None:
    """These were deleted, and they are recordings of someone's family.

    An invented credit often precedes real audio. Matching the prefix alone
    discards the recording to remove the noise; the tail after the prefix has
    to look like a name for the line to be a credit.
    """
    assert subtitle_boilerplate(text) is False


def test_a_sentence_that_merely_opens_like_a_credit_survives() -> None:
    assert subtitle_boilerplate(
        "Subtitles by hand are a pain, so we built a tool that makes them.") is False


def test_accents_and_dots_do_not_hide_a_credit() -> None:
    """"Amara.org" against "amara org", "réalisés" against "realises"."""
    assert subtitle_boilerplate("Subtitles by the Amara.org community") is True
    assert subtitle_boilerplate("Subtitulos realizados por Amara") is True


def test_normalisation_leaves_japanese_intact() -> None:
    """NFKD splits dakuten -- ご becomes こ -- so every Japanese phrase stopped
    matching. Folding is confined to Latin letters."""
    assert subtitle_boilerplate("ご視聴ありがとうございました") is True


@pytest.mark.parametrize("text", ["Happy birthday everyone", "I love you", "Yes absolutely"])
def test_a_short_phrase_is_not_all_repetition(text: str) -> None:
    """repeat_share returned 1.00 for any three-word text.

    One trigram repeats itself by construction, so the shortest real
    utterances scored as perfectly degenerate.
    """
    assert repeat_share(text) == 0.0
    assert judge_transcript(text, duration_s=20.0).keep is True
