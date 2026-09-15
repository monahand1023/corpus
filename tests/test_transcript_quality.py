"""Signals separating invented transcript text from real speech.

Fixtures are shapes taken from a real real archive. Each rule
exists because a specific transcript looked wrong and the reason was chased
down; the tests keep those reasons enforced.
"""

from __future__ import annotations

import pytest

from corpus.transcripts import (
    CAPTION_SIGNOFF_TAILS,
    SUBTITLE_BOILERPLATE,
    TranscriptVerdict,
    impossible_speech_rate,
    judge_transcript,
    only_unspoken_languages,
    repeat_share,
    strip_caption_tail,
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
    """A fabricated credit line often sits in front of genuine audio.

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
    # Well under the rejection threshold is what matters; an exact 0.0 would
    # only hold for the word-unit path, and short text falls back to
    # character n-grams so that "Mmmmmm..." -- a single token -- is still seen.
    assert repeat_share(text) < 0.5
    assert judge_transcript(text, duration_s=20.0).keep is True


# --- defects found by pre-publication review -------------------------------

@pytest.mark.parametrize("text", [
    "字幕由大力中文字幕组提供", "字幕製作 中文字幕组", "翻译由志愿者提供",
])
def test_a_cjk_credit_prefix_can_fire(text: str) -> None:
    """These were unreachable: the tail guard rejected any CJK tail.

    The guard spares real speech following a fabricated Latin credit, but a
    CJK credit's tail is ALWAYS CJK, so those prefixes could never match --
    the same "cannot fire" defect this list has had twice before.
    """
    assert subtitle_boilerplate(text) is True


@pytest.mark.parametrize("text", [
    "Gracias por ver el vídeo", "merci d'avoir regardé", "Subtítulos",
])
def test_decomposed_and_composed_input_match_alike(text: str) -> None:
    """macOS emits NFD routinely, and folding examined each character.

    A standalone combining mark survived untouched, so "vídeo" written as
    an i followed by a combining accent could not match "video".
    """
    import unicodedata
    assert (subtitle_boilerplate(unicodedata.normalize("NFC", text))
            is subtitle_boilerplate(unicodedata.normalize("NFD", text)) is True)


@pytest.mark.parametrize("text", [
    "merci d’avoir regardé",   # U+2019, a transcriber's default apostrophe
    "¡Gracias por ver!",
    "¿Gracias?",
])
def test_typographic_punctuation_does_not_hide_a_signoff(text: str) -> None:
    assert subtitle_boilerplate(text) is True


def test_language_tags_are_compared_case_insensitively() -> None:
    """A tag arriving as "EN" against expected {"en"} rejected real speech."""
    assert only_unspoken_languages(["EN"], {"en"}) is False
    assert only_unspoken_languages(["en"], {"EN"}) is False
    assert only_unspoken_languages(["NN"], {"en"}) is True


def test_duration_is_required_so_the_rate_signal_cannot_be_skipped() -> None:
    """It defaulted to 0.0, which fails the rate check's own guard.

    A caller omitting it silently got one fewer signal than it thought.
    """
    import inspect
    param = inspect.signature(judge_transcript).parameters["duration_s"]
    assert param.default is inspect.Parameter.empty


# --- strip_caption_tail ----------------------------------------------------
# For the shape subtitle_boilerplate deliberately will not touch: a sign-off
# glued to the end of real speech. Discarding a whole transcript for merely
# containing one measurably deletes real recordings (12.3% of the reference
# archive), so the sign-off is cut off and the speech is kept.


def test_a_sign_off_is_cut_from_the_end_of_real_speech() -> None:
    assert strip_caption_tail("まって、まってご視聴ありがとうございました") == "まって、まって"


def test_an_english_sign_off_is_cut_with_its_punctuation() -> None:
    assert (
        strip_caption_tail("The gate is round the side Thank you for watching!")
        == "The gate is round the side"
    )


def test_text_that_is_only_a_sign_off_becomes_empty() -> None:
    assert strip_caption_tail("¡Gracias por ver el video!") == ""


def test_the_longest_matching_phrase_wins() -> None:
    # "gracias por ver" is also in the set; matching it first would strand
    # "el video" as though it were speech.
    assert strip_caption_tail("Bueno. Gracias por ver el video") == "Bueno."


# --- the two ways an earlier, unanchored version destroyed real content ----


def test_a_sign_off_phrase_in_mid_sentence_is_left_alone() -> None:
    # A recorded meeting. Matching this phrase anywhere it appeared turned it
    # into "for coming today" -- the defect that made the match end-anchored.
    speech = "So speaking of time and budget, thank you very much for coming today."
    assert strip_caption_tail(speech) == speech


def test_a_phrase_that_continues_into_another_word_is_not_a_tail() -> None:
    # "Takk for ating medieting." -> "ieting." when the match was unanchored.
    assert strip_caption_tail("Takk for ating medieting.") == "Takk for ating medieting."


def test_generic_politeness_is_never_stripped() -> None:
    # These ARE in SUBTITLE_BOILERPLATE, because a whole 30-minute transcript
    # reading "Thank you." is invented. One window of a longer recording
    # reading "Thank you." is a person talking.
    for spoken in ["Thank you.", "Okay.", "Gracias.", "ありがとうございます"]:
        assert strip_caption_tail(spoken) == spoken


def test_the_tail_set_is_a_subset_of_the_whole_text_set() -> None:
    # The two lists must not drift: anything safe to strip from a tail is by
    # definition boilerplate, but not the reverse.
    extra = CAPTION_SIGNOFF_TAILS - SUBTITLE_BOILERPLATE
    assert extra == set(), f"not also whole-text boilerplate: {sorted(extra)}"


def test_no_tail_phrase_is_short_enough_to_be_ordinary_speech() -> None:
    # The guard on the judgement call behind the set: every member is a fixed
    # caption phrase, not something a person says in passing.
    too_short = {p for p in CAPTION_SIGNOFF_TAILS if len(p.split()) < 2 and len(p) < 8}
    assert too_short == set(), f"suspiciously short: {sorted(too_short)}"


def test_empty_and_blank_text_survive() -> None:
    assert strip_caption_tail("") == ""
    assert strip_caption_tail("   ") == "   "


# --- trailing credit lines -------------------------------------------------
# Credit lines cannot live in a phrase list: they end in a name that is not
# knowable in advance. What is knowable is that ONLY a name follows.


def test_a_window_that_is_only_a_credit_line_becomes_empty() -> None:
    assert strip_caption_tail("Субтитры создавал DimaTorzok") == ""
    assert strip_caption_tail("Teksting av Nicolai Winther") == ""


def test_a_credit_line_is_cut_off_the_end_and_the_rest_stays() -> None:
    assert strip_caption_tail("🎵🎵🎵 🎵🎵🎵 Teksting av Nicolai Winther") == "🎵🎵🎵 🎵🎵🎵"


def test_real_speech_following_a_fabricated_credit_survives() -> None:
    # This exact recording -- someone filming their family -- was DELETED by an
    # earlier prefix rule that matched the credit and discarded the transcript.
    # The credit is invented; everything after it is a person talking.
    speech = "Субтитры создавал DimaTorzok 山が正面 見えてる?見えてる!今も撮ってる!"
    assert strip_caption_tail(speech) == speech


def test_a_long_tail_after_a_prefix_is_speech_not_a_name() -> None:
    sentence = (
        "Hello there, subtitles by me and my long real sentence continues on here"
    )
    assert strip_caption_tail(sentence) == sentence


# --- accents and dakuten ---------------------------------------------------
# The phrase lists are stored normalised, which folds accents away, but the
# text being cleaned keeps them. Case-insensitivity does NOT bridge that: í and
# i are different characters. This is the "cannot fire" defect that left 22 of
# 45 entries in these lists unreachable, and it recurred here.


def test_an_accented_spelling_matches_its_folded_phrase() -> None:
    # Stored as "suscribete al canal"; the transcript says "¡Suscríbete al canal".
    assert (
        strip_caption_tail("Cierra la puerta por favor ¡Suscríbete al canal")
        == "Cierra la puerta por favor"
    )


def test_opening_punctuation_goes_with_the_sign_off() -> None:
    # Spanish opens a clause with ¡. Leaving it behind produces a stray "¡".
    assert strip_caption_tail("Sí ¡Suscríbete") == "Sí"


def test_an_accented_credit_prefix_still_fires() -> None:
    assert strip_caption_tail("Subtítulos realizados por la comunidad") == ""


def test_a_real_word_sharing_the_stem_is_not_a_sign_off() -> None:
    sentence = "Me suscribo a la revista todos los años"
    assert strip_caption_tail(sentence) == sentence


def test_voiced_kana_survive_the_decomposition_used_for_matching() -> None:
    # Matching happens on NFD text, which splits ご into こ + U+3099. If the
    # combining range omits those marks every CJK phrase silently stops
    # matching; if the result is not recomposed, callers get decomposed kana.
    assert strip_caption_tail("まって、まってご視聴ありがとうございました") == "まって、まって"


def test_returned_text_is_recomposed_not_decomposed() -> None:
    import unicodedata

    out = strip_caption_tail("ご飯を食べた ご視聴ありがとうございました")
    assert out == unicodedata.normalize("NFC", out)
    assert "ご飯" in out


def test_text_with_no_sign_off_is_returned_byte_identical() -> None:
    """Callers detect contamination by comparing output against input.

    An earlier version normalised and collapsed whitespace on the way out even
    when it removed nothing, so clean text came back cosmetically different --
    and an index audit comparing the two would report contamination that was
    not there.
    """
    for untouched in [
        "already   collapsed\ttext with  gaps",
        "Thank you.",
        "We went to the park and fed the ducks.",
        "  leading and trailing  ",
    ]:
        assert strip_caption_tail(untouched) == untouched
