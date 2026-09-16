"""Signals separating invented transcript text from real speech.

Fixtures are shapes taken from a real real archive. Each rule
exists because a specific transcript looked wrong and the reason was chased
down; the tests keep those reasons enforced.
"""

from __future__ import annotations

import pytest

from corpus.transcripts import (
    CAPTION_SIGNOFF_TAILS,
    DEFAULT_MAX_CHARS_PER_SECOND,
    DEFAULT_MAX_LOOPING_SHARE,
    SUBTITLE_BOILERPLATE,
    TranscriptVerdict,
    impossible_speech_rate,
    judge_transcript,
    looping_share,
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
# Genuinely varied, not one sentence repeated. It used to be `... * 4`, which
# made it a LOOP -- the fixture contradicted its own name, and only the
# looping signal noticed. A rate test whose text is degenerate cannot show
# that the rate signal is what rejected it.
_FAST_BUT_VARIED = (
    "the quarterly numbers came in ahead of plan and the team has already "
    "started work on the next phase of migration which we expect to finish "
    "before the end of the year while hiring continues across both sites and "
    "the vendor contract is renegotiated ahead of renewal in the spring which "
    "should free up budget for the tooling work everyone has been asking about "
    "since the reorganisation was announced at the all hands in February and "
    "the backlog was rewritten around the new priorities agreed with finance "
    "after the audit closed out with no material findings this time and the "
    "regional leads signed off on the revised dates without much argument"
)


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
        "I'm going to put on some diamonds. Uh, the lollipops? Yeah, grab a "
        "couple for the car. Did you want the blue ones or should I leave "
        "them? Leave them, we can come back tomorrow before the train. Okay "
        "but bring the small bag this time, the big one was a nightmare on "
        "the escalator. Fine. Are we eating first or after we check out of "
        "the hotel? After, I think, it gets busy around noon and the queue "
        "goes round the corner.",
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


def test_the_fast_screen_never_hides_a_phrase_from_the_real_match() -> None:
    """The screen is an optimisation, so it must only ever reject text the
    full pass would have left alone.

    A screen that rejects something the real rule would have changed is the
    worst kind of bug here: it is invisible, it silently stops the filter
    firing for one phrase, and the phrase list has already had two rounds of
    entries that could not fire at all.
    """
    preamble = "We were talking about the weekend and then. "
    for phrase in CAPTION_SIGNOFF_TAILS:
        text = preamble + phrase
        assert strip_caption_tail(text) != text, f"screened out: {phrase!r}"


def test_the_fast_screen_passes_every_credit_prefix() -> None:
    from corpus.transcripts.quality import SUBTITLE_CREDIT_PREFIXES

    for prefix in SUBTITLE_CREDIT_PREFIXES:
        text = f"Some ordinary speech here. {prefix} Someone"
        assert strip_caption_tail(text) != text, f"screened out: {prefix!r}"


def test_a_sign_off_survives_screening_at_the_end_of_a_long_chunk() -> None:
    # Only the tail is screened, so a long chunk must not hide its own ending.
    long_text = "We walked down to the water and talked about the trip. " * 90
    cleaned = strip_caption_tail(long_text + " ご視聴ありがとうございました")
    assert cleaned.endswith("about the trip.")
    assert strip_caption_tail(long_text) == long_text


def test_text_merely_mentioning_a_sign_off_word_is_untouched() -> None:
    """Passing the fast screen is not the same as matching a phrase.

    The screen looks for ONE word of a phrase anywhere in the tail, so
    ordinary prose mentioning "watching" or "subscribe" reaches the full pass
    and removes nothing. An earlier version still returned the normalised,
    whitespace-collapsed copy in that case, so the output differed from the
    input with nothing taken out -- and callers compare the two to decide
    whether a sign-off was present. It reported 5,585 contaminated chunks in a
    mail archive that had one.
    """
    for untouched in [
        "I love  watching the birds from the  kitchen window",
        "Please subscribe me to the  newsletter, thanks",
        "The subtitles by then were  already out of sync",
        "Amara asked about  dinner",
        "vielen Dank fuer  alles, sagte er",
    ]:
        assert strip_caption_tail(untouched) == untouched


# ---------------------------------------------------------------------------
# Sign-offs that name their object.
#
# Found by running the shipped model against generated non-speech and then
# checking the documented claims one by one. The tail patterns are anchored
# with `$`, so they fired only when the phrase itself ended the text -- every
# real-world variant naming its object survived, English included.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Thanks for watching this video.",
        "Thank you for watching my video.",
        "Thanks for watching the video.",
        "Gracias por ver este video.",
        "Merci d'avoir regardé cette vidéo.",
        "Obrigado por assistir a este vídeo.",
        "Grazie per la visione di questo video.",
        "Please subscribe to the channel.",
    ],
)
def test_a_signoff_that_names_its_object_is_still_a_signoff(text: str) -> None:
    assert strip_caption_tail(text) == ""


@pytest.mark.parametrize(
    "text",
    [
        # The reason this is a vocabulary test and not a length test. Bounding
        # the remainder by word count -- the way credit lines are bounded --
        # would cut every one of these, and they are real speech. The first is
        # the exact shape of family audio this filter exists not to delete.
        "Thanks for watching the kids.",
        "Thanks for watching the kids while we were out.",
        "Thank you for watching my daughter yesterday.",
        "Please subscribe to the mailing list before Friday.",
        "Gracias por ver a los niños el sábado.",
        "We spent the afternoon watching the boats come in.",
    ],
)
def test_real_speech_that_merely_resembles_a_signoff_is_untouched(text: str) -> None:
    assert strip_caption_tail(text) == text


def test_the_object_pass_does_not_reach_past_its_vocabulary() -> None:
    # "video" is in the vocabulary and "rental" is not, so the match is
    # disqualified rather than truncated to the last recognised word.
    text = "Thanks for watching the video rental place for me."
    assert strip_caption_tail(text) == text


def test_a_talk_ending_in_thank_you_very_much_survives_the_new_pass() -> None:
    # The property the whole tail design exists for: this phrase is in
    # SUBTITLE_BOILERPLATE but deliberately NOT in CAPTION_SIGNOFF_TAILS,
    # because dropping transcripts containing it deleted 732 real recordings.
    talk = "and it changed how we staffed the team for two quarters. Thank you very much."
    assert strip_caption_tail(talk) == talk


# ---------------------------------------------------------------------------
# Loops.
#
# Found by transcribing 200 real recordings and reading what survived. Six
# were the transcriber stuck in a loop -- "I am going to draw a small map."
# repeated to fill the window. Every one passed `repeat_share`, whose highest
# score across all 74 kept transcripts was 0.250 against a 0.9 threshold: the
# metric divides one trigram's count by the total, so the denominator grows
# with the text and a looped phrase can never approach 1.0.
# ---------------------------------------------------------------------------


def test_repeat_share_structurally_cannot_see_a_looped_phrase() -> None:
    # Pins the reason a second signal exists, so nobody "simplifies" it away.
    looped = "I am going to draw a small map. " * 8
    assert repeat_share(looped) < 0.2
    assert looping_share(looped) > 0.7


def test_repeat_share_still_sees_one_unit_hammered() -> None:
    # The shape it DOES see; neither signal subsumes the other.
    assert repeat_share("okay okay okay okay okay okay okay") > 0.9


@pytest.mark.parametrize(
    "text",
    [
        "I'm going to eat the rest of the food. " * 9,
        "and this is how it looks " * 40,
    ],
)
def test_a_looping_transcriber_is_rejected(text: str) -> None:
    assert not judge_transcript(text, duration_s=30.0, languages=["en"]).keep
    assert judge_transcript(text, duration_s=30.0, languages=["en"]).reason == (
        "looping_repetition"
    )


@pytest.mark.parametrize(
    "text",
    [
        # Real speech that repeats. These are the recordings the archive is
        # kept FOR, and a share threshold alone rejects the first outright --
        # it scores 0.667. The unit floor is what protects them.
        "Papa! Papa! Papa! Papa!",
        "Mama, mama, mama!",
        "No no no no!",
        "Look! Look at that! Look!",
        "Hey! Hey! Come here!",
        "おいしい、おいしい、おいしい",
    ],
)
def test_someone_repeating_a_word_is_not_a_loop(text: str) -> None:
    assert looping_share(text) == 0.0
    assert judge_transcript(text, duration_s=12.0, languages=["en"]).keep


def test_ordinary_long_speech_scores_zero() -> None:
    real = (
        "we drove up to the lake and the kids fed the ducks all afternoon before "
        "the rain started and we had to run back to the car with the picnic things"
    )
    assert looping_share(real) == 0.0
    assert judge_transcript(real, duration_s=30.0, languages=["en"]).keep


def test_the_looping_threshold_is_in_the_policy_fingerprint() -> None:
    # A signal absent from the fingerprint leaves stale verdicts alive: files
    # rejected under the old threshold would never be retried under a new one.
    from corpus.transcripts.pipeline import Settings

    assert "max_looping_share" in Settings().as_policy("m")
    loose = Settings(max_looping_share=0.95).as_policy("m")
    assert loose != Settings().as_policy("m")


def test_a_repeated_signoff_is_removed_completely() -> None:
    # Single-pass stripping left one copy behind, because each pass only
    # removes what is at the END. Found in a live index: two chunks reading
    # "Please subscribe. Please subscribe." after the filter had run.
    assert strip_caption_tail("Please subscribe. Please subscribe.") == ""
    assert strip_caption_tail("Thanks for watching. Thanks for watching this video.") == ""


def test_repeated_stripping_still_stops_at_real_speech() -> None:
    text = "and then we drove home. Thank you for watching. Thanks for watching."
    assert strip_caption_tail(text) == "and then we drove home."


def test_stripping_reaches_a_fixed_point_without_eating_the_text() -> None:
    plain = "an ordinary sentence with no sign-off in it at all"
    assert strip_caption_tail(plain) is plain


# ---------------------------------------------------------------------------
# The fast paths. Both are pure speedups, and the tests exist to keep them
# that way: each pins the property that makes skipping work safe, not the
# speed itself.
# ---------------------------------------------------------------------------


def test_the_length_guard_bounds_on_word_characters_not_raw_length() -> None:
    """The adversarial case a raw-length guard gets wrong.

    `subtitle_boilerplate` now rejects on length before normalising, because
    normalising 2 KB of prose to compare against an 80-character sign-off was
    70% of an index-quality scan. That is only safe if the bound counts what
    SURVIVES normalisation: padding normalises away, so a long raw string can
    still be a short sign-off.
    """
    padded = "." * 400 + "Thank you for watching"
    assert len(padded) > 400
    assert subtitle_boilerplate(padded) is True


def test_a_long_document_is_rejected_without_normalising_it() -> None:
    prose = "the quarterly numbers came in ahead of plan and the team moved on. " * 40
    assert subtitle_boilerplate(prose) is False


def test_the_guard_respects_caller_supplied_phrase_lists() -> None:
    # The bound is derived from the lists actually in use; a caller passing a
    # longer phrase must not have it silently cut off by a cached limit.
    long_phrase = "thanks so very much indeed for watching this entire video today"
    assert subtitle_boilerplate(long_phrase, phrases=frozenset({long_phrase})) is True


def test_tail_search_before_substitution_leaves_output_identical() -> None:
    """`strip_caption_tail` searches a bounded window, then substitutes.

    The sign-off must still be removed when it follows a long head, which is
    what proves the window is a search optimisation and not a truncation.
    """
    head = "and then we drove home and the kids fell asleep in the back. " * 40
    assert strip_caption_tail(head + "Thanks for watching") == head.strip()
    assert strip_caption_tail(head + "Thanks for watching this video.") == head.strip()


# ---------------------------------------------------------------------------
# The speech-rate ceiling, re-measured.
#
# Found by the dormancy check: this filter had never fired. It is NOT dead the
# way repeat_share was -- that sat 3.6x from its threshold and could not be
# reached. This one is reachable and correctly aimed; the problem is which
# side the margin is on.
#
# The ceiling was set when the fastest genuine content measured 14.7 c/s, a
# comfortable 70% of headroom. The archive has since grown to 7,186
# transcripts and real speech now reaches 24.85 c/s -- 0.6% below the ceiling.
# The threshold never moved. The data did.
# ---------------------------------------------------------------------------

# Measured across 7,186 real transcripts, 2026-09-16.
FASTEST_REAL_SPEECH = 24.85
# The documented decode loop: over a thousand characters from an 11s clip.
MEASURED_DECODE_LOOP = 59.91


def test_the_fastest_real_speech_measured_is_kept_with_room_to_spare() -> None:
    """A French speaker mid-conversation, at 24.85 characters per second.

    Under the old 25.0 ceiling this survived by 0.6%. One slightly faster
    speaker, or a denser orthography, and the filter deletes a real recording
    -- the failure every other threshold here is deliberately tuned to avoid.
    """
    duration = 60.0
    text = "x" * int(FASTEST_REAL_SPEECH * duration)
    assert impossible_speech_rate(text, duration) is False


def test_there_is_real_headroom_above_the_fastest_measured_speech() -> None:
    # The property that was missing: a margin big enough that ordinary
    # variation between speakers cannot cross it.
    assert DEFAULT_MAX_CHARS_PER_SECOND >= FASTEST_REAL_SPEECH * 1.5


def test_a_decode_loop_is_still_caught() -> None:
    """Raising the ceiling must not cost the detection it exists for."""
    duration = 10.7
    text = "x" * int(MEASURED_DECODE_LOOP * duration)
    assert impossible_speech_rate(text, duration) is True


def test_the_ceiling_sits_between_the_two_measurements() -> None:
    # Pins the reasoning, not just the number. If someone re-tightens this,
    # this test says what the number was chosen against.
    assert FASTEST_REAL_SPEECH < DEFAULT_MAX_CHARS_PER_SECOND < MEASURED_DECODE_LOOP


# ---------------------------------------------------------------------------
# The loop ceiling, re-measured against a FULL archive.
#
# The first version of this threshold was validated against 74 transcripts and
# then applied to 7,186. On the full archive real material reaches 0.5984 --
# the ceiling was 0.600, a margin of 0.3% -- and it removed 241 transcripts,
# a substantial fraction of them real.
#
# REAL FAMILY SPEECH IS GENUINELY REPETITIVE. Songs, chants, a parent
# repeating an instruction, a child repeating a word. That is not a tuning
# detail; it is the shape of the data, and it is the same lesson repeat_share
# already carried ("at 0.6 a reference archive lost a clip of a child
# repeating one word").
# ---------------------------------------------------------------------------

# Measured across 7,186 real transcripts, 2026-09-16.
HIGHEST_REAL_LOOPING = 0.5984


def test_a_birthday_song_with_the_talk_that_follows_it_is_kept() -> None:
    """The shape of the real case, which scored 0.6075 and was deleted.

    A sung phrase repeated many times, then the ordinary conversation around
    it. The fixture is synthetic on purpose -- the real one is family audio
    and does not belong in a public repository -- but it reproduces the shape
    and lands in the same band (0.641 against the archive's 0.6075).

    Its English-language counterpart survived at 0.5955 while this one was
    deleted at 0.6075. Same family, same event, separated by 0.012.
    """
    song = "happy birthday to you " * 14
    after = ("there you go blow them out well done that was great shall we do "
             "the cake now wait not yet careful there you are")
    assert judge_transcript(song + after, duration_s=60.0, languages=["en"]).keep


def test_a_parent_repeating_an_instruction_is_kept() -> None:
    # 0.671 -- the shape of a real 0.703 rejection, a parent telling a child
    # not to hit something because it will break.
    real = "careful do not hit it hard it will break careful now " * 3
    assert judge_transcript(real, duration_s=30.0, languages=["en"]).keep


def test_a_genuine_decode_loop_is_still_rejected() -> None:
    # The shape of the archive's clearest loop: one short phrase, nothing
    # else, filling the window. That one scored 0.986 and appeared byte
    # identical across several unrelated files.
    loop = "and this is how it looks " * 40
    verdict = judge_transcript(loop, duration_s=60.0, languages=["en"])
    assert not verdict.keep and verdict.reason == "looping_repetition"


def test_the_loop_ceiling_clears_real_material_by_a_real_margin() -> None:
    """The property that was missing, stated as a property.

    An indexed loop is recoverable noise. A deleted family recording is not,
    so the margin belongs on that side.
    """
    assert DEFAULT_MAX_LOOPING_SHARE >= HIGHEST_REAL_LOOPING * 1.25


# --- the escape window in looping_share --------------------------------------


# A SHORT phrase is what reaches the window: one or two words repeated gives
# 8-25 word-trigrams. A five-word phrase repeated eight times gives 38 and
# never enters it -- the first version of this test made that mistake and
# passed while the bug was live.
@pytest.mark.parametrize("reps", [10, 14, 20, 25])
def test_a_short_decode_loop_does_not_escape_the_signal(reps: int) -> None:
    """A pure repetition of 8-25 words scored exactly 0.0.

    The unit basis switches: below four word-trigrams the signal falls back to
    character 6-grams, which yield plenty of units. Between four and
    twenty-three trigrams it uses trigrams -- and those sit BELOW the
    twenty-four-unit floor, so the signal returns 0.0 and the text escapes.

    The band that escapes is exactly the shape the filter exists to catch: one
    short phrase repeated to fill a window.
    """
    assert looping_share("Blimvara " * reps) > 0.5


def test_the_signal_is_monotonic_in_repetition() -> None:
    # The symptom that exposed it: more repetition scored LOWER.
    shares = [looping_share("Blimvara " * n) for n in (5, 10, 20, 40)]
    assert min(shares) > 0.5, shares


def test_a_short_word_repeated_a_few_times_is_now_protected() -> None:
    """Deliberately changed when the ceiling moved to 0.85.

    Five repetitions of one word scores 0.78. Under the old 0.6 ceiling this
    was rejected; it is now kept, and that is the trade the re-measurement
    bought -- the same band held a family singing Happy Birthday. A surviving
    loop is recoverable noise in the index; a deleted recording is not.
    """
    assert judge_transcript(
        "Blimvara " * 5, duration_s=30.0, languages=["en"]
    ).keep


def test_short_real_repeats_are_still_protected() -> None:
    # The floor's actual purpose must survive the fix: a child repeating a
    # word is not a decode loop.
    for text in ("Papa! Papa! Papa! Papa!", "Mama, mama, mama!", "No no no no!"):
        assert looping_share(text) == 0.0, text
