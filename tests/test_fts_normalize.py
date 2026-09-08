from __future__ import annotations

import unicodedata

from corpus.util.fts_normalize import _CJK, MAX_FTS_TERMS, fts_terms, normalize_for_fts


def test_latin_text_passes_through_unchanged() -> None:
    assert normalize_for_fts("the quarterly planning meeting") == "the quarterly planning meeting"


def test_accents_are_left_for_unicode61_to_fold() -> None:
    # We must NOT strip marks ourselves; unicode61 folds them on both sides.
    assert normalize_for_fts("café") == "café"
    assert normalize_for_fts("reunión") == "reunión"


def test_eszett_folds_to_ss() -> None:
    assert normalize_for_fts("Größe") == "Grösse"


def test_cjk_run_becomes_overlapping_bigrams() -> None:
    assert normalize_for_fts("東京で会議") == "東京 京で で会 会議"


def test_single_character_cjk_run_is_emitted_as_is() -> None:
    assert normalize_for_fts("猫 cat") == "猫 cat"


def test_mixed_script_keeps_latin_intact() -> None:
    assert normalize_for_fts("Tokyo 東京 office") == "Tokyo 東京 office"


# --- The regression guard for the kana-corruption defect. ---
# NFKD + combining-mark stripping maps がっこう (school) onto かっこう (cuckoo).
# These MUST normalize differently.
def test_voiced_kana_are_not_collapsed_onto_unvoiced() -> None:
    assert normalize_for_fts("バス") != normalize_for_fts("ハス")   # bus vs lotus
    assert normalize_for_fts("ガード") != normalize_for_fts("カード")  # guard vs card
    assert normalize_for_fts("がっこう") != normalize_for_fts("かっこう")  # school vs cuckoo
    assert normalize_for_fts("ばか") != normalize_for_fts("はか")   # idiot vs grave


def test_fts_terms_extracts_unicode_tokens() -> None:
    assert fts_terms("東京の会議") == ['"東京"', '"京の"', '"の会"', '"会議"']


def test_fts_terms_keeps_accented_words_whole() -> None:
    assert fts_terms("reunión de café") == ['"reunión"', '"de"', '"café"']


def test_fts_terms_quotes_hyphenated_identifiers() -> None:
    assert fts_terms("PROJ-123") == ['"PROJ-123"']


def test_fts_terms_empty_for_punctuation_only() -> None:
    assert fts_terms("!!! ???") == []


# --- Fix-round-1 regression coverage for the homoglyph defect. ---
# A prior version of `_CJK`'s "CJK Compatibility Ideographs" range started at
# U+8C48 (a CJK Unified Ideograph) instead of U+F900 (the intended
# Compatibility Ideograph). The two glyphs are visually identical -- U+F900
# canonically decomposes to U+8C48 -- so the substitution was invisible on
# screen and in the diff. The bug widened the range into one contiguous span
# that also swallowed Hangul Syllables, Yi, and Vai. A codepoint-boundary
# assertion (below) pins the invariant directly; the pass-through tests pin
# the user-visible symptom.


def test_cjk_ranges_match_exact_unicode_block_boundaries() -> None:
    # `_CJK` is a concatenation of six "<start>-<end>" segments, each exactly
    # 3 characters (start char, literal hyphen, end char), in this order.
    expected_boundaries = [
        (0x3040, 0x309F),  # Hiragana
        (0x30A0, 0x30FF),  # Katakana
        (0xFF66, 0xFF9F),  # Halfwidth Katakana
        (0x3400, 0x4DBF),  # CJK Extension A
        (0x4E00, 0x9FFF),  # CJK Unified Ideographs
        (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    ]
    assert len(_CJK) == len(expected_boundaries) * 3
    for i, (start, end) in enumerate(expected_boundaries):
        segment = _CJK[i * 3 : i * 3 + 3]
        assert segment[1] == "-"
        assert (ord(segment[0]), ord(segment[2])) == (start, end)


def test_hangul_passes_through_unchanged() -> None:
    # Hangul Syllables (U+AC00-U+D7A3) must NOT be absorbed into the CJK run:
    # Korean is out of scope and would need its own handling (see module
    # docstring). Characters are built from their Unicode names rather than
    # pasted as literals, for the same reason the range fix avoids literals.
    han_gug_eo = (
        unicodedata.lookup("HANGUL SYLLABLE HAN")
        + unicodedata.lookup("HANGUL SYLLABLE GUG")
        + unicodedata.lookup("HANGUL SYLLABLE EO")
    )
    ga_na_da = (
        unicodedata.lookup("HANGUL SYLLABLE GA")
        + unicodedata.lookup("HANGUL SYLLABLE NA")
        + unicodedata.lookup("HANGUL SYLLABLE DA")
    )
    assert normalize_for_fts(han_gug_eo) == han_gug_eo
    assert normalize_for_fts(ga_na_da) == ga_na_da


# --- FIX 4: unbounded / duplicated CJK query terms. ---


def test_fts_terms_deduplicates_repeated_terms_preserving_order() -> None:
    # "東京の会議 東京の会議" naively yields 8 bigram terms (4 repeated); only
    # the first occurrence of each should survive, in first-seen order.
    assert fts_terms("東京の会議 東京の会議") == ['"東京"', '"京の"', '"の会"', '"会議"']


def test_fts_terms_deduplicates_repeated_latin_words() -> None:
    assert fts_terms("meeting notes meeting") == ['"meeting"', '"notes"']


def test_fts_terms_caps_at_max_fts_terms() -> None:
    # 100 distinct single-character CJK runs -> 100 distinct one-character
    # "bigrams" (each run is too short to bigram, so it's emitted as-is);
    # all are unique, so the cap -- not dedup -- is what bites here.
    long_query = " ".join(chr(0x4E00 + i) for i in range(100))  # distinct CJK Unified Ideographs
    terms = fts_terms(long_query)
    assert len(terms) == MAX_FTS_TERMS


def test_yi_and_vai_pass_through_unchanged() -> None:
    # Cheap extra coverage for the other scripts the same defect swallowed.
    yi = unicodedata.lookup("YI SYLLABLE IT")
    vai = unicodedata.lookup("VAI SYLLABLE EE")
    assert normalize_for_fts(yi) == yi
    assert normalize_for_fts(vai) == vai
