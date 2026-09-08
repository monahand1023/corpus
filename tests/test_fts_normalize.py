from __future__ import annotations

from corpus.util.fts_normalize import fts_terms, normalize_for_fts


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
