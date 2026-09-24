"""Granular coverage for `corpus.util.dedup`.

Adopted from a consumer repo during fork consolidation, which is how the gap
surfaced: two downstream archives each had eight tests for this module and the
engine that owns it had none beyond two coarse assertions in `test_util.py`.

What `fingerprint` is for: the same document re-exported several times differs
only in URLs, timestamps and whitespace, and should collapse to one entry.
The line it must not cross is collapsing documents that genuinely differ —
notably translations, which are the same *document* but not the same *content*
and must stay separately retrievable.
"""

from __future__ import annotations

from corpus.util.dedup import fingerprint, normalize_for_dedup


def test_normalize_strips_urls() -> None:
    a = "Read more at https://example.com/foo and www.example.com"
    b = "Read more at https://example.com/bar and www.different.com"

    assert normalize_for_dedup(a) == normalize_for_dedup(b)


def test_normalize_strips_dates_in_either_format() -> None:
    a = "Updated 2026-05-01 — see attached doc."
    b = "Updated 2024-11-15T08:30:00Z — see attached doc."

    assert normalize_for_dedup(a) == normalize_for_dedup(b)


def test_normalize_collapses_whitespace() -> None:
    assert normalize_for_dedup("foo   bar\n\n\tbaz") == normalize_for_dedup("foo bar baz")


def test_normalize_lowercases() -> None:
    assert normalize_for_dedup("Hello World") == normalize_for_dedup("HELLO world")


def test_fingerprint_collapses_a_re_export() -> None:
    # The case this exists for: one document saved repeatedly, differing only
    # in the host it was fetched from and the day it was saved.
    a = """# Voice Platform — Technical Design Review
The system runs on Cloud Run at https://service.example.com/api
Last updated 2026-04-01."""
    b = """# Voice Platform — Technical Design Review
The system runs on Cloud Run at https://different.example.com/api
Last updated 2026-04-20."""

    assert fingerprint(a) == fingerprint(b)


def test_fingerprint_keeps_translations_distinct() -> None:
    # The boundary. These are the same document and must NOT collapse: an
    # archive that silently drops the Spanish copy answers Spanish queries
    # with nothing, and reports success while doing it.
    en = "Customer Block List: a list of users who cannot purchase."
    es = "Lista de bloqueo de clientes: una lista de usuarios que no pueden comprar."

    assert fingerprint(en) != fingerprint(es)


def test_fingerprint_is_stable_across_calls() -> None:
    assert fingerprint("Stable content here") == fingerprint("Stable content here")


def test_fingerprint_is_32_hex_characters() -> None:
    fp = fingerprint("anything")

    assert len(fp) == 32
    assert all(c in "0123456789abcdef" for c in fp)


def test_near_duplicates_keeps_the_first_and_names_it(caplog) -> None:
    """The per-load tracker ten connectors each hand-wrote."""
    import logging

    from corpus.util.dedup import NearDuplicates

    dupes = NearDuplicates("notes")
    assert not dupes.seen_before("Quarterly report  2024-01-05", "a.md")
    with caplog.at_level(logging.INFO, logger="corpus.util.dedup"):
        assert dupes.seen_before("quarterly report 2025-02-06", "copy of a.md")
    assert "matches 'a.md'" in caplog.text
    assert not dupes.seen_before("something else entirely", "b.md")
