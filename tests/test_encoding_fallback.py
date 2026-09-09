"""Tests for `corpus.util.encoding` — the shared UTF-8 -> CP932 -> latin-1
text-decoding fallback used by the markdown/text/html/rtf connectors (see
`corpus/util/encoding.py`'s module docstring for why this exists: a real
index had ~970 chunks of U+FFFD replacement-character soup from the old
`errors="replace"` behavior, found while chasing the related zip-filename
mojibake bug).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from corpus.util.encoding import decode_text_with_fallback, read_text_with_fallback


def test_utf8_content_decodes_normally() -> None:
    text = "Plain ASCII and 日本語のテキスト both decode as UTF-8."
    assert decode_text_with_fallback(text.encode("utf-8")) == text


def test_cp932_content_is_recovered_not_mangled() -> None:
    text = "会議メモ: 予算は前年比で増加した。"
    raw = text.encode("cp932")
    # Sanity check the fixture: this really isn't valid UTF-8, so the old
    # errors="replace" behavior would have produced replacement characters.
    assert raw.decode("utf-8", errors="replace") != text
    assert "�" in raw.decode("utf-8", errors="replace")

    assert decode_text_with_fallback(raw) == text


def test_genuinely_undecodable_bytes_fall_back_to_latin1_without_raising() -> None:
    # Deliberately does NOT start with a byte-order mark: the point here is
    # the terminal latin-1 backstop, and a leading ff fe / fe ff would instead
    # exercise the BOM tier (which decodes rather than falls through).
    raw = bytes([0x41, 0x80, 0x81, 0xFE, 0xFF, 0x00])
    result = decode_text_with_fallback(raw)  # must not raise
    assert result == raw.decode("latin-1")


def test_crlf_and_cr_line_endings_are_normalized_to_lf() -> None:
    assert decode_text_with_fallback(b"line one\r\nline two\rline three\n") == (
        "line one\nline two\nline three\n"
    )


def test_read_text_with_fallback_reads_a_real_file(tmp_path: Path) -> None:
    path = tmp_path / "note.txt"
    text = "領収書 2026年"
    path.write_bytes(text.encode("cp932"))
    assert read_text_with_fallback(path) == text


def test_read_text_with_fallback_raises_oserror_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        read_text_with_fallback(tmp_path / "nope.txt")


# --- byte-order marks -------------------------------------------------------
#
# These are regression tests for a real archive measurement: 7 of the 8 files
# the UTF-8/CP932/latin-1 ladder turned into pure garbage were BOM-carrying
# UTF-16. Both cp932 and latin-1 "succeed" on UTF-16 bytes and hand back one
# NUL per ASCII character, so the damage is silent — unlike the U+FFFD the
# ladder was written to replace.


@pytest.mark.parametrize(
    ("encoding", "signature"),
    [
        ("utf-16-le", b"\xff\xfe"),
        ("utf-16-be", b"\xfe\xff"),
        ("utf-32-le", b"\xff\xfe\x00\x00"),
        ("utf-32-be", b"\x00\x00\xfe\xff"),
    ],
)
def test_bom_encodings_decode_and_lose_no_characters(encoding: str, signature: bytes) -> None:
    # Mixed scripts: ASCII would decode "successfully" under latin-1 too, so
    # the Japanese is what proves the right codec ran rather than a lucky one.
    text = "Home Page\n日本の伝説によると\n"
    raw = signature + text.encode(encoding)

    result = decode_text_with_fallback(raw)

    assert result == text
    assert "\x00" not in result  # the silent-garbage signature
    assert "﻿" not in result  # the BOM itself must be stripped


def test_utf8_bom_is_stripped_rather_than_indexed() -> None:
    # Plain UTF-8 decoding of a BOM'd file succeeds, so this one never looked
    # broken — it just quietly prefixed a zero-width U+FEFF onto the first
    # chunk of every such document.
    raw = b"\xef\xbb\xbf" + b"Notes\n"

    assert decode_text_with_fallback(raw) == "Notes\n"


def test_utf32_le_bom_is_not_mistaken_for_utf16_le() -> None:
    # ff fe 00 00 starts with the entire UTF-16-LE mark, so a shortest-first
    # BOM table decodes every UTF-32-LE file as UTF-16-LE and silently
    # interleaves NULs.
    text = "abc\n"
    raw = b"\xff\xfe\x00\x00" + text.encode("utf-32-le")

    result = decode_text_with_fallback(raw)

    assert result == text
    assert "\x00" not in result


def test_damaged_bom_file_falls_through_to_the_ladder() -> None:
    # An odd trailing byte makes this undecodable as UTF-16. The BOM stated an
    # intent the file doesn't keep; some text still beats an exception.
    raw = b"\xff\xfe" + "ok".encode("utf-16-le") + b"\x41"

    result = decode_text_with_fallback(raw)  # must not raise

    assert isinstance(result, str)


def test_bom_check_leaves_ordinary_utf8_untouched() -> None:
    # The BOM tier runs first on every file, so it has to be inert for the
    # overwhelmingly common case.
    text = "plain ascii and 日本語, no mark\n"

    assert decode_text_with_fallback(text.encode("utf-8")) == text
