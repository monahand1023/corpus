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
    raw = bytes([0xFF, 0xFE, 0x00, 0x01, 0x80, 0x81])
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
