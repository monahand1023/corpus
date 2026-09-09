from __future__ import annotations

from pathlib import Path

import pytest

from corpus.connectors.rtf import RtfConnector

RTF_HELLO = r"{\rtf1\ansi\deff0 {\fonttbl {\f0 Times;}}\f0\fs24 Hello from RTF.\par Second line.\par}"


def test_extracts_plain_text(tmp_path: Path) -> None:
    (tmp_path / "note.rtf").write_text(RTF_HELLO, encoding="utf-8")
    docs = list(RtfConnector(source_type="rtfs", path=tmp_path).load())
    assert len(docs) == 1
    assert "Hello from RTF." in docs[0].raw["body"]
    assert "Second line." in docs[0].raw["body"]
    assert docs[0].title == "note"


def test_skips_empty_rtf(tmp_path: Path) -> None:
    (tmp_path / "empty.rtf").write_text(r"{\rtf1\ansi}", encoding="utf-8")
    assert list(RtfConnector(source_type="rtfs", path=tmp_path).load()) == []


def test_missing_dir_raises() -> None:
    with pytest.raises(FileNotFoundError):
        list(RtfConnector(source_type="rtfs", path="/nonexistent").load())


def test_dedupes_identical_rtf(tmp_path: Path) -> None:
    (tmp_path / "a.rtf").write_text(RTF_HELLO, encoding="utf-8")
    (tmp_path / "b.rtf").write_text(RTF_HELLO, encoding="utf-8")
    assert len(list(RtfConnector(source_type="rtfs", path=tmp_path).load())) == 1


def test_cp932_encoded_rtf_is_decoded_correctly_not_replaced(tmp_path: Path) -> None:
    """Some older Japanese-locale RTF exports embed raw CP932 bytes directly
    (via `\\ansicpg932`) rather than escaping every non-ASCII character as
    `\\'hh`. See the identical regression test in test_text_connector.py —
    this connector shares the same `errors="replace"` -> fallback fix (see
    `corpus.util.encoding`)."""
    rtf = (
        r"{\rtf1\ansi\ansicpg932\deff0 {\fonttbl {\f0 MS Mincho;}}"
        r"\f0\fs24 会議メモ\par}"
    )
    (tmp_path / "memo.rtf").write_bytes(rtf.encode("cp932"))
    docs = list(RtfConnector(source_type="rtfs", path=tmp_path).load())
    assert len(docs) == 1
    assert "会議メモ" in docs[0].raw["body"]
    assert "�" not in docs[0].raw["body"]
