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
