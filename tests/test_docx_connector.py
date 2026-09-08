from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from corpus.connectors.docx import DocxConnector


def _make_mock_docx(paragraphs: list[str], tables: list[list[list[str]]] | None = None,
                    title: str | None = None) -> MagicMock:
    doc = MagicMock()
    doc.paragraphs = [MagicMock(text=t) for t in paragraphs]
    tbls = []
    for rows in tables or []:
        tbl = MagicMock()
        tbl.rows = [MagicMock(cells=[MagicMock(text=c) for c in row]) for row in rows]
        tbls.append(tbl)
    doc.tables = tbls
    doc.core_properties = MagicMock(title=title)
    return doc


def test_loads_paragraphs_and_tables(tmp_path: Path) -> None:
    (tmp_path / "report.docx").write_bytes(b"PK fake")
    doc = _make_mock_docx(
        ["First paragraph.", "", "Second paragraph."],
        tables=[[["Name", "Qty"], ["Widget", "3"]]],
        title="Quarterly Report",
    )
    with patch("docx.Document", return_value=doc):
        docs = list(DocxConnector(source_type="docs", path=tmp_path).load())
    assert len(docs) == 1
    body = docs[0].raw["body"]
    assert docs[0].title == "Quarterly Report"
    assert "First paragraph." in body
    assert "Second paragraph." in body
    assert "Name\tQty" in body
    assert "Widget\t3" in body


def test_title_falls_back_to_filename_stem(tmp_path: Path) -> None:
    (tmp_path / "untitled-doc.docx").write_bytes(b"PK fake")
    with patch("docx.Document", return_value=_make_mock_docx(["Body text here."])):
        docs = list(DocxConnector(source_type="docs", path=tmp_path).load())
    assert docs[0].title == "untitled-doc"


def test_skips_empty_documents(tmp_path: Path) -> None:
    (tmp_path / "empty.docx").write_bytes(b"PK fake")
    with patch("docx.Document", return_value=_make_mock_docx(["", "   "])):
        docs = list(DocxConnector(source_type="docs", path=tmp_path).load())
    assert docs == []


def test_skips_unreadable_documents(tmp_path: Path) -> None:
    (tmp_path / "corrupt.docx").write_bytes(b"not a docx")
    with patch("docx.Document", side_effect=Exception("bad zip")):
        docs = list(DocxConnector(source_type="docs", path=tmp_path).load())
    assert docs == []


def test_missing_dir_raises() -> None:
    with pytest.raises(FileNotFoundError):
        list(DocxConnector(source_type="docs", path="/nonexistent").load())


def test_dedupes_identical_documents(tmp_path: Path) -> None:
    (tmp_path / "a.docx").write_bytes(b"PK")
    (tmp_path / "b.docx").write_bytes(b"PK")
    with patch("docx.Document", return_value=_make_mock_docx(["Same body."])):
        docs = list(DocxConnector(source_type="docs", path=tmp_path).load())
    assert len(docs) == 1
