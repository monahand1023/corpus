from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from corpus.connectors.docx import DocxConnector

# The connector checks a file's real signature before parsing, so a
# fixture standing in for a readable .docx has to carry a genuine zip
# local-file-header magic. (b"PK fake" merely looked zip-ish.)


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
    (tmp_path / "report.docx").write_bytes(b"PK\x03\x04 fake")
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
    (tmp_path / "untitled-doc.docx").write_bytes(b"PK\x03\x04 fake")
    with patch("docx.Document", return_value=_make_mock_docx(["Body text here."])):
        docs = list(DocxConnector(source_type="docs", path=tmp_path).load())
    assert docs[0].title == "untitled-doc"


def test_skips_empty_documents(tmp_path: Path) -> None:
    (tmp_path / "empty.docx").write_bytes(b"PK\x03\x04 fake")
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
    (tmp_path / "a.docx").write_bytes(b"PK\x03\x04")
    (tmp_path / "b.docx").write_bytes(b"PK\x03\x04")
    with patch("docx.Document", return_value=_make_mock_docx(["Same body."])):
        docs = list(DocxConnector(source_type="docs", path=tmp_path).load())
    assert len(docs) == 1


def test_lazy_parse_failure_is_contained_to_the_file(tmp_path: Path) -> None:
    """python-docx parses lazily too: a malformed file can construct fine and
    only raise when `.paragraphs` is touched. That access must be inside the
    per-file guard, or one bad file aborts the whole source."""
    (tmp_path / "broken.docx").write_bytes(b"PK\x03\x04 fake")
    (tmp_path / "fine.docx").write_bytes(b"PK\x03\x04 fake")

    def doc_factory(path: str) -> MagicMock:
        d = MagicMock()
        if Path(path).name == "broken.docx":
            type(d).paragraphs = property(
                lambda self: (_ for _ in ()).throw(KeyError("word/document.xml"))
            )
        else:
            d.paragraphs = [MagicMock(text="Readable body.")]
            d.tables = []
            d.core_properties = MagicMock(title=None)
        return d

    conn = DocxConnector(source_type="docs", path=tmp_path)
    with patch("docx.Document", side_effect=doc_factory):
        docs = list(conn.load())

    assert len(docs) == 1, "the readable docx must still be ingested"
    assert docs[0].title == "fine"
    assert conn.failed_files == 1
