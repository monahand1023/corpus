from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from corpus.connectors.pdf import PdfConnector


def _make_mock_pdf(pages_text: list[str], title: str | None = None) -> MagicMock:
    """Build a mock pypdf.PdfReader with the given pages and metadata."""
    reader = MagicMock()
    reader.pages = [MagicMock(extract_text=MagicMock(return_value=t)) for t in pages_text]
    if title is not None:
        reader.metadata = MagicMock(title=title)
    else:
        reader.metadata = None
    return reader


def test_loads_pdf_files_with_mocked_reader(tmp_path: Path) -> None:
    # Create empty placeholder files; the PdfReader is mocked so contents don't matter
    (tmp_path / "doc-a.pdf").write_bytes(b"%PDF-1.4 fake")
    (tmp_path / "doc-b.pdf").write_bytes(b"%PDF-1.4 fake")

    readers = {
        "doc-a.pdf": _make_mock_pdf(["Page 1 of A.", "Page 2 of A."], title="Doc A Title"),
        "doc-b.pdf": _make_mock_pdf(["Single page B."]),
    }

    def reader_factory(path: str) -> MagicMock:
        return readers[Path(path).name]

    with patch("pypdf.PdfReader", side_effect=reader_factory):
        docs = list(PdfConnector(source_type="papers", path=tmp_path).load())

    by_title = {d.title: d for d in docs}
    assert "Doc A Title" in by_title  # title from PDF metadata
    assert "doc-b" in by_title  # falls back to filename stem
    assert "Page 1 of A." in by_title["Doc A Title"].raw["body"]
    assert "Page 2 of A." in by_title["Doc A Title"].raw["body"]
    assert by_title["Doc A Title"].raw["page_count"] == 2


def test_skips_pdfs_with_no_extractable_text(tmp_path: Path) -> None:
    (tmp_path / "scanned.pdf").write_bytes(b"%PDF-1.4 fake")
    reader = _make_mock_pdf(["", "  "])  # empty pages — simulates scanned PDF

    with patch("pypdf.PdfReader", return_value=reader):
        docs = list(PdfConnector(source_type="papers", path=tmp_path).load())

    assert docs == []


def test_skips_pdfs_that_fail_to_open(tmp_path: Path) -> None:
    (tmp_path / "corrupt.pdf").write_bytes(b"not really a PDF")
    with patch("pypdf.PdfReader", side_effect=Exception("malformed")):
        docs = list(PdfConnector(source_type="papers", path=tmp_path).load())
    assert docs == []


def test_missing_dir_raises() -> None:
    with pytest.raises(FileNotFoundError):
        list(PdfConnector(source_type="papers", path="/nonexistent").load())


def test_dedupes_identical_pdfs(tmp_path: Path) -> None:
    (tmp_path / "v1.pdf").write_bytes(b"x")
    (tmp_path / "v2.pdf").write_bytes(b"x")
    reader = _make_mock_pdf(["Same content in both."])
    with patch("pypdf.PdfReader", return_value=reader):
        docs = list(PdfConnector(source_type="papers", path=tmp_path).load())
    assert len(docs) == 1
