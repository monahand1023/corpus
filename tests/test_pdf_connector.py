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


# ---------------------------------------------------------------------------
# failed_files: the optional per-file failure counter the ingester reads to
# decide whether pruning is safe. See Ingester._reported_failures.
# ---------------------------------------------------------------------------

def test_unreadable_pdf_increments_failed_files(tmp_path: Path) -> None:
    (tmp_path / "corrupt.pdf").write_bytes(b"not really a PDF")
    conn = PdfConnector(source_type="papers", path=tmp_path)
    with patch("pypdf.PdfReader", side_effect=Exception("malformed")):
        assert list(conn.load()) == []
    assert conn.failed_files == 1


def test_failed_page_counts_once_even_though_document_is_yielded(tmp_path: Path) -> None:
    """A partially-read PDF is still unsafe to prune against: the shorter body
    produces fewer chunks, so the tail chunk ids vanish from seen_ids."""
    (tmp_path / "partial.pdf").write_bytes(b"%PDF-1.4 fake")
    good = MagicMock(extract_text=MagicMock(return_value="Readable page."))
    bad1 = MagicMock(extract_text=MagicMock(side_effect=Exception("bad page")))
    bad2 = MagicMock(extract_text=MagicMock(side_effect=Exception("bad page")))
    reader = MagicMock(pages=[good, bad1, bad2], metadata=None)

    conn = PdfConnector(source_type="papers", path=tmp_path)
    with patch("pypdf.PdfReader", return_value=reader):
        docs = list(conn.load())

    assert len(docs) == 1, "document should still be yielded"
    assert conn.failed_files == 1, "two failed pages in one file count once"


def test_title_metadata_failure_does_not_count(tmp_path: Path) -> None:
    """The title falls back to the filename stem; the body is untouched, so the
    chunk ids are unchanged and pruning stays safe."""
    (tmp_path / "doc.pdf").write_bytes(b"%PDF-1.4 fake")
    reader = MagicMock(pages=[MagicMock(extract_text=MagicMock(return_value="Body text here."))])
    type(reader).metadata = property(lambda self: (_ for _ in ()).throw(Exception("bad metadata")))

    conn = PdfConnector(source_type="papers", path=tmp_path)
    with patch("pypdf.PdfReader", return_value=reader):
        docs = list(conn.load())

    assert len(docs) == 1
    assert docs[0].title == "doc"
    assert conn.failed_files == 0


def test_failed_files_resets_between_runs(tmp_path: Path) -> None:
    """A reused connector instance must not suppress pruning forever on the
    strength of a failure from an earlier run."""
    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4 fake")
    conn = PdfConnector(source_type="papers", path=tmp_path)

    with patch("pypdf.PdfReader", side_effect=Exception("locked")):
        list(conn.load())
    assert conn.failed_files == 1

    reader = MagicMock(pages=[MagicMock(extract_text=MagicMock(return_value="Now readable."))],
                       metadata=None)
    with patch("pypdf.PdfReader", return_value=reader):
        docs = list(conn.load())
    assert len(docs) == 1
    assert conn.failed_files == 0, "counter must reset at the start of load()"


def test_lazy_parse_failure_is_contained_to_the_file(tmp_path: Path) -> None:
    """pypdf parses lazily: an encrypted or malformed PDF constructs fine and
    only raises when `.pages` is first touched. That access must be inside the
    per-file guard, or one bad file aborts the whole source — and because
    pypdf's errors derive from Exception (not ValueError/OSError), the CLI's
    per-source handler would not catch it either, taking down `--all`."""
    from pypdf.errors import FileNotDecryptedError

    (tmp_path / "encrypted.pdf").write_bytes(b"%PDF-1.4 fake")
    (tmp_path / "fine.pdf").write_bytes(b"%PDF-1.4 fake")

    def reader_factory(path: str) -> MagicMock:
        r = MagicMock()
        if Path(path).name == "encrypted.pdf":
            type(r).pages = property(
                lambda self: (_ for _ in ()).throw(FileNotDecryptedError("File has not been decrypted"))
            )
        else:
            r.pages = [MagicMock(extract_text=MagicMock(return_value="Readable body."))]
            r.metadata = None
        return r

    conn = PdfConnector(source_type="papers", path=tmp_path)
    with patch("pypdf.PdfReader", side_effect=reader_factory):
        docs = list(conn.load())

    assert len(docs) == 1, "the readable PDF must still be ingested"
    assert docs[0].title == "fine"
    assert conn.failed_files == 1
