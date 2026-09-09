from __future__ import annotations

import csv
from pathlib import Path

import pytest

from corpus.connectors.csv_ import FULL_INDEX_MAX_ROWS, CsvConnector


def test_full_index_for_small_csv_with_header(tmp_path: Path) -> None:
    (tmp_path / "budget.csv").write_text(
        "name,revenue,active\nAcme,1200.50,true\nGlobex,980,false\n", encoding="utf-8"
    )
    docs = list(CsvConnector(source_type="sheets", path=tmp_path).load())
    assert len(docs) == 1
    body = docs[0].raw["body"]
    assert docs[0].title == "budget"
    assert "**Rows:** 2  **Columns:** 3" in body
    assert "`name` — text" in body
    assert "`revenue` — number" in body
    assert "`active` — boolean" in body
    assert "## Data" in body
    assert "Acme\t1200.50\ttrue" in body
    assert "Globex\t980\tfalse" in body


def test_summary_mode_for_large_csv(tmp_path: Path) -> None:
    rows = [f"{i},{i * 10}" for i in range(1, 501)]
    (tmp_path / "export.csv").write_text("id,value\n" + "\n".join(rows) + "\n", encoding="utf-8")

    docs = list(CsvConnector(source_type="sheets", path=tmp_path).load())
    body = docs[0].raw["body"]
    assert "**Rows:** 500  **Columns:** 2" in body
    assert "## Sample rows (first 10, last 10 of 500)" in body
    assert "## Data" not in body
    # head and tail present
    assert "1\t10" in body
    assert "500\t5000" in body
    # a row from the untouched middle must NOT be in the sample
    assert "250\t2500" not in body


def test_row_count_at_threshold_is_still_full_indexed(tmp_path: Path) -> None:
    rows = [f"{i},x" for i in range(1, FULL_INDEX_MAX_ROWS + 1)]
    (tmp_path / "at_threshold.csv").write_text(
        "id,label\n" + "\n".join(rows) + "\n", encoding="utf-8"
    )
    docs = list(CsvConnector(source_type="sheets", path=tmp_path).load())
    body = docs[0].raw["body"]
    assert "## Data" in body
    assert "## Sample rows" not in body


def test_headerless_numeric_list_synthesizes_column(tmp_path: Path) -> None:
    (tmp_path / "numbers.csv").write_text("12\n13\n14\n", encoding="utf-8")
    docs = list(CsvConnector(source_type="sheets", path=tmp_path).load())
    body = docs[0].raw["body"]
    assert "`col_1` — integer" in body
    assert "12" in body and "13" in body and "14" in body


def test_headerless_text_list_keeps_every_value(tmp_path: Path) -> None:
    """A single-column list of values with no header: all 3 items must
    survive in the index, not have the first one consumed as a fake header."""
    (tmp_path / "fruits.csv").write_text("apple\nbanana\ncherry\n", encoding="utf-8")
    docs = list(CsvConnector(source_type="sheets", path=tmp_path).load())
    body = docs[0].raw["body"]
    assert "`col_1` — text" in body
    assert "apple" in body
    assert "banana" in body
    assert "cherry" in body


def test_ragged_rows_are_flagged_and_preserved(tmp_path: Path) -> None:
    (tmp_path / "ragged.csv").write_text(
        "name,amount\nAcme,100\nGlobex,200,extra\nBadRow\n", encoding="utf-8"
    )
    docs = list(CsvConnector(source_type="sheets", path=tmp_path).load())
    body = docs[0].raw["body"]
    assert "row lengths vary" in body
    assert "Globex\t200\textra" in body  # extra cell not truncated
    assert "BadRow" in body  # short row not dropped


def test_embedded_newline_in_quoted_field_is_flattened(tmp_path: Path) -> None:
    (tmp_path / "notes.csv").write_bytes(
        b'name,notes\n"Acme","Called on\nMon and Tue"\n'
    )
    docs = list(CsvConnector(source_type="sheets", path=tmp_path).load())
    body = docs[0].raw["body"]
    assert "Called on Mon and Tue" in body
    assert "Called on\nMon" not in body  # the embedded newline itself is gone


def test_latin1_fallback_for_non_utf8_file(tmp_path: Path) -> None:
    (tmp_path / "accents.csv").write_bytes("name,amount\ncafé,50\nthé,20\n".encode("latin-1"))
    docs = list(CsvConnector(source_type="sheets", path=tmp_path).load())
    body = docs[0].raw["body"]
    assert "café" in body
    assert "thé" in body


def test_empty_file_is_skipped_not_failed(tmp_path: Path) -> None:
    (tmp_path / "empty.csv").write_bytes(b"")
    conn = CsvConnector(source_type="sheets", path=tmp_path)
    docs = list(conn.load())
    assert docs == []
    assert conn.failed_files == 0
    assert conn.skipped_files == 0


def test_binary_content_is_skipped_not_failed(tmp_path: Path) -> None:
    """A NUL byte means this will never be valid CSV -- permanent, not
    transient, so it belongs in skipped_files."""
    (tmp_path / "binary.csv").write_bytes(b"\x00\x01\x02not really text")
    conn = CsvConnector(source_type="sheets", path=tmp_path)
    docs = list(conn.load())
    assert docs == []
    assert conn.skipped_files == 1
    assert conn.failed_files == 0


def test_field_larger_than_limit_counts_as_failed(tmp_path: Path) -> None:
    original_limit = csv.field_size_limit()
    csv.field_size_limit(64)
    try:
        (tmp_path / "huge_field.csv").write_text("a,b\n" + "x" * 200 + ",y\n", encoding="utf-8")
        conn = CsvConnector(source_type="sheets", path=tmp_path)
        docs = list(conn.load())
        assert docs == []
        assert conn.failed_files == 1
        assert conn.skipped_files == 0
    finally:
        csv.field_size_limit(original_limit)


def test_header_only_file_indexes_schema_with_no_data_rows(tmp_path: Path) -> None:
    (tmp_path / "template.csv").write_text("a,b,c\n", encoding="utf-8")
    docs = list(CsvConnector(source_type="sheets", path=tmp_path).load())
    assert len(docs) == 1
    body = docs[0].raw["body"]
    assert "**Rows:** 0  **Columns:** 3" in body
    assert "(no data rows)" in body


def test_tsv_delimiter_is_autodetected(tmp_path: Path) -> None:
    (tmp_path / "scores.tsv").write_text("name\tscore\nAlice\t90\nBob\t85\n", encoding="utf-8")
    conn = CsvConnector(
        source_type="sheets", path=tmp_path, glob="**/*.tsv", default_delimiter="\t"
    )
    docs = list(conn.load())
    body = docs[0].raw["body"]
    assert "`name` — text" in body
    assert "`score` — integer" in body
    assert "Alice\t90" in body


def test_missing_dir_raises() -> None:
    with pytest.raises(FileNotFoundError):
        list(CsvConnector(source_type="sheets", path="/nonexistent").load())


def test_dedupes_identical_files(tmp_path: Path) -> None:
    (tmp_path / "a.csv").write_text("name,amount\nAcme,100\n", encoding="utf-8")
    (tmp_path / "b.csv").write_text("name,amount\nAcme,100\n", encoding="utf-8")
    docs = list(CsvConnector(source_type="sheets", path=tmp_path).load())
    assert len(docs) == 1
