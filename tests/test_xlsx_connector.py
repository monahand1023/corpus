from __future__ import annotations

from pathlib import Path

import pytest

from corpus.connectors.xlsx import XlsxConnector


def _write_workbook(path: Path, sheets: dict[str, list[list[object]]]) -> None:
    import openpyxl

    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for name, rows in sheets.items():
        ws = wb.create_sheet(title=name)
        for row in rows:
            ws.append(row)
    wb.save(path)


def test_loads_sheets_with_headings(tmp_path: Path) -> None:
    _write_workbook(tmp_path / "budget.xlsx", {
        "Q1": [["Item", "Cost"], ["Rent", 1200], ["Power", 90]],
        "Notes": [["Reviewed by Dan"]],
    })
    docs = list(XlsxConnector(source_type="sheets", path=tmp_path).load())
    assert len(docs) == 1
    body = docs[0].raw["body"]
    assert "Q1" in body
    assert "Item\tCost" in body
    assert "Rent\t1200" in body
    assert "Notes" in body
    assert "Reviewed by Dan" in body
    assert docs[0].title == "budget"


def test_skips_empty_workbook(tmp_path: Path) -> None:
    _write_workbook(tmp_path / "blank.xlsx", {"Sheet1": []})
    assert list(XlsxConnector(source_type="sheets", path=tmp_path).load()) == []


def test_skips_unreadable_workbook(tmp_path: Path) -> None:
    (tmp_path / "corrupt.xlsx").write_bytes(b"not a workbook")
    assert list(XlsxConnector(source_type="sheets", path=tmp_path).load()) == []


def test_missing_dir_raises() -> None:
    with pytest.raises(FileNotFoundError):
        list(XlsxConnector(source_type="sheets", path="/nonexistent").load())
