"""Legacy Excel `.xls` connector via `xlrd` (BSD).

`.xls` is OLE2/BIFF — a completely different container from `.xlsx`'s zipped
XML — so `openpyxl` cannot read it and `xlsx.py` cannot be pointed at it.
Output shape is deliberately identical to `xlsx.py`'s (one workbook per
document, sheet name as a heading, non-empty rows tab-joined), so a corpus
holding both eras of the format retrieves them the same way.

Install: `pip install corpus-rag[xls]` or `uv add xlrd`.

Two differences from `xlsx.py`, both forced by the format:

**Dates are floats.** BIFF stores a date as a serial number with the meaning
supplied by the workbook's epoch flag (1900 on Windows, 1904 on classic Mac),
so a raw read turns every date into something like `40744.0`. Cells whose
declared type is `XL_CELL_DATE` are converted through `xlrd.xldate` using the
book's own `datemode`, which is the only way to get the year right for a
Mac-authored workbook.

**Formulas have no cached values to ask for.** `xlsx.py` passes
`data_only=True` and gets whatever Excel last computed. `xlrd` gives the
cached result directly for a formula cell, with no equivalent flag — a
workbook saved by a tool that never computed its formulas yields blanks
either way, which is the same limitation `xlsx.py` documents.

`xlrd` 2.x deliberately dropped `.xlsx` support to avoid exactly the
confusion this pairing could cause; it reads `.xls` only, which is precisely
what is wanted here.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from corpus.connectors.discovery import discover_files
from corpus.types import SourceDocument
from corpus.util.dedup import fingerprint

logger = logging.getLogger(__name__)


def _cell_text(cell: Any, datemode: int) -> str:
    """One cell as text, resolving the two BIFF types that lie about themselves.

    `xlrd` hands back a bare float for both dates and numbers; without the
    type check a date column reads as five-digit serial numbers and a whole
    workbook's chronology becomes unsearchable.
    """
    import xlrd

    value = cell.value
    if cell.ctype == xlrd.XL_CELL_EMPTY or value is None:
        return ""
    if cell.ctype == xlrd.XL_CELL_DATE:
        try:
            when = xlrd.xldate.xldate_as_datetime(value, datemode)
            return str(when.isoformat(sep=" "))
        except (ValueError, OverflowError, xlrd.xldate.XLDateError):
            # Out-of-range serials exist in real files (a 0 in a date-formatted
            # column). Fall through to the raw value rather than losing the cell.
            return str(value)
    if cell.ctype == xlrd.XL_CELL_BOOLEAN:
        return "TRUE" if value else "FALSE"
    if cell.ctype == xlrd.XL_CELL_NUMBER and float(value).is_integer():
        # 2011.0 is noise in a spreadsheet dump; 2011 is a searchable token.
        return str(int(value))
    return str(value).strip()


class XlsConnector:
    def __init__(
        self,
        source_type: str,
        path: Path | str,
        glob: str = "**/*.xls",
    ):
        self.source_type = source_type
        self._root = Path(os.path.expanduser(str(path))).resolve()
        self._glob = glob
        self.failed_files = 0
        self.skipped_files = 0

    def load(self) -> Iterable[SourceDocument]:
        # Reset per run: a stale count from a previous load() would suppress
        # orphan pruning forever. See xlsx.py for the full reasoning.
        self.failed_files = 0
        self.skipped_files = 0
        import xlrd

        if not self._root.is_dir():
            raise FileNotFoundError(
                f"Xls source '{self.source_type}': directory not found: {self._root}"
            )

        seen: dict[str, str] = {}
        for path in discover_files(self._root, self._glob):
            try:
                book = xlrd.open_workbook(str(path), on_demand=True)
            except xlrd.XLRDError as e:
                # The common case by far: an `.xlsx` (or a CSV) misnamed `.xls`,
                # which xlrd 2.x refuses by design. Permanent for this file, so
                # skipped_files — counting it as failed_files would block
                # pruning on every future run for a condition that never clears.
                logger.info(
                    "%s: skipping '%s' — not a legacy .xls workbook: %s",
                    self.source_type,
                    path.name,
                    e,
                )
                self.skipped_files += 1
                continue
            except Exception as e:
                logger.warning("Xls source '%s': cannot open %s: %s", self.source_type, path, e)
                self.failed_files += 1
                continue

            parts: list[str] = []
            try:
                for sheet in book.sheets():
                    rows: list[str] = []
                    for r in range(sheet.nrows):
                        cells = [_cell_text(c, book.datemode) for c in sheet.row(r)]
                        if any(cells):
                            rows.append("\t".join(cells).rstrip("\t"))
                    if rows:
                        parts.append(f"{sheet.name}\n" + "\n".join(rows))
            except Exception as e:
                logger.warning("Xls source '%s': error reading %s: %s", self.source_type, path, e)
                self.failed_files += 1
                continue
            finally:
                book.release_resources()

            body = "\n\n".join(parts).strip()
            if not body:
                logger.info("%s: skipping '%s' — no cell content", self.source_type, path.name)
                self.skipped_files += 1
                continue

            source_key = str(path.relative_to(self._root))
            fp = fingerprint(body)
            if fp in seen:
                logger.info(
                    "%s: skipping near-duplicate '%s' (matches '%s')",
                    self.source_type,
                    source_key,
                    seen[fp],
                )
                continue
            seen[fp] = source_key

            yield SourceDocument(
                source_type=self.source_type,
                source_key=source_key,
                title=path.stem,
                url=None,
                raw={"body": body, "path": str(path)},
            )
