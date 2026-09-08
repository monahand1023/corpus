"""Excel `.xlsx` connector via `openpyxl` (MIT).

One workbook becomes one SourceDocument. Each sheet contributes its name as a
heading followed by its non-empty rows, cells tab-joined.

Spreadsheets retrieve poorly whatever you do — the goal here is findability, not
fidelity. Formulas are read as their cached values (`data_only=True`), so a
workbook never opened by Excel may have empty formula cells.

Install: `pip install corpus-rag[xlsx]` or `uv add openpyxl`.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from pathlib import Path

from corpus.connectors.discovery import discover_files
from corpus.types import SourceDocument
from corpus.util.dedup import fingerprint

logger = logging.getLogger(__name__)


class XlsxConnector:
    def __init__(
        self,
        source_type: str,
        path: Path | str,
        glob: str = "**/*.xlsx",
    ):
        self.source_type = source_type
        self._root = Path(os.path.expanduser(str(path))).resolve()
        self._glob = glob

    def load(self) -> Iterable[SourceDocument]:
        import openpyxl

        if not self._root.is_dir():
            raise FileNotFoundError(
                f"Xlsx source '{self.source_type}': directory not found: {self._root}"
            )

        seen: dict[str, str] = {}
        for path in discover_files(self._root, self._glob):
            try:
                wb = openpyxl.load_workbook(str(path), data_only=True, read_only=True)
            except Exception as e:
                logger.warning("Xlsx source '%s': cannot open %s: %s", self.source_type, path, e)
                continue

            parts: list[str] = []
            try:
                for ws in wb.worksheets:
                    rows: list[str] = []
                    for row in ws.iter_rows(values_only=True):
                        cells = ["" if v is None else str(v).strip() for v in row]
                        if any(cells):
                            rows.append("\t".join(cells).rstrip("\t"))
                    if rows:
                        parts.append(f"{ws.title}\n" + "\n".join(rows))
            except Exception as e:
                logger.warning("Xlsx source '%s': error reading %s: %s", self.source_type, path, e)
                continue
            finally:
                wb.close()

            body = "\n\n".join(parts).strip()
            if not body:
                logger.info("%s: skipping '%s' — no cell content", self.source_type, path.name)
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
