"""Word `.docx` connector via `python-docx` (MIT).

Each file becomes one SourceDocument whose body is the document's paragraphs
followed by its tables. Paragraphs and tables are separate top-level
collections in the underlying XML, not interleaved in reading order, so a
table that appears between two paragraphs in the original document still
lands after ALL paragraphs here — this is not full document-order fidelity.
Chunking is delegated to the shared markdown chunker.

Limitations:
  - Headers, footers, footnotes, comments, and tracked-change markup are not
    extracted; only body paragraphs and table cells.
  - Legacy `.doc` (OLE2) is a different format entirely and is not supported —
    it needs LibreOffice or `antiword`.

Install: `pip install corpus-rag[docx]` or `uv add python-docx`.
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


class DocxConnector:
    def __init__(
        self,
        source_type: str,
        path: Path | str,
        glob: str = "**/*.docx",
    ):
        self.source_type = source_type
        self._root = Path(os.path.expanduser(str(path))).resolve()
        self._glob = glob

    def load(self) -> Iterable[SourceDocument]:
        import docx

        if not self._root.is_dir():
            raise FileNotFoundError(
                f"Docx source '{self.source_type}': directory not found: {self._root}"
            )

        seen: dict[str, str] = {}
        for path in discover_files(self._root, self._glob):
            try:
                document = docx.Document(str(path))
            except Exception as e:  # python-docx raises many types for bad files
                logger.warning("Docx source '%s': cannot open %s: %s", self.source_type, path, e)
                continue

            parts = [p.text.strip() for p in document.paragraphs if p.text.strip()]
            for table in document.tables:
                for row in table.rows:
                    cells = [c.text.strip() for c in row.cells]
                    if any(cells):
                        parts.append("\t".join(cells))
            body = "\n\n".join(parts).strip()

            if not body:
                logger.info(
                    "%s: skipping '%s' — no extractable text", self.source_type, path.name
                )
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

            title = path.stem
            try:
                meta_title = document.core_properties.title
                if meta_title and str(meta_title).strip():
                    title = str(meta_title).strip()
            except Exception:
                pass

            yield SourceDocument(
                source_type=self.source_type,
                source_key=source_key,
                title=title,
                url=None,
                raw={"body": body, "path": str(path)},
            )
