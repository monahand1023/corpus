"""PDF connector via `pypdf` (pure Python, MIT license).

Each PDF → one SourceDocument with `body` = concatenated page text. Chunking
runs through the shared markdown chunker, which size-splits at paragraph
boundaries.

Limitations:
  - Scanned/image-only PDFs return empty text. Run OCR (`ocrmypdf`, `tesseract`)
    before pointing the connector at them.
  - Complex layouts (multi-column papers, tables) may extract in non-reading-
    order. For high-quality paper retrieval, consider `pymupdf4llm` instead
    (AGPL-licensed; left out of the default deps for licensing flexibility).
  - The `title` is read from PDF metadata when present, falling back to the
    filename stem.

Install: `pip install corpus-rag[pdf]` or `uv add pypdf`.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from pathlib import Path

from corpus.types import SourceDocument
from corpus.util.dedup import fingerprint

logger = logging.getLogger(__name__)


class PdfConnector:
    def __init__(
        self,
        source_type: str,
        path: Path | str,
        glob: str = "**/*.pdf",
    ):
        self.source_type = source_type
        self._root = Path(os.path.expanduser(str(path))).resolve()
        self._glob = glob

    def load(self) -> Iterable[SourceDocument]:
        from pypdf import PdfReader

        if not self._root.is_dir():
            raise FileNotFoundError(
                f"PDF source '{self.source_type}': directory not found: {self._root}"
            )

        seen: dict[str, str] = {}
        for path in sorted(self._root.glob(self._glob)):
            if not path.is_file():
                continue
            try:
                reader = PdfReader(str(path))
            except Exception as e:  # pypdf raises many subclasses; treat any read failure as skip
                logger.warning("PDF source '%s': cannot open %s: %s", self.source_type, path, e)
                continue

            page_texts: list[str] = []
            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text() or ""
                except Exception as e:
                    logger.debug("page %d of %s: extract_text failed: %s", i, path, e)
                    continue
                if page_text.strip():
                    page_texts.append(page_text)
            body = "\n\n".join(page_texts).strip()

            if not body:
                logger.info(
                    "%s: skipping '%s' — no extractable text (scanned PDF?)",
                    self.source_type,
                    path.name,
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

            # Pull title from PDF metadata if available
            title = path.stem
            try:
                meta = reader.metadata
                if meta and meta.title:
                    title = str(meta.title).strip() or title
            except Exception:
                pass

            yield SourceDocument(
                source_type=self.source_type,
                source_key=source_key,
                title=title,
                url=None,
                raw={"body": body, "path": str(path), "page_count": len(reader.pages)},
            )
