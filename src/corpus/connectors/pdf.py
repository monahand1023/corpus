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

from corpus.connectors.discovery import discover_files
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
        self.failed_files = 0
        self.skipped_files = 0

    def load(self) -> Iterable[SourceDocument]:
        # Per-file read failures are counted, not just logged: a skipped file
        # yields no document, so its chunk ids vanish from `seen_ids` and the
        # ingester's orphan sweep would delete already-indexed content. The
        # ingester reads this counter and suppresses pruning when it is
        # non-zero. Reset per run so a reused instance cannot suppress pruning
        # forever on the strength of an old failure.
        self.failed_files = 0
        self.skipped_files = 0
        from pypdf import PdfReader

        # Lazy like PdfReader above: pypdf is an optional extra.
        from pypdf.errors import FileNotDecryptedError

        if not self._root.is_dir():
            raise FileNotFoundError(
                f"PDF source '{self.source_type}': directory not found: {self._root}"
            )

        seen: dict[str, str] = {}
        for path in discover_files(self._root, self._glob):
            try:
                reader = PdfReader(str(path))
                # pypdf parses LAZILY: the constructor succeeds for an encrypted
                # or malformed file and the failure only surfaces when `.pages`
                # is first touched (FileNotDecryptedError, and friends). Force
                # that here, inside the per-file guard — outside it, one bad PDF
                # aborts the entire source, and since pypdf's errors derive from
                # Exception rather than ValueError/OSError, cli/ingest.py's
                # per-source handler would not catch it either, taking down an
                # entire `--all` run.
                pages = list(reader.pages)
            except FileNotDecryptedError:
                # PERMANENT, so it counts as a skip rather than a failure.
                # `failed_files` means "might succeed next time" and suppresses
                # orphan pruning for the whole source; an encrypted PDF will
                # never decrypt without its password, so counting it there
                # switches pruning OFF FOREVER for that source. Measured on a
                # real library: a handful of password-protected PDFs in one source meant
                # it could never prune a deleted document again, on any run.
                # `skipped_files` is the right bucket -- reported and visible,
                # but it does not gate pruning, because the absence is not a
                # surprise and will not resolve itself.
                logger.info(
                    "%s: skipping '%s' -- encrypted, no password available",
                    self.source_type, path.name,
                )
                self.skipped_files += 1
                continue
            except Exception as e:  # pypdf raises many subclasses; treat any read failure as skip
                logger.warning("PDF source '%s': cannot open %s: %s", self.source_type, path, e)
                self.failed_files += 1
                continue

            page_texts: list[str] = []
            page_failed = False
            for i, page in enumerate(pages):
                try:
                    page_text = page.extract_text() or ""
                except Exception as e:
                    logger.debug("page %d of %s: extract_text failed: %s", i, path, e)
                    page_failed = True
                    continue
                if page_text.strip():
                    page_texts.append(page_text)
            if page_failed:
                # Counted ONCE per file, however many pages failed. The document
                # is still yielded, but with a shorter body — which produces
                # fewer chunks, so the tail chunk ids disappear from the
                # ingester's `seen_ids` and its orphan sweep would delete the
                # content those pages used to occupy. Being yielded does not
                # make a partially-read file safe to prune against.
                self.failed_files += 1
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
                raw={"body": body, "path": str(path), "page_count": len(pages)},
            )
