"""HTML connector via `trafilatura`.

Trafilatura is purpose-built for boilerplate stripping — it identifies the
main content of a page (article body, blog post, wiki entry) and discards
nav menus, ads, comments, related-links sidebars. Output is plain text
(or markdown, if requested) suitable for direct chunking.

Each `.html` / `.htm` file → one SourceDocument with `body` = main content,
plus extracted metadata (title, author, date) when trafilatura can find it.

Install: `pip install corpus-rag[html]` or `uv add trafilatura`.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from pathlib import Path

from corpus.types import SourceDocument
from corpus.util.dedup import fingerprint

logger = logging.getLogger(__name__)


class HtmlConnector:
    def __init__(
        self,
        source_type: str,
        path: Path | str,
        glob: str = "**/*.html",
    ):
        self.source_type = source_type
        self._root = Path(os.path.expanduser(str(path))).resolve()
        self._glob = glob

    def load(self) -> Iterable[SourceDocument]:
        import trafilatura

        if not self._root.is_dir():
            raise FileNotFoundError(
                f"HTML source '{self.source_type}': directory not found: {self._root}"
            )

        seen: dict[str, str] = {}
        for path in sorted(self._root.glob(self._glob)):
            if not path.is_file():
                continue
            try:
                raw_html = path.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                logger.debug("cannot read %s: %s", path, e)
                continue

            extracted = trafilatura.extract(
                raw_html,
                include_comments=False,
                include_tables=True,
                favor_recall=True,  # err toward including marginal content
            )
            if not extracted or not extracted.strip():
                logger.info(
                    "%s: skipping '%s' — trafilatura found no main content",
                    self.source_type,
                    path.name,
                )
                continue

            # Metadata extraction (title, author, date)
            meta = trafilatura.extract_metadata(raw_html) or None
            title = path.stem
            author = None
            url = None
            created_at = None
            if meta is not None:
                title = (getattr(meta, "title", None) or path.stem).strip() or path.stem
                author = getattr(meta, "author", None)
                url = getattr(meta, "url", None)
                created_at = getattr(meta, "date", None)

            source_key = str(path.relative_to(self._root))
            fp = fingerprint(extracted)
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
                title=title,
                url=url,
                created_at=created_at,
                raw={
                    "body": extracted,
                    "path": str(path),
                    "author": author,
                },
            )
