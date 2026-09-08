"""Rich Text Format `.rtf` connector via `striprtf` (pure Python, BSD).

Each file becomes one SourceDocument of plain text. Title comes from the
filename stem — RTF has no reliable title metadata.

Install: `pip install corpus-rag[rtf]` or `uv add striprtf`.
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


class RtfConnector:
    def __init__(
        self,
        source_type: str,
        path: Path | str,
        glob: str = "**/*.rtf",
    ):
        self.source_type = source_type
        self._root = Path(os.path.expanduser(str(path))).resolve()
        self._glob = glob
        self.failed_files = 0

    def load(self) -> Iterable[SourceDocument]:
        # Per-file read failures are counted, not just logged: a skipped file
        # yields no document, so its chunk ids vanish from `seen_ids` and the
        # ingester's orphan sweep would delete already-indexed content. The
        # ingester reads this counter and suppresses pruning when it is
        # non-zero. Reset per run so a reused instance cannot suppress pruning
        # forever on the strength of an old failure.
        self.failed_files = 0
        from striprtf.striprtf import rtf_to_text

        if not self._root.is_dir():
            raise FileNotFoundError(
                f"Rtf source '{self.source_type}': directory not found: {self._root}"
            )

        seen: dict[str, str] = {}
        for path in discover_files(self._root, self._glob):
            try:
                raw = path.read_text(encoding="utf-8", errors="replace")
                body = rtf_to_text(raw, errors="ignore").strip()  # type: ignore[no-untyped-call]
            except Exception as e:
                logger.warning("Rtf source '%s': cannot read %s: %s", self.source_type, path, e)
                self.failed_files += 1
                continue

            if not body:
                logger.info("%s: skipping '%s' — no extractable text", self.source_type, path.name)
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
