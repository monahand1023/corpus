"""Plain-text connector — points at a directory of `.txt` files.

Stripped-down version of the markdown connector: no frontmatter parsing,
no YAML, just file-by-file. Title comes from the filename stem. Each file
becomes one SourceDocument; chunking is delegated to the shared markdown
chunker, which falls back gracefully to paragraph-boundary splitting when
no markdown headings are present.
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


class TextConnector:
    def __init__(
        self,
        source_type: str,
        path: Path | str,
        glob: str = "**/*.txt",
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
        if not self._root.is_dir():
            raise FileNotFoundError(
                f"Text source '{self.source_type}': directory not found: {self._root}"
            )
        seen: dict[str, str] = {}
        for path in discover_files(self._root, self._glob):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                logger.debug("cannot read %s: %s", path, e)
                self.failed_files += 1
                continue
            if not text.strip():
                continue

            source_key = str(path.relative_to(self._root))
            fp = fingerprint(text)
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
                raw={"body": text, "path": str(path)},
            )
