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

    def load(self) -> Iterable[SourceDocument]:
        if not self._root.is_dir():
            raise FileNotFoundError(
                f"Text source '{self.source_type}': directory not found: {self._root}"
            )
        seen: dict[str, str] = {}
        for path in sorted(self._root.glob(self._glob)):
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                logger.debug("cannot read %s: %s", path, e)
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
