"""Generic markdown directory connector + chunker.

Points at any directory of `.md` / `.markdown` files (configurable glob).
Parses optional YAML frontmatter for `title`, `url`, `id`, dates. Each file
becomes a `SourceDocument` with body chunks emitted via the shared markdown
chunker.

This is the **reference connector** — when you want to support PDF / HTML /
Slack-export / your-flavor-of-the-week, copy this file as a starting point.
See `docs/adding_a_source.md` for the walkthrough.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from pathlib import Path

from corpus.chunkers.markdown import chunk_markdown_body, parse_markdown
from corpus.types import Chunk, ChunkKind, ChunkMetadata, SourceDocument
from corpus.util.dedup import fingerprint
from corpus.util.hash import chunk_id, sha256
from corpus.util.scrub import scrub
from corpus.util.tokens import estimate_tokens

logger = logging.getLogger(__name__)


class MarkdownConnector:
    """Walks `path` for matching files. Skips near-duplicates (same body
    fingerprint) within a single load() call to handle re-exports of the
    same content under different filenames."""

    def __init__(
        self,
        source_type: str,
        path: Path | str,
        glob: str = "**/*.md",
    ):
        self.source_type = source_type
        self._root = Path(os.path.expanduser(str(path))).resolve()
        self._glob = glob

    def load(self) -> Iterable[SourceDocument]:
        if not self._root.is_dir():
            raise FileNotFoundError(
                f"Markdown source '{self.source_type}': directory not found: {self._root}"
            )
        seen: dict[str, str] = {}
        for md_path in sorted(self._root.glob(self._glob)):
            if not md_path.is_file():
                continue
            try:
                text = md_path.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                logger.debug("cannot read %s: %s", md_path, e)
                continue
            parsed = parse_markdown(text)
            fm = parsed.frontmatter
            # Stable source key: frontmatter `id` if present, else relative path.
            source_key = fm.get("id") or str(md_path.relative_to(self._root))

            fp = fingerprint(parsed.body)
            if fp in seen:
                logger.info(
                    "%s: skipping near-duplicate '%s' (matches earlier '%s')",
                    self.source_type,
                    source_key,
                    seen[fp],
                )
                continue
            seen[fp] = source_key

            # Frontmatter dates take precedence; fall back to filesystem mtime/ctime
            # so timeline + recent_activity tools work for plain markdown without
            # requiring users to add date frontmatter.
            from datetime import UTC, datetime
            stat = md_path.stat()
            fs_modified = datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat()
            fs_created = datetime.fromtimestamp(stat.st_ctime, tz=UTC).isoformat()

            yield SourceDocument(
                source_type=self.source_type,
                source_key=source_key,
                title=fm.get("title") or md_path.stem,
                url=fm.get("url"),
                created_at=fm.get("created") or fs_created,
                updated_at=fm.get("modified") or fm.get("updated") or fs_modified,
                raw={
                    "frontmatter": fm,
                    "body": parsed.body,
                    "path": str(md_path),
                },
            )


class MarkdownChunker:
    """Splits a markdown SourceDocument into chunks via the shared chunker."""

    def __init__(self, source_type: str):
        self.source_type = source_type

    def chunk(self, doc: SourceDocument) -> list[Chunk]:
        body = doc.raw.get("body", "")
        title = doc.title
        url = doc.url
        pieces = chunk_markdown_body(body)
        if not pieces:
            pieces = [body.strip()] if body.strip() else []

        chunks: list[Chunk] = []
        for i, piece in enumerate(pieces):
            # Title goes in the first chunk; later chunks get a "[title]" prefix
            # so retrieval results carry their source's name in the chunk text.
            content = f"{title}\n\n{piece}" if i == 0 else f"[{title}]\n\n{piece}"
            content = scrub(content)
            metadata = ChunkMetadata(
                source_type=self.source_type,
                source_key=doc.source_key,
                chunk_kind=ChunkKind.SECTION,
                chunk_index=i,
                title=title,
                url=url,
                created_at=doc.created_at,
                updated_at=doc.updated_at,
                token_count=estimate_tokens(content),
            )
            chunks.append(
                Chunk(
                    id=chunk_id(self.source_type, doc.source_key, ChunkKind.SECTION, i),
                    content=content,
                    content_hash=sha256(content),
                    metadata=metadata,
                )
            )
        return chunks
