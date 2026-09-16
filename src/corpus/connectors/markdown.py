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
from corpus.connectors.discovery import discover_files
from corpus.transcripts.quality import strip_caption_tail
from corpus.types import Chunk, ChunkKind, ChunkMetadata, SourceDocument
from corpus.util.dedup import fingerprint
from corpus.util.encoding import read_text_with_fallback
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
                f"Markdown source '{self.source_type}': directory not found: {self._root}"
            )
        seen: dict[str, str] = {}
        for md_path in discover_files(self._root, self._glob):
            try:
                text = read_text_with_fallback(md_path)
            except OSError as e:
                logger.debug("cannot read %s: %s", md_path, e)
                self.failed_files += 1
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


def _strip_nul(text: str) -> str:
    """Remove NUL characters from chunk text.

    Every connector routes through `MarkdownChunker` (see
    `connectors/registry.py` — all 13 `_build_*` functions return one), so
    this is the single point every chunk of every source type passes
    through, and the right place for normalization that is not
    format-specific.

    NUL is never meaningful document content, but real extractions emit it:
    measured on one archive, 239 chunks carried NUL bytes, all from PDF
    extraction where a page's font had no usable encoding and the extractor
    handed back raw bytes as if they were text. It survives into the store
    and then breaks consumers that treat text as C strings or tokenize it —
    FTS5 indexing and terminal display both truncate at the first NUL, which
    silently hides the rest of an otherwise-fine chunk.

    Deliberately NUL only, not all C0 controls: form feed (\x0c) is a real
    page separator in PDF text and carries structure worth keeping. Stripping
    is unconditional rather than a rejection gate — chunks around the NUL are
    frequently good text, and a whole-chunk quality gate measured on the same
    archive would have discarded recoverable documents (several PDFs there
    have intact text behind a shifted font encoding, which reads as noise but
    is not).
    """
    return text.replace("\x00", "") if "\x00" in text else text


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
        index = 0
        for piece in pieces:
            # Transcription sign-offs ride in on every connector, not just the
            # transcript one. A voicemail transcript inside an email, a meeting
            # transcript pasted into a note, an Otter export saved as a .docx:
            # all arrive here, and stripping only in `connectors/transcripts.py`
            # attached the cleanup to the CONNECTOR rather than to the CONTENT.
            # `corpus-survey index-quality` detected the leftovers across every
            # source type and then told the operator to edit a connector by
            # hand. The vocabulary is closed and matches only at a tail, so
            # this is safe on ordinary prose.
            piece = strip_caption_tail(piece)
            if not piece.strip():
                # The whole piece was boilerplate. Indexing it would give
                # search a result that answers nothing.
                continue
            # Title goes in the first chunk; later chunks get a "[title]" prefix
            # so retrieval results carry their source's name in the chunk text.
            i = index
            index += 1
            content = f"{title}\n\n{piece}" if i == 0 else f"[{title}]\n\n{piece}"
            content = _strip_nul(content)
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
