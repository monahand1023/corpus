# Adding a new source type

`corpus` ships one built-in connector: `markdown`. For everything else — PDFs, HTML pages, Slack exports, Jira API dumps, your custom JSON format — you write the connector.

This doc walks through it with a worked example: **a JSON-files connector** that reads a directory of `.json` files, each one containing one document.

The pattern is:

1. **Write a `Connector`** — walks your data, yields one `SourceDocument` per logical "document"
2. **Write a `Chunker`** — splits a `SourceDocument` into one or more `Chunk`s
3. **Register the pair** in `src/corpus/connectors/registry.py`
4. **Reference the new type** in your `corpus.toml`

## 1. Write the connector

The `Connector` protocol:

```python
from collections.abc import Iterable
from corpus.types import SourceDocument

class Connector(Protocol):
    source_type: str
    def load(self) -> Iterable[SourceDocument]: ...
```

`source_type` is a free-form lowercase identifier — your choice. It becomes the chunk's `source_type` field and the `name` users put in their `corpus.toml`.

Example: `src/corpus/connectors/json_files.py`

```python
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from corpus.types import SourceDocument


class JsonFilesConnector:
    """Reads a directory of .json files. Each file = one document.

    Expects each JSON to have at least: id, title. Optional: url, body,
    created_at, updated_at, author, plus any custom fields (kept in raw).
    """

    def __init__(self, source_type: str, path: Path | str, glob: str = "*.json"):
        self.source_type = source_type
        self._root = Path(path).expanduser().resolve()
        self._glob = glob

    def load(self) -> Iterable[SourceDocument]:
        for json_path in sorted(self._root.glob(self._glob)):
            data = json.loads(json_path.read_text())
            yield SourceDocument(
                source_type=self.source_type,
                source_key=str(data["id"]),
                title=data["title"],
                url=data.get("url"),
                created_at=data.get("created_at"),
                updated_at=data.get("updated_at"),
                raw=data,  # the chunker reads from here
            )
```

## 2. Write the chunker

The `Chunker` protocol takes a `SourceDocument` and emits one or more `Chunk`s:

```python
from corpus.types import Chunk, ChunkKind, ChunkMetadata, SourceDocument
from corpus.util.hash import chunk_id, sha256
from corpus.util.scrub import scrub
from corpus.util.tokens import MAX_CHUNK_TOKENS, estimate_tokens

MAX_CHUNK_CHARS = MAX_CHUNK_TOKENS * 4


class JsonFilesChunker:
    def __init__(self, source_type: str):
        self.source_type = source_type

    def chunk(self, doc: SourceDocument) -> list[Chunk]:
        raw = doc.raw
        title = doc.title
        body = (raw.get("body") or "").strip()

        # Simple strategy: one chunk per document. If the body is huge, split
        # it into ~512-token pieces.
        pieces = self._split(body) if body else [""]

        chunks: list[Chunk] = []
        for i, piece in enumerate(pieces):
            content = f"{title}\n\n{piece}" if i == 0 else f"[{title}]\n\n{piece}"
            content = scrub(content)  # strip credentials before embedding
            metadata = ChunkMetadata(
                source_type=self.source_type,
                source_key=doc.source_key,
                chunk_kind=ChunkKind.SECTION,
                chunk_index=i,
                title=title,
                url=doc.url,
                author=raw.get("author"),
                created_at=doc.created_at,
                updated_at=doc.updated_at,
                token_count=estimate_tokens(content),
                extra={"category": raw.get("category")},  # any custom fields
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

    def _split(self, text: str) -> list[str]:
        if len(text) <= MAX_CHUNK_CHARS:
            return [text]
        # Naive char-boundary split. For real text, split at paragraphs.
        return [text[i : i + MAX_CHUNK_CHARS] for i in range(0, len(text), MAX_CHUNK_CHARS)]
```

## 3. Register it

Edit `src/corpus/connectors/registry.py`:

```python
from corpus.connectors.json_files import JsonFilesConnector, JsonFilesChunker

def _build_json_files(cfg: SourceConfig) -> tuple[JsonFilesConnector, JsonFilesChunker]:
    connector = JsonFilesConnector(
        source_type=cfg.name,
        path=cfg.path,
        glob=cfg.glob or "*.json",
    )
    return connector, JsonFilesChunker(source_type=cfg.name)


CONNECTOR_REGISTRY: dict[str, _ConnectorFactory] = {
    "markdown": _build_markdown,
    "json_files": _build_json_files,   # ← add this
}
```

## 4. Use it in `corpus.toml`

```toml
[[sources]]
name = "my_articles"
type = "json_files"      # matches the registry key
path = "~/exports/articles"
glob = "*.json"
```

Then:

```sh
corpus-ingest --source my_articles -v
corpus-query "your question"
```

## Tips

- **Set `metadata.author`** if you want `who_did_what(person)` to work for your source.
- **Set `metadata.extra.parent`** if your docs have hierarchy (Jira parent epics, threaded forum posts, etc.). `expand_context`'s `parent` mode uses this.
- **Honor the size cap.** Chunks over ~2,000 chars start eating context budget on the Claude side. The chars/4 token heuristic is in `util/tokens.py`.
- **Run scrub** in the chunker before computing `content_hash`. Otherwise dedup will think two near-identical docs are different just because secret patterns differ.
- **Use deterministic `chunk_id`s.** `chunk_id(source_type, source_key, kind, index)` from `util/hash.py` does this — guarantees idempotent re-ingestion.
- **Test your connector**: write a `tests/test_my_connector.py` that builds 2-3 fake files in a `tmp_path` and exercises load() + chunk().
