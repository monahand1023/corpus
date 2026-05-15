"""SQLite + sqlite-vec store. One file. UPSERT semantics. Orphan deletion.

SCHEMA HAZARD — do not `VACUUM` this DB without rebuilding the vec/fts
indices. The `chunks` table's auto-rowid is the foreign key into both
`chunks_vec` and `chunks_fts`; VACUUM can re-pack rowids and silently
break the linkage. Re-ingest from scratch is the safe recovery path.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import sqlite_vec

from corpus.types import Chunk


@dataclass(frozen=True)
class UpsertResult:
    upserted: int
    skipped: int


@dataclass
class StoredChunk:
    id: str
    source_type: str
    source_key: str
    content: str
    metadata: dict[str, Any]
    title: str | None
    url: str | None
    distance: float | None = None


class EmbeddingDimMismatch(RuntimeError):
    """Raised when the requested embedding dim doesn't match the existing
    schema. Switching dims after data has been ingested would silently
    corrupt retrieval."""


class ChunkStore:
    def __init__(self, db_path: Path | str, *, embedding_dim: int):
        """Open or create the chunk store.

        `embedding_dim` is checked against the existing schema if the DB
        already has data — if they mismatch (e.g., user changed embedder
        model without re-ingesting), we raise rather than silently corrupt.

        Threading: each thread gets its own sqlite3.Connection via thread-local
        storage. WAL mode handles concurrent readers; ingest (single-threaded)
        is the only writer path. The MCP server uses `asyncio.to_thread` so
        each tool invocation lands on its own worker thread and gets its own
        connection — no shared cursor state.
        """
        self._db_path = Path(db_path)
        self._embedding_dim = embedding_dim
        self._tls = threading.local()
        # All opened connections, tracked so close() can shut them all down.
        # Guarded by a lock since connections open on arbitrary worker threads.
        self._all_conns: list[sqlite3.Connection] = []
        self._conns_lock = threading.Lock()
        # First connection: opened on the constructing thread, used for schema
        # setup + the dim guard, then registered as that thread's TLS conn so
        # it's reused rather than orphaned.
        init_conn = self._open_connection()
        self._tls.conn = init_conn
        self._init_schema(init_conn)
        self._guard_embedding_dim(init_conn)

    def _open_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        with self._conns_lock:
            self._all_conns.append(conn)
        return conn

    @property
    def _conn(self) -> sqlite3.Connection:
        """Per-thread connection. Lazy-created on first access from each thread."""
        existing = getattr(self._tls, "conn", None)
        if existing is None:
            existing = self._open_connection()
            self._tls.conn = existing
        return existing

    def _init_schema(self, conn: sqlite3.Connection | None = None) -> None:
        conn = conn or self._conn
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS chunks (
              id TEXT PRIMARY KEY,
              source_type TEXT NOT NULL,
              source_key TEXT NOT NULL,
              chunk_kind TEXT NOT NULL,
              chunk_index INTEGER NOT NULL,
              content TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              metadata TEXT NOT NULL,
              title TEXT,
              url TEXT,
              author TEXT,
              created_at TEXT,
              updated_at TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_source_type ON chunks(source_type);
            CREATE INDEX IF NOT EXISTS idx_chunks_source_key ON chunks(source_key);

            CREATE TABLE IF NOT EXISTS schema_meta (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS summaries (
              source_type TEXT NOT NULL,
              source_key TEXT NOT NULL,
              summary TEXT NOT NULL,
              doc_hash TEXT NOT NULL,
              model TEXT NOT NULL,
              generated_at TEXT NOT NULL,
              token_count INTEGER,
              PRIMARY KEY (source_type, source_key)
            );
            """
        )
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(embedding float[{self._embedding_dim}])"
        )
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(content, tokenize = 'porter unicode61')"
        )
        conn.commit()

    def _guard_embedding_dim(self, conn: sqlite3.Connection | None = None) -> None:
        """If the DB has previous data, the embedding dim must match."""
        conn = conn or self._conn
        row = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'embedding_dim'"
        ).fetchone()
        if row is None:
            # First open — record the chosen dim.
            conn.execute(
                "INSERT INTO schema_meta (key, value) VALUES ('embedding_dim', ?)",
                (str(self._embedding_dim),),
            )
            conn.commit()
            return
        existing = int(row["value"])
        if existing != self._embedding_dim:
            raise EmbeddingDimMismatch(
                f"DB was created with embedding_dim={existing}, but this run "
                f"requests {self._embedding_dim}. Re-ingest from scratch with "
                f"the new dim, or revert your embedder.model in corpus.toml."
            )

    @contextmanager
    def _txn(self):
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def get_known_hashes(self, ids: Sequence[str]) -> dict[str, str]:
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        rows = self._conn.execute(
            f"SELECT id, content_hash FROM chunks WHERE id IN ({placeholders})",
            tuple(ids),
        ).fetchall()
        return {row["id"]: row["content_hash"] for row in rows}

    def upsert(self, chunk: Chunk, embedding: Sequence[float]) -> bool:
        if len(embedding) != self._embedding_dim:
            raise ValueError(
                f"embedding dim mismatch: got {len(embedding)}, expected {self._embedding_dim}"
            )

        existing = self._conn.execute(
            "SELECT content_hash FROM chunks WHERE id = ?", (chunk.id,)
        ).fetchone()
        if existing and existing["content_hash"] == chunk.content_hash:
            return False

        meta_json = chunk.metadata.model_dump_json()
        cur = self._conn.execute(
            """
            INSERT INTO chunks (
              id, source_type, source_key, chunk_kind, chunk_index,
              content, content_hash, metadata, title, url, author, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              chunk_kind = excluded.chunk_kind,
              chunk_index = excluded.chunk_index,
              content = excluded.content,
              content_hash = excluded.content_hash,
              metadata = excluded.metadata,
              title = excluded.title,
              url = excluded.url,
              author = excluded.author,
              created_at = excluded.created_at,
              updated_at = excluded.updated_at
            RETURNING rowid
            """,
            (
                chunk.id,
                chunk.metadata.source_type,
                chunk.metadata.source_key,
                chunk.metadata.chunk_kind.value,
                chunk.metadata.chunk_index,
                chunk.content,
                chunk.content_hash,
                meta_json,
                chunk.metadata.title,
                chunk.metadata.url,
                chunk.metadata.author,
                chunk.metadata.created_at,
                chunk.metadata.updated_at,
            ),
        )
        rowid = cur.fetchone()["rowid"]
        blob = sqlite_vec.serialize_float32(list(embedding))
        self._conn.execute("DELETE FROM chunks_vec WHERE rowid = ?", (rowid,))
        self._conn.execute("INSERT INTO chunks_vec(rowid, embedding) VALUES (?, ?)", (rowid, blob))
        self._conn.execute("DELETE FROM chunks_fts WHERE rowid = ?", (rowid,))
        self._conn.execute("INSERT INTO chunks_fts(rowid, content) VALUES (?, ?)", (rowid, chunk.content))
        return True

    def upsert_batch(self, items: Iterable[tuple[Chunk, Sequence[float]]]) -> UpsertResult:
        upserted = 0
        skipped = 0
        with self._txn():
            for chunk, embedding in items:
                if self.upsert(chunk, embedding):
                    upserted += 1
                else:
                    skipped += 1
        return UpsertResult(upserted=upserted, skipped=skipped)

    def delete_by_source(self, source_type: str) -> int:
        rowids = [
            row["rowid"]
            for row in self._conn.execute(
                "SELECT rowid FROM chunks WHERE source_type = ?", (source_type,)
            )
        ]
        with self._txn():
            for rid in rowids:
                self._conn.execute("DELETE FROM chunks_vec WHERE rowid = ?", (rid,))
                self._conn.execute("DELETE FROM chunks_fts WHERE rowid = ?", (rid,))
            self._conn.execute("DELETE FROM chunks WHERE source_type = ?", (source_type,))
        return len(rowids)

    def delete_orphans(self, source_type: str, seen_ids: set[str]) -> int:
        existing = self._conn.execute(
            "SELECT id, rowid FROM chunks WHERE source_type = ?", (source_type,)
        ).fetchall()
        orphans = [(row["id"], row["rowid"]) for row in existing if row["id"] not in seen_ids]
        if not orphans:
            return 0
        with self._txn():
            for _id, rowid in orphans:
                self._conn.execute("DELETE FROM chunks_vec WHERE rowid = ?", (rowid,))
                self._conn.execute("DELETE FROM chunks_fts WHERE rowid = ?", (rowid,))
            placeholders = ",".join("?" for _ in orphans)
            self._conn.execute(
                f"DELETE FROM chunks WHERE id IN ({placeholders})",
                tuple(o[0] for o in orphans),
            )
        return len(orphans)

    def fts_search(
        self,
        query: str,
        top_k: int,
        filter_sources: Sequence[str] | None = None,
    ) -> list[StoredChunk]:
        import re as _re

        tokens = _re.findall(r"[A-Za-z0-9_\-]+", query)
        if not tokens:
            return []
        match_terms = []
        for t in tokens:
            if "-" in t and t.count("-") <= 3:
                match_terms.append(f'"{t}"')
            else:
                match_terms.append(t)
        match_expr = " OR ".join(match_terms)

        over_fetch = top_k * 3 if filter_sources else top_k
        try:
            rows = self._conn.execute(
                """
                SELECT c.id, c.source_type, c.source_key, c.content, c.metadata,
                       c.title, c.url, f.rank
                FROM chunks_fts f
                JOIN chunks c ON c.rowid = f.rowid
                WHERE f.content MATCH ?
                ORDER BY f.rank
                LIMIT ?
                """,
                (match_expr, over_fetch),
            ).fetchall()
        except Exception:
            return []

        results: list[StoredChunk] = []
        filter_set = set(filter_sources) if filter_sources else None
        for row in rows:
            if filter_set and row["source_type"] not in filter_set:
                continue
            results.append(
                StoredChunk(
                    id=row["id"],
                    source_type=row["source_type"],
                    source_key=row["source_key"],
                    content=row["content"],
                    metadata=json.loads(row["metadata"]),
                    title=row["title"],
                    url=row["url"],
                    distance=row["rank"],
                )
            )
            if len(results) >= top_k:
                break
        return results

    def vector_search(
        self,
        query_embedding: Sequence[float],
        top_k: int,
        filter_sources: Sequence[str] | None = None,
    ) -> list[StoredChunk]:
        blob = sqlite_vec.serialize_float32(list(query_embedding))
        over_fetch = top_k * 3 if filter_sources else top_k
        rows = self._conn.execute(
            """
            SELECT c.id, c.source_type, c.source_key, c.content, c.metadata,
                   c.title, c.url, v.distance
            FROM chunks_vec v
            JOIN chunks c ON c.rowid = v.rowid
            WHERE v.embedding MATCH ? AND k = ?
            ORDER BY v.distance
            """,
            (blob, over_fetch),
        ).fetchall()

        results: list[StoredChunk] = []
        filter_set = set(filter_sources) if filter_sources else None
        for row in rows:
            if filter_set and row["source_type"] not in filter_set:
                continue
            results.append(
                StoredChunk(
                    id=row["id"],
                    source_type=row["source_type"],
                    source_key=row["source_key"],
                    content=row["content"],
                    metadata=json.loads(row["metadata"]),
                    title=row["title"],
                    url=row["url"],
                    distance=row["distance"],
                )
            )
            if len(results) >= top_k:
                break
        return results

    def get_by_id(self, chunk_id: str) -> StoredChunk | None:
        row = self._conn.execute(
            "SELECT id, source_type, source_key, content, metadata, title, url FROM chunks WHERE id = ?",
            (chunk_id,),
        ).fetchone()
        if not row:
            return None
        return StoredChunk(
            id=row["id"],
            source_type=row["source_type"],
            source_key=row["source_key"],
            content=row["content"],
            metadata=json.loads(row["metadata"]),
            title=row["title"],
            url=row["url"],
        )

    def get_by_source_key(self, source_type: str, source_key: str) -> list[StoredChunk]:
        rows = self._conn.execute(
            """
            SELECT id, source_type, source_key, content, metadata, title, url
            FROM chunks
            WHERE source_type = ? AND source_key = ?
            ORDER BY chunk_index
            """,
            (source_type, source_key),
        ).fetchall()
        return [
            StoredChunk(
                id=row["id"],
                source_type=row["source_type"],
                source_key=row["source_key"],
                content=row["content"],
                metadata=json.loads(row["metadata"]),
                title=row["title"],
                url=row["url"],
            )
            for row in rows
        ]

    def find_recent(
        self,
        since: str,
        filter_sources: Sequence[str] | None = None,
        limit: int = 50,
    ) -> list[StoredChunk]:
        clauses = ["updated_at >= ?"]
        params: list[Any] = [since]
        if filter_sources:
            placeholders = ",".join("?" for _ in filter_sources)
            clauses.append(f"source_type IN ({placeholders})")
            params.extend(filter_sources)

        sql = f"""
            SELECT id, source_type, source_key, content, metadata, title, url, updated_at
            FROM chunks
            WHERE {" AND ".join(clauses)}
            ORDER BY updated_at DESC
            LIMIT ?
        """
        params.append(limit)
        rows = self._conn.execute(sql, tuple(params)).fetchall()
        return [
            StoredChunk(
                id=r["id"],
                source_type=r["source_type"],
                source_key=r["source_key"],
                content=r["content"],
                metadata=json.loads(r["metadata"]),
                title=r["title"],
                url=r["url"],
            )
            for r in rows
        ]

    def get_summary(self, source_type: str, source_key: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            """
            SELECT summary, doc_hash, model, generated_at, token_count
            FROM summaries WHERE source_type = ? AND source_key = ?
            """,
            (source_type, source_key),
        ).fetchone()
        if not row:
            return None
        return dict(row)

    def upsert_summary(
        self,
        source_type: str,
        source_key: str,
        summary: str,
        doc_hash: str,
        model: str,
        token_count: int | None = None,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO summaries (
              source_type, source_key, summary, doc_hash, model, generated_at, token_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_type, source_key) DO UPDATE SET
              summary = excluded.summary,
              doc_hash = excluded.doc_hash,
              model = excluded.model,
              generated_at = excluded.generated_at,
              token_count = excluded.token_count
            """,
            (
                source_type,
                source_key,
                summary,
                doc_hash,
                model,
                datetime.now(UTC).isoformat(),
                token_count,
            ),
        )
        self._conn.commit()

    def known_summary_hashes(self, source_type: str) -> dict[str, str]:
        rows = self._conn.execute(
            "SELECT source_key, doc_hash FROM summaries WHERE source_type = ?",
            (source_type,),
        ).fetchall()
        return {r["source_key"]: r["doc_hash"] for r in rows}

    def list_source_keys(self, source_type: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT source_key FROM chunks WHERE source_type = ? ORDER BY source_key",
            (source_type,),
        ).fetchall()
        return [r["source_key"] for r in rows]

    def stats(self) -> dict[str, Any]:
        total = self._conn.execute("SELECT COUNT(*) AS c FROM chunks").fetchone()["c"]
        by_source = {
            row["s"]: row["c"]
            for row in self._conn.execute(
                "SELECT source_type s, COUNT(*) c FROM chunks GROUP BY source_type"
            )
        }
        return {"total": total, "by_source": by_source}

    def close(self) -> None:
        """Close every connection opened across all threads."""
        import contextlib

        with self._conns_lock:
            for conn in self._all_conns:
                with contextlib.suppress(Exception):
                    conn.close()
            self._all_conns.clear()
