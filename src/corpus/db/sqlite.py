"""SQLite + sqlite-vec store. One file. UPSERT semantics. Orphan deletion.

SCHEMA HAZARD — do not `VACUUM` this DB without rebuilding the vec/fts
indices. The `chunks` table's auto-rowid is the foreign key into both
`chunks_vec` and `chunks_fts`; VACUUM can re-pack rowids and silently
break the linkage. Re-ingest from scratch is the safe recovery path.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import sqlite_vec

from corpus.types import Chunk
from corpus.util.fts_normalize import fts_terms, normalize_for_fts

logger = logging.getLogger(__name__)

FTS_VERSION = "2"


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
    # Per-DOC summary (from the `summaries` table), attached at query time by the
    # Retriever so the reranker can score against summary+content. NOT persisted on
    # the chunk row; None unless the retriever populated it.
    summary: str | None = None


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
        self._migrate_fts(init_conn)

    def _open_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        # The MCP server is this engine's primary consumer and can hold the
        # DB open indefinitely; without a busy_timeout, opening the store
        # from a second process (e.g. `corpus-ingest` while `corpus-mcp` is
        # running) raises `database is locked` immediately instead of
        # waiting briefly for the writer to finish its transaction.
        conn.execute("PRAGMA busy_timeout = 5000")
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

    def _migrate_fts(self, conn: sqlite3.Connection | None = None) -> None:
        """Rebuild chunks_fts when its normalization is stale.

        Runs automatically at open rather than as a CLI command: a single-user
        DB whose migration must be remembered is a migration that gets skipped,
        leaving a half-normalized index with nothing surfaced. Local and free —
        content is re-read from `chunks`, so no embeddings are recomputed.
        """
        conn = conn or self._conn
        row = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'fts_version'"
        ).fetchone()
        if row is not None and row["value"] == FTS_VERSION:
            return
        conn.execute("DELETE FROM chunks_fts")
        # Stream the cursor rather than `.fetchall()`: materializing every row
        # up front measured 2.90s / ~146MB RSS at 72,158 chunks (~8s at 205k).
        # Iterating the cursor pulls rows incrementally instead of holding the
        # whole table in Python memory at once. Writing to chunks_fts /
        # schema_meta (different tables) while this SELECT cursor on `chunks`
        # is still open is safe on a single connection.
        count = 0
        for r in conn.execute("SELECT rowid, content FROM chunks"):
            # Explicit rowid: the chunks_fts <-> chunks join depends on it.
            conn.execute(
                "INSERT INTO chunks_fts(rowid, content) VALUES (?, ?)",
                (r["rowid"], normalize_for_fts(r["content"])),
            )
            count += 1
        # The fts_version stamp is written only AFTER every insert above,
        # inside this same transaction (single commit below) -- so a
        # migration that crashes mid-stream leaves no stamp and retries
        # cleanly from scratch on next open. Do not reorder this earlier.
        conn.execute(
            "INSERT INTO schema_meta (key, value) VALUES ('fts_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (FTS_VERSION,),
        )
        conn.commit()
        if count:
            logger.info("rebuilt FTS index for %d chunks (fts_version=%s)", count, FTS_VERSION)

    @contextmanager
    def _txn(self) -> Iterator[sqlite3.Connection]:
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
        self._conn.execute(
            "INSERT INTO chunks_fts(rowid, content) VALUES (?, ?)",
            (rowid, normalize_for_fts(chunk.content)),
        )
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
        # Delete chunks by rowid in the loop (same as vec/fts) rather than a single
        # `WHERE id IN (?,?,...)` — a one-shot IN clause blows SQLite's variable cap
        # (~32k) when a re-ingest produces hundreds of thousands of orphans (e.g. a
        # full re-chunk). Per-rowid deletes have no such limit.
        with self._txn():
            for _id, rowid in orphans:
                self._conn.execute("DELETE FROM chunks_vec WHERE rowid = ?", (rowid,))
                self._conn.execute("DELETE FROM chunks_fts WHERE rowid = ?", (rowid,))
                self._conn.execute("DELETE FROM chunks WHERE rowid = ?", (rowid,))
        return len(orphans)

    def source_types(self) -> list[str]:
        """Distinct source types present in the store. Backed by idx_chunks_source_type."""
        rows = self._conn.execute(
            "SELECT DISTINCT source_type FROM chunks ORDER BY source_type"
        ).fetchall()
        return [row["source_type"] for row in rows]

    def fts_search(
        self,
        query: str,
        top_k: int,
        filter_sources: Sequence[str] | None = None,
    ) -> list[StoredChunk]:
        match_terms = fts_terms(query)
        if not match_terms:
            return []
        match_expr = " OR ".join(match_terms)

        # INTENTIONALLY a post-hoc Python filter, NOT a rowid-IN pre-filter
        # like vector_search uses. A rowid-IN pre-filter was tried here and
        # measured: it defeats FTS5's rank-ordered LIMIT short-circuit, so
        # SQLite falls back to `USE TEMP B-TREE FOR ORDER BY` -- a full
        # MATCH + sort of every row that matches the query terms, once per
        # source type in the retriever's per-source loop. Measured ~1561x
        # slower than this global-query form at 40k chunks / 4 source types
        # (mean 7.3s/query, p99 17.5s) -- fatal for sub-300ms search. This is
        # safe unlike vector_search's starvation risk: BM25 only matches
        # chunks that actually contain the query's terms, so a small source
        # that genuinely matches is far likelier to survive a fixed-size
        # over-fetch window than to be crowded out by sheer volume the way
        # every chunk (via distance) competes in vector search. See
        # Retriever.query for the per-source vector / global-FTS split.
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
        except sqlite3.OperationalError as e:
            logger.warning("FTS query failed for %r: %s", match_expr, e)
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
        # filter_sources is a genuine PRE-filter on the ANN query (a rowid
        # subquery constraining the k-nearest search itself), not a post-hoc
        # filter over a fixed global over-fetch window. A post-filter cannot
        # be made reliable by over-fetching more: at extreme skew (a source
        # with a handful of chunks buried among hundreds of thousands from
        # another) the required over-fetch multiplier is unbounded. The
        # pre-filter keeps k == top_k because k now applies to the
        # already-constrained candidate set.
        if filter_sources:
            placeholders = ",".join("?" for _ in filter_sources)
            rows = self._conn.execute(
                f"""
                SELECT c.id, c.source_type, c.source_key, c.content, c.metadata,
                       c.title, c.url, v.distance
                FROM chunks_vec v
                JOIN chunks c ON c.rowid = v.rowid
                WHERE v.embedding MATCH ? AND k = ?
                  AND v.rowid IN (SELECT rowid FROM chunks WHERE source_type IN ({placeholders}))
                ORDER BY v.distance
                """,
                (blob, top_k, *filter_sources),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT c.id, c.source_type, c.source_key, c.content, c.metadata,
                       c.title, c.url, v.distance
                FROM chunks_vec v
                JOIN chunks c ON c.rowid = v.rowid
                WHERE v.embedding MATCH ? AND k = ?
                ORDER BY v.distance
                """,
                (blob, top_k),
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
