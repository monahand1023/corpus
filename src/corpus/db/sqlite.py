"""SQLite + sqlite-vec store. One file. UPSERT semantics. Orphan deletion.

SCHEMA HAZARD — do not `VACUUM` this DB without rebuilding the vec/fts
indices. The `chunks` table's auto-rowid is the foreign key into both
`chunks_vec` and `chunks_fts`; VACUUM can re-pack rowids and silently
break the linkage. Re-ingest from scratch is the safe recovery path.

KNOWN DEBT: that warning is the whole defence, and a warning cannot stop a
database browser, a backup tool, or a future maintenance script from running
VACUUM. The structural fix is an explicit `INTEGER PRIMARY KEY` column on
`chunks` — SQLite guarantees such a column is a stable alias for the rowid
and never renumbers it, VACUUM included. Not done here because it changes
the schema of every existing store and needs a migration that rewrites
`chunks_vec` and `chunks_fts` alongside it; worth doing before this engine
is used anywhere the database might be touched by tooling nobody controls.
"""

from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
import tempfile
import threading
import tomllib
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

# Bump whenever `normalize_for_fts` changes: a stored index built by an older
# normalization no longer agrees with the query path, and only a rebuild
# reconciles them.
#   "2" -> CJK runs rewritten as overlapping bigrams.
#   "3" -> those runs separated from adjacent Latin/digits, which `unicode61`
#          would otherwise fuse into one token (`Public会議室`).
FTS_VERSION = "3"

# Above this many chunks, an FTS rebuild stops being something to do silently
# while opening a store. Measured: a 70k-chunk store rebuilt in ~3s, so a store of
# this size is a few seconds; a multi-million-chunk store is minutes of held writer lock, with
# any concurrent ingest failing on its busy timeout meanwhile.
AUTO_FTS_MIGRATION_MAX_CHUNKS = 200_000

# sqlite-vec's vec0 refuses a KNN query with k above this, raising
# `OperationalError: k value in knn query too large`. Nothing in this engine
# bounded it, so a caller asking for a large enough top_k crashed the whole
# retrieval instead of getting the most it could give — reachable from any
# code path that widens a candidate pool.
VEC0_MAX_K = 4096

# Ceiling on how far `fts_search` will widen its window trying to satisfy a
# source filter. Bounds the worst case — a filter matching a source that has
# no rows for the query at all would otherwise widen until it had scanned
# every matching row.
_FTS_MAX_OVER_FETCH = 5_000

# System temp root, resolved once. Symlinks matter here: on macOS
# `tempfile.gettempdir()` returns a `/var/folders/...` path that is itself a
# symlink into `/private/var/folders/...`, and pytest's `tmp_path` fixture
# resolves through that link -- comparing unresolved paths would silently
# fail to match every `tmp_path`-based test fixture and reintroduce the
# warning-flood this exclusion exists to prevent.
_TEMP_ROOT = Path(tempfile.gettempdir()).resolve()

# The name this project's own pyproject.toml declares -- used to confirm a
# "src" parent above the installed package really is *this* checkout before
# extending the in-tree-index warning to the whole repo root (see
# `_corpus_engine_roots`), not some unrelated project that happens to use a
# src/ layout.
_PROJECT_NAME = "corpus-rag"


def _corpus_engine_roots() -> list[Path]:
    """Directories where a consumer's index must never live: the installed
    `corpus` package's own directory, and -- only when running from a source
    checkout -- that checkout's repo root too.

    Both are derived from *this module's* `__file__`, i.e. wherever Python's
    import machinery actually resolved the `corpus` package via `sys.path`.
    That is deliberate: a check based on `Path.cwd()` would misbehave for
    `uv run` (or any invocation) launched from a directory other than the
    checkout -- cwd reveals nothing about where corpus is installed, so it
    could both miss a real in-tree index and, separately, misfire against an
    unrelated cwd. `__file__`-based resolution has neither failure mode: it
    names where the package's bytes physically live, and that is invariant
    across every invocation style -- a normal (non-editable) install, an
    editable install, or pytest's `pythonpath` sys.path injection (which
    resolves `__file__` the same way an editable install does).

    A normal site-packages install's package directory has no meaningful
    "repo root" above it -- just more of the venv -- so it gets no
    escalation: only `package_root` itself is protected. An editable
    install / dev checkout has the shape ".../<repo>/src/corpus"; only then,
    and only after confirming `<repo>/pyproject.toml` actually declares this
    project, is `<repo>` added too. That escalation matters: the real
    incident this guards against put files at the *repo root*
    (next to pyproject.toml), not inside src/corpus/ itself.
    """
    package_root = Path(__file__).resolve().parent.parent
    roots = [package_root]
    if package_root.parent.name != "src":
        return roots
    candidate_root = package_root.parent.parent
    pyproject = candidate_root / "pyproject.toml"
    if not pyproject.is_file():
        return roots
    try:
        project_name = tomllib.loads(pyproject.read_text()).get("project", {}).get("name")
    except (OSError, tomllib.TOMLDecodeError):
        project_name = None
    if project_name == _PROJECT_NAME:
        roots.append(candidate_root)
    return roots


def _warn_if_inside_corpus_repo(db_path: Path) -> None:
    """Emit a loud (but non-fatal) warning if `db_path` resolves inside
    corpus's own package directory or checkout root.

    That location is never correct for a consumer's index: corpus is a
    library, and an index built from someone's personal documents belongs in
    a *consumer* project's own directory (with its own config), not inside
    the engine that merely builds and serves it. The danger isn't
    hypothetical -- an untracked file living there survives only as long as
    `.gitignore` happens to stay correct, and is exactly what `git clean
    -fdx` deletes and a stray `git add -f` would publish.

    This is deliberately a warning, not a hard error: an existing
    installation with a database already in this location must not be
    locked out of its own data by an upgrade. Skipped entirely for
    `:memory:` and for anything under the system temp directory (where the
    whole test suite's `tmp_path`-based fixtures live) so routine test runs
    never see it.
    """
    if str(db_path) == ":memory:":
        return
    try:
        resolved = db_path.resolve()
    except OSError:
        return
    if resolved.is_relative_to(_TEMP_ROOT):
        return
    for root in _corpus_engine_roots():
        if resolved.is_relative_to(root):
            logger.warning(
                "%s resolves inside corpus's own source tree (%s). Storing a "
                "consumer's index there is always a mistake: corpus is a "
                "library, and personal data belongs in a separate consumer "
                "project's own directory, with its own config, not inside "
                "corpus itself. This is a warning, not an error -- an "
                "existing database here still opens -- but move it out "
                "before it grows.",
                resolved,
                root,
            )
            return

# Defaults for the orphan-pruning blast-radius guard (see OrphanPruneRefused).
# Mirrored in corpus.config.PruningConfig, which is the normal way a caller
# overrides them via corpus.toml; these are the fallback when no config is
# threaded through (e.g. a caller using ChunkStore directly).
DEFAULT_MAX_ORPHAN_RATIO = 0.20
DEFAULT_MIN_CHUNKS_FOR_GUARD = 50

# Defaults for the SQLite memory-tuning pragmas (see corpus.config.
# PerformanceConfig for the full rationale and the benchmark these come
# from -- mirrored here as the fallback for a caller using ChunkStore
# directly, same pattern as the orphan-guard defaults above).
DEFAULT_CACHE_SIZE_MB = 64
DEFAULT_MMAP_SIZE_MB = 1024
DEFAULT_TEMP_STORE_MEMORY = True


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
    # Contextual-Retrieval blurb, persisted in the chunk's own `context`
    # column. Unlike `summary` this IS stored, because the chunk's embedding
    # and FTS row are computed over `context + content` — see
    # `corpus.contextual`.
    context: str | None = None


class EmbeddingDimMismatch(RuntimeError):
    """Raised when the requested embedding dim doesn't match the existing
    schema. Switching dims after data has been ingested would silently
    corrupt retrieval."""


class ReadOnlyStoreError(RuntimeError):
    """Raised by any mutating `ChunkStore` method when the store was opened
    with `read_only=True`. A read-only store's connection is opened via a
    `mode=ro` SQLite URI, so a write would eventually fail anyway — this
    check exists to fail immediately with a message that names the actual
    cause, rather than surfacing SQLite's generic 'attempt to write a
    readonly database' from wherever the query happens to be."""


class OrphanPruneRefused(RuntimeError):
    """Raised by `delete_orphans` when pruning would delete more than
    `max_orphan_ratio` of a source's existing chunks (and that source has at
    least `min_chunks_for_guard` chunks to begin with).

    This is the blast-radius guard for a connector that honestly reports
    zero failures while silently under-yielding documents — e.g. one that
    skips files whose fingerprint looks unchanged and, on a later run,
    yields nothing for the source at all. `failed_files` can't catch this:
    the connector isn't reporting any failure. Only the *shape* of the
    result — almost everything for this source vanished — is evidence
    something is wrong, which is exactly what this guard checks.

    Pass `force=True` (wired to `--prune-anyway` in the ingest CLI) when the
    drop is a genuine bulk deletion rather than a connector bug."""

    def __init__(
        self, source_type: str, existing: int, orphans: int, ratio: float, max_orphan_ratio: float
    ):
        self.source_type = source_type
        self.existing = existing
        self.orphans = orphans
        self.ratio = ratio
        self.max_orphan_ratio = max_orphan_ratio
        super().__init__(
            f"refusing to prune '{source_type}': {orphans}/{existing} existing chunks "
            f"({ratio:.0%}) would be deleted as orphans, exceeding the {max_orphan_ratio:.0%} "
            "guard. This usually means a connector under-yielded documents while "
            "reporting no failures, not a genuine bulk deletion. Re-run with "
            "--prune-anyway if it is."
        )


class ChunkStore:
    def __init__(
        self,
        db_path: Path | str,
        *,
        embedding_dim: int,
        read_only: bool = False,
        allow_expensive_migration: bool = False,
        cache_size_mb: int = DEFAULT_CACHE_SIZE_MB,
        mmap_size_mb: int = DEFAULT_MMAP_SIZE_MB,
        temp_store_memory: bool = DEFAULT_TEMP_STORE_MEMORY,
    ):
        """Open or create the chunk store.

        `embedding_dim` is checked against the existing schema if the DB
        already has data — if they mismatch (e.g., user changed embedder
        model without re-ingesting), we raise rather than silently corrupt.

        `read_only=True` opens the DB through a `mode=ro` SQLite URI and never
        runs `_migrate_fts` — for a query-only caller (the MCP server, the
        query CLI), this guarantees opening the store to search it can never
        rewrite it. A schema migration is otherwise a side effect of opening
        ANY store, including one you only meant to inspect: on a large store
        it can take real time and rewrite a large fraction of the file, and
        running it against, say, a backup you're comparing before/after would
        destroy the very state being compared. See `_migrate_fts` for the
        migration itself and its WARNING logging. Any method that would
        mutate raises `ReadOnlyStoreError` on a read-only store. Requires the
        file to already exist — read-only mode never creates a store.

        `cache_size_mb` / `mmap_size_mb` / `temp_store_memory` are the
        memory-tuning pragmas from `corpus.config.PerformanceConfig` (see
        there for the benchmark behind the defaults) — applied to every
        connection this store opens, read-only or not; mmap composes fine
        with read-only mode; it was in fact benchmarked exclusively through
        read-only `mode=ro` connections, matching how the MCP server and
        query CLI actually use it.

        Threading: each thread gets its own sqlite3.Connection via thread-local
        storage. WAL mode handles concurrent readers; ingest (single-threaded)
        is the only writer path. The MCP server uses `asyncio.to_thread` so
        each tool invocation lands on its own worker thread and gets its own
        connection — no shared cursor state.
        """
        self._db_path = Path(db_path)
        _warn_if_inside_corpus_repo(self._db_path)
        self._embedding_dim = embedding_dim
        self._read_only = read_only
        self._closed = False
        # Set only by the explicit `corpus-migrate-fts` maintenance command.
        # A rebuild is one long transaction holding the sole writer lock, so
        # it must be something an operator chose, not something a constructor
        # did on their behalf. See _migrate_fts.
        self._allow_expensive_migration = allow_expensive_migration
        self._cache_size_mb = cache_size_mb
        self._mmap_size_mb = mmap_size_mb
        self._temp_store_memory = temp_store_memory
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
        try:
            if read_only:
                self._guard_embedding_dim(init_conn)
            else:
                self._init_schema(init_conn)
                self._guard_embedding_dim(init_conn)
                self._migrate_fts(init_conn)
        except BaseException:
            # Setup failed, so no caller holds this store and nobody will ever
            # call close() on it. Without this the connection survives until
            # garbage collection with an open write transaction, holding
            # SQLite's single writer lock — every later writer then fails on
            # its busy timeout, and in a long-running process (an MCP server)
            # that lock is held for the life of the process. Found by a test
            # that crashed an FTS rebuild mid-way and then could not reopen
            # the database at all.
            #
            # Rollback before close so the partial transaction is discarded
            # explicitly rather than relying on close()'s implicit behaviour.
            with contextlib.suppress(Exception):
                init_conn.rollback()
            with contextlib.suppress(Exception):
                init_conn.close()
            with self._conns_lock, contextlib.suppress(ValueError):
                self._all_conns.remove(init_conn)
            self._tls.conn = None
            raise

    def _require_writable(self, action: str) -> None:
        if self._read_only:
            raise ReadOnlyStoreError(
                f"{self._db_path} was opened with read_only=True; {action} is not permitted."
            )

    def _open_connection(self) -> sqlite3.Connection:
        if self._read_only:
            if not self._db_path.exists():
                raise FileNotFoundError(
                    f"cannot open {self._db_path} read-only: file does not exist "
                    "(read-only mode never creates a store)"
                )
            # mode=ro: SQLite refuses any write against this connection at the
            # OS/file level, as a second line of defense behind
            # _require_writable's explicit checks.
            conn = sqlite3.connect(
                f"file:{self._db_path.as_posix()}?mode=ro", uri=True, check_same_thread=False
            )
        else:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        if not self._read_only:
            # journal_mode/synchronous govern write behavior; skip them on a
            # read-only connection, which cannot legally change either.
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
        # The MCP server is this engine's primary consumer and can hold the
        # DB open indefinitely; without a busy_timeout, opening the store
        # from a second process (e.g. `corpus-ingest` while `corpus-mcp` is
        # running) raises `database is locked` immediately instead of
        # waiting briefly for the writer to finish its transaction.
        conn.execute("PRAGMA busy_timeout = 5000")
        # Memory tuning -- see corpus.config.PerformanceConfig for the
        # benchmark behind these three. Short version: mmap_size is the
        # pragma that actually matters for vector search (measured
        # 3.18x-3.29x once its window covers the whole store); cache_size
        # and temp_store measured no effect on that workload but cost
        # little and are kept as a safety net for other access patterns.
        # Applied on every connection, read-only or not.
        conn.execute(f"PRAGMA cache_size = -{self._cache_size_mb * 1024}")
        conn.execute(f"PRAGMA mmap_size = {self._mmap_size_mb * 1024 * 1024}")
        if self._temp_store_memory:
            conn.execute("PRAGMA temp_store = MEMORY")
        with self._conns_lock:
            self._all_conns.append(conn)
        return conn

    @property
    def _conn(self) -> sqlite3.Connection:
        """Per-thread connection. Lazy-created on first access from each thread.

        Raises after `close()`. Without this check, a thread that had already
        opened a connection keeps a thread-local reference to the closed
        object — `close()` can shut every connection down but cannot reach
        into another thread's `threading.local()` to clear it — so the next
        query there fails with sqlite3's "Cannot operate on a closed
        database", from a call site that has nothing to do with closing.

        Deliberately NOT reopened silently: using a store after closing it is
        a caller bug, and quietly resurrecting one hides it. The error says
        which mistake was made.
        """
        if self._closed:
            raise ReadOnlyStoreError(
                "this ChunkStore has been closed; open a new one rather than "
                "reusing it (close() shuts down every thread's connection)"
            )
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

            -- One row per source, rewritten at the end of every successful
            -- ingest. Exists so the NEXT run can notice that a connector
            -- yielded far fewer documents than it used to — a collapse that
            -- is invisible when pruning is suppressed, because nothing is
            -- deleted and the run looks entirely normal.
            CREATE TABLE IF NOT EXISTS source_yield (
              source_type TEXT PRIMARY KEY,
              documents INTEGER NOT NULL,
              chunks INTEGER NOT NULL,
              recorded_at TEXT NOT NULL,
              -- How many inputs the connector permanently gave up on. A RISE
              -- means files it used to read are now classified unreadable —
              -- a parser regression, a permission change, a new "unsupported"
              -- rule — and since a skip does not suppress pruning, those
              -- files' indexed chunks are deleted as if they had vanished.
              skipped INTEGER NOT NULL DEFAULT 0,
              -- The resolved path this source last read from. Two different
              -- folders can normalize to the same source name, and when they
              -- do the second ingest OVERWRITES the first's chunks rather
              -- than pruning them, so nothing else in the system notices.
              source_path TEXT
            );

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
        self._migrate_context_column(conn)
        conn.commit()

    def _migrate_context_column(self, conn: sqlite3.Connection | None = None) -> None:
        """Add the `context` column to an existing store, in place.

        Contextual Retrieval keeps its blurb in its own column rather than
        rewriting `content`, so the canonical text is never mutated and a bad
        contextualization run is undone by clearing one column instead of
        re-ingesting. Added by ALTER rather than in the CREATE above so a
        database written before this feature migrates on first open.

        Silently does nothing on a read-only connection: a query-only caller
        (the MCP server) has no business writing schema, and a store old
        enough to lack the column simply has no contexts to read.
        """
        conn = conn or self._conn
        if self._read_only:
            return
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(chunks)")}
        if "context" not in cols:
            conn.execute("ALTER TABLE chunks ADD COLUMN context TEXT")
        yield_cols = {r["name"] for r in conn.execute("PRAGMA table_info(source_yield)")}
        if yield_cols and "source_path" not in yield_cols:
            conn.execute("ALTER TABLE source_yield ADD COLUMN source_path TEXT")
        if yield_cols and "skipped" not in yield_cols:
            conn.execute("ALTER TABLE source_yield ADD COLUMN skipped INTEGER NOT NULL DEFAULT 0")

    def _guard_embedding_dim(self, conn: sqlite3.Connection | None = None) -> None:
        """If the DB has previous data, the embedding dim must match."""
        conn = conn or self._conn
        row = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'embedding_dim'"
        ).fetchone()
        if row is None:
            if self._read_only:
                # Nothing recorded yet (an empty store) and a read-only
                # connection can't record one either. Nothing to compare
                # against, so there's nothing to guard.
                return
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

    def fts_version(self) -> str | None:
        """The FTS normalization version this store is stamped with, or None.

        None means the store predates the stamp entirely and its index is
        whatever the code of the day wrote.
        """
        row = self._conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'fts_version'"
        ).fetchone()
        return row["value"] if row else None

    def _migrate_fts(self, conn: sqlite3.Connection | None = None) -> None:
        """Rebuild chunks_fts when its normalization is stale — if it is cheap.

        A SMALL store migrates automatically at open. That was the original
        reasoning and it still holds at small scale: a single-user database
        whose migration must be remembered is a migration that gets skipped,
        leaving a half-normalized index with nothing surfaced. It is local and
        free — content is re-read from `chunks`, nothing is re-embedded.

        A LARGE store does not. Above `AUTO_FTS_MIGRATION_MAX_CHUNKS` the
        method logs loudly and returns, leaving the index stale until someone
        runs `corpus-migrate-fts` deliberately. Three reasons, all of which
        only bite at scale:

          - The whole rebuild is ONE transaction, so it holds SQLite's single
            writer lock from first delete to final commit. On a large
            store that is many minutes during which any other writer — an
            ingest, a contextualization run — fails on its busy timeout.
          - A library constructor is not a maintenance command. Any script,
            test, admin tool, or MCP server that happens to open the store
            read-write would trigger it.
          - It rewrites the database file, and in WAL mode the writes
            accumulate before checkpoint, so required free space is not
            bounded by the file's current size.

        Being one transaction is what makes an interruption safe: a killed
        process rolls back to the old index and the old version stamp, and the
        next attempt starts clean. There is no durable half-migrated state.

        Skipped entirely on a read-only connection, which is how a store
        should be opened for inspection.
        """
        conn = conn or self._conn
        row = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'fts_version'"
        ).fetchone()
        if row is not None and row["value"] == FTS_VERSION:
            return
        from_version = row["value"] if row is not None else None
        pending = conn.execute("SELECT COUNT(*) AS c FROM chunks").fetchone()["c"]
        if pending > AUTO_FTS_MIGRATION_MAX_CHUNKS and not self._allow_expensive_migration:
            logger.warning(
                "%s: full-text index is STALE (fts_version %r, expected %r) and "
                "has %d chunks — too many to rebuild automatically while opening "
                "the store. Search still works, but text this normalization "
                "handles (notably CJK, which is indexed as overlapping bigrams) "
                "will not match. Run `corpus-migrate-fts` when no other process "
                "is writing to this database. Deploy the current code EVERYWHERE "
                "first: an older writer appending to a freshly migrated index "
                "writes unnormalized rows into a store stamped as current, which "
                "reintroduces the problem silently for new content.",
                self._db_path,
                from_version,
                FTS_VERSION,
                pending,
            )
            return
        # A brand-new, still-empty store also has no fts_version stamp yet —
        # that is ordinary first-time schema setup, not a migration of
        # existing data, so it does not warrant a WARNING. Only chunks that
        # already exist are actually at risk of the silent-rewrite failure
        # mode this logging exists for.
        existing_chunks = conn.execute("SELECT COUNT(*) AS c FROM chunks").fetchone()["c"]
        if existing_chunks:
            logger.warning(
                "migrating FTS index for %s: fts_version %r -> %r, about to rebuild "
                "%d existing chunk(s). This rewrites the database file on disk. Open "
                "with ChunkStore(path, read_only=True) if you need to inspect a "
                "store's current on-disk state unmodified.",
                self._db_path,
                from_version,
                FTS_VERSION,
                existing_chunks,
            )
        conn.execute("DELETE FROM chunks_fts")
        # Stream the cursor rather than `.fetchall()`: materializing every row
        # up front measured 2.90s / ~146MB RSS at ~70k chunks (~8s at 205k).
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
            logger.warning(
                "migrated FTS index for %s: rebuilt %d chunk(s), fts_version now %r",
                self._db_path,
                count,
                FTS_VERSION,
            )

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
        self._require_writable("upsert")
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
        self._require_writable("upsert_batch")
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
        self._require_writable("delete_by_source")
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

    def delete_orphans(
        self,
        source_type: str,
        seen_ids: set[str],
        *,
        max_orphan_ratio: float = DEFAULT_MAX_ORPHAN_RATIO,
        min_chunks_for_guard: int = DEFAULT_MIN_CHUNKS_FOR_GUARD,
        force: bool = False,
    ) -> int:
        """Delete every chunk of `source_type` whose id is absent from
        `seen_ids`.

        Blast-radius guard: refuses (raising `OrphanPruneRefused`, deleting
        nothing) when the chunks about to be pruned exceed `max_orphan_ratio`
        of the source's existing chunk count, unless that source has fewer
        than `min_chunks_for_guard` chunks to begin with (too small a blast
        radius to matter -- see `OrphanPruneRefused` and
        `corpus.config.PruningConfig` for the full rationale). Pass
        `force=True` to delete anyway, e.g. for a deliberate bulk deletion.
        """
        self._require_writable("delete_orphans")
        existing = self._conn.execute(
            "SELECT id, rowid FROM chunks WHERE source_type = ?", (source_type,)
        ).fetchall()
        orphans = [(row["id"], row["rowid"]) for row in existing if row["id"] not in seen_ids]
        if not orphans:
            return 0

        if not force and len(existing) > min_chunks_for_guard:
            ratio = len(orphans) / len(existing)
            # `>` and not `>=`, deliberately. A review flagged that a
            # deletion of exactly max_orphan_ratio is permitted, which is
            # true. `>=` was tried and reverted: `max_orphan_ratio = 1.0` is
            # a legal config value documented as "never refuse", and `>=`
            # inverts it into "refuse every complete deletion" — trading a
            # marginal gain against exactly 20.000% for a footgun on a
            # setting someone chose on purpose.
            if ratio > max_orphan_ratio:
                logger.error(
                    "refusing to prune '%s': %d/%d existing chunks (%.0f%%) would be "
                    "deleted, exceeding the %.0f%% guard (enforced above %d chunks). "
                    "Re-run with --prune-anyway if this is a genuine bulk deletion.",
                    source_type,
                    len(orphans),
                    len(existing),
                    ratio * 100,
                    max_orphan_ratio * 100,
                    min_chunks_for_guard,
                )
                raise OrphanPruneRefused(
                    source_type, len(existing), len(orphans), ratio, max_orphan_ratio
                )

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
        # That reasoning stands, and the fast path below is still the one
        # that runs for virtually every query. The claim that once sat here
        # about the STARVATION risk did not stand: it said a source that
        # genuinely matches is "far likelier to survive a fixed-size
        # over-fetch window" than to be crowded out. Measured, it is not —
        # 100 chunks of one source and 20 of another, all containing the
        # query term, filtered to the smaller source returned ZERO results,
        # because the first 15 by rank were all the larger source.
        #
        # So: fast path first, and fall back to the correct-but-slower
        # pre-filter ONLY when the fast path came up short. An uncrowded
        # filter never pays for the fallback; a crowded one gets a right
        # answer slowly instead of an empty one quickly. Widening the window
        # instead was tried and rejected — it cannot fix a source whose rows
        # all sort below any bounded window (a source inserted last ties on
        # rank and sorts by rowid), so it bought latency and still returned
        # nothing.
        if not filter_sources:
            return self._fts_rows(match_expr, top_k, None, top_k)

        filter_set = set(filter_sources)
        fast = self._fts_rows(match_expr, top_k * 3, filter_set, top_k)
        if len(fast) >= top_k:
            return fast
        return self._fts_rows_prefiltered(match_expr, top_k, filter_sources)

    def _fts_rows_prefiltered(
        self, match_expr: str, top_k: int, filter_sources: Sequence[str]
    ) -> list[StoredChunk]:
        """Restrict to the named sources IN SQL, so a crowded filter still
        returns rows. Slower — this is the form measured ~1561x slower than
        the global query at 40k chunks, because it defeats FTS5's rank-ordered
        LIMIT short-circuit and falls back to sorting every matching row — so
        it runs only after the fast path has already come up short."""
        placeholders = ",".join("?" for _ in filter_sources)
        try:
            rows = self._conn.execute(
                f"""
                SELECT c.id, c.source_type, c.source_key, c.content, c.metadata,
                       c.title, c.url, c.context, f.rank
                FROM chunks_fts f
                JOIN chunks c ON c.rowid = f.rowid
                WHERE f.content MATCH ?
                  AND c.source_type IN ({placeholders})
                ORDER BY f.rank
                LIMIT ?
                """,
                (match_expr, *filter_sources, top_k),
            ).fetchall()
        except sqlite3.OperationalError as e:
            logger.warning("FTS pre-filtered query failed for %r: %s", match_expr, e)
            return []
        return [
            StoredChunk(
                id=row["id"],
                source_type=row["source_type"],
                source_key=row["source_key"],
                content=row["content"],
                metadata=json.loads(row["metadata"]),
                title=row["title"],
                url=row["url"],
                context=row["context"],
                distance=row["rank"],
            )
            for row in rows
        ]

    def _fts_rows(
        self,
        match_expr: str,
        over_fetch: int,
        filter_set: set[str] | None,
        top_k: int,
    ) -> list[StoredChunk]:
        try:
            rows = self._conn.execute(
                """
                SELECT c.id, c.source_type, c.source_key, c.content, c.metadata,
                       c.title, c.url, c.context, f.rank
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
                    context=row["context"],
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
                       c.title, c.url, c.context, v.distance
                FROM chunks_vec v
                JOIN chunks c ON c.rowid = v.rowid
                WHERE v.embedding MATCH ? AND k = ?
                  AND v.rowid IN (SELECT rowid FROM chunks WHERE source_type IN ({placeholders}))
                ORDER BY v.distance
                """,
                (blob, min(top_k, VEC0_MAX_K), *filter_sources),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT c.id, c.source_type, c.source_key, c.content, c.metadata,
                       c.title, c.url, c.context, v.distance
                FROM chunks_vec v
                JOIN chunks c ON c.rowid = v.rowid
                WHERE v.embedding MATCH ? AND k = ?
                ORDER BY v.distance
                """,
                (blob, min(top_k, VEC0_MAX_K)),
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
                    context=row["context"],
                    distance=row["distance"],
                )
            )
            if len(results) >= top_k:
                break
        return results

    def get_by_id(self, chunk_id: str) -> StoredChunk | None:
        row = self._conn.execute(
            "SELECT id, source_type, source_key, content, metadata, title, url, context "
            "FROM chunks WHERE id = ?",
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
            context=row["context"],
        )

    def set_context(
        self, chunk_id: str, context: str, embedding: Sequence[float]
    ) -> bool:
        """Attach a Contextual-Retrieval blurb to a chunk and re-index it.

        `context` goes in its own column; `content` is never touched, so a bad
        run is reverted by clearing one column rather than re-ingesting. The
        vector and FTS rows are then replaced with ones computed over
        `context + "\n\n" + content` — that replacement is the whole point,
        since a context nobody searches against buys nothing.

        Idempotent: re-calling overwrites. Returns False for an unknown id
        rather than raising, so a stale batch result (a chunk deleted between
        submit and apply) is a counted no-op instead of a crashed run.
        """
        if len(embedding) != self._embedding_dim:
            raise ValueError(
                f"embedding dim mismatch: got {len(embedding)}, "
                f"expected {self._embedding_dim}"
            )
        row = self._conn.execute(
            "SELECT rowid, content FROM chunks WHERE id = ?", (chunk_id,)
        ).fetchone()
        if row is None:
            return False
        rowid = row["rowid"]
        combined = f"{context}\n\n{row['content']}"
        with self._txn() as conn:
            conn.execute(
                "UPDATE chunks SET context = ? WHERE id = ?", (context, chunk_id)
            )
            blob = sqlite_vec.serialize_float32(list(embedding))
            conn.execute("DELETE FROM chunks_vec WHERE rowid = ?", (rowid,))
            conn.execute(
                "INSERT INTO chunks_vec(rowid, embedding) VALUES (?, ?)", (rowid, blob)
            )
            conn.execute("DELETE FROM chunks_fts WHERE rowid = ?", (rowid,))
            conn.execute(
                "INSERT INTO chunks_fts(rowid, content) VALUES (?, ?)",
                # normalize_for_fts, exactly as `upsert` does. The query path
                # searches for the normalized form (CJK runs become overlapping
                # bigrams via `fts_terms`), so a row written raw here is
                # unreachable by any CJK query — the chunk would be silently
                # dropped out of BM25 the moment it gained a context.
                (rowid, normalize_for_fts(combined)),
            )
        return True

    def clear_context(self, source_type: str) -> int:
        """Drop every context for a source and restore its plain-text FTS rows.

        The escape hatch for a contextualization run that produced bad blurbs.
        Note what this does NOT do: it cannot restore the pre-context
        embeddings, because those were overwritten in place. Vectors stay as
        they are until the source is re-ingested or re-contextualized — which
        is why this prints a count rather than claiming a clean revert.
        """
        rows = self._conn.execute(
            "SELECT rowid, content FROM chunks WHERE source_type = ? AND context IS NOT NULL",
            (source_type,),
        ).fetchall()
        with self._txn() as conn:
            for r in rows:
                conn.execute("DELETE FROM chunks_fts WHERE rowid = ?", (r["rowid"],))
                conn.execute(
                    "INSERT INTO chunks_fts(rowid, content) VALUES (?, ?)",
                    # Same normalization as `upsert` and `set_context`: this
                    # restores the pre-context row, and a raw one would be
                    # just as unreachable as the bug this mirrors.
                    (r["rowid"], normalize_for_fts(r["content"])),
                )
            conn.execute(
                "UPDATE chunks SET context = NULL WHERE source_type = ?", (source_type,)
            )
        return len(rows)

    def chunks_missing_context(
        self, source_type: str, limit: int | None = None
    ) -> list[StoredChunk]:
        """Chunks of this source with no context yet.

        Ordered by (source_key, chunk_index) so every chunk of a document
        arrives together. That adjacency is what lets a batch send the parent
        document once as a cached prompt prefix instead of once per chunk —
        the difference between paying full input price per chunk and paying
        roughly a tenth of it.
        """
        sql = (
            "SELECT id, source_type, source_key, content, metadata, title, url, context "
            "FROM chunks WHERE source_type = ? AND context IS NULL "
            "ORDER BY source_key, chunk_index"
        )
        params: list[Any] = [source_type]
        if limit is not None:
            sql += " LIMIT ?"
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
                context=r["context"],
            )
            for r in rows
        ]

    def context_coverage(self) -> dict[str, dict[str, int]]:
        """Per-source total and contextualized counts."""
        rows = self._conn.execute(
            "SELECT source_type, COUNT(*) AS total, "
            "SUM(CASE WHEN context IS NOT NULL THEN 1 ELSE 0 END) AS with_context "
            "FROM chunks GROUP BY source_type"
        ).fetchall()
        return {
            r["source_type"]: {
                "total": r["total"],
                "with_context": r["with_context"] or 0,
            }
            for r in rows
        }

    def doc_body(self, source_type: str, source_key: str) -> str:
        """Reconstruct a document's body from its stored chunks.

        Each chunk carries a title preamble the chunker prepended; stripping
        it before joining keeps that boilerplate from being repeated once per
        chunk inside the prompt a contextualizer pays for.
        """
        return "\n\n".join(
            c.content.split("\n\n", 1)[-1]
            for c in self.get_by_source_key(source_type, source_key)
        )

    def last_yield(self, source_type: str) -> tuple[int, int] | None:
        """(documents, chunks) from this source's last recorded ingest, or None."""
        row = self._conn.execute(
            "SELECT documents, chunks FROM source_yield WHERE source_type = ?",
            (source_type,),
        ).fetchone()
        return (row["documents"], row["chunks"]) if row else None

    def last_skipped(self, source_type: str) -> int | None:
        """How many inputs this source permanently skipped last run, if known."""
        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(source_yield)")}
        if "skipped" not in cols:
            return None
        row = self._conn.execute(
            "SELECT skipped FROM source_yield WHERE source_type = ?", (source_type,)
        ).fetchone()
        return row["skipped"] if row else None

    def last_source_path(self, source_type: str) -> str | None:
        """The resolved path this source last read from, if recorded."""
        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(source_yield)")}
        if "source_path" not in cols:
            return None
        row = self._conn.execute(
            "SELECT source_path FROM source_yield WHERE source_type = ?", (source_type,)
        ).fetchone()
        return row["source_path"] if row else None

    def record_yield(
        self,
        source_type: str,
        documents: int,
        chunks: int,
        source_path: str | None = None,
        skipped: int = 0,
    ) -> None:
        """Record what this source yielded, for the next run to compare against.

        Written only after a run completes, so an aborted or crashed ingest
        never installs a low-water mark that makes the NEXT run's collapse
        look normal.
        """
        with self._txn() as conn:
            conn.execute(
                "INSERT INTO source_yield "
                "(source_type, documents, chunks, recorded_at, source_path, skipped) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(source_type) DO UPDATE SET "
                "documents = excluded.documents, chunks = excluded.chunks, "
                "recorded_at = excluded.recorded_at, source_path = excluded.source_path, "
                "skipped = excluded.skipped",
                (
                    source_type,
                    documents,
                    chunks,
                    datetime.now(UTC).isoformat(),
                    source_path,
                    skipped,
                ),
            )

    def get_by_source_key(self, source_type: str, source_key: str) -> list[StoredChunk]:
        rows = self._conn.execute(
            """
            SELECT id, source_type, source_key, content, metadata, title, url, context
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
                context=row["context"],
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
            SELECT id, source_type, source_key, content, metadata, title, url, context, updated_at
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
        self._require_writable("upsert_summary")
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
            self._closed = True
            for conn in self._all_conns:
                with contextlib.suppress(Exception):
                    conn.close()
            self._all_conns.clear()
