"""Re-embed an archive in place, without destroying what was paid for.

THE GAP THIS FILLS. There is no other supported way to change embedding
provider, model or dim on an existing archive. A re-ingest re-embeds NOTHING
-- `ChunkStore.upsert` returns early when the content hash is unchanged -- and
the dim guard's own error says "Re-ingest from scratch with the new dim",
which in practice means `corpus-reset --all`. That destroys everything stored
only in the database and nowhere else:

    one archive:  a quarter-million contextualized chunks
    another:      tens of thousands of blurbs and summaries

All of it Anthropic spend, with no on-disk backup. The safe operation is much
narrower than a re-ingest: only the VECTORS change. `chunks`, `chunks_fts`,
`summaries`, every context blurb and every piece of metadata stay exactly as
they are.

WHY A STAGING TABLE, rather than writing vectors as they arrive.

  - The expensive part is the API calls, over up to 1.4M chunks. Staging means
    an interruption never pays for the same chunk twice.
  - `chunks_vec` is a `vec0` virtual table with the dimension baked into its
    schema, so changing dim means DROPPING and recreating it. Doing that first
    and filling it as embeddings arrive would leave a half-populated vector
    table on any interruption -- an archive that returns wrong neighbours and
    says nothing about it.

So the swap is one transaction at the end, and it refuses to run until every
chunk is staged.

WHAT IS EMBEDDED. `content` alone, which is what `Ingester._flush` embeds. A
chunk's `context` feeds the FTS text, not the vector, so re-embedding does not
need it and does not disturb it.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import sqlite_vec

from corpus.util.tokens import estimate_tokens

Embed = Callable[[list[str]], list[list[float] | None]]

_STAGING = "reembed_staging"

DEFAULT_BATCH_SIZE = 128


class ReembedIncomplete(RuntimeError):
    """The swap was asked for while chunks were still unstaged."""


def _connect(db_path: Path | str) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def _ensure_staging(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"""CREATE TABLE IF NOT EXISTS {_STAGING} (
              rowid_ref INTEGER PRIMARY KEY,
              dim       INTEGER NOT NULL,
              embedding BLOB NOT NULL
            )"""
    )


def _current_dim(conn: sqlite3.Connection) -> int | None:
    row = conn.execute(
        "SELECT value FROM schema_meta WHERE key = 'embedding_dim'"
    ).fetchone()
    return int(row["value"]) if row else None


@dataclass(frozen=True)
class ReembedPlan:
    """What a re-embed would do, priced before anything is spent."""

    chunks: int
    current_dim: int | None
    new_dim: int
    estimated_tokens: int
    already_staged: int = 0

    def describe(self) -> str:
        lines = [
            f"chunks to re-embed : {self.chunks:,}",
            f"embedding dim      : {self.current_dim} -> {self.new_dim}",
            f"estimated tokens   : ~{self.estimated_tokens:,}",
        ]
        if self.already_staged:
            lines.append(
                f"already staged     : {self.already_staged:,} "
                "(a previous run; these are not paid for again)"
            )
        lines.append(
            "Context blurbs, summaries, metadata and BM25 rows are untouched: "
            "only the vectors change."
        )
        return "\n".join(lines)


def reembed_plan(db_path: Path | str, *, new_dim: int) -> ReembedPlan:
    """Price the run. Opens read-only and writes nothing.

    The token figure is the chars/4 heuristic over the text that would
    actually be sent, so it is an estimate and not a quote -- non-English text
    runs materially higher.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        total = conn.execute("SELECT count(*) AS c FROM chunks").fetchone()["c"]
        tokens = 0
        for row in conn.execute("SELECT content FROM chunks"):
            tokens += estimate_tokens(row["content"] or "")
        try:
            staged = conn.execute(
                f"SELECT count(*) AS c FROM {_STAGING} WHERE dim = ?", (new_dim,)
            ).fetchone()["c"]
        except sqlite3.Error:
            staged = 0
        return ReembedPlan(
            chunks=total,
            current_dim=_current_dim(conn),
            new_dim=new_dim,
            estimated_tokens=tokens,
            already_staged=staged,
        )
    finally:
        conn.close()


def staged_count(db_path: Path | str, *, new_dim: int | None = None) -> int:
    """How many chunks are already embedded and waiting to be swapped in."""
    conn = _connect(db_path)
    try:
        _ensure_staging(conn)
        if new_dim is None:
            row = conn.execute(f"SELECT count(*) AS c FROM {_STAGING}").fetchone()
        else:
            row = conn.execute(
                f"SELECT count(*) AS c FROM {_STAGING} WHERE dim = ?", (new_dim,)
            ).fetchone()
        return int(row["c"])
    finally:
        conn.close()


def stage_embeddings(
    db_path: Path | str,
    *,
    new_dim: int,
    embed: Embed,
    batch_size: int = DEFAULT_BATCH_SIZE,
    limit: int | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> int:
    """Embed every chunk not yet staged for `new_dim`. Returns how many it added.

    Resumable by construction: a chunk already in the staging table for this
    dim is skipped, so an interruption costs only the batch in flight. Staging
    for a DIFFERENT dim is discarded first -- mixing vector widths in one
    table would produce a store that cannot be searched and would be found out
    only at query time.

    `limit` is for sampling a run before committing to it, and is what makes
    an incomplete staging table reachable in a test.
    """
    conn = _connect(db_path)
    try:
        _ensure_staging(conn)
        conn.execute(f"DELETE FROM {_STAGING} WHERE dim != ?", (new_dim,))
        conn.commit()

        rows = conn.execute(
            f"""SELECT c.rowid AS rid, c.content AS content
                  FROM chunks c
                  LEFT JOIN {_STAGING} s
                    ON s.rowid_ref = c.rowid AND s.dim = ?
                 WHERE s.rowid_ref IS NULL
                 ORDER BY c.rowid""",
            (new_dim,),
        ).fetchall()
        if limit is not None:
            rows = rows[:limit]

        added = 0
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            vectors = embed([r["content"] or "" for r in batch])
            payload = [
                (r["rid"], new_dim, sqlite_vec.serialize_float32(list(v)))
                for r, v in zip(batch, vectors, strict=True)
                if v is not None
            ]
            # Committed per batch, which is what makes a resume cheap: the
            # alternative loses every embedding bought since the last failure.
            conn.executemany(
                f"INSERT OR REPLACE INTO {_STAGING} (rowid_ref, dim, embedding)"
                " VALUES (?, ?, ?)",
                payload,
            )
            conn.commit()
            added += len(payload)
            if on_progress:
                on_progress(added, len(rows))
        return added
    finally:
        conn.close()


def swap_in_staged(db_path: Path | str, *, new_dim: int) -> int:
    """Replace `chunks_vec` with the staged vectors. Returns how many moved.

    Refuses while any chunk is unstaged. A half-swapped vector table is an
    archive that returns wrong neighbours and says nothing about it, which is
    strictly worse than one that has not been touched.

    One transaction: the drop, the recreate, every insert, and the dim stamp.
    An interruption rolls back to the old vectors, and the staging table -- the
    part that cost money -- is still there for the retry.
    """
    conn = _connect(db_path)
    try:
        _ensure_staging(conn)
        total = conn.execute("SELECT count(*) AS c FROM chunks").fetchone()["c"]
        staged = conn.execute(
            f"SELECT count(*) AS c FROM {_STAGING} WHERE dim = ?", (new_dim,)
        ).fetchone()["c"]
        if staged < total:
            raise ReembedIncomplete(
                f"{staged:,} of {total:,} chunks are staged for dim {new_dim}. "
                "Run the staging pass to completion first: swapping now would "
                "leave the archive searchable but wrong."
            )

        conn.execute("BEGIN")
        conn.execute("DROP TABLE IF EXISTS chunks_vec")
        conn.execute(
            "CREATE VIRTUAL TABLE chunks_vec USING vec0"
            f"(embedding float[{new_dim}])"
        )
        moved = 0
        for row in conn.execute(
            f"SELECT rowid_ref, embedding FROM {_STAGING} WHERE dim = ?"
            " ORDER BY rowid_ref",
            (new_dim,),
        ):
            conn.execute(
                "INSERT INTO chunks_vec(rowid, embedding) VALUES (?, ?)",
                (row["rowid_ref"], row["embedding"]),
            )
            moved += 1
        conn.execute(
            "INSERT INTO schema_meta (key, value) VALUES ('embedding_dim', ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (str(new_dim),),
        )
        conn.execute(f"DELETE FROM {_STAGING}")
        conn.commit()
        return moved
    finally:
        conn.close()


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "Embed",
    "ReembedIncomplete",
    "ReembedPlan",
    "reembed_plan",
    "stage_embeddings",
    "staged_count",
    "swap_in_staged",
]
