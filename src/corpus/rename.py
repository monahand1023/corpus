"""Rename a source in place, without re-ingesting or re-embedding it.

WHY THIS IS NOT AN UPDATE STATEMENT. `source_type` is not just a label -- it
is the first field of the chunk id:

    id = f"{source_type}:{source_key}:{chunk_kind}:{chunk_index}"

So renaming the column alone leaves ids that no longer match what the next
ingest computes, and that ingest inserts a second copy of every chunk instead
of recognising the existing rows -- silently doubling the source, at full
embedding cost, with no error anywhere. The rename therefore recomputes ids,
and updates the copy of `source_type` inside the metadata JSON so a reader
that trusts the metadata does not disagree with the column beside it.

`summaries` is keyed by `(source_type, source_key)` too. Leaving those behind
orphans paid work that `get_summary` can then never find.

WHAT IS DELIBERATELY NOT TOUCHED. Vectors and BM25 rows are keyed by `rowid`,
which does not change, so they are left exactly as they are -- and that is the
whole point: renaming a source must not cost an embedding bill.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from corpus.types import SOURCE_TYPE_PATTERN


class SourceNotFound(LookupError):
    """No chunks carry the source being renamed."""


class ForeignChunkIds(ValueError):
    """The stored ids were not built by this engine's `chunk_id`.

    A rename recomputes every id from (source_type, source_key, kind, index).
    That is only safe when the chunker that WROTE them derives ids the same
    way -- otherwise the rename imposes a scheme the consumer's own chunker
    will not reproduce, and the next ingest sees every chunk as new.
    """


class TargetExists(ValueError):
    """The new name is already in use.

    Merging two sources is a different operation with different consequences:
    chunk ids would collide and one source would silently overwrite the other.
    """


def rename_source(db_path: Path | str, old: str, new: str) -> int:
    """Rename `old` to `new`, returning how many chunks moved.

    One transaction: either every id, column and metadata copy moves together,
    or none does. A half-renamed source is one whose ids match neither name.
    """
    if not re.fullmatch(SOURCE_TYPE_PATTERN, new):
        raise ValueError(
            f"{new!r} is not a valid source name (must match "
            f"{SOURCE_TYPE_PATTERN}). A name no corpus.toml can express is a "
            "source that can never be ingested again."
        )

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT id, source_key, chunk_kind, chunk_index, metadata"
            " FROM chunks WHERE source_type = ?",
            (old,),
        ).fetchall()
        if not rows:
            raise SourceNotFound(
                f"no chunks with source_type={old!r}. Nothing was changed."
            )
        clash = conn.execute(
            "SELECT count(*) AS c FROM chunks WHERE source_type = ?", (new,)
        ).fetchone()["c"]
        if clash:
            raise TargetExists(
                f"{clash:,} chunks already use source_type={new!r}. Renaming "
                "onto an existing source would collide their chunk ids and "
                "silently overwrite one with the other."
            )

        # VERIFY THE SCHEME BEFORE IMPOSING IT. A consumer is free to derive
        # chunk ids however it likes -- one here uses a truncated sha256 --
        # and rewriting those into this engine's readable form leaves ids its
        # chunker will never produce again. The next ingest then treats every
        # chunk as new: the whole source is re-embedded, the renamed rows
        # become orphans, and the prune deletes them. Nothing errors anywhere
        # along the way.
        #
        # Checked against the OLD name, which is what the stored ids encode.
        foreign = [
            row["id"]
            for row in rows
            if row["id"]
            != f"{old}:{row['source_key']}:{row['chunk_kind']}:{row['chunk_index']}"
        ]
        if foreign:
            raise ForeignChunkIds(
                f"{len(foreign):,} of {len(rows):,} chunks in {old!r} have ids "
                "this engine did not build. Renaming would rewrite them into "
                "a scheme the chunker that wrote them does not use, and the "
                "next ingest would re-embed the whole source and prune what "
                "was renamed. Rename the source in that consumer's own "
                "configuration and re-ingest instead."
            )

        conn.execute("BEGIN")
        for row in rows:
            new_id = (
                f"{new}:{row['source_key']}:{row['chunk_kind']}:{row['chunk_index']}"
            )
            metadata = row["metadata"]
            try:
                parsed = json.loads(metadata)
                parsed["source_type"] = new
                metadata = json.dumps(parsed)
            except (TypeError, ValueError):
                # Unparseable metadata is left as-is rather than dropped: the
                # column and the id are what retrieval uses, and discarding a
                # document's metadata to fix its label would be the larger
                # loss.
                pass
            conn.execute(
                "UPDATE chunks SET id = ?, source_type = ?, metadata = ?"
                " WHERE id = ?",
                (new_id, new, metadata, row["id"]),
            )
        conn.execute(
            "UPDATE summaries SET source_type = ? WHERE source_type = ?",
            (new, old),
        )
        # The ingest baseline (yield, skips, path) is keyed by name as well;
        # left behind, the renamed source's next ingest has nothing to compare
        # against. A database older than the table has nothing to move.
        if conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'source_yield'"
        ).fetchone():
            conn.execute(
                "UPDATE source_yield SET source_type = ? WHERE source_type = ?",
                (new, old),
            )
        conn.commit()
        return len(rows)
    finally:
        conn.close()


__all__ = ["SourceNotFound", "TargetExists", "rename_source"]
