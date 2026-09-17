"""Renaming a source rewrote chunk ids into the ENGINE's scheme, always.

`rename_source` builds every new id as

    f"{new}:{source_key}:{chunk_kind}:{chunk_index}"

regardless of what the stored ids actually look like. That is correct for
an archive whose chunker uses `corpus.util.hash.chunk_id`, and silently
destructive for one that does not.

A consumer here computes ids as a truncated sha256 instead. Renaming its
source would rewrite every id into the engine's readable form -- after
which its own chunker still produces hashes, so on the NEXT ingest every
chunk looks new: the whole archive is re-embedded, the renamed rows become
orphans, and the prune deletes them. Nothing reports a problem at any
point; it just costs a full re-embedding and leaves duplicates in between.

So the rename verifies the scheme it is about to impose is the scheme
already in use, and refuses otherwise. Cheaper than making every consumer
adopt one identity function, and it protects consumers nobody here knows
about.
"""

from __future__ import annotations

import hashlib
import sqlite3

import pytest


def _store(tmp_path, ids_are_hashes: bool):
    db = tmp_path / "x.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE chunks (id TEXT PRIMARY KEY, source_type TEXT,"
        " source_key TEXT, chunk_kind TEXT, chunk_index INTEGER,"
        " content TEXT, content_hash TEXT, metadata TEXT)"
    )
    conn.execute(
        "CREATE TABLE summaries (source_type TEXT, source_key TEXT, summary TEXT)"
    )
    for i in range(3):
        key = f"m{i}.eml"
        if ids_are_hashes:
            cid = hashlib.sha256(f"mail|{key}|section|0".encode()).hexdigest()[:32]
        else:
            cid = f"mail:{key}:section:0"
        conn.execute(
            "INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?)",
            (cid, "mail", key, "section", 0, "body", "h", "{}"),
        )
    conn.commit()
    conn.close()
    return db


def test_a_foreign_id_scheme_is_refused(tmp_path):
    from corpus.rename import ForeignChunkIds, rename_source

    db = _store(tmp_path, ids_are_hashes=True)
    with pytest.raises(ForeignChunkIds):
        rename_source(db, "mail", "email")


def test_the_engine_scheme_still_renames(tmp_path):
    from corpus.rename import rename_source

    db = _store(tmp_path, ids_are_hashes=False)
    assert rename_source(db, "mail", "email") == 3

    conn = sqlite3.connect(db)
    ids = [r[0] for r in conn.execute("SELECT id FROM chunks")]
    assert all(i.startswith("email:") for i in ids), ids
    conn.close()


def test_a_refusal_changes_nothing(tmp_path):
    """A guard that half-applies is worse than none."""
    from corpus.rename import ForeignChunkIds, rename_source

    db = _store(tmp_path, ids_are_hashes=True)
    conn = sqlite3.connect(db)
    before = sorted(r[0] for r in conn.execute("SELECT id FROM chunks"))
    conn.close()

    with pytest.raises(ForeignChunkIds):
        rename_source(db, "mail", "email")

    conn = sqlite3.connect(db)
    after = sorted(r[0] for r in conn.execute("SELECT id FROM chunks"))
    types = {r[0] for r in conn.execute("SELECT DISTINCT source_type FROM chunks")}
    conn.close()
    assert after == before
    assert types == {"mail"}
