"""A chunk the embedder declined vanished from both totals.

`_flush` skips any chunk whose embedding came back None:

    for chunk, emb in zip(to_embed, embeddings, strict=True):
        if emb is None:
            continue

No counter, no log. It returns `upserted, skipped + already`, so a dropped
chunk appears in NEITHER, and the run reports success with a quietly
smaller number.

Today `None` only ever means "blank input" by the Embedder contract, so
the blast radius is currently nil -- the contract is the only thing holding
it. An embedder that later returns None for a FAILED item would silently
un-index content while reporting `failed_files == 0`, which also leaves
the pruning gate open: nothing would stop the sweep deleting the chunks
that failed to embed.

Counted and named, so the number cannot be lost even if the contract
changes underneath it.
"""

from __future__ import annotations


class _PartialEmbedder:
    """Returns None for one chunk, as a failing embedder would."""

    dim = 4
    total_tokens_used = 0

    def embed_documents(self, texts):
        return [None if "DROP" in t else [0.1, 0.2, 0.3, 0.4] for t in texts]

    def embed_query(self, text):
        return [0.1, 0.2, 0.3, 0.4]


def test_a_chunk_the_embedder_declined_is_counted(tmp_path):
    from corpus.db.sqlite import ChunkStore
    from corpus.ingester import Ingester
    from corpus.types import Chunk, ChunkKind, ChunkMetadata

    store = ChunkStore(tmp_path / "x.db", embedding_dim=4)
    ing = Ingester.__new__(Ingester)
    ing._store = store
    ing._embedder = _PartialEmbedder()
    ing._owned_store = False

    def _chunk(i: int, text: str) -> Chunk:
        return Chunk(
            id=f"c{i}",
            content=text,
            content_hash=f"h{i}",
            metadata=ChunkMetadata(
                source_type="notes",
                source_key="a.md",
                chunk_kind=ChunkKind.SECTION,
                chunk_index=i,
                title="t",
            ),
        )

    upserted, _skipped, dropped = ing._flush(
        [_chunk(0, "keep this"), _chunk(1, "DROP this one")]
    )

    assert upserted == 1
    assert dropped == 1, (
        "a chunk the embedder declined was in neither total -- the run would "
        "report success with a quietly smaller index"
    )
    store.close()
