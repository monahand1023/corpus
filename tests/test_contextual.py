"""Tests for `corpus.contextual` — Contextual Retrieval.

The technique prepends an LLM-written sentence to each chunk before embedding
so a fragment retrieves on what it is about, not only on the words it happens
to contain. What these tests protect is mostly economics and safety: the
prompt-caching adjacency that makes it affordable, the spend policy that keeps
it off chunks it would harm, and the store contract that keeps `content`
recoverable if a run goes wrong.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from corpus.contextual.batch_builder import build_batch_requests, build_window_mapping
from corpus.contextual.contextualizer import (
    DEFAULT_MIN_TOKENS,
    MAX_CHUNK_CHARS,
    MAX_DOC_CHARS,
    build_context_prompt,
    estimate_chunk_cost_tokens,
    should_contextualize,
)
from corpus.db.sqlite import ChunkStore, StoredChunk
from corpus.types import Chunk, ChunkKind, ChunkMetadata

DIM = 8


def _chunk(i: int, source_key: str = "doc.md", content: str = "body text") -> Chunk:
    return Chunk(
        id=f"notes:{source_key}:{i}",
        content=content,
        content_hash=f"hash{i}",
        metadata=ChunkMetadata(
            source_type="notes",
            source_key=source_key,
            chunk_kind=ChunkKind.SECTION,
            chunk_index=i,
            title="Doc",
            token_count=len(content) // 4,
        ),
    )


def _store(tmp_path: Path) -> ChunkStore:
    return ChunkStore(tmp_path / "t.db", embedding_dim=DIM)


def _emb(seed: float = 0.1) -> list[float]:
    return [seed] * DIM


# --- the spend policy -------------------------------------------------------


def test_short_chunks_are_not_contextualized() -> None:
    # A blurb rivalling the chunk in length makes the embedding describe the
    # blurb rather than the content — actively worse, not merely wasteful.
    assert should_contextualize(10) is False
    assert should_contextualize(DEFAULT_MIN_TOKENS) is True
    assert should_contextualize(500) is True


def test_unknown_token_count_declines_to_spend() -> None:
    # The safe direction: skip rather than pay blind on a chunk of unknown size.
    assert should_contextualize(None) is False


def test_min_tokens_is_tunable_per_call() -> None:
    # Source code needs a far higher bar than prose — it already carries the
    # identifiers people search for. That decision is the caller's.
    assert should_contextualize(100, min_tokens=200) is False
    assert should_contextualize(300, min_tokens=200) is True


# --- prompt construction ----------------------------------------------------


def test_oversized_document_is_truncated() -> None:
    prompt = build_context_prompt("x" * (MAX_DOC_CHARS * 2), "chunk")

    assert len(prompt) < MAX_DOC_CHARS * 2
    assert "[truncated]" in prompt


def test_oversized_chunk_is_truncated() -> None:
    # The model places the chunk; it does not summarize it, so sending all of
    # a long chunk buys nothing and is billed.
    prompt = build_context_prompt("doc", "y" * (MAX_CHUNK_CHARS * 2))

    assert prompt.count("y") == MAX_CHUNK_CHARS


def test_prompt_contains_both_halves() -> None:
    prompt = build_context_prompt("THE DOCUMENT", "THE CHUNK")

    assert "THE DOCUMENT" in prompt
    assert "THE CHUNK" in prompt


# --- windowing, which is what makes it affordable ---------------------------


def test_chunks_of_one_document_share_a_request() -> None:
    # The parent document is a cached prompt prefix. Splitting one document's
    # chunks across requests re-pays full input price for the body each time,
    # so this adjacency IS the cost model.
    chunks = [
        StoredChunk(
            id=f"i{i}",
            source_type="notes",
            source_key="same.md",
            content=f"chunk {i}",
            metadata={"chunk_index": i},
            title="T",
            url=None,
        )
        for i in range(5)
    ]

    requests = build_batch_requests(chunks, {"same.md": "the body"}, window_size=40)

    assert len(requests) == 1


def test_windows_split_at_the_configured_size() -> None:
    chunks = [
        StoredChunk(
            id=f"i{i}",
            source_type="notes",
            source_key="big.md",
            content=f"chunk {i}",
            metadata={"chunk_index": i},
            title="T",
            url=None,
        )
        for i in range(10)
    ]

    requests = build_batch_requests(chunks, {"big.md": "body"}, window_size=4)

    assert len(requests) == 3  # 4 + 4 + 2


def test_window_mapping_matches_what_was_sent() -> None:
    # A custom_id must map to exactly the chunks that went out under it —
    # otherwise an applied batch attaches contexts to the wrong chunks, which
    # is silent and corpus-wide.
    chunks = [
        StoredChunk(
            id=f"i{i}",
            source_type="notes",
            source_key="d.md",
            content=f"c{i}",
            metadata={"chunk_index": i},
            title="T",
            url=None,
        )
        for i in range(6)
    ]

    requests = build_batch_requests(chunks, {"d.md": "body"}, window_size=4)
    mapping = build_window_mapping(chunks, window_size=4)

    assert {r["custom_id"] for r in requests} == set(mapping)
    assert sum(len(v) for v in mapping.values()) == 6


# --- the store contract -----------------------------------------------------


def test_set_context_leaves_content_untouched(tmp_path: Path) -> None:
    # `content` is canonical. A bad contextualization run must be revertible
    # without re-ingesting, which requires never overwriting it.
    store = _store(tmp_path)
    store.upsert_batch([(_chunk(0, content="original text"), _emb())])

    assert store.set_context("notes:doc.md:0", "situating sentence", _emb(0.2)) is True

    got = store.get_by_id("notes:doc.md:0")
    assert got is not None
    assert got.content == "original text"
    store.close()


def test_context_is_searchable_after_being_set(tmp_path: Path) -> None:
    # The whole point: the FTS/vector rows must cover context + content, or the
    # context is stored and never matched.
    store = _store(tmp_path)
    store.upsert_batch([(_chunk(0, content="opaque fragment"), _emb())])
    store.set_context("notes:doc.md:0", "concerning the Antarctic survey", _emb(0.2))

    hits = store.fts_search("Antarctic", top_k=5)

    assert [h.id for h in hits] == ["notes:doc.md:0"]
    store.close()


def test_set_context_is_idempotent(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.upsert_batch([(_chunk(0), _emb())])

    store.set_context("notes:doc.md:0", "first", _emb(0.2))
    store.set_context("notes:doc.md:0", "second", _emb(0.3))

    assert store.get_by_id("notes:doc.md:0").context == "second"  # type: ignore[union-attr]
    assert len(store.fts_search("second", top_k=5)) == 1
    store.close()


def test_set_context_on_unknown_id_is_a_no_op(tmp_path: Path) -> None:
    # A chunk can be deleted between batch submit and batch apply. That must
    # be a counted no-op, not a crash that abandons the rest of the results.
    store = _store(tmp_path)

    assert store.set_context("notes:gone.md:0", "ctx", _emb()) is False
    store.close()


def test_set_context_rejects_a_wrong_size_embedding(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.upsert_batch([(_chunk(0), _emb())])

    with pytest.raises(ValueError, match="dim mismatch"):
        store.set_context("notes:doc.md:0", "ctx", [0.1] * (DIM + 1))
    store.close()


def test_chunks_missing_context_excludes_those_that_have_one(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.upsert_batch([(_chunk(0), _emb()), (_chunk(1), _emb())])
    store.set_context("notes:doc.md:0", "done", _emb(0.2))

    missing = store.chunks_missing_context("notes")

    assert [c.id for c in missing] == ["notes:doc.md:1"]
    store.close()


def test_chunks_missing_context_groups_documents_together(tmp_path: Path) -> None:
    # Ordering by (source_key, chunk_index) is what lets the batch builder
    # send each parent document once as a cached prefix.
    store = _store(tmp_path)
    store.upsert_batch(
        [
            (_chunk(1, "b.md"), _emb()),
            (_chunk(0, "a.md"), _emb()),
            (_chunk(0, "b.md"), _emb()),
            (_chunk(1, "a.md"), _emb()),
        ]
    )

    keys = [(c.source_key, c.metadata["chunk_index"]) for c in store.chunks_missing_context("notes")]

    assert keys == [("a.md", 0), ("a.md", 1), ("b.md", 0), ("b.md", 1)]
    store.close()


def test_clear_context_restores_plain_text_search(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.upsert_batch([(_chunk(0, content="opaque fragment"), _emb())])
    store.set_context("notes:doc.md:0", "concerning the Antarctic survey", _emb(0.2))

    cleared = store.clear_context("notes")

    assert cleared == 1
    assert store.get_by_id("notes:doc.md:0").context is None  # type: ignore[union-attr]
    assert store.fts_search("Antarctic", top_k=5) == []
    assert len(store.fts_search("opaque", top_k=5)) == 1
    store.close()


def test_context_coverage_reports_per_source(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.upsert_batch([(_chunk(0), _emb()), (_chunk(1), _emb())])
    store.set_context("notes:doc.md:0", "ctx", _emb(0.2))

    coverage = store.context_coverage()

    assert coverage["notes"] == {"total": 2, "with_context": 1}
    store.close()


def test_doc_body_strips_the_repeated_title_preamble(tmp_path: Path) -> None:
    # The preamble is on every chunk. Leaving it in means paying for it once
    # per chunk inside the prompt.
    store = _store(tmp_path)
    store.upsert_batch(
        [
            (_chunk(0, content="Doc\n\nfirst part"), _emb()),
            (_chunk(1, content="[Doc]\n\nsecond part"), _emb()),
        ]
    )

    body = store.doc_body("notes", "doc.md")

    assert "first part" in body and "second part" in body
    assert "[Doc]" not in body
    store.close()


# --- cost estimation --------------------------------------------------------


def test_estimate_counts_a_capped_document() -> None:
    uncached, out = estimate_chunk_cost_tokens(MAX_DOC_CHARS * 10, 400)

    assert uncached <= (MAX_DOC_CHARS + MAX_CHUNK_CHARS) // 4 + 1
    assert out > 0


def test_estimate_of_a_cached_document_counts_only_the_chunk() -> None:
    # Callers pass doc_chars=0 for the second and later chunks of a document,
    # because the body is already a cache hit by then.
    uncached, _ = estimate_chunk_cost_tokens(0, 400)

    assert uncached == 100


# --- coverage reporting -----------------------------------------------------


def test_coverage_is_absent_for_a_store_that_never_contextualized(tmp_path: Path) -> None:
    # An install that has never run corpus-contextualize should see no extra
    # output at all, rather than "0 contextualized, 0%" on every line.
    store = _store(tmp_path)
    store.upsert_batch([(_chunk(0), _emb())])

    coverage = store.context_coverage()

    assert coverage["notes"]["with_context"] == 0
    store.close()


# --- placeholder contexts ---------------------------------------------------
#
# A model asked to situate an empty or fully-redacted chunk has nothing to
# work with and says so. Measured on a real archive, 0.037% of generated
# contexts came back this way. Storing one would prepend a meaningless token
# to the chunk's embedding AND mark the chunk done, so it would never be
# reconsidered — the answer is honest, but keeping it is not free.


@pytest.mark.parametrize(
    "placeholder",
    ["<UNKNOWN>", "unknown", "N/A", "None", "Empty or redacted document section.", "", "   "],
)
def test_placeholder_contexts_are_rejected(placeholder: str) -> None:
    from corpus.contextual.contextualizer import is_useful_context

    assert is_useful_context(placeholder) is False


def test_a_real_context_is_accepted() -> None:
    from corpus.contextual.contextualizer import is_useful_context

    assert is_useful_context(
        "From the July 2015 meeting minutes thread, the section on billing changes."
    )


def test_a_short_but_meaningful_context_is_still_rejected() -> None:
    # Under the length floor the blurb cannot situate anything, whatever it
    # says — and the chunk stays eligible for a later, better attempt.
    from corpus.contextual.contextualizer import is_useful_context

    assert is_useful_context("A PR-FAQ.") is False


# --- custom_id validity -----------------------------------------------------
#
# Anthropic rejects a custom_id outside ^[a-zA-Z0-9_-]{1,64}$ — and rejects
# the WHOLE batch, after the requests have been built and sent. corpus source
# keys are file paths, so the id is hashed rather than interpolated. This was
# a live failure: a real run died on a path-keyed source with a 400, while a
# sibling archive keyed by hex hashes never hit it, which is exactly how the
# constraint came to be written in a comment beside the field and enforced
# nowhere.


@pytest.mark.parametrize(
    "source_key",
    [
        "notes/2024/meeting minutes.md",
        "Archive/Field Notes/会議メモ.pptx",
        "a_b/c_d/e_f.md",
        "deep/" * 40 + "leaf.md",
        "with'quote\"and:colon.txt",
        "emoji-🎵-in-name.mp3",
    ],
)
def test_custom_id_is_valid_for_any_source_key(source_key: str) -> None:
    from corpus.contextual.batch_builder import CUSTOM_ID_PATTERN, window_custom_id

    assert CUSTOM_ID_PATTERN.match(window_custom_id("notes", source_key, 0))


def test_every_built_request_carries_a_valid_custom_id() -> None:
    chunks = [
        StoredChunk(
            id=f"i{i}",
            source_type="notes",
            source_key="Archive/Field Notes/会議メモ (2010).pptx",
            content=f"chunk {i}",
            metadata={"chunk_index": i},
            title="T",
            url=None,
        )
        for i in range(9)
    ]

    from corpus.contextual.batch_builder import CUSTOM_ID_PATTERN

    requests = build_batch_requests(chunks, {chunks[0].source_key: "body"}, window_size=4)

    assert len(requests) == 3
    for request in requests:
        assert CUSTOM_ID_PATTERN.match(request["custom_id"]), request["custom_id"]


def test_distinct_source_keys_never_share_an_id() -> None:
    # Sanitizing instead of hashing would collapse "a/b.md" and "a_b.md" onto
    # one id and apply one document's contexts to another's chunks.
    from corpus.contextual.batch_builder import window_custom_id

    ids = {
        window_custom_id("notes", "a/b.md", 0),
        window_custom_id("notes", "a_b.md", 0),
        window_custom_id("notes", "a-b.md", 0),
        window_custom_id("other", "a/b.md", 0),
        window_custom_id("notes", "a/b.md", 1),
    }

    assert len(ids) == 5


def test_the_id_is_stable_across_calls() -> None:
    from corpus.contextual.batch_builder import window_custom_id

    assert window_custom_id("notes", "x/y.md", 2) == window_custom_id("notes", "x/y.md", 2)


def test_window_chunk_ids_round_trips_a_hashed_id() -> None:
    # The old implementation parsed the id with split("_"), which was already
    # wrong for any source_key containing an underscore.
    chunks = [
        StoredChunk(
            id=f"i{i}",
            source_type="notes",
            source_key="has_underscores/in_path.md",
            content=f"c{i}",
            metadata={"chunk_index": i},
            title="T",
            url=None,
        )
        for i in range(6)
    ]

    from corpus.contextual.batch_builder import window_chunk_ids

    requests = build_batch_requests(chunks, {chunks[0].source_key: "b"}, window_size=4)
    first = window_chunk_ids(chunks, requests[0]["custom_id"], window_size=4)

    assert first == ["i0", "i1", "i2", "i3"]


# --- FTS normalization on the context write path ----------------------------
#
# `upsert` writes normalize_for_fts(content) and the query path searches for
# that normalized form — CJK runs become overlapping bigrams via fts_terms.
# A row written RAW is therefore unreachable by any CJK query, so a Japanese
# chunk would silently drop out of BM25 the moment it gained a context. This
# went unnoticed because ASCII normalizes to itself, so every English test
# passed.


def test_contextualized_cjk_chunk_stays_searchable(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.upsert_batch([(_chunk(0, content="東京で会議をしました"), _emb())])
    assert len(store.fts_search("東京", top_k=5)) == 1, "precondition: findable before"

    store.set_context("notes:doc.md:0", "A meeting note from the Tokyo office.", _emb(0.2))

    assert len(store.fts_search("東京", top_k=5)) == 1, "lost to BM25 after contextualizing"
    store.close()


def test_a_cjk_context_is_itself_searchable(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.upsert_batch([(_chunk(0, content="opaque fragment"), _emb())])

    store.set_context("notes:doc.md:0", "予算報告書からの抜粋", _emb(0.2))

    assert len(store.fts_search("予算", top_k=5)) == 1
    store.close()


def test_clear_context_restores_a_searchable_cjk_row(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.upsert_batch([(_chunk(0, content="東京で会議をしました"), _emb())])
    store.set_context("notes:doc.md:0", "context", _emb(0.2))

    store.clear_context("notes")

    assert len(store.fts_search("東京", top_k=5)) == 1
    store.close()
