"""The cost estimate for a run that spends real money per chunk.

`--dry-run` is documented as the intended first step and "the only warning
before" a paid run. An estimate that is wrong in the cheap direction is how a
user approves a bill they did not expect -- and this project's standing rule
is that paid runs are approved from a stated cost.

The subtle part is the caching model: the parent document is sent ONCE as a
cached prefix, so counting it per chunk would overstate a big document's cost
by roughly its chunk count.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from corpus.cli.contextualize import _estimate


def _chunk(key: str, content: str, token_count: int):
    return SimpleNamespace(
        source_key=key, content=content, metadata={"token_count": token_count}
    )


def _store(chunks, bodies):
    store = MagicMock()
    store.chunks_missing_context.return_value = chunks
    store.doc_body.side_effect = lambda _s, key: bodies[key]
    return store


def test_nothing_eligible_costs_nothing() -> None:
    store = _store([], {})
    assert _estimate(store, "notes", min_tokens=50) == (0, 0, 0)


def test_chunks_below_the_token_floor_are_not_counted() -> None:
    """The floor exists so tiny chunks are not paid for; if the estimate
    ignored it the quote would exceed the real spend."""
    chunks = [_chunk("a.md", "tiny", token_count=5)]
    eligible, _, _ = _estimate(_store(chunks, {"a.md": "body"}), "notes", min_tokens=50)
    assert eligible == 0


def test_the_parent_document_is_counted_once_per_DOCUMENT(monkeypatch) -> None:
    """The whole point of the caching model.

    Two chunks of one document must not be charged for that document twice.
    """
    body = "d" * 4000
    two_chunks_one_doc = [
        _chunk("a.md", "chunk one" * 20, token_count=100),
        _chunk("a.md", "chunk two" * 20, token_count=100),
    ]
    one_chunk_one_doc = [_chunk("a.md", "chunk one" * 20, token_count=100)]

    _, in_two, _ = _estimate(_store(two_chunks_one_doc, {"a.md": body}), "n", 50)
    _, in_one, _ = _estimate(_store(one_chunk_one_doc, {"a.md": body}), "n", 50)

    # The second chunk adds its own text but NOT another copy of the document.
    doc_tokens = len(body) // 4
    assert in_two - in_one < doc_tokens, (
        "the parent document was charged twice for one document"
    )


def test_two_separate_documents_are_each_counted() -> None:
    body = "d" * 4000
    chunks = [
        _chunk("a.md", "chunk" * 20, token_count=100),
        _chunk("b.md", "chunk" * 20, token_count=100),
    ]
    bodies = {"a.md": body, "b.md": body}
    _, in_two_docs, _ = _estimate(_store(chunks, bodies), "n", 50)
    _, in_one_doc, _ = _estimate(
        _store(chunks[:1], {"a.md": body}), "n", 50)

    doc_tokens = len(body) // 4
    assert in_two_docs - in_one_doc >= doc_tokens * 0.9, (
        "a second document must add its own cached prefix"
    )


def test_the_estimate_scales_with_eligible_chunks() -> None:
    body = "d" * 400
    def run(n):
        chunks = [_chunk(f"{i}.md", "c" * 400, token_count=100) for i in range(n)]
        return _estimate(_store(chunks, {f"{i}.md": body for i in range(n)}), "n", 50)
    small, large = run(2), run(8)
    assert large[0] > small[0]
    assert large[1] > small[1] and large[2] > small[2]
