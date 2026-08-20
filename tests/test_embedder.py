from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import voyageai.error as ve

from corpus.embedder.voyage import VoyageEmbedder

DIM = 1024


def fake_response(texts: list[str]) -> object:
    obj = MagicMock()
    obj.embeddings = [[0.0] * DIM for _ in texts]
    obj.total_tokens = sum(len(t) // 4 for t in texts)
    return obj


@pytest.fixture
def embedder() -> VoyageEmbedder:
    e = VoyageEmbedder(model="voyage-3-large", api_key="dummy")
    e._client = MagicMock()
    e._client.embed = MagicMock(side_effect=lambda texts, **kw: fake_response(texts))
    return e


def test_doc_input_type(embedder: VoyageEmbedder) -> None:
    embedder.embed_documents(["one", "two"])
    assert embedder._client.embed.call_args.kwargs["input_type"] == "document"


def test_query_input_type(embedder: VoyageEmbedder) -> None:
    embedder.embed_query("question")
    assert embedder._client.embed.call_args.kwargs["input_type"] == "query"


def test_empty_strings_skipped(embedder: VoyageEmbedder) -> None:
    out = embedder.embed_documents(["real", "", "  ", "another"])
    assert out[1] is None
    assert out[2] is None
    assert embedder._client.embed.call_args.kwargs["texts"] == ["real", "another"]


def test_size_error_triggers_split(embedder: VoyageEmbedder) -> None:
    calls = {"n": 0}

    def side(texts, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ve.InvalidRequestError("too big")
        return fake_response(texts)

    embedder._client.embed.side_effect = side
    out = embedder.embed_documents(["a", "b", "c", "d"])
    assert all(v is not None for v in out)
    assert calls["n"] == 3


def test_empty_query_raises(embedder: VoyageEmbedder) -> None:
    with pytest.raises(ValueError):
        embedder.embed_query("")


def test_missing_api_key_raises(monkeypatch) -> None:
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="VOYAGE_API_KEY"):
        VoyageEmbedder(model="voyage-3-large")


# --- token-accurate batch packing ----------------------------------------
# Regression: batches were packed by a 400,000-char budget annotated
# "~100K tokens" (the chars/4 rule). Real token density is not constant — it
# ranged 3.66 to 1.51 chars/token on a mixed corpus — so at 1.51 that budget is
# 264K tokens against a hard 120K cap. Voyage rejected every oversized batch and
# the recursive halving made ingest ~300x slower than the API itself.


def _tokenizing_embedder(chars_per_token: float) -> VoyageEmbedder:
    """Embedder whose local tokenizer reports a chosen density."""
    e = VoyageEmbedder(model="voyage-3-large", api_key="dummy")
    e._client = MagicMock()
    e._client.embed = MagicMock(side_effect=lambda texts, **kw: fake_response(texts))
    e._client.tokenize = MagicMock(
        side_effect=lambda texts, **kw: [
            ["t"] * max(1, int(len(t) / chars_per_token)) for t in texts
        ]
    )
    return e


def test_packs_by_real_tokens_not_chars() -> None:
    """Dense text must produce more batches than sparse text of equal length."""
    from corpus.embedder.voyage import MAX_TOKENS_PER_BATCH

    texts = ["x" * 5_000 for _ in range(60)]   # 300K chars either way

    sparse = _tokenizing_embedder(4.0)
    dense = _tokenizing_embedder(1.5)
    n_sparse = len(sparse._pack_batches(list(range(len(texts))), texts))
    n_dense = len(dense._pack_batches(list(range(len(texts))), texts))

    assert n_dense > n_sparse, "dense content must be split into more batches"
    for idxs, batch in dense._pack_batches(list(range(len(texts))), texts):
        assert sum(len(t) / 1.5 for t in batch) <= MAX_TOKENS_PER_BATCH
        assert len(idxs) == len(batch)


def test_no_batch_exceeds_the_hard_token_cap() -> None:
    """The exact failure: 128 legal-count inputs totalling 164,556 tokens."""
    from corpus.embedder.voyage import MAX_TOKENS_PER_BATCH

    e = _tokenizing_embedder(1.51)
    texts = ["x" * 1_940 for _ in range(400)]
    for _idx, batch in e._pack_batches(list(range(len(texts))), texts):
        assert sum(len(t) / 1.51 for t in batch) <= MAX_TOKENS_PER_BATCH


def test_input_count_cap_still_applies() -> None:
    from corpus.embedder.voyage import MAX_INPUTS_PER_BATCH

    e = _tokenizing_embedder(4.0)
    texts = ["tiny" for _ in range(1000)]
    batches = e._pack_batches(list(range(len(texts))), texts)
    assert all(len(b) <= MAX_INPUTS_PER_BATCH for _i, b in batches)
    assert sum(len(b) for _i, b in batches) == 1000


def test_token_cap_is_under_voyages_hard_limit() -> None:
    from corpus.embedder.voyage import MAX_TOKENS_PER_BATCH

    assert MAX_TOKENS_PER_BATCH < 120_000


def test_falls_back_to_pessimistic_chars_when_tokenizer_unavailable() -> None:
    """Offline or an older SDK must degrade to over-splitting, never overflow."""
    from corpus.embedder.voyage import FALLBACK_CHARS_PER_TOKEN, MAX_TOKENS_PER_BATCH

    e = VoyageEmbedder(model="voyage-3-large", api_key="dummy")
    e._client = MagicMock()
    e._client.embed = MagicMock(side_effect=lambda texts, **kw: fake_response(texts))
    e._client.tokenize = MagicMock(side_effect=RuntimeError("no tokenizer here"))

    texts = ["x" * 2_000 for _ in range(300)]
    batches = e._pack_batches(list(range(len(texts))), texts)
    assert e._tokenizer_ok is False
    # Even at the densest ratio ever observed, no batch may exceed the cap.
    for _idx, batch in batches:
        assert sum(len(t) / 1.51 for t in batch) <= MAX_TOKENS_PER_BATCH
    assert FALLBACK_CHARS_PER_TOKEN < 1.51


def test_tokenizer_failure_is_warned_once_not_per_batch(caplog) -> None:
    e = VoyageEmbedder(model="voyage-3-large", api_key="dummy")
    e._client = MagicMock()
    e._client.embed = MagicMock(side_effect=lambda texts, **kw: fake_response(texts))
    e._client.tokenize = MagicMock(side_effect=RuntimeError("nope"))
    with caplog.at_level("WARNING"):
        for _ in range(5):
            e._pack_batches([0], ["some text"])
    assert sum("tokenizer unavailable" in r.message for r in caplog.records) == 1


def test_tokenizer_returning_wrong_arity_falls_back_safely() -> None:
    """A tokenizer that returns the wrong number of counts must degrade to the
    char fallback, not crash packing with a zip length mismatch."""
    e = VoyageEmbedder(model="voyage-3-large", api_key="dummy")
    e._client = MagicMock()
    e._client.embed = MagicMock(side_effect=lambda texts, **kw: fake_response(texts))
    e._client.tokenize = MagicMock(side_effect=lambda texts, **kw: [["t"]])  # always 1
    batches = e._pack_batches([0, 1, 2], ["aa", "bb", "cc"])
    assert e._tokenizer_ok is False
    assert sum(len(b) for _i, b in batches) == 3
