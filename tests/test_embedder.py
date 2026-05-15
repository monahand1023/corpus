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
