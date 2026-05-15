from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from corpus.embedder.gemini import GeminiEmbedder


def _fake_response(texts: list[str], dim: int = 1536) -> MagicMock:
    """Mock the Gemini SDK's embed_content response shape."""
    embeddings = [MagicMock(values=[0.1] * dim) for _ in texts]
    return MagicMock(embeddings=embeddings)


@pytest.fixture
def embedder(monkeypatch) -> GeminiEmbedder:
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    with patch("google.genai.Client"):
        e = GeminiEmbedder(model="gemini-embedding-001", dim=1536)
    e._client = MagicMock()
    e._client.models.embed_content = MagicMock(
        side_effect=lambda model, contents, config: _fake_response(contents)
    )
    return e


def test_documents_use_retrieval_document_task_type(embedder: GeminiEmbedder) -> None:
    embedder.embed_documents(["one", "two"])
    config = embedder._client.models.embed_content.call_args.kwargs["config"]
    assert config.task_type == "RETRIEVAL_DOCUMENT"


def test_query_uses_retrieval_query_task_type(embedder: GeminiEmbedder) -> None:
    embedder.embed_query("question")
    config = embedder._client.models.embed_content.call_args.kwargs["config"]
    assert config.task_type == "RETRIEVAL_QUERY"


def test_empty_strings_skipped(embedder: GeminiEmbedder) -> None:
    out = embedder.embed_documents(["real", "", "  ", "another"])
    assert out[1] is None
    assert out[2] is None
    sent = embedder._client.models.embed_content.call_args.kwargs["contents"]
    assert sent == ["real", "another"]


def test_empty_query_raises(embedder: GeminiEmbedder) -> None:
    with pytest.raises(ValueError):
        embedder.embed_query("")


def test_dim_passed_to_config(embedder: GeminiEmbedder) -> None:
    embedder.embed_documents(["x"])
    config = embedder._client.models.embed_content.call_args.kwargs["config"]
    assert config.output_dimensionality == 1536


def test_missing_api_key_raises(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        GeminiEmbedder()
