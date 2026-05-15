from __future__ import annotations

import pytest

from corpus.embedder.factory import make_embedder


def test_unknown_provider_raises() -> None:
    with pytest.raises(ValueError, match="Unknown embedder provider"):
        make_embedder(provider="not_a_provider", model="x")


def test_voyage_factory_returns_voyage_embedder(monkeypatch) -> None:
    monkeypatch.setenv("VOYAGE_API_KEY", "dummy")
    embedder = make_embedder(provider="voyage", model="voyage-3-large")
    from corpus.embedder.voyage import VoyageEmbedder
    assert isinstance(embedder, VoyageEmbedder)


def test_gemini_factory_returns_gemini_embedder(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    embedder = make_embedder(provider="gemini", model="gemini-embedding-001", dim=1536)
    from corpus.embedder.gemini import GeminiEmbedder
    assert isinstance(embedder, GeminiEmbedder)
