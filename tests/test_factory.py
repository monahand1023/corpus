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


def test_make_embedder_missing_voyage_sdk_gives_actionable_error(monkeypatch) -> None:
    """When voyageai isn't installed (base install, no [voyage] extra), the
    factory should raise an ImportError that names the extra to install."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "corpus.embedder.voyage" or name.startswith("voyageai"):
            raise ImportError("No module named 'voyageai'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"pip install corpus-rag\[voyage\]"):
        make_embedder(provider="voyage", model="voyage-3-large")


def test_make_embedder_missing_gemini_sdk_gives_actionable_error(monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "corpus.embedder.gemini" or name.startswith("google"):
            raise ImportError("No module named 'google'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"pip install corpus-rag\[gemini\]"):
        make_embedder(provider="gemini", model="gemini-embedding-001", dim=1536)
