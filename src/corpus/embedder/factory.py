"""Embedder factory — dispatch by provider name from corpus.toml.

Built-in providers:
  - voyage  — Voyage AI (default; voyage-3-large, 1024-dim)
  - gemini  — Google Gemini (gemini-embedding-001, 768/1536/3072-dim)

Add a new provider by writing an `Embedder`-conforming class and registering
it here. The Embedder Protocol is in `embedder/base.py`.
"""

from __future__ import annotations

from corpus.embedder.base import Embedder


def make_embedder(
    provider: str,
    model: str,
    *,
    api_key: str | None = None,
    dim: int | None = None,
) -> Embedder:
    if provider == "voyage":
        from corpus.embedder.voyage import VoyageEmbedder

        return VoyageEmbedder(model=model, api_key=api_key)

    if provider == "gemini":
        from corpus.embedder.gemini import GeminiEmbedder

        return GeminiEmbedder(model=model, api_key=api_key, dim=dim)

    raise ValueError(
        f"Unknown embedder provider '{provider}'. "
        "Built-in providers: voyage, gemini. "
        "Add yours in src/corpus/embedder/factory.py."
    )
