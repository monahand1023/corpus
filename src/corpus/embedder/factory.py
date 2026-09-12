"""Embedder factory — dispatch by provider name from corpus.toml.

Built-in providers:
  - voyage  — Voyage AI (default; voyage-4-large, 1024-dim)
  - gemini  — Google Gemini (gemini-embedding-001, 768/1536/3072-dim)
  - hash    — deterministic zero-dep hashing embedder (no API key; tests/CI only)

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
        try:
            from corpus.embedder.voyage import VoyageEmbedder
        except ImportError as e:
            raise ImportError(
                "The 'voyage' embedder requires the voyageai SDK, which is not "
                "installed. Install it with `pip install corpus-rag[voyage]`."
            ) from e

        return VoyageEmbedder(model=model, api_key=api_key)

    if provider == "gemini":
        # NOT wrapped in try/except ImportError, unlike `voyage` directly above.
        # The asymmetry is deliberate and reflects where each SDK is imported:
        # `voyage.py` does `import voyageai` at module level, so importing the
        # module genuinely raises when the SDK is absent and the wrapper there
        # is live. `gemini.py` imports `google.genai` lazily inside
        # `GeminiEmbedder.__init__`, so this import cannot raise for a missing
        # SDK -- a wrapper here would be dead code that merely looks like
        # protection. The real handler lives at the lazy import in
        # `GeminiEmbedder.__init__`, which raises with the same install hint.
        from corpus.embedder.gemini import GeminiEmbedder

        return GeminiEmbedder(model=model, api_key=api_key, dim=dim)

    if provider == "hash":
        from corpus.embedder.hash import HashEmbedder

        return HashEmbedder(dim=dim or 256)

    raise ValueError(
        f"Unknown embedder provider '{provider}'. "
        "Built-in providers: voyage, gemini, hash. "
        "Add yours in src/corpus/embedder/factory.py."
    )
