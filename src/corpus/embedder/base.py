"""Embedder protocol — the contract any embedder backend must satisfy.

Two methods, one tracking attribute. Document and query embeddings are
asymmetric (different `task_type` / `input_type`) because that's mandatory
for both Voyage's voyage-3-large and Gemini's gemini-embedding-001 to hit
their advertised retrieval quality.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    total_tokens_used: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float] | None]:
        """Returns parallel list of embeddings. Empty inputs map to None.

        The provider MUST send the document-asymmetric task type
        (`input_type='document'` for Voyage, `task_type='RETRIEVAL_DOCUMENT'`
        for Gemini)."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Returns a single embedding. Empty input raises.

        The provider MUST send the query-asymmetric task type."""
        ...
