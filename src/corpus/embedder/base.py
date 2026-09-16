"""Embedder protocol — the contract any embedder backend must satisfy.

Two methods, one tracking attribute. Document and query embeddings are
asymmetric (different `task_type` / `input_type`) because that's mandatory
for both Voyage's voyage-4-large and Gemini's gemini-embedding-001 to hit
their advertised retrieval quality.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    total_tokens_used: int

    # False when the provider returns no usage figure, so `total_tokens_used`
    # stays at 0 whatever the run did. Without this a Gemini-backed ingest
    # printed "tokens billed: 0" for half a million chunks -- "not measured"
    # rendered identically to "measured, and it was nothing".
    #
    # Declared here so mypy checks it, but READ with a default elsewhere
    # (`getattr(embedder, "counts_tokens", True)`): a backend that omits it is
    # assumed to count, because the flag exists to declare an ABSENCE of
    # measurement and a missing declaration must not create one silently.
    # Note this does tighten `isinstance(x, Embedder)`, which checks attribute
    # names -- nothing in corpus uses it, and the getattr is what the pipeline
    # actually depends on.
    counts_tokens: bool = True

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
