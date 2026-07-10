"""Eval query set over `examples/sample_corpus` (source types: note, faq).

Queries deliberately share distinctive keywords with their target doc: the CI
gate embeds with the lexical `hash` embedder, so a query with no lexical overlap
would test nothing. On the real private corpora (voyage/gemini embedder) you can
and should paraphrase harder to stress semantic retrieval. See docs/eval.md.

Each EvalQuery:
  - query: the natural-language question
  - expected_keys: source_keys, any one of which counts as a hit (logical OR)
  - source_filter: optional list[str] restricting the search to source types
  - source_type: optional tag used ONLY to bucket the per-source-type breakdown
  - note: free-text; printed, never scored
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvalQuery:
    query: str
    expected_keys: list[str] = field(default_factory=list)
    source_filter: list[str] | None = None
    source_type: str | None = None
    note: str = ""


EVAL_QUERIES: list[EvalQuery] = [
    # --- note source type ---
    EvalQuery("corpus calls itself a straightforward single-user local RAG framework", ["architecture"], source_type="note"),
    EvalQuery("how does the sqlite-vec store upsert chunks with content-hash dedup during ingest", ["architecture"], source_type="note"),
    EvalQuery("why keep configuration in corpus.toml instead of hardcoding it in python, since it changes across deployments", ["configuration"], source_type="note"),
    EvalQuery("what settings live in corpus.toml", ["configuration"], source_type="note"),
    EvalQuery("which MCP tools does corpus expose to Claude", ["mcp-tools"], source_type="note"),
    EvalQuery("what does expand_context return: siblings, references, parent", ["mcp-tools"], source_type="note"),
    EvalQuery("which embedding provider is the default and what dimension", ["embeddings"], source_type="note"),
    EvalQuery("asymmetric document and query input types for embeddings", ["embeddings"], source_type="note"),
    EvalQuery("how is a markdown document split into retrievable chunks", ["chunking"], source_type="note"),
    EvalQuery("how are vector search and BM25 combined by reciprocal rank fusion", ["hybrid-search"], source_type="note"),
    EvalQuery("why doesn't reciprocal rank fusion need score normalization when combining BM25 and vector similarity", ["hybrid-search"], source_type="note"),
    EvalQuery("auto-tuned BM25 weight for identifier-shaped queries", ["hybrid-search"], source_type="note"),
    EvalQuery("how does the BGE cross-encoder rerank the candidate pool", ["reranking"], source_type="note"),
    EvalQuery("does corpus remove secrets and credentials before storing", ["scrubbing"], source_type="note"),
    EvalQuery("content-hash dedup skips unchanged chunks on re-ingest", ["deduplication"], source_type="note"),
    EvalQuery("how does corpus detect a near-duplicate document body, re-exported under a different filename, before storing it twice", ["deduplication"], source_type="note"),
    EvalQuery("how do I write a connector for a new source type", ["connectors"], source_type="note"),
    EvalQuery("measure per-stage retrieval latency p50 p95 p99", ["benchmarking"], source_type="note"),
    # --- faq source type ---
    EvalQuery("do I need an api key to run corpus", ["faq-api-keys"], source_type="faq"),
    EvalQuery("can I run retrieval without any cloud api key", ["faq-api-keys"], source_type="faq"),
    EvalQuery("where is my data stored, is it a local sqlite file", ["faq-storage"], source_type="faq"),
    EvalQuery("single-file sqlite-vec local storage", ["faq-storage"], source_type="faq"),
    EvalQuery("is my personal data private, does it leave my machine", ["faq-privacy"], source_type="faq"),
    EvalQuery("what source formats can corpus ingest: pdf html markdown", ["faq-sources"], source_type="faq"),
    EvalQuery("how much does embedding cost", ["faq-cost"], source_type="faq"),
    EvalQuery("how do I refresh the corpus after files change", ["faq-updating"], source_type="faq"),
    EvalQuery("can I keep multiple separate corpora", ["faq-multi-corpus"], source_type="faq"),
    EvalQuery("retrieval returns nothing, what should I check", ["faq-troubleshooting"], source_type="faq"),
    # --- source_filter demonstrations ---
    EvalQuery("reciprocal rank fusion", ["hybrid-search"], source_filter=["note"], source_type="note",
              note="filter restricts to note source type"),
    EvalQuery("api key required for embedder", ["faq-api-keys"], source_filter=["faq"], source_type="faq",
              note="filter restricts to faq source type"),
    # --- negatives (informational; excluded from scoring) ---
    EvalQuery("how do I deploy corpus to a kubernetes cluster", note="negative — not covered by the corpus"),
    EvalQuery("what is the capital of france", note="negative — off-topic"),
]
