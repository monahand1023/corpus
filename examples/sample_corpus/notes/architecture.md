---
title: "How corpus works"
id: architecture
---

# Architecture

`corpus` is a local single-user RAG framework. The pipeline is straightforward:

1. **Connectors** walk source directories and yield `SourceDocument`s.
2. **Chunkers** split each document into retrievable chunks.
3. **Scrub** removes secrets and credentials.
4. **Embedder** (Voyage `voyage-3-large` by default) turns each chunk into a 1024-dim vector.
5. **Store** (SQLite + sqlite-vec) UPSERTs the chunks with content-hash dedup.

At query time:

1. **Embed** the query asymmetrically (`input_type='query'`).
2. **Hybrid search**: vector KNN + BM25 FTS5, fused via reciprocal rank fusion.
3. **Dedupe** by `(source_type, source_key)` so one long doc doesn't fill top-K.
4. **Diversity cap** (`max_per_source_type=3`) ensures multi-source results.
5. **Optional re-rank** via local BGE cross-encoder.

The MCP server exposes 8 tools to Claude Code over stdio.
