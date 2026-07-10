---
title: "Hybrid search and RRF"
id: hybrid-search
---

# Hybrid search and RRF

Every query runs two rankers in parallel: a vector KNN search over the embedded chunks, and a BM25 full-text search via SQLite's FTS5 extension. The two ranked lists are fused with reciprocal rank fusion (RRF) — no score normalization needed, since RRF just sums `weight / (k + rank)` across rankers for each chunk id.

The BM25 weight is auto-tuned per query: prose-shaped questions get a low BM25 weight so vector search dominates, while identifier-shaped queries — backticked code, quoted phrases, or anything matching a configured `[[references]]` pattern — bump the BM25 weight up to 1.0, because lexical match matters more when someone's searching for a specific ticket key or function name.

After fusion, results are deduped by source key — `(source_type, source_key)` — so one long doc doesn't flood top-K and crowd out other relevant documents; a `max_per_source_type` diversity cap enforces the same thing across source types.
