---
title: "Retrieval returns nothing?"
id: faq-troubleshooting
---

# Retrieval returns nothing?

Work through it in order. First, confirm ingest actually ran and wrote chunks — `corpus_stats` or a direct `SELECT source_type, COUNT(*) FROM chunks GROUP BY source_type` should show non-zero counts for the source you expect.

Second, check the embedding dim matches between `corpus.toml` and what's already in the database; a mismatch raises at startup rather than silently returning nothing, but it's worth double-checking after switching models.

Third, if you're filtering by source, loosen the source_filter — a typo'd or too-narrow filter returns an empty result even though the corpus has matching content elsewhere.

Finally, try hybrid search (`hybrid = true`, the default) rather than vector-only — a malformed FTS query silently falls back to empty BM25 results, but hybrid still returns the vector branch.
