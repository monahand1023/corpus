---
title: "How do I update the corpus?"
id: faq-updating
---

# How do I update the corpus?

Edit your files, then re-run `corpus-ingest --all` (or `--source name` for just one). There's no daemon or file watcher — ingestion only happens when you run the command. It is idempotent, so running it again after no changes is cheap and safe.

Content-hash dedup skips unchanged chunks — only chunks whose content actually changed get re-embedded and re-upserted. And orphan deletion removes stale ones: chunks whose source file was deleted or moved out of the configured `path` get pruned from the database on the next run, so retrieval never surfaces content that no longer exists.
