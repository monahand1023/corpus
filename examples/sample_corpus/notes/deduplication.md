---
title: "Deduplication"
id: deduplication
---

# Deduplication

`corpus` dedups at two different points. First, at load time, each connector fingerprints a document's body (URLs, dates, casing, and whitespace stripped before hashing), so a near-duplicate body fingerprint — the same content re-exported under a different filename — is skipped within a single ingest run rather than stored twice.

Second, at ingest time, every chunk's content hash is compared against what's already in the store. Content-hash dedup skips unchanged chunks on re-ingest entirely: no re-embedding, no API cost, nothing rewritten. Only chunks whose hash actually changed get re-embedded and re-upserted, which is what makes `corpus-ingest` idempotent and cheap to run repeatedly.

Finally, orphan deletion removes vanished docs: if a source file that previously produced chunks is deleted or moved out of the configured `path`, the next ingest prunes those now-orphaned chunks from the database, so retrieval never returns stale results for content that no longer exists.
