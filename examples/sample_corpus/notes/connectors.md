---
title: "Writing a connector"
id: connectors
---

# Writing a connector

`corpus` ships four built-in connectors: markdown, text, pdf, and html. Markdown is the reference connector — when you want to support a new source type, copy it for PDF, HTML, Slack export, Jira dump, or whatever-flavor-of-the-week source you actually have, and adapt it.

A connector is small: it walks a directory (via `discover_files`, which skips symlinks and rejects paths outside the source root) and yields `SourceDocument` objects — one per logical document, with a `source_key`, title, optional URL, dates, and a `raw` payload the chunker will read from.

A paired chunker then splits them: it takes each `SourceDocument` and emits one or more `Chunk` objects, running `scrub()` before hashing and assigning deterministic `chunk_id`s so re-ingestion stays idempotent. Register the connector/chunker pair in `connectors/registry.py`, then reference the new `type` from a `[[sources]]` block in `corpus.toml`. See `docs/adding_a_source.md` for a full worked JSON-files example.
