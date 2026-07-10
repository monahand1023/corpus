---
title: "Embedding providers"
id: embeddings
---

# Embedding providers

`corpus` turns each chunk into a vector via a pluggable embedder. The default provider is Voyage, model `voyage-3-large`, producing a 1024 dimension vector — `dim = 1024` in `corpus.toml` must match whatever the embedder model actually outputs. Gemini's `gemini-embedding-001` is the other built-in cloud option, at 768/1536/3072 dimension depending on config. A third provider, `hash`, needs no API key at all and exists for tests and CI, not real retrieval quality.

Both Voyage and Gemini embed asymmetrically: documents get a document input type at ingest time, and the query gets a separate query input type at search time. Mixing these up quietly hurts retrieval quality even though nothing errors.

An embedding dim guard checks `embedder.dim` against the dimension already recorded in the SQLite schema at startup and refuses to proceed whenever you change models or dim after data has already been ingested — preventing silent vector corruption.
