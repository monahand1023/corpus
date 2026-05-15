---
title: "Configuring corpus"
id: configuration
---

# Configuration

Everything that varies between deployments lives in `corpus.toml`:

- `[corpus] db_path` — where the SQLite file lives
- `[embedder]` — provider, model, dim (must match the model's output)
- `[retriever]` — top_k, max_per_source_type, hybrid on/off
- `[[sources]]` — repeatable; each defines a Connector to load
- `[[references]]` — optional; reference-pattern regexes for `expand_context`

The DB enforces an embedding-dim guard at startup: if your existing data was embedded at 1024 dim and you change the model to one with 512 dim, the store refuses to open. This prevents silent retrieval corruption.

## Why corpus.toml instead of code

Configuration is the most-frequently-changed part of any RAG setup. Keeping it out of source means you can:

- Try a different embedder without editing Python
- Add/remove sources without redeploying
- Share the codebase across multiple personal corpora (work archive vs research papers vs personal notes) — one `corpus.toml` per directory
