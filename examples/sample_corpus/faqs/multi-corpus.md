---
title: "Can I run multiple corpora?"
id: faq-multi-corpus
---

# Can I run multiple corpora?

Yes. `corpus` has no notion of a single global corpus — keep one `corpus.toml` per directory, each pointing at its own `db_path` database file: one for work, one for research, one for personal notes, however you want to split things up.

Since `corpus-mcp` is spawned with `--config /absolute/path/to/corpus.toml`, you can wire up several MCP server entries in `~/.claude.json`, each with a different config path. Nothing is shared between corpora except the `corpus` codebase itself — no cross-corpus dedup, no shared cache, no shared database.
