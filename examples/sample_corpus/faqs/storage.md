---
title: "Where is my data stored?"
id: faq-storage
---

# Where is my data stored?

Everything lives in one local SQLite file, at whatever path you set for `db_path` in `corpus.toml` (`./corpus.db` by default). There's no separate vector database service to run or pay for — vectors are stored in that same file via the `sqlite-vec` extension, alongside a BM25 full-text index and the chunk metadata.

Nothing about storage touches the cloud. The one exception, covered elsewhere, is that the text you ingest and your queries are sent to your chosen embedding API (Voyage or Gemini) to be turned into vectors — but the resulting database, the index, and search itself all run entirely on your machine.
