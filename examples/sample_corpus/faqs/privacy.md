---
title: "Is my data private?"
id: faq-privacy
---

# Is my data private?

`corpus` is local-first: one process, one SQLite file, no accounts, no multi-tenancy, no network service beyond stdio for the MCP server. The only place your text leaves your machine at all is the embedding step — only the embedder API (Voyage or Gemini) ever sees chunk text and query text, because turning them into vectors requires their cloud endpoint.

No other data leaves your machine: retrieval, the vector and BM25 index, the reranker, and the MCP server all run locally, and nothing is uploaded, logged remotely, or shared with any other service. Secrets and credentials are also scrubbed out of chunk text before it's ever embedded or stored.
