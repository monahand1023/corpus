---
title: "How much does it cost?"
id: faq-cost
---

# How much does it cost?

`corpus` itself is free and local — SQLite, sqlite-vec, and the reranker all run on your machine at no cost. The embedding cost is per token, charged by your provider (Voyage or Gemini) at ingest time when chunks are turned into vectors; querying also costs a small number of tokens per query embedding.

Because of content-hash dedup, that embedding cost is effectively a one-time cost per chunk — re-running `corpus-ingest` after small edits only re-embeds what actually changed, not the whole corpus. Voyage's free tier covers roughly 200M tokens.

Per-doc summaries (`corpus-summarize`) are an optional extra cost, billed against your Anthropic key for Claude-Haiku calls — skip it if you don't need summaries.
