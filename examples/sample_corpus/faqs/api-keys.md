---
title: "Do I need an API key?"
id: faq-api-keys
---

# Do I need an API key?

It depends which embedder you pick. The Voyage and Gemini embedders both need an API key — `VOYAGE_API_KEY` or `GEMINI_API_KEY` in your `.env` — because turning text into vectors happens through their cloud API, both at ingest time and at query time.

The `hash` embedder needs no key at all: it's a deterministic, zero-dependency feature-hashing embedder meant for tests, CI, and a no-API-key on-ramp, not for real retrieval quality.

Separately, an Anthropic key is only needed if you run `corpus-summarize` for optional per-document Claude-Haiku summaries — core ingest and search never require it.
