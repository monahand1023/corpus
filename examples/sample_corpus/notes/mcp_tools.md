---
title: "MCP tools reference"
id: mcp-tools
---

# MCP tools

When wired to Claude Code, `corpus` exposes 8 tools:

## search_knowledge

Hybrid BM25 + vector search. Returns the top-K most relevant chunks. Auto-tunes the BM25 weight based on whether the query looks identifier-shaped vs prose.

## expand_context

Given a chunk_id from a prior search result, returns related chunks:

- **Siblings**: other chunks of the same document (read the rest of the page)
- **References**: chunks of any IDs matched by `[[references]]` patterns in `corpus.toml`
- **Parent**: if the chunk's `metadata.extra.parent` is set, fetch the parent's chunks

This is what enables multi-hop investigation: Claude searches, finds an entry point, then chases references across documents.

## get_doc

Pull every chunk of a specific `(source_type, source_key)`. Use when you want to read a whole doc in order rather than the highest-scoring fragments.

## timeline / who_did_what / recent_activity

Specialized non-semantic queries: chronological-by-date, author-filtered, and last-N-days-by-source.

## get_summary

Returns the cached Claude-Haiku summary for a document. Only available after running `corpus-summarize`.

## corpus_stats

Total chunks + per-source breakdown. Health check.
