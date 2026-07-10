---
title: "What sources are supported?"
id: faq-sources
---

# What sources are supported?

Four connectors ship in the box: markdown, text, PDF, and HTML. Markdown and text need no extras; PDF needs `pip install corpus-rag[pdf]` (uses `pypdf`), and HTML needs `corpus-rag[html]` (uses `trafilatura` for boilerplate-stripped extraction). Each is configured with a `[[sources]]` block naming a `type`, a `path`, and a `glob`.

Beyond those four, connectors are extensible for any source you actually have — Slack exports, Jira dumps, EPUB books, a custom JSON format. Copy the markdown connector as a starting point, write a small `Connector` + `Chunker` pair, and register it in `connectors/registry.py`. See `docs/adding_a_source.md` for a full worked example.
