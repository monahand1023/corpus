---
title: "How chunking works"
id: chunking
---

# How chunking works

The markdown chunker splits a document's body into sections at header boundaries, cutting on H2/H3 headings while ignoring any heading that appears inside a fenced code block. Long sections get further size-split at paragraph breaks, always snapping around code fences so a split never lands inside one. Small heading-heavy fragments get coalesced back together up to the chunk size cap, so a document with many tiny headers doesn't explode into a flood of one-line chunks.

Every chunk carries the document's title so results are self-describing out of context: the first chunk carries the title as a plain prefix, and every later chunk gets a bracketed title prefix like `[How chunking works]` so you can tell which document a mid-document chunk came from.

Each chunk also gets a token count estimate recorded in its metadata (roughly chars/4), used for size guards and reporting.
