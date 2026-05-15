# MCP integration

`corpus` ships an MCP server that exposes your corpus to Claude Code (or any MCP-aware client) over stdio. Once wired, Claude can search your archive, chase references, fetch summaries, and synthesize cross-source answers.

## Wiring it up

Add to `~/.claude.json`:

```json
{
  "mcpServers": {
    "corpus": {
      "type": "stdio",
      "command": "uv",
      "args": [
        "--directory",
        "/absolute/path/to/your/corpus",
        "run",
        "corpus-mcp"
      ],
      "env": {}
    }
  }
}
```

Use an **absolute path** to your corpus repo — Claude Code spawns from an arbitrary CWD and won't resolve relative paths reliably. Use an absolute path to `uv` too if your `uv` isn't on Claude Code's `PATH` (`/opt/homebrew/bin/uv` on Mac Homebrew).

After editing `~/.claude.json`, restart Claude Code. The eight MCP tools will appear in `/mcp`.

## The eight tools

### `search_knowledge`

Hybrid BM25 + vector search.

**Args:**
- `query` (str): natural-language question or search terms
- `source_types` (list[str] or str, optional): restrict to specific source types
- `top_k` (int, 1-20, default 5): result count

**Returns:** top-K chunks, each formatted with `[i] source_type:source_key d=<distance>`, title, URL, and content.

**Use it for:** the entry-point query. "How does X work?" / "What did we decide about Y?"

### `expand_context`

Chase references from a chunk you already have.

**Args:**
- `chunk_id` (str): ID from a prior `search_knowledge` result
- `include` (list[str] or str, optional): subset of `["siblings", "references", "parent"]`. Default: all three.
- `max_results` (int, 1-30, default 10)

**Returns:** related chunks — siblings (other chunks of the same doc), references (chunks matching `[[references]]` patterns mentioned in the seed), parent (if `metadata.extra.parent` is set).

**Use it for:** investigation. Search finds an entry point; expand pulls in adjacent material without re-embedding.

### `get_doc`

Pull every chunk of a specific document, in order.

**Args:**
- `source_type` (str)
- `source_key` (str)

**Returns:** all chunks of `(source_type, source_key)` ordered by `chunk_index`. Lets Claude read a whole doc rather than just the highest-scoring fragment.

### `timeline`

Search results reordered chronologically instead of by relevance.

**Args:**
- `topic` (str)
- `top_k` (int, 1-50, default 15)
- `since` / `until` (ISO date strings, optional)

**Returns:** chunks matching `topic`, sorted ascending by `updated_at`. Use for "walk me through what happened with X" questions.

### `recent_activity`

Chunks updated within the last N days.

**Args:**
- `days` (int, 1-365, default 7)
- `source_types` (list[str] or str, optional): restrict to specific sources
- `top_k` (int, 1-50, default 15)

**Returns:** newest-first list, deduped to one chunk per source doc.

### `get_summary`

Return the cached Claude-Haiku summary of a doc.

**Args:**
- `source_type` (str)
- `source_key` (str)

**Returns:** one-paragraph summary (~120 words) capturing the doc's intent, decisions, and concrete facts. **Requires** running `corpus-summarize` first; returns a helpful error if no summary exists.

### `corpus_stats`

Health check: total chunks + per-source breakdown.

**Args:** none

**Returns:** `Total chunks: N` plus a line per source type with its chunk count.

## The investigation pattern

The high-leverage flow is **search → expand → synthesize**:

1. Claude calls `search_knowledge("how does payment retry work?")` — gets top-5 entry points
2. Picks the most promising chunk (a Jira ticket); calls `expand_context(that_chunk.id)` — pulls in the rest of the ticket, the linked PR, the parent epic
3. Synthesizes from the full investigation, not just the top-5

Without `expand_context`, the model would only ever see 5 isolated chunks. With it, one search becomes the entry point to a multi-hop investigation across documents.

## Hygiene rules

Three things the server does that matter for stability:

1. **stderr-only logging.** The MCP protocol uses stdout for JSON-RPC; a stray `print()` would corrupt the stream. The server's logger is explicitly bound to stderr.

2. **Blocking calls wrapped in `asyncio.to_thread`.** SQLite + Voyage/Gemini SDKs are synchronous. Wrapping them in `to_thread` keeps the asyncio event loop responsive, so Claude isn't blocked by long-running queries.

3. **API-key check at startup.** If your configured provider needs an API key that isn't set, the server exits with code 2 and a clear error rather than failing on the first tool call. This shows up as "server failed to start" in Claude Code, easier to debug than an opaque MCP timeout.

## When you change code

The running MCP subprocess caches its Python bytecode in memory. If you change source files in `src/corpus/`, the subprocess won't see the changes until you restart Claude Code.

**For most changes** (retriever logic, chunker, embedder, util), iterate via the CLI instead — same code paths, no restart needed:

```sh
uv run corpus-query "your test question" -k 10
uv run pytest tests/ -q
```

**For MCP-specific changes** (tool schemas, FastMCP wiring, output formatting), restart Claude Code. There's no `corpus-mcp --reload` because the protocol is bound to the subprocess lifetime.

## Multiple corpora

If you want different MCP servers for different archives (work archive vs personal notes, for instance), give each its own entry:

```json
{
  "mcpServers": {
    "work-corpus": {
      "type": "stdio",
      "command": "uv",
      "args": ["--directory", "/path/to/work-corpus", "run", "corpus-mcp"]
    },
    "personal-corpus": {
      "type": "stdio",
      "command": "uv",
      "args": ["--directory", "/path/to/personal-corpus", "run", "corpus-mcp"]
    }
  }
}
```

Each clone has its own `corpus.toml`, `corpus.db`, and config. Tools become namespaced per server in Claude Code.

## Tool argument coercion

Two MCP-specific quirks the server handles silently:

1. `source_types` accepts either `"jira"` or `["jira", "pr"]` — LLMs occasionally send a single string when the schema asks for a list. The server normalizes both shapes.

2. `include` on `expand_context` accepts the same single-string-or-list polymorphism.

If you build your own tools, follow the same convention — `list[T] | str | None` in the type hint, normalize in the handler. Reduces "the LLM almost called it right" failures.
