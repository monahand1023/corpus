# MCP integration

`corpus` ships an MCP server that exposes your corpus to Claude Code (or any MCP-aware client) over stdio. Once wired, Claude can search your archive, chase references, fetch summaries, and synthesize cross-source answers.

## Wiring it up

Add to `~/.claude.json`. **Pass the absolute path to your `corpus.toml`** via `--config`:

```json
{
  "mcpServers": {
    "corpus": {
      "type": "stdio",
      "command": "corpus-mcp",
      "args": ["--config", "/absolute/path/to/your/corpus.toml"],
      "env": {}
    }
  }
}
```

Why the absolute path? Claude Code spawns the MCP server from an arbitrary CWD (often `$HOME`). Without `--config`, `corpus-mcp` looks for `./corpus.toml` in that CWD — which usually isn't where your config lives.

If you'd rather avoid `--config` and you're a uv user, the alternative is to spawn through `uv`, which sets the CWD for you:

```json
{
  "mcpServers": {
    "corpus": {
      "type": "stdio",
      "command": "uv",
      "args": ["--directory", "/path/to/repo", "run", "corpus-mcp"]
    }
  }
}
```

Use an absolute path to `uv` (`/opt/homebrew/bin/uv` on Mac Homebrew) if Claude Code's PATH doesn't include it.

After editing `~/.claude.json`, restart Claude Code. The seven MCP tools will appear in `/mcp`.

## Claude Desktop wiring

Claude Desktop also supports stdio MCP servers using the same mechanism. Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or the equivalent path on your OS:

```json
{
  "mcpServers": {
    "corpus": {
      "command": "uv",
      "args": [
        "--directory", "/absolute/path/to/your/corpus",
        "run", "corpus-mcp"
      ]
    }
  }
}
```

Use the absolute path to `uv` if Claude Desktop's PATH doesn't include it (e.g. `/opt/homebrew/bin/uv` on Mac Homebrew, `~/.cargo/bin/uv` on Linux).

Restart Claude Desktop after editing the config. The corpus tools appear in the tool panel alongside any other MCP servers you have configured.

> **Note:** Claude Desktop uses stdio (it spawns the server process itself) — not HTTP/SSE. If you try to use a `"url"` field, Claude Desktop will reject the config entry. Stdio works on macOS, Windows, and Linux without any extra server setup.

## The seven tools

### `search_knowledge`

Hybrid BM25 + vector search.

**Args:**
- `query` (str): natural-language question or search terms
- `source_types` (list[str] or str, optional): restrict to specific source types
- `top_k` (int, 1-20, default 5): result count

**Returns:** top-K chunks, each formatted with `[i] source_type:source_key d=<distance>`, title, URL, and content. The result is prefixed with an "untrusted retrieved content" banner (see Hygiene rules) — `get_doc`, `expand_context`, and `timeline` carry it too.

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

4. **Sanitized tool errors.** If a tool handler hits an unexpected exception, it returns a generic "an internal error occurred (see server logs)" message and logs the detail to stderr — internal paths and state aren't leaked back to the model.

5. **Untrusted-content labeling.** The content-returning tools (`search_knowledge`, `get_doc`, `expand_context`, `timeline`) prefix their output with a banner telling the model to treat retrieved corpus text as data, not instructions. It's a prompt-injection mitigation for adversarial corpus content — a speed bump, not a guarantee.

## When you change code

The running MCP subprocess caches its Python bytecode in memory. If you change source files in `src/corpus/`, the subprocess won't see the changes until you restart Claude Code.

**For most changes** (retriever logic, chunker, embedder, util), iterate via the CLI instead — same code paths, no restart needed:

```sh
uv run corpus-query "your test question" -k 10
uv run pytest tests/ -q
```

**For MCP-specific changes** (tool schemas, FastMCP wiring, output formatting), restart Claude Code. There's no `corpus-mcp --reload` because the protocol is bound to the subprocess lifetime.

## Building your own MCP server

If your consumer needs its own tool vocabulary, import the shared machinery
from `corpus.mcp_util` rather than reimplementing it — and deliberately NOT
from `corpus.mcp_server`, which constructs a FastMCP instance and reconfigures
the root logger at import time. Importing that module would give you a second
server and a rewired logger as side effects.

```python
from corpus.mcp_util import (
    UNTRUSTED_PREFIX,      # the data-not-instructions banner
    QueryTimer,            # elapsed-ms measurement
    format_chunk_block,    # the [i] source:key / title / URL / content renderer
    record_query,          # append to the query log, never raises
    safe_tool,             # decorator: generic error text, detail to the log
)

@mcp.tool(description="...")
@safe_tool
async def search_knowledge(query: str, top_k: int = 5) -> str:
    chunks = ...
    return UNTRUSTED_PREFIX + "\n\n---\n\n".join(
        format_chunk_block(i, c) for i, c in enumerate(chunks, 1)
    )
```

`format_chunk_block` takes an optional `extra` callable returning one more
header line — useful when your chunks have no URL and traceability comes from
somewhere else (an originating file path, a message sender, a folder).

**Mark every tool that returns indexed text.** Corpus content is not written by
the person running the server: an archive of mail, tickets, pull requests or
shared documents is full of text other people wrote, and anyone who ever wrote
into it could have included something shaped like an instruction — years before
it was indexed, with no idea it would be fed to a model. Unmarked, it reaches
the model indistinguishable from the operator's own words. Tools returning only
aggregates (counts, coverage summaries) need no banner; anything carrying
document text, titles, or a generated summary OF document text does. A summary
is generated FROM the content, so an injection can survive summarisation.

This is a framing, not a sandbox. It does not make injection impossible — it
removes the case where there is no defence at all.

## Server lifetime: stdio versus a daemon transport

These two transports fail in opposite directions, and the difference decides
which guards a server needs.

**stdio needs none of them.** The transport *is* the pipe from the client, so
one client session means exactly one process — by definition, not by accident.
A machine running several Claude sessions will show several copies of every
stdio server, and that is correct: two clients cannot share one pipe. They are
not leaked, they are owned. Closing the pipe ends them (measurably, in about
300ms), so killing a client — however violently — takes its servers with it.

Do not put a run-lock on a stdio path. The first session would win and every
later one would find the server refusing to start, which is a worse failure
than the one the lock prevents.

**An HTTP/SSE server has neither property.** Nothing owns the process, so
nothing reaps it: when the shell that launched one exits, the server is
reparented to init and keeps listening indefinitely. Two guards supply what
the pipe would have:

```python
from corpus.mcp_util import (
    claim_single_instance,  # flock run-lock; returns the holder's PID, or None
    port_holder,            # human-readable "who has this port", or None
    exit_when_orphaned,     # shut down once reparented to init
)

if args.transport == "http":
    holder = claim_single_instance(f"my-archive-http-{args.port}")
    if holder is not None:
        logger.info("already serving port %d as pid %d — nothing to do",
                    args.port, holder)
        return                      # idempotent: asking twice is not an error
    conflict = port_holder(args.port)
    if conflict is not None:
        logger.error("%s — free it or pass --port", conflict)
        sys.exit(2)
    exit_when_orphaned()
    mcp.run(transport="sse")
```

`claim_single_instance` uses an `flock`, not a PID file, and the difference is
the point: the kernel releases an flock when the holder dies by any means,
including SIGKILL. There is no stale lock to detect and no cleanup path that
can be skipped, so a crashed server never blocks its own replacement.

The lock and the port are separate resources and can disagree — an unrelated
program can hold the port while the lock is free — so `port_holder` probes it
and turns that case into one readable line instead of a bind traceback from
inside the ASGI server. It races by construction; the lock is what enforces
exclusion, the probe only explains the common case.

`exit_when_orphaned` returns False and does nothing when the parent is already
init, because that means deliberate daemonisation (nohup, launchd, a container
entrypoint) — arming there would kill exactly the servers meant to persist.

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
