"""MCP stdio server exposing the corpus to Claude Code.

Seven tools: search, expand_context, get_doc, timeline, recent_activity,
get_summary, corpus_stats.

Hygiene rules:
  - stdout is the MCP protocol channel — never `print()`. Logging is on stderr.
  - Sqlite + Voyage calls are blocking — wrapped in `asyncio.to_thread`.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from corpus.config import ConfigError, CorpusConfig
from corpus.credentials import describe_search, resolve_dotenv
from corpus.db.sqlite import ChunkStore, StoredChunk
from corpus.embedder.base import Embedder
from corpus.embedder.factory import make_embedder
from corpus.mcp_util import (
    UNTRUSTED_PREFIX,
    QueryTimer,
    format_chunk_block,
    record_query,
    safe_tool,
)
from corpus.retriever import Retriever

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("corpus.mcp")

mcp = FastMCP(
    name="corpus",
    instructions=(
        "Search the user's personal knowledge archive — a generalized RAG over "
        "their own notes, documents, papers, web clippings, and similar material "
        "(the configured source types vary per install; call `corpus_stats` to see "
        "which exist). Answer plain-language questions by choosing the right tool; "
        "the user does NOT pass parameters — infer them from the question.\n"
        "• Any topic / 'what do I have on X', 'find where I wrote about Y', "
        "conceptual or keyword lookup → `search_knowledge` (semantic + BM25 hybrid; "
        "pass source_types to scope to one kind of material).\n"
        "• 'Show me the whole note/doc', 'read all of <X>', 'pull up the full <doc>' → "
        "`get_doc` (every chunk of one document, in order) — usually after a "
        "search_knowledge hit gives you its source_type + source_key.\n"
        "• 'What's around this', 'related sections', 'what does it reference', "
        "neighbors/siblings of a result → `expand_context` on a chunk_id.\n"
        "• 'When did X happen', 'how did Y evolve', anything chronological → "
        "`timeline`.\n"
        "• 'What have I added/changed lately', 'what's new this week' → "
        "`recent_activity`.\n"
        "• 'Give me the gist of <doc>', a one-paragraph TL;DR of a single document → "
        "`get_summary` (only populated after running `corpus-summarize`; falls back "
        "to a hint if absent).\n"
        "• 'How big is the archive', 'what's in here', counts per source → "
        "`corpus_stats`.\n"
        "Typical flow: `search_knowledge` to locate material → `get_doc` / "
        "`expand_context` to read it in full → cite concrete details (names, dates, "
        "identifiers) from the source text."
    ),
)

_store: ChunkStore | None = None
_embedder: Embedder | None = None
_retriever: Retriever | None = None
_config: CorpusConfig | None = None
_config_path_override: str | None = None  # set by main() from --config flag


def _init() -> tuple[ChunkStore, Embedder, Retriever, CorpusConfig]:
    global _store, _embedder, _retriever, _config
    if _config is None:
        _config = CorpusConfig.load(_config_path_override)
    if _store is None:
        # read_only=True: every tool this server exposes is a read (search,
        # get_doc, timeline, ...) -- none ingest or otherwise mutate. Opening
        # the store any other way risks silently running a schema migration
        # (see ChunkStore._migrate_fts) as a side effect of a search request.
        _store = ChunkStore.from_config(_config, read_only=True)
    if _embedder is None:
        _embedder = make_embedder(
            provider=_config.embedder.provider,
            model=_config.embedder.model,
            dim=_config.embedder.dim,
        )
    if _retriever is None:
        _retriever = Retriever(
            store=_store,
            embedder=_embedder,
            reference_patterns=_config.compiled_references(),
        )
    return _store, _embedder, _retriever, _config


def _format_chunk_block(idx: int, c: StoredChunk) -> str:
    return format_chunk_block(idx, c)


def _log_path() -> Path | None:
    """Where to append served queries, or None when logging is off.

    Defaults beside the store rather than the cwd: an MCP server's working
    directory is whatever launched it, and a log that lands somewhere
    different on each launch is worse than no log.
    """
    from corpus.query_log import configured_log_path

    return configured_log_path(_config)


def _record(
    tool: str, query: str, chunks: list[StoredChunk], elapsed_ms: float, **extra: object
) -> None:
    """Append one served query. Never raises -- see corpus.query_log."""
    record_query(
        _log_path(),
        tool=tool,
        query=query,
        chunks=chunks,
        elapsed_ms=elapsed_ms,
        include_results=_config is not None and _config.query_log.include_results,
        **extra,
    )


@mcp.tool(description="Semantic + BM25 hybrid search over the corpus. Returns top-K chunks.")
@safe_tool
async def search_knowledge(
    query: Annotated[str, Field(description="Natural-language question or search terms")],
    source_types: Annotated[
        list[str] | str | None,
        Field(description="Optional source-type filter. Single string or list."),
    ] = None,
    top_k: Annotated[int, Field(description="How many chunks to return (1-20).", ge=1, le=20)] = 5,
) -> str:
    filter_sources: list[str] | None
    if source_types is None:
        filter_sources = None
    elif isinstance(source_types, str):
        filter_sources = [source_types]
    else:
        filter_sources = [str(s) for s in source_types]
    store, _, retriever, cfg = _init()
    if filter_sources:
        # A source type that does not exist returns nothing and is
        # indistinguishable from a genuine miss, so a typo or a guessed name
        # reads as "the corpus has nothing on this" — the caller then stops
        # looking. Naming the real ones costs one cheap query and turns a
        # dead end into a correction.
        stats = await asyncio.to_thread(store.stats)
        known = set(stats["by_source"])
        unknown = [s for s in filter_sources if s not in known]
        if unknown:
            return (
                f"Unknown source type(s): {', '.join(sorted(unknown))}. "
                f"This corpus has: {', '.join(sorted(known))}."
            )
    with QueryTimer() as timer:
        result = await asyncio.to_thread(
            retriever.query,
            query,
            top_k,
            filter_sources,
            # Documented in configuration.md, the README and
            # corpus.toml.example -- and read by nobody. This call passed its
            # arguments positionally, so the cap could not arrive at all and
            # every MCP search used the hardcoded default however corpus.toml
            # was written. This is the path that actually serves Claude.
            max_per_source_type=cfg.retriever.max_per_source_type,
        )
    chunks = result.chunks
    _record(
        "search_knowledge",
        query,
        chunks,
        timer.elapsed_ms,
        top_k=top_k,
        filters=filter_sources,
    )
    if not chunks:
        return f"No results for: {query}"
    return UNTRUSTED_PREFIX + "\n\n---\n\n".join(
        _format_chunk_block(i, c) for i, c in enumerate(chunks, 1)
    )


@mcp.tool(
    description=(
        "Fetch every chunk for a specific document (source_type + source_key). "
        "Use after search_knowledge to read a full doc in order."
    )
)
@safe_tool
async def get_doc(
    source_type: Annotated[str, Field(description="Source type from corpus.toml, e.g. 'notes'")],
    source_key: Annotated[str, Field(description="Document identifier as stored")],
) -> str:
    store, _, _, _ = _init()
    chunks = await asyncio.to_thread(store.get_by_source_key, source_type, source_key)
    if not chunks:
        return f"No chunks found for {source_type}:{source_key}"
    return UNTRUSTED_PREFIX + "\n\n---\n\n".join(
        _format_chunk_block(i, c) for i, c in enumerate(chunks, 1)
    )


@mcp.tool(
    description=(
        "Chase references from a chunk. Returns: (a) siblings = other chunks of "
        "the same document; (b) references = chunks matched by the [[references]] "
        "patterns in corpus.toml; (c) parent = if metadata.extra.parent is set, "
        "the parent's chunks."
    )
)
@safe_tool
async def expand_context(
    chunk_id: Annotated[str, Field(description="Chunk ID from a prior result")],
    include: Annotated[list[str] | str | None, Field(description="Subset of [siblings, references, parent]")] = None,
    max_results: Annotated[int, Field(description="Cap on returned chunks (1-30).", ge=1, le=30)] = 10,
) -> str:
    if include is None:
        include_types = ["siblings", "references", "parent"]
    elif isinstance(include, str):
        include_types = [include]
    else:
        include_types = list(include)
    _, _, retriever, _ = _init()
    chunks = await asyncio.to_thread(retriever.expand_context, chunk_id, include_types, max_results)
    if not chunks:
        return f"No related chunks found for {chunk_id}."
    return UNTRUSTED_PREFIX + "\n\n---\n\n".join(
        _format_chunk_block(i, c) for i, c in enumerate(chunks, 1)
    )


@mcp.tool(description="Search results reordered chronologically instead of by relevance.")
@safe_tool
async def timeline(
    topic: Annotated[str, Field(description="Topic to trace through time")],
    top_k: Annotated[int, Field(description="Events to return (1-50)", ge=1, le=50)] = 15,
    since: Annotated[str | None, Field(description="ISO date lower bound")] = None,
    until: Annotated[str | None, Field(description="ISO date upper bound")] = None,
) -> str:
    _, _, retriever, _ = _init()
    chunks = await asyncio.to_thread(retriever.timeline, topic, top_k, since, until, None)
    if not chunks:
        return f"No timeline events for: {topic}"
    out: list[str] = []
    for i, c in enumerate(chunks, 1):
        ts = (c.metadata or {}).get("updated_at") or (c.metadata or {}).get("created_at") or "?"
        ts_short = ts[:10] if isinstance(ts, str) else "?"
        out.append(f"[{i}] {ts_short} — {c.source_type}:{c.source_key}\n{c.title or ''}\n{c.content[:600]}")
    return UNTRUSTED_PREFIX + "\n\n---\n\n".join(out)


@mcp.tool(description="Chunks updated within the last N days, newest first.")
@safe_tool
async def recent_activity(
    days: Annotated[int, Field(description="How many days back (1-365)", ge=1, le=365)] = 7,
    source_types: Annotated[list[str] | str | None, Field(description="Optional source-type filter")] = None,
    top_k: Annotated[int, Field(description="How many docs (1-50)", ge=1, le=50)] = 15,
) -> str:
    filter_sources: list[str] | None
    if source_types is None:
        filter_sources = None
    elif isinstance(source_types, str):
        filter_sources = [source_types]
    else:
        filter_sources = [str(s) for s in source_types]
    _, _, retriever, _ = _init()
    chunks = await asyncio.to_thread(retriever.recent_activity, days, filter_sources, top_k)
    if not chunks:
        return f"No activity in the last {days} days"
    return UNTRUSTED_PREFIX + "\n\n---\n\n".join(
        _format_chunk_block(i, c) for i, c in enumerate(chunks, 1)
    )


@mcp.tool(
    description=(
        "Return the Claude-generated summary for a document. Only available "
        "if you've run `corpus-summarize` on the source type."
    )
)
@safe_tool
async def get_summary(
    source_type: Annotated[str, Field(description="Source type, e.g. 'notes'")],
    source_key: Annotated[str, Field(description="Document identifier")],
) -> str:
    store, _, _, _ = _init()
    summary = await asyncio.to_thread(store.get_summary, source_type, source_key)
    if not summary:
        return f"No summary cached for {source_type}:{source_key}. Run `corpus-summarize --source {source_type}` to generate one."
    # Marked untrusted like every other tool that returns corpus-derived
    # text. A summary is not raw corpus content, which is why it was missed —
    # but it is GENERATED FROM corpus content, so a document carrying
    # adversarial instructions can steer what the summary says, and the
    # summary is then handed to a model as if it were trustworthy.
    return (
        UNTRUSTED_PREFIX
        + f"{source_type}:{source_key} (model={summary['model']}, "
        + f"generated={summary['generated_at']}):\n\n{summary['summary']}"
    )


@mcp.tool(description="Total chunks and per-source counts. Health check.")
@safe_tool
async def corpus_stats() -> str:
    store, _, _, _ = _init()
    stats = await asyncio.to_thread(store.stats)
    coverage = await asyncio.to_thread(store.context_coverage)
    lines = [f"Total chunks: {stats['total']:,}"]
    for src, n in sorted(stats["by_source"].items()):
        # Contextual-Retrieval coverage matters to a caller deciding how much
        # to trust a thin result set: an uncontextualized source retrieves
        # noticeably worse on fragments, and that is a property of the index,
        # not of the query. Shown only where some coverage exists, so an
        # install that has never run `corpus-contextualize` sees no noise.
        with_context = coverage.get(src, {}).get("with_context", 0)
        if with_context:
            pct = with_context / n * 100 if n else 0.0
            lines.append(f"  {src}: {n:,} ({with_context:,} contextualized, {pct:.0f}%)")
        else:
            lines.append(f"  {src}: {n:,}")
    return "\n".join(lines)


# The env vars each embedder reads its key from; any one will do. `hash`
# needs none -- it is the offline embedder for tests and dry runs.
_PROVIDER_KEYS: dict[str, tuple[str, ...]] = {
    "voyage": ("VOYAGE_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "hash": (),
}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="corpus MCP stdio server (spawned by Claude Code)"
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Absolute path to corpus.toml. RECOMMENDED when wiring via "
            "~/.claude.json — Claude Code spawns the server from an arbitrary "
            "CWD, so a relative path won't find the right config. Defaults to "
            "./corpus.toml in the current working directory."
        ),
    )
    args = parser.parse_args()

    # Cache the config path so _init() loads from the same place when MCP
    # tools are invoked.
    global _config_path_override
    _config_path_override = args.config

    # Resolve credentials predictably, now that --config is known: an
    # already-set env var wins, then a .env beside --config, then a .env in
    # the cwd. This library never reads secrets out of its own source tree
    # (see corpus.credentials) -- a checkout of corpus holding a live API key
    # is not a precondition for using it.
    resolve_dotenv(args.config)

    try:
        config = CorpusConfig.load(args.config)
    except ConfigError as e:
        logger.error("%s", e)
        sys.exit(2)

    keys = _PROVIDER_KEYS.get(config.embedder.provider)
    if keys is None:
        logger.error("Unknown embedder provider in corpus.toml: %s", config.embedder.provider)
        sys.exit(2)
    if keys and not any(os.environ.get(k) for k in keys):
        logger.error(
            "Embedder provider '%s' requires one of: %s. corpus looked for a "
            ".env: %s. Set the variable in your shell, or add it to one of "
            "those files.",
            config.embedder.provider, ", ".join(keys), describe_search(args.config),
        )
        sys.exit(2)

    if not Path(config.db_path).exists():
        logger.warning(
            "DB at %s does not exist. Run `corpus-ingest --source <name>` to create it.",
            config.db_path,
        )
    logger.info("starting corpus MCP server (stdio)")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
