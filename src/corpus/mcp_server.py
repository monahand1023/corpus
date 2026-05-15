"""MCP stdio server exposing the corpus to Claude Code.

Eight tools: search, expand_context, timeline, who_did_what, recent_activity,
get_summary, get_doc, corpus_stats.

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

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import Field

from corpus.config import CorpusConfig
from corpus.db.sqlite import ChunkStore
from corpus.embedder.base import Embedder
from corpus.embedder.factory import make_embedder
from corpus.retriever import Retriever

# .env from CWD, then from this file's parent (handy when Claude Code spawns
# from arbitrary cwds).
load_dotenv()
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("corpus.mcp")

mcp = FastMCP(
    name="corpus",
    instructions=(
        "Search a personal archive. Use `search_knowledge` for natural-language "
        "questions; `expand_context` to chase references from a chunk; "
        "`get_doc` to pull every chunk of a specific document; "
        "`timeline` / `who_did_what` / `recent_activity` for non-semantic "
        "queries; `get_summary` for a Claude-generated one-paragraph summary "
        "of a doc (only available after running `corpus-summarize`)."
    ),
)

_store: ChunkStore | None = None
_embedder: Embedder | None = None
_retriever: Retriever | None = None
_config: CorpusConfig | None = None


def _init() -> tuple[ChunkStore, Embedder, Retriever, CorpusConfig]:
    global _store, _embedder, _retriever, _config
    if _config is None:
        _config = CorpusConfig.load()
    if _store is None:
        _store = ChunkStore(_config.db_path, embedding_dim=_config.embedder.dim)
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


def _format_chunk_block(idx: int, c: object) -> str:
    distance = getattr(c, "distance", None)
    distance_str = f" d={distance:.4f}" if distance is not None else ""
    header = f"[{idx}] {c.source_type}:{c.source_key}{distance_str}"
    title = f"\nTitle: {c.title}" if c.title else ""
    url = f"\nURL: {c.url}" if c.url else ""
    return f"{header}{title}{url}\n\n{c.content}"


@mcp.tool(description="Semantic + BM25 hybrid search over the corpus. Returns top-K chunks.")
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
    _, _, retriever, _ = _init()
    result = await asyncio.to_thread(
        retriever.query, query, top_k, filter_sources
    )
    chunks = result.chunks
    if not chunks:
        return f"No results for: {query}"
    return "\n\n---\n\n".join(_format_chunk_block(i, c) for i, c in enumerate(chunks, 1))


@mcp.tool(
    description=(
        "Fetch every chunk for a specific document (source_type + source_key). "
        "Use after search_knowledge to read a full doc in order."
    )
)
async def get_doc(
    source_type: Annotated[str, Field(description="Source type from corpus.toml, e.g. 'notes'")],
    source_key: Annotated[str, Field(description="Document identifier as stored")],
) -> str:
    store, _, _, _ = _init()
    chunks = await asyncio.to_thread(store.get_by_source_key, source_type, source_key)
    if not chunks:
        return f"No chunks found for {source_type}:{source_key}"
    return "\n\n---\n\n".join(_format_chunk_block(i, c) for i, c in enumerate(chunks, 1))


@mcp.tool(
    description=(
        "Chase references from a chunk. Returns: (a) siblings = other chunks of "
        "the same document; (b) references = chunks matched by the [[references]] "
        "patterns in corpus.toml; (c) parent = if metadata.extra.parent is set, "
        "the parent's chunks."
    )
)
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
    return "\n\n---\n\n".join(_format_chunk_block(i, c) for i, c in enumerate(chunks, 1))


@mcp.tool(description="Search results reordered chronologically instead of by relevance.")
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
    return "\n\n---\n\n".join(out)


@mcp.tool(
    description=(
        "Chunks where a specific person is author / assignee / reporter. "
        "Only works for source types whose connectors populate those fields."
    )
)
async def who_did_what(
    person: Annotated[str, Field(description="Person name fragment")],
    top_k: Annotated[int, Field(description="How many unique docs (1-50)", ge=1, le=50)] = 15,
    since: Annotated[str | None, Field(description="ISO date lower bound")] = None,
    until: Annotated[str | None, Field(description="ISO date upper bound")] = None,
) -> str:
    _, _, retriever, _ = _init()
    chunks = await asyncio.to_thread(retriever.who_did_what, person, top_k, since, until)
    if not chunks:
        return f"No chunks involving '{person}'"
    return "\n\n---\n\n".join(_format_chunk_block(i, c) for i, c in enumerate(chunks, 1))


@mcp.tool(description="Chunks updated within the last N days, newest first.")
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
    return "\n\n---\n\n".join(_format_chunk_block(i, c) for i, c in enumerate(chunks, 1))


@mcp.tool(
    description=(
        "Return the Claude-generated summary for a document. Only available "
        "if you've run `corpus-summarize` on the source type."
    )
)
async def get_summary(
    source_type: Annotated[str, Field(description="Source type, e.g. 'notes'")],
    source_key: Annotated[str, Field(description="Document identifier")],
) -> str:
    store, _, _, _ = _init()
    summary = await asyncio.to_thread(store.get_summary, source_type, source_key)
    if not summary:
        return f"No summary cached for {source_type}:{source_key}. Run `corpus-summarize --source {source_type}` to generate one."
    return f"{source_type}:{source_key} (model={summary['model']}, generated={summary['generated_at']}):\n\n{summary['summary']}"


@mcp.tool(description="Total chunks and per-source counts. Health check.")
async def corpus_stats() -> str:
    store, _, _, _ = _init()
    stats = await asyncio.to_thread(store.stats)
    lines = [f"Total chunks: {stats['total']:,}"]
    for src, n in sorted(stats["by_source"].items()):
        lines.append(f"  {src}: {n:,}")
    return "\n".join(lines)


def main() -> None:
    try:
        config = CorpusConfig.load()
    except FileNotFoundError as e:
        logger.error("%s", e)
        sys.exit(2)

    expected_key = {
        "voyage": "VOYAGE_API_KEY",
        "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    }.get(config.embedder.provider)
    if expected_key is None:
        logger.error("Unknown embedder provider in corpus.toml: %s", config.embedder.provider)
        sys.exit(2)
    keys = expected_key if isinstance(expected_key, tuple) else (expected_key,)
    if not any(os.environ.get(k) for k in keys):
        logger.error(
            "Embedder provider '%s' requires one of: %s. Set in .env before starting.",
            config.embedder.provider, ", ".join(keys),
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
