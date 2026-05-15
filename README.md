# corpus

Single-user RAG framework for personal archives. Local-first, MCP-native.

Point it at any directory of markdown files (notes, Obsidian vault, exported docs) and get:

- Semantic + BM25 hybrid search with auto-tuned fusion weights
- Source-diversity-aware retrieval (no single doc floods top-K)
- Multi-hop reference chasing via `expand_context`
- Optional cross-encoder re-ranker (local, BGE)
- Optional per-document Claude-Haiku summaries
- Eight MCP tools wired into Claude Code over stdio

**Stack:** Python 3.12 + uv • Voyage embeddings • SQLite + sqlite-vec • FastMCP. No AWS, no Docker, no Terraform.

---

## Why this exists

A useful personal knowledge system shouldn't require running a vector database service, paying for a SaaS, or trusting your archive to someone else's cloud. `corpus` runs entirely on your machine: one Python process, one SQLite file, one MCP server. The whole system is small enough to read end-to-end.

You add a `corpus.toml`, point it at your data, run `corpus-ingest`, and Claude Code can search it.

## Quick start

```sh
# 1. Clone and install
git clone <this repo> corpus
cd corpus
uv sync

# 2. Set up your config
cp corpus.toml.example corpus.toml
cp .env.example .env
# Edit corpus.toml: point [[sources]] at your data
# Edit .env: paste your VOYAGE_API_KEY (free tier covers ~200M tokens)

# 3. Run a first ingest
uv run corpus-ingest --source notes -v

# 4. Try it from the CLI
uv run corpus-query "the question you wish you could ask your archive"

# 5. Wire it to Claude Code (see "MCP server" below)
```

The example corpus in `examples/sample_corpus/` lets you try the full pipeline before pointing it at real data — just run `corpus-ingest --source sample` after copying `examples/corpus.toml.example` to `corpus.toml`.

## Configuration

Everything that varies between deployments lives in `corpus.toml`. See `corpus.toml.example` for the annotated template.

```toml
[corpus]
db_path = "./corpus.db"

[embedder]
provider = "voyage"
model = "voyage-3-large"
dim = 1024                # must match the model's output dim

[retriever]
top_k = 5
max_per_source_type = 3   # diversity cap
hybrid = true             # vector + BM25 via RRF

[[sources]]
name = "notes"            # free-form; used as source_type everywhere
type = "markdown"         # which built-in connector to use
path = "~/Documents/notes"
glob = "**/*.md"

[[references]]
# Optional. When set, `expand_context` chases these patterns across docs and
# the BM25 weight auto-tunes higher when the user's query contains a match.
pattern = '\b[A-Z]{2,}-\d+\b'
source_type = "tickets"
description = "Jira-style ticket keys"
```

Schema hazard: changing `embedder.dim` after data has been ingested would silently corrupt retrieval. `corpus` validates the dim against the existing schema at startup and refuses to proceed on mismatch.

## MCP server (Claude Code integration)

Wire `corpus` into Claude Code by adding to `~/.claude.json`:

```json
{
  "mcpServers": {
    "corpus": {
      "type": "stdio",
      "command": "uv",
      "args": ["--directory", "/path/to/your/corpus", "run", "corpus-mcp"],
      "env": {}
    }
  }
}
```

Eight tools exposed:

| Tool | Purpose |
|---|---|
| `search_knowledge` | Hybrid BM25+vector search with dedupe + diversity |
| `expand_context` | Chase references from a chunk — siblings, cited docs, parent |
| `get_doc` | Pull every chunk of a specific document |
| `timeline` | Search results reordered chronologically |
| `who_did_what` | Chunks involving a specific person (needs author metadata) |
| `recent_activity` | Chunks updated in the last N days |
| `get_summary` | Cached Claude-Haiku summary (after running `corpus-summarize`) |
| `corpus_stats` | Health check — total chunks + per-source counts |

The **investigation pattern** is the high-leverage flow: Claude calls `search_knowledge` to find entry points, then `expand_context` on the top result to pull in adjacent material (other chunks of the same doc, referenced doc IDs, parent links), then synthesizes from the full picture.

## CLI reference

```sh
corpus-ingest --source notes -v              # ingest one source
corpus-ingest --all                          # ingest everything in corpus.toml
corpus-query "your question" -k 10           # ad-hoc search
corpus-query "question" --source notes       # source-filtered
corpus-query "question" --rerank             # local BGE reranker (opt-in)
corpus-eval --queries tests/eval_queries.py  # recall@k against your queries
corpus-summarize --source notes --dry-run    # estimate Haiku spend
corpus-summarize --source notes              # run it
corpus-mcp                                   # stdio MCP server (Claude spawns)
```

## Built-in connectors

| `type` | Default glob | Extra needed | Notes |
|---|---|---|---|
| `markdown` | `**/*.md` | — | YAML frontmatter parsed (`title`, `id`, `url`, dates) |
| `text` | `**/*.txt` | — | Plain text; title from filename stem |
| `pdf` | `**/*.pdf` | `pip install corpus-rag[pdf]` | Uses `pypdf`. Scanned PDFs need OCR first. |
| `html` | `**/*.html` | `pip install corpus-rag[html]` | Uses `trafilatura` for boilerplate-stripped main-content extraction |

## Adding a new source type

For Slack exports, JSON dumps, an internal API archive, EPUB books — write your own connector. The shipped connectors at `src/corpus/connectors/` are the reference implementations; copy any of them as a starting point.

See [`docs/adding_a_source.md`](docs/adding_a_source.md) for the walkthrough with a worked JSON-files example.

## Eval

`corpus-eval` runs hand-written known-answer queries against the live corpus and reports recall@K. It's a regression signal — run it after changing chunking, switching embedders, or tweaking retrieval.

Write your queries in a Python module exporting `EVAL_QUERIES` (the shipped `tests/eval_queries.py` is a commented placeholder template). Each `EvalQuery` has a `query`, a list of `expected_keys` (any one in top-K = pass), an optional `source_filter`, and a `note`. Tips: paraphrase away from doc titles to stress semantic retrieval; list multiple `expected_keys` when several docs are valid answers; add a few negative queries (empty `expected_keys`) to confirm the corpus correctly fails on absent topics.

```sh
corpus-eval --top-k 5            # baseline
corpus-eval --rerank             # with the BGE reranker
corpus-eval --no-hybrid          # vector-only baseline
```

## Benchmarking

`corpus-benchmark` measures per-stage retrieval latency (embed / vector / FTS / fusion / dedupe) with p50/p95/p99 + throughput.

```sh
corpus-benchmark --runs 20                      # latency profile
corpus-benchmark --queries tests/eval_queries.py
corpus-benchmark --compare voyage gemini        # embed-latency A/B
corpus-benchmark --json out.json
```

Typical profile on an M-series Mac, few-thousand-chunk corpus: `embed` dominates at 100–300ms (provider API round-trip), while `vector_search` / `fts_search` / `fusion` / `dedupe` are collectively under ~5ms. The optimization lever is "fewer or concurrent embed calls," not "faster SQLite." If `vector_search` exceeds ~50ms you've outgrown brute-force `vec0` (~100K chunks) and want HNSW indexing.

`--compare` measures embedder-API latency only — it does **not** compare retrieval quality, because two providers' vectors aren't comparable against one DB. For quality, ingest each provider into its own corpus and run `corpus-eval` against each.

## Project layout

```
src/corpus/
  types.py              # Pydantic models (Chunk, ChunkMetadata, SourceDocument, ChunkKind)
  config.py             # corpus.toml loader
  retriever.py          # query → embed → hybrid search → dedupe → diversity → rerank
  ingester.py           # orchestrator: connector → chunker → embed → upsert
  mcp_server.py         # FastMCP stdio server (8 tools)
  db/sqlite.py          # ChunkStore — UPSERT, FTS5, vec0, summaries, embedding-dim guard
  embedder/
    base.py                    # Embedder protocol
    voyage.py  gemini.py       # Provider implementations
    factory.py                 # make_embedder(provider, model) dispatch
  reranker/local.py     # BGE cross-encoder (optional, lazy-loaded)
  summarizer/anthropic_summarizer.py  # Claude Haiku per-doc summaries (optional)
  connectors/
    base.py                    # Connector + Chunker protocols
    markdown.py                # YAML-frontmatter markdown (default ref impl)
    text.py                    # Plain .txt files
    pdf.py                     # PDFs via pypdf (optional [pdf] extra)
    html.py                    # HTML via trafilatura (optional [html] extra)
    registry.py                # Maps `type` string → (Connector, Chunker) factory
  chunkers/
    markdown.py                # Shared markdown chunking logic
  util/
    hash.py        scrub.py        # SHA-256, deterministic chunk IDs, secret regex
    dedup.py       rrf.py          # Near-dup fingerprinting, reciprocal rank fusion
    tokens.py                      # Char-based token heuristic for chunking decisions
  cli/
    init.py  ingest.py  query.py  eval.py        # 9 console scripts
    summarize.py  benchmark.py  list_sources.py  reset.py
tests/
docs/
examples/sample_corpus/    # Synthetic markdown for try-before-you-config
```

## Tests

```sh
uv run pytest tests/ -q
```

Unit tests cover the storage layer, embedder (mocked), retriever (dedupe/diversity/references), all four connectors (markdown / text / pdf / html), RRF, scrub, dedup, config loader, the init wizard, factory dispatch, and benchmark helpers. Run `uv run pytest tests/ -q` to see the current count.

## Documentation

| Doc | What it covers |
|---|---|
| [`docs/configuration.md`](docs/configuration.md) | Every `corpus.toml` setting + env var, including the Voyage-vs-Gemini embedder choice |
| [`docs/mcp_integration.md`](docs/mcp_integration.md) | Claude Code wiring + all 8 tools, the investigation pattern |
| [`docs/adding_a_source.md`](docs/adding_a_source.md) | Walkthrough for writing a custom connector |
| [`docs/troubleshooting.md`](docs/troubleshooting.md) | Common problems and the actual fixes |

Architecture overview, benchmarking, and eval methodology are covered inline in this README (sections above).

## License

MIT — see [LICENSE](LICENSE).
