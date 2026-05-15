# Configuration reference

Everything that varies between deployments lives in `corpus.toml`. This doc is the canonical reference for every section.

`corpus-init` generates a starter `corpus.toml` for you interactively. Edit by hand from there.

## Top-level structure

```toml
[corpus]       # Where the DB lives.
[embedder]     # Which provider, model, and dim.
[retriever]    # Defaults for the query pipeline.
[[sources]]    # Repeatable. Each = one connector instance.
[[references]] # Optional. Reference patterns for expand_context.
```

## `[corpus]` — global

| Setting | Type | Default | Notes |
|---|---|---|---|
| `db_path` | path | `./corpus.db` | Where SQLite stores everything. Relative paths resolve from the corpus.toml's directory. |

```toml
[corpus]
db_path = "./corpus.db"
```

## `[embedder]` — provider, model, and dim

| Setting | Type | Default | Notes |
|---|---|---|---|
| `provider` | str | `voyage` | One of `voyage`, `gemini`. See the comparison below. |
| `model` | str | `voyage-3-large` | Model name. Must match what the provider supports. |
| `dim` | int | `1024` | Embedding dimension. MUST match the model's output; guarded at startup. |

```toml
[embedder]
provider = "voyage"
model = "voyage-3-large"
dim = 1024
```

For Gemini:

```toml
[embedder]
provider = "gemini"
model = "gemini-embedding-001"
dim = 1536    # also valid: 768, 3072 — Matryoshka representation
```

### Choosing a provider

| | Voyage `voyage-3-large` | Gemini `gemini-embedding-001` |
|---|---|---|
| **Quality (MTEB retrieval)** | Top of leaderboard | Comparable, ~1-3 points lower |
| **Free tier** | 200M tokens lifetime | ~1,500 requests/day rolling |
| **Card required for free tier** | Yes (to lift rate limits) | No |
| **Default dim** | 1024 (fixed) | 768 / 1536 / 3072 (Matryoshka) |
| **Asymmetric retrieval** | `input_type=document/query` | `task_type=RETRIEVAL_DOCUMENT/QUERY` |
| **Signup** | https://dash.voyageai.com/ | https://aistudio.google.com/apikey |

**Pick Voyage** if you want best-in-class retrieval quality and don't mind putting a card on file (you won't be charged below 200M tokens — the card just unlocks normal rate limits).

**Pick Gemini** if you don't want a card on file, your corpus fits under ~1,500 requests/day, or you want Matryoshka dim selection (768 to save space, 3072 for max recall).

For Gemini, the `dim` choice is a real knob: `768` for tight storage, `1536` recommended balanced default, `3072` for highest recall on hard corpora. You can't change it after ingest — the dim guard refuses (see below).

### The dim guard

The DB stores `embedding_dim` in a `schema_meta` table on first open. If you change `dim` later (different model, different Matryoshka cut), the next open raises `EmbeddingDimMismatch` rather than silently corrupting retrieval. To switch: `corpus-reset --all` then re-ingest.

### Adding a provider

The embedder layer is a Protocol (`src/corpus/embedder/base.py`). To add OpenAI, Cohere, Ollama-local, etc: write a class implementing `embed_documents` + `embed_query`, register it in `embedder/factory.py`'s `make_embedder` dispatch. ~100 lines, following the Voyage/Gemini pattern.

## `[retriever]` — query-time defaults

| Setting | Type | Default | Notes |
|---|---|---|---|
| `top_k` | int | `5` | Default result count when callers don't override. 1-20. |
| `max_per_source_type` | int? | `3` | Diversity cap; no single source type fills more than this. `null` to disable. |
| `hybrid` | bool | `true` | Enable BM25+vector fusion. Disable for vector-only baseline. |

```toml
[retriever]
top_k = 5
max_per_source_type = 3
hybrid = true
```

These are *defaults*. CLI flags (`-k`, `--no-hybrid`) and MCP tool args override per call.

## `[[sources]]` — repeatable

Each `[[sources]]` block configures one connector instance.

| Setting | Type | Notes |
|---|---|---|
| `name` | str | Free-form identifier (`^[a-z][a-z0-9_]*$`). Used as the chunk's `source_type` everywhere. |
| `type` | str | Which built-in connector to use. One of `markdown`, `text`, `pdf`, `html`. |
| `path` | path | Where to find the source files. `~` expands to home. |
| `glob` | str? | File pattern relative to `path`. Default depends on `type`. |

```toml
[[sources]]
name = "notes"
type = "markdown"
path = "~/Documents/notes"
glob = "**/*.md"

[[sources]]
name = "papers"
type = "pdf"
path = "~/Documents/papers"

[[sources]]
name = "bookmarks"
type = "html"
path = "~/exports/saved-articles"
```

Multiple sources of the same `type` are fine — they just need different `name`s.

### Built-in `type` options

| `type` | Default glob | Extra needed |
|---|---|---|
| `markdown` | `**/*.md` | none |
| `text` | `**/*.txt` | none |
| `pdf` | `**/*.pdf` | `pip install corpus-rag[pdf]` |
| `html` | `**/*.html` | `pip install corpus-rag[html]` |

Adding a new `type` means registering a new connector — see [`adding_a_source.md`](adding_a_source.md).

## `[[references]]` — optional

Reference patterns drive two things:

1. **`expand_context`**: when a chunk's content contains a match, the retriever fetches chunks of the matching source.
2. **BM25 auto-weighting**: when a user's query contains a match, `fts_weight` jumps from 0.25 to 1.0 (BM25-heavy).

| Setting | Type | Notes |
|---|---|---|
| `pattern` | str (regex) | Compiled at startup. Use raw-string form in TOML (single-quoted). |
| `source_type` | str | Which source the matched IDs live in. Must match an existing `[[sources]] name`. |
| `description` | str? | Free-form note; not used at runtime. |

```toml
[[references]]
pattern = '\b[A-Z]{2,}-\d+\b'
source_type = "tickets"
description = "Jira-style ticket keys (PROJ-123)"

[[references]]
pattern = '\b[a-z][a-z0-9_-]+#\d+\b'
source_type = "pulls"
description = "GitHub PR refs (repo#NNN)"
```

If `[[references]]` is empty or absent:

- `expand_context` mode `references` returns nothing — siblings + parent only.
- `_auto_fts_weight` falls back to a generic heuristic (looks for backtick-quoted code or quoted phrases) before defaulting to 0.25.

## Environment variables

| Var | Required? | Used by | Notes |
|---|---|---|---|
| `VOYAGE_API_KEY` | when `provider="voyage"` | embedder | https://dash.voyageai.com/ |
| `GEMINI_API_KEY` | when `provider="gemini"` | embedder | https://aistudio.google.com/apikey (or `GOOGLE_API_KEY`) |
| `ANTHROPIC_API_KEY` | when running `corpus-summarize` | summarizer | https://console.anthropic.com/ |

`corpus` uses `python-dotenv` — set these in `.env` (gitignored), in your shell, or in your MCP server's environment block.

## Validation

`CorpusConfig.load()` validates the file via Pydantic at startup. Common errors:

- **Source name pattern**: `name = "BAD-NAME"` → `ValidationError`. Names must be lowercase identifiers.
- **Reference source_type matches no source**: pattern validation passes, but `expand_context` will find no chunks. Not an error; documented behavior.
- **Missing `[[sources]]`**: `corpus-ingest` errors with "No source named X" when called.
- **Embedder dim mismatch**: caught at `ChunkStore.__init__` open time, not config load.
