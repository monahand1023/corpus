# Configuration reference

Everything that varies between deployments lives in `corpus.toml`. This doc is the canonical reference for every section.

`corpus-init` generates a starter `corpus.toml` for you interactively. Edit by hand from there.

## Top-level structure

```toml
[corpus]       # Where the DB lives.
[embedder]     # Which provider, model, and dim.
[retriever]    # Defaults for the query pipeline.
[pruning]      # Orphan-deletion blast-radius guard.
[performance]  # SQLite memory tuning (cache_size/mmap_size/temp_store).
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
| `model` | str | `voyage-4-large` | Model name. Must match what the provider supports. |
| `dim` | int | `1024` | Embedding dimension. MUST match the model's output; guarded at startup. |

```toml
[embedder]
provider = "voyage"
model = "voyage-4-large"
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

| | Voyage `voyage-4-large` | Gemini `gemini-embedding-001` |
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

### Why embedders are optional extras

Neither embedder ships in the base install. You pick one at install time:

```sh
pip install 'corpus-rag[voyage]'   # Voyage (recommended, best quality)
pip install 'corpus-rag[gemini]'   # Gemini (free tier, no card)
```

An embedder is **mandatory** to ingest or query — the extra just makes *which*
one explicit. Bare `pip install corpus-rag` gives you the minimal,
provider-agnostic base; `corpus-ingest` will then tell you to add an embedder
extra.

**Why it's split out.** `voyageai` pulls in a large transitive dependency tree
(`langchain-core`, `pillow`, `ffmpeg-python`, `future`) that `corpus` itself
never imports. Baking it into the base meant every install — including
Gemini-only users — carried that weight and its supply-chain surface. Making it
opt-in keeps the base install small and provider-agnostic, matching the "one
small local-first process" design.

**Pros:** smaller, faster base install; smaller dependency & CVE surface;
Gemini users don't pay for Voyage's tree; symmetric provider design.

**Cons / tradeoffs:** a bare `pip install corpus-rag` no longer works end-to-end
out of the box — you must add an embedder extra, one extra thing to remember.
And if you choose `[voyage]`, its transitive tree still comes along; that's
`voyageai`'s own dependency set, not something `corpus` can strip.

**Guidance:** most people want `[voyage]` (quality) or `[gemini]` (free, no
card). The bare base is for advanced users wiring a custom embedder (see
"Adding a provider" below).

### The dim guard

The DB stores `embedding_dim` in a `schema_meta` table on first open. If you change `dim` later (different model, different Matryoshka cut), the next open raises `EmbeddingDimMismatch` rather than silently corrupting retrieval. To switch: `corpus-reset --all` then re-ingest.

### Adding a provider

The embedder layer is a Protocol (`src/corpus/embedder/base.py`). To add OpenAI, Cohere, Ollama-local, etc: write a class implementing `embed_documents` + `embed_query`, register it in `embedder/factory.py`'s `make_embedder` dispatch. ~100 lines, following the Voyage/Gemini pattern.

## `[retriever]` — query-time defaults

| Setting | Type | Default | Notes |
|---|---|---|---|
| `top_k` | int | `5` | Default result count when callers don't override. 1-20. |
| `max_per_source_type` | int? | `3` | Diversity cap: how many results each source type gets *before* any backfill. `null` to disable. See below — this is a preference, not a hard ceiling. |
| `hybrid` | bool | `true` | Enable BM25+vector fusion. Disable for vector-only baseline. |

```toml
[retriever]
top_k = 5
max_per_source_type = 3
hybrid = true
```

**`max_per_source_type` is a preference, not a hard ceiling.** It shapes the
front of the result list, then any slots it left empty are filled from the
chunks it displaced, best-scoring first. So a type CAN exceed the cap when the
alternative is returning fewer results than were asked for.

It used to be absolute, and that was wrong: the cap multiplied by the number of
source types became the real ceiling on every answer, whatever `top_k` said. A
store with three source types and a cap of 3 could never return more than 9
results — a tool advertising 30 would quietly return a fraction of them, and a
query whose matches were mostly one type returned barely a handful. A short
result now means the candidate pool was genuinely exhausted, which is the only
honest reason to return fewer than were requested.

If you want a hard ceiling per type, lower `top_k` instead — that is the knob
that means "return no more than this".

These are *defaults*. CLI flags (`-k`, `--no-hybrid`) and MCP tool args override per call.

## `[pruning]` — orphan-deletion blast-radius guard

| Setting | Type | Default | Notes |
|---|---|---|---|
| `max_orphan_ratio` | float | `0.20` | Refuse to prune a source when the chunks about to be deleted exceed this fraction of its existing chunk count. |
| `min_chunks_for_guard` | int | `50` | Only enforce the ratio above this many existing chunks; a small/new source can legitimately cross 20% on a handful of deletions. |

```toml
[pruning]
max_orphan_ratio = 0.20
min_chunks_for_guard = 50
```

Every `corpus-ingest` run deletes chunks whose id is no longer present in what the connector just yielded ("orphans") — that's how a deleted or renamed file gets removed from the index. The only other protection against this happening for the wrong reason is `failed_files == 0`, which a connector with a genuine bug can still report honestly: one that silently yields fewer documents than it should (e.g. skips files it believes are unchanged) reports zero failures right up until the run that deletes the whole source's index.

This guard adds a second, independent check: if pruning would remove more than `max_orphan_ratio` of a source's existing chunks, `corpus-ingest` refuses — it logs the actual numbers, leaves the index untouched, and exits non-zero — rather than deleting first and hoping the numbers looked right. Pass `--prune-anyway` when the drop really is a deliberate bulk deletion:

```sh
corpus-ingest --source photos --prune-anyway
```

`--prune-anyway` also overrides the pre-existing `failed_files` gate (see [`troubleshooting.md`](troubleshooting.md)); both are the same "I've reviewed this, delete anyway" escape hatch.

## `[query_log]` — record served queries (opt-in, local)

| Setting | Type | Default | Notes |
|---|---|---|---|
| `enabled` | bool | `false` | Off unless you turn it on. Recording what someone searched for is a decision they make, not one they discover. |
| `path` | string | *(beside the store)* | Defaults to `queries.jsonl` next to `db_path`, NOT the working directory — an MCP server's cwd is whatever launched it, and a log that lands somewhere different on each launch is worse than no log. |
| `include_results` | bool | `true` | Record each result's `(source_type, source_key)`. Set false to log only the query text and timing. |

```toml
[query_log]
enabled = true
```

Appends one JSON object per served query: the query text, each result's
`(source_type, source_key)`, and elapsed milliseconds. Enough to judge later
whether a result was any good, and to rebuild an evaluation set from real
usage — without copying document text into a second place.

**Why you would want this.** Any gold set you write by hand or mine from your
own corpus partly measures its own construction: title-mined queries measure
lexical overlap, and queries written by a model echo the source's vocabulary.
Tuning `fts_weight`, reranking or contextualization against such a set can
produce a clean, confident, wrong answer — a sweep that looks monotone on one
synthesised set can reverse on another. Queries people actually ran are the
only unbiased population, and they can only be collected going forward, so
every day without logging is data that cannot be recovered later.

Writing failures are swallowed after one warning: a full disk or a read-only
directory must never take down retrieval for the sake of a log. Delete the
file to erase the history; nothing transmits it anywhere.

**`corpus-doctor` audits this log automatically** once `enabled = true` — it
resolves the same path the server writes to, so you do not pass `--query-log`.
It reports the count of distinct real queries, separates them from synthetic
and test-fixture entries, and refuses to draw conclusions below 30. That last
part matters: a low count is evidence of low USAGE only once the log is known
clean and known to have been exercised, and is otherwise indistinguishable
from logging that never worked.

## `[performance]` — SQLite memory tuning

| Setting | Type | Default | Notes |
|---|---|---|---|
| `cache_size_mb` | int | `64` | SQLite's own page cache. Measured **no effect** on vector search (see below); kept modest as a safety net for other access patterns. |
| `mmap_size_mb` | int | `1024` | Memory-mapped I/O window. This is the setting that matters — see below. Raise it if your store is bigger than 1GiB. |
| `temp_store_memory` | bool | `true` | Keep SQLite's temp tables in memory instead of a temp file on disk. Measured no effect at corpus's current query sizes; free to leave on. |

```toml
[performance]
cache_size_mb = 64
mmap_size_mb = 1024
temp_store_memory = true
```

SQLite's own default page cache is tiny (~2MB) — fine for a small corpus, not for the hundreds-of-MB-to-multi-GB indexes a real personal archive reaches. Before picking these defaults, each pragma was isolated individually (not just bundled together) on a synthetic 150,000-chunk / 711MB store:

- **`mmap_size_mb` is responsible for effectively the entire speedup**: measured 3.18x-3.29x on vector search once the memory-mapped window covered the whole store, only 1.37x covering about a third of it. `corpus`'s vector index does an exhaustive scan per query (no ANN index), so there's no "hot" subset a bigger page cache could protect — but memory-mapped I/O still helps because it turns every page touch into a direct memory read instead of a `read()` syscall + copy, a win that applies uniformly regardless of caching. **If your `corpus.db` is bigger than 1GiB, raise `mmap_size_mb` to (at least) match it** to get the benefit on the whole store.
- **`cache_size_mb` measured no effect**, confirmed twice: cold reopened connections, and a single persistent connection re-running the same queries six times in a row. Kept modest rather than large, both because it showed no benefit here and because — unlike `mmap_size_mb` — its memory is a real, non-evictable-the-same-way allocation, so a large value has a real cost.
- **`temp_store_memory` measured no effect either.** `EXPLAIN QUERY PLAN` confirms vector search's final sort does use a temp b-tree, but it only orders the small `top_k` result set — trivially cheap on disk or in memory either way at that size.

Fixed defaults rather than scaled off detected physical memory: there's no portable way to read total RAM from the standard library (no `os.sysconf` on Windows), and `corpus` keeps its base install dependency-minimal by design (see "Why embedders are optional extras" above) — pulling in a dependency just for this felt like the wrong tradeoff against a one-line override here. `mmap_size_mb`'s generous default is safe on constrained hardware anyway: it's a ceiling on a lazily-paged-in, evictable mapping, not a memory reservation, so setting it larger than your store costs nothing until pages are actually touched.

## `[[sources]]` — repeatable

Each `[[sources]]` block configures one connector instance.

| Setting | Type | Notes |
|---|---|---|
| `name` | str | Free-form identifier (`^[a-z][a-z0-9_]*$`). Used as the chunk's `source_type` everywhere. |
| `type` | str | Which built-in connector to use. One of `markdown`, `text`, `pdf`, `html`. |
| `path` | path | Where to find the source files. `~` expands to home. |
| `glob` | str? | File pattern relative to `path`. Default depends on `type`. |
| `exclude` | list[str] | Paths this source must not index. See below. |

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

### `exclude` — paths this source must not index

`fnmatch` patterns, matched against **both** the path relative to `path` and
the basename, so all of these work without you having to know which form the
implementation wanted:

```toml
[[sources]]
name = "work"
type = "docx"
path = "~/Work"
exclude = [
  "Old Backup",                  # that folder, and everything under it
  "*Backup*",                    # any folder or file with Backup in its name
  "**/Caches/*",                 # glob form
  "Archive/report.docx",         # one exact file
  "copy of *",                   # by basename
]
```

A directory name matches **exactly**: `"Backup"` is the folder called
`Backup`, not every folder with `Backup` in its name — the same way
`"report.md"` is not `"final report.md"`. Write `*Backup*` for that.

**What this is for.** `corpus-survey duplicates` reports documents whose every
passage also appears under another document, which are the ones that can be
dropped losslessly. On one archive that was 1,119 documents holding 3% of the
index. Naming them was useless until there was a way to act:

```sh
corpus-survey duplicates --db data/corpus.db                       # look
corpus-survey duplicates --db data/corpus.db --excludes-for work   # paste
```

Duplicates come in **pairs** and both members qualify, so the report lists
exactly one of each — dropping the whole set would delete the content rather
than deduplicate it. The copy under a folder named `Backup`, or inside a
`.zip` whose contents are also extracted, is the one suggested.


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

**Note on trust:** `pattern` values are your own regexes, compiled and run
against query text and chunk content on the query path. They're trusted config,
but as a defensive bound each pattern is only scanned against the first
`MAX_REGEX_SCAN_CHARS` (20K) characters of input, so a pathological
catastrophic-backtracking pattern can't be fed an unbounded string. Avoid
nested unbounded quantifiers like `(a+)+`.

## Environment variables

| Var | Required? | Used by | Notes |
|---|---|---|---|
| `VOYAGE_API_KEY` | when `provider="voyage"` | embedder | https://dash.voyageai.com/ |
| `GEMINI_API_KEY` | when `provider="gemini"` | embedder | https://aistudio.google.com/apikey (or `GOOGLE_API_KEY`) |
| `ANTHROPIC_API_KEY` | when running `corpus-summarize` | summarizer | https://console.anthropic.com/ |

`corpus` uses `python-dotenv`, with a predictable, explicit precedence: an already-set shell/CI environment variable wins; then a `.env` next to whatever `--config` you passed; then a `.env` in the current directory; then a clear error naming what's missing and where it looked. corpus never reads a credential out of its own installed source tree. See "Credentials" in the [README](../README.md#credentials) for the full precedence order.

## Validation

`CorpusConfig.load()` validates the file via Pydantic at startup. Invalid TOML,
a bad value, or a missing file raises `ConfigError`; the CLI entrypoints catch
it and print a clean one-line `error: ...` message (no traceback) before
exiting non-zero. Common errors:

- **Source name pattern**: `name = "BAD-NAME"` → `ConfigError`. Names must be lowercase identifiers (`^[a-z][a-z0-9_]*$`).
- **Embedder dim**: `dim` must be a positive integer (`> 0`); `dim = 0` or negative → `ConfigError`.
- **Reference source_type matches no source**: pattern validation passes, but `expand_context` will find no chunks. Not an error; documented behavior.
- **Missing `[[sources]]`**: `corpus-ingest` errors with "No source named X" when called.
- **Embedder dim mismatch** (vs an existing DB): caught at `ChunkStore.__init__` open time, not config load.
