# Troubleshooting

Common problems and the actual fix, not the canned "did you try restarting?"

## Setup errors

### `error: corpus.toml not found` (or `... is not valid TOML` / `... has invalid values`)

You ran a CLI before generating the config, or the file has a syntax/validation
problem. The CLIs print a clean one-line `error:` message and exit non-zero
(no traceback). Generate or fix it:

```sh
corpus-init                                      # interactive wizard
corpus-init --quiet                              # non-interactive (defaults; for CI)
# or
cp corpus.toml.example corpus.toml               # manual
```

### `ImportError: The 'voyage' embedder requires the voyageai SDK`

The embedders are optional extras (see [configuration.md](configuration.md#why-embedders-are-optional-extras)).
You installed the bare base but no embedder. Add one:

```sh
pip install corpus-rag[voyage]      # or: corpus-rag[gemini]
```

Same message shape for `gemini` (`google-genai`).

### `VOYAGE_API_KEY missing` / `GEMINI_API_KEY missing`

Embedder couldn't find an API key.

```sh
cp .env.example .env
# edit .env, paste your key
```

`.env` is loaded via `python-dotenv`, next to whatever `--config` you passed if there is one there, else from the current directory — see "Credentials" in the [README](../README.md#credentials) for the full precedence. If you set the var in your shell `.zshrc` / `.bashrc`, it'll also work (a real environment variable always wins over any `.env`). Don't commit `.env` — it's gitignored by default.

### `EmbeddingDimMismatch: DB was created with embedding_dim=X, but this run requests Y`

You changed `embedder.model` (or `dim`) in `corpus.toml` after ingesting. The DB's vector column is fixed-width — switching dims would silently corrupt retrieval.

Fix:

```sh
rm corpus.db   # delete and re-ingest from scratch
corpus-ingest --all -v
```

### `ImportError: PDF connector requires the [pdf] extra`

You configured `type = "pdf"` but `pypdf` isn't installed.

```sh
uv add pypdf      # or: pip install corpus-rag[pdf]
```

Same pattern for `html` (`trafilatura`), `gemini` (`google-genai`), `reranker` (`sentence-transformers`), `summarizer` (`anthropic`).

### `ValidationError: source_type must match ^[a-z][a-z0-9_]*$`

Your `[[sources]] name` has uppercase, hyphens, or starts with a digit. Fix the name:

```toml
# Bad
name = "My-Notes"   # uppercase + hyphens

# Good
name = "my_notes"   # lowercase + underscores
```

## Ingest errors

### Ingest crashes mid-run on a specific file

For PDF/HTML: the parser library hit a malformed file. Check the verbose log to see which one:

```sh
corpus-ingest --source papers -v 2>&1 | grep -i 'cannot\|fail'
```

The connector already swallows individual file errors and continues. If the whole run dies, the failure is in chunking, embedding, or DB writing — share the traceback.

### Voyage rate limit (3 RPM / 10K TPM)

You don't have a card on file. The free tier requires payment-method-on-file to lift account rate limits (you're still in the 200M-token free zone, but throttled to 3 requests/minute until a card is added).

Add a card at https://dash.voyageai.com/billing. You won't be charged unless you exceed 200M tokens.

### Gemini "Resource exhausted" / quota errors

You hit the ~1,500 requests/day rolling limit. Wait a few hours for the window to roll, or switch to Voyage.

### `pruning REFUSED — blast-radius guard tripped` / exit code 1

`corpus-ingest` refused to delete orphaned chunks because doing so would remove more than `max_orphan_ratio` (default 20%) of that source's existing chunks. The printed line names the actual numbers (existing, orphans, percentage). Two possible causes:

1. **A connector bug.** The connector silently yielded far fewer documents than it should while still reporting `failed_files = 0` — this is exactly the failure mode the guard exists to catch (see [`configuration.md`](configuration.md#pruning--orphan-deletion-blast-radius-guard)). Investigate before doing anything else; don't just re-run with `--prune-anyway`.
2. **A genuine bulk deletion.** You actually removed most of that source (deleted a folder, changed the connector's `path`/`glob`). If you've confirmed this, force it through:

```sh
corpus-ingest --source NAME --prune-anyway
```

The index is left untouched either way until you decide — nothing is deleted on refusal.

### Re-ingest is slow even though content didn't change

The content-hash skip works at the chunk level. Common reasons it doesn't kick in:

1. **You changed chunker logic.** Different splitting = different chunks = different IDs = all-new.
2. **You changed embedder model or dim.** The DB guard should have caught this, but if you nuked the DB first, all chunks re-embed.
3. **You changed `util/scrub.py` patterns.** Different scrubbing = different content = different hash. Worth it for correctness.

## Retrieval problems

### Top-K results are dominated by one document

The dedupe-by-source-key post-filter should prevent this. If it isn't:

1. Make sure your chunker emits distinct `source_key` values per document. If all chunks of doc A share `source_key="doc-a"`, dedupe works. If they have different keys, they're treated as separate docs.

2. Drop `max_per_source_type` lower:

   ```toml
   [retriever]
   max_per_source_type = 2     # was 3
   ```

3. Inspect what's happening:

   ```sh
   corpus-query "your query" -k 20    # see the raw fused order
   ```

### Top-K has 0 results

Either the corpus is empty or the query matches nothing.

```sh
corpus-list                     # chunk counts per source (CLI); the MCP equivalent is the corpus_stats tool. Or directly:
sqlite3 corpus.db "SELECT source_type, COUNT(*) FROM chunks GROUP BY source_type"
```

If counts are non-zero but queries return nothing, the most likely cause is a malformed FTS query (special characters tripping FTS5's MATCH parser). The store falls back to empty FTS results silently, but the vector branch should still return. Try `--no-hybrid` to isolate.

### Negative queries (topics not in corpus) return high-confidence-looking results

Vector search ALWAYS returns the top-K closest chunks regardless of how irrelevant they are. The distance is the signal. Look at it:

```sh
corpus-query "Snowflake setup" -k 5
# Distances of 0.95+ across the board → nothing relevant in the corpus
# Distances 0.85-0.90 → maybe relevant, examine the content
# Distances < 0.85 → genuinely relevant match
```

The exact thresholds depend on the embedder; calibrate against known-good queries.

### `recent_activity` returns nothing for plain markdown/text

Dates come from frontmatter (`created` / `modified` / `updated`) if present,
otherwise the connector falls back to the file's modification time. Files from a
`git clone` all share the checkout time, which may be older than the window you
asked for (e.g. the default 7 days), so `recent_activity(days=7)` looks empty.
Fixes: widen the window (`recent_activity` with a larger `days`), or add date
frontmatter to your docs:

```markdown
---
title: My note
modified: 2026-07-01
---
```

### Re-rank made retrieval worse on my eval

This is a real and common observation. Cross-encoder rerankers optimize for full-content semantic relevance — they can pick a chunk with rich content over a chunk with a perfect title-match. If your eval queries are paraphrases of titles, plain hybrid search may outperform reranker-augmented hybrid search.

Try:

```sh
corpus-eval                       # baseline
corpus-eval --rerank              # with reranker
# Compare recall@5
```

If the reranker hurts, leave it off. It tends to help on harder queries (paraphrased intent, long content) and hurt on title-shaped queries.

## MCP problems

### Tools don't show up in Claude Code

1. Restart Claude Code after editing `~/.claude.json`. The MCP discovery runs at app launch.
2. Check the MCP status: `/mcp` in Claude Code shows connection state for each server.
3. Check stderr: launch the server manually to see what it logs:

   ```sh
   uv --directory /path/to/corpus run corpus-mcp
   # Type something then Ctrl-C; logs go to stderr
   ```

### Tool call returns "No source named X"

Either `source_type` in the tool call is misspelled or that source isn't in your `corpus.toml`. Use `corpus_stats` to see what's actually loaded.

### Tool call gets "SQLite objects created in a thread can only be used in that same thread"

You're running an older version that doesn't pass `check_same_thread=False` to sqlite3. Pull latest — this was fixed early in development.

### Changes to MCP code aren't reflected in Claude Code

The MCP subprocess caches imports. Restart Claude Code (full app restart, not just a new window). Or do most of your iteration via the CLI commands, which always pick up fresh code.

## Performance problems

### Queries are slow

Run the benchmark to see where time goes:

```sh
corpus-benchmark --runs 10
```

Typical numbers for a few-thousand-chunk corpus on M-series Mac:

- embed_query: 80-200ms (network round-trip dominates)
- vector_search: 1-5ms
- fts_search: 1-3ms
- fusion / dedupe: <1ms

If `embed_query` is 500ms+, your network to the embedding provider is the bottleneck. Gemini and Voyage are both routed via global Anthropic / Google networks; usually fast, but cellular hotspot will obviously hurt.

If `vector_search` is >50ms, your corpus is past the ~100K-chunk point where brute-force `vec0` starts feeling slow. First check `[performance] mmap_size_mb` in `corpus.toml` (see [`configuration.md`](configuration.md#performance--sqlite-memory-tuning)) — it needs to cover your `corpus.db`'s actual file size to give its full benefit (measured 3x+ on a store where it does, only ~1.4x when it covers a third of the file); `corpus-list` prints the chunk count, `ls -lh corpus.db` the file size. If it's already sized to the file and still slow, you're past what pragma tuning alone fixes — time to add HNSW indexing or partition by source type.

### `corpus-summarize` is rate-limit-throttled

Default `--concurrency 8` is safe under Anthropic standard tiers. Lower it (`--concurrency 4`) if you're on a stricter tier or seeing 429s.

### The DB file is large

```sh
ls -lh corpus.db
```

Most of the size is embeddings (1024 floats × 4 bytes × chunk count). 50K chunks ≈ 200 MB. Compressed not. Acceptable for personal use; if you need to ship the DB around, gzip cuts it ~3x.

### Opening a store rewrote it (a schema migration ran)

`ChunkStore(path)` runs a one-time FTS index migration automatically when it opens a store whose `fts_version` is stale — normal, and how a single-user DB stays correct without a separate "remember to migrate" step. When it runs against a store that already has chunks, it logs at WARNING with the path and the before/after chunk count, so it's never silent.

If you need to open a store WITHOUT any chance of this happening — e.g. to inspect a backup's on-disk state, or compare before/after some other change — open it read-only, which never migrates and raises a clear error on any write attempt:

```python
from corpus.db.sqlite import ChunkStore
store = ChunkStore(path, embedding_dim=1024, read_only=True)
```

`corpus-mcp` and `corpus-query` already open the store this way, since neither ever writes to it.

## Security notes

- **`corpus-eval` / `corpus-benchmark` execute the `--queries` file you pass.**
  They `import` it as a Python module to read `EVAL_QUERIES`, so only point them
  at query files you trust (same as running any Python script).
- **Ingestion won't follow symlinks or read outside a source's `path`.** A
  symlink inside a source directory, or a `..` in a `glob`, is skipped — so a
  planted `notes.md -> ~/.ssh/id_rsa` can't be ingested and surfaced to Claude.
- **Retrieved corpus content is labeled untrusted to the MCP client.** Search /
  doc / timeline results are prefixed with a banner telling the model to treat
  them as data, not instructions — a mitigation, not a guarantee, against
  prompt injection from adversarial corpus content.

## When you really need help

If none of the above applies:

1. Open an issue with: your `corpus.toml` (redacted), your Python version (`uv run python --version`), the full traceback, what you ran.
2. The repo's small enough that you can probably find the line of code involved — `grep -r "your_error_message" src/` is often productive.
