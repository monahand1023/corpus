# Troubleshooting

Common problems and the actual fix, not the canned "did you try restarting?"

## Setup errors

### `FileNotFoundError: corpus.toml not found`

You ran a CLI before generating the config. Either:

```sh
corpus-init                                      # interactive wizard
# or
cp corpus.toml.example corpus.toml               # manual
```

### `VOYAGE_API_KEY missing` / `GEMINI_API_KEY missing`

Embedder couldn't find an API key.

```sh
cp .env.example .env
# edit .env, paste your key
```

`.env` is loaded via `python-dotenv` at process start. If you set the var in your shell `.zshrc` / `.bashrc`, it'll also work, but `.env` is the standard place. Don't commit `.env` — it's gitignored by default.

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
corpus-stats                    # via MCP, or:
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

If `vector_search` is >50ms, your corpus is past the ~100K-chunk point where brute-force `vec0` starts feeling slow. Time to add HNSW indexing or partition by source type.

### `corpus-summarize` is rate-limit-throttled

Default `--concurrency 8` is safe under Anthropic standard tiers. Lower it (`--concurrency 4`) if you're on a stricter tier or seeing 429s.

### The DB file is large

```sh
ls -lh corpus.db
```

Most of the size is embeddings (1024 floats × 4 bytes × chunk count). 50K chunks ≈ 200 MB. Compressed not. Acceptable for personal use; if you need to ship the DB around, gzip cuts it ~3x.

## When you really need help

If none of the above applies:

1. Open an issue with: your `corpus.toml` (redacted), your Python version (`uv run python --version`), the full traceback, what you ran.
2. The repo's small enough that you can probably find the line of code involved — `grep -r "your_error_message" src/` is often productive.
