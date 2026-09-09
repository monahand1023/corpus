# Changelog

All notable changes to `corpus-rag` are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). This project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **`corpus-ingest --path DIR`** — ingest whatever is in a folder. Detects which
  built-in connectors apply and ingests each matching file type as its own
  source, with no `[[sources]]` block to write. `corpus.toml` still supplies the
  database path, embedder, and retriever settings; only the sources are
  superseded. Detected sources are namespaced by folder (`documents_pdf`,
  `inbox_pdf`) because orphan pruning is scoped by source type — two folders
  sharing a bare `pdf` name in one database would delete each other's chunks.
  Cannot be combined with `--source` or `--all`.
- **Orphan-pruning blast-radius guard.** `delete_orphans` now refuses (and
  exits non-zero from the CLI) when the chunks it's about to delete for a
  source exceed a configurable fraction of that source's existing chunks —
  `[pruning] max_orphan_ratio` (default 20%), enforced only above
  `min_chunks_for_guard` (default 50) chunks so small/new sources aren't
  blocked by noise. Closes a real gap: the only prior protection was a
  connector's `failed_files` count, which stays 0 for a connector that
  silently under-yields documents while still honestly reporting success —
  found by ingesting a large archive twice and comparing the numbers, which a
  reviewer does not reliably do. `corpus-ingest --source NAME --prune-anyway`
  overrides the guard (as it already did for `failed_files`) for a genuine
  bulk deletion. See [`configuration.md`](docs/configuration.md#pruning--orphan-deletion-blast-radius-guard).
- **`skipped_files` connector counter**, alongside the existing `failed_files`.
  Both are optional, defensively-read attributes with full backwards
  compatibility for connectors exposing neither. `failed_files` means "might
  succeed on a later run" and suppresses pruning for the source, as before.
  `skipped_files` means "this connector has permanently decided it can never
  read this input" (e.g. a directory containing thousands of files in a format
  the connector doesn't support) — it does NOT suppress pruning, since the
  absence isn't evidence of a bug, but it IS reported on `IngestResult` and
  logged so a large permanent-skip count stays visible instead of disappearing.
- **`ChunkStore(path, read_only=True)`.** Opens the store through a `mode=ro`
  SQLite URI and never runs the FTS schema migration; any mutating method call
  raises a clear `ReadOnlyStoreError` naming the cause. `corpus-mcp` and
  `corpus-query` now open the store this way, since neither ever writes to it.
  Closes a real footgun: opening a store — including a *backup*, to inspect
  its pre-migration state — silently ran a schema migration as a side effect,
  rewriting the file (626 MB → 679 MB on a ~70k-chunk store) with no warning
  and no way to opt out. A migration that does run against a store with
  existing chunks is now logged at WARNING (not INFO) with the path, before
  and after, including the row count rebuilt.

### Fixed
- **One unreadable PDF no longer aborts an entire source.** `pypdf` and
  `python-docx` parse lazily: the constructor succeeds on a file that cannot
  actually be read, and the failure surfaces later on first access to `.pages`
  or `.paragraphs`. Both accesses sat outside the per-file guard, so an
  encrypted PDF took down the whole source — and since
  `pypdf.errors.FileNotDecryptedError` derives from `Exception` rather than
  `ValueError`/`OSError`, it escaped the CLI's per-source handler too and would
  abort an entire `--all` run. Found by ingesting 3,247 real PDFs, where a
  single encrypted file left the source with zero chunks; after the fix, 2,719
  documents and tens of thousands of chunks.
- **A real `mypy` type error in the test suite** (`upsert_batch` was handed a
  `list[float] | None` where a `Sequence[float]` was required — the embedder's
  optional-embedding return type, not asserted away before use). CI only ran
  `mypy src/`, so this was invisible: `mypy src tests` surfaces it alongside
  ~225 unrelated `no-untyped-def`-style errors from Mock-heavy test helpers.
  Fixed the real bug; made the src-only scope explicit via `[tool.mypy]
  exclude` (with rationale) rather than leaving it an unstated accident that
  could hide the next real one.

## [0.3.0] - 2026-09-08

### Added
- **Multilingual / CJK full-text search.** `unicode61` (FTS5's tokenizer)
  cannot segment Japanese, Chinese, or Korean-adjacent scripts — with no
  spaces between words, a whole sentence became one token and a query like
  `東京` never matched inside it. CJK runs (Hiragana, Katakana, Halfwidth
  Katakana, CJK Unified Ideographs + Extension A, CJK Compatibility
  Ideographs — Hangul deliberately excluded) are now rewritten as
  overlapping character bigrams on both the index and query paths
  (`corpus.util.fts_normalize`), so two-character CJK words match. Query
  terms are also de-duplicated and capped at 64 to bound worst-case query
  latency and keep repeated terms from skewing BM25 rank. **Measured index
  growth on the FTS table: ~2.4x for Japanese content, 0% for English** —
  size accordingly if a large fraction of your corpus is CJK.
- **`docx`, `xlsx`, and `rtf` connectors** (`pip install 'corpus-rag[docx]'` /
  `[xlsx]` / `[rtf]`, or `[all]`), bringing the built-in connector count to
  seven. `docx` extracts body paragraphs and tables via `python-docx`; `xlsx`
  extracts each sheet's non-empty rows via `openpyxl` (formulas read as
  cached values); `rtf` extracts plain text via `striprtf` (pure Python).
  See [`docs/adding_a_source.md`](docs/adding_a_source.md) and the README's
  "Built-in connectors" table.
- **Per-source candidate budgets in the retriever.** Vector search now fetches
  candidates per source type (sized off `top_k` and the number of source
  types) before fusing globally, instead of one flat global fetch — a
  dominant source type could otherwise consume the entire candidate pool by
  sheer volume (every chunk has *some* distance to the query) and starve a
  small source out of the results entirely, regardless of relevance. BM25/FTS
  deliberately keeps a single global query rather than the same per-source
  loop — see the comment on `ChunkStore.fts_search` for the measured ~1561x
  regression that a per-source pre-filter caused there.

### Fixed
- **A missing optional-extra install no longer aborts `corpus-ingest --all`.**
  Each connector imports its third-party library lazily inside `load()`, so
  the "install `corpus-rag[docx]`"-style friendly error in the registry's
  `_build_*` factories was unreachable dead code — a user missing an extra
  got a raw `ModuleNotFoundError` traceback, and because the CLI only caught
  `(ValueError, FileNotFoundError)`, `--all` aborted and silently skipped
  every source configured after the broken one. The registry factories now
  probe the real import, and the CLI's catch widens to also catch `OSError`
  (of which `FileNotFoundError` is a subclass) and `ImportError` — one
  misconfigured source now fails with an actionable message and `--all`
  continues to the rest.
- Broken multilingual full-text search (see "Multilingual / CJK full-text
  search" above) — CJK queries previously matched nothing.
- FTS index migration streams the cursor instead of `.fetchall()`-ing every
  row up front (measured 2.90s / ~146MB RSS at ~70k chunks), and the DB
  connection now sets a 5-second `busy_timeout` so opening the store while an
  MCP server holds it open waits briefly instead of raising `database is
  locked` immediately.

### Changed
- **Ingestion enumeration completeness is now an explicit contract.** A
  connector that cannot fully enumerate its source (missing directory,
  unmounted volume) must raise rather than yield a partial list — a partial
  enumeration previously looked identical to "these files were deleted" and
  triggered orphan deletion against content that was never actually gone.

- **A file the connector cannot read now suppresses orphan pruning.** This
  extends the enumeration-completeness contract above from whole sources down
  to individual files. Connectors swallow per-file read errors and continue, so
  a file that was momentarily locked yielded no document — and because
  `delete_orphans` removes any chunk whose id is absent from the run, its
  already-indexed content was silently deleted. Connectors now count files they
  skipped and expose the count as an optional `failed_files` attribute; the
  ingester skips pruning entirely when it is non-zero. `corpus-ingest
  --prune-anyway` overrides for a named source (refused with `--all`).
  Partially-read files count too: a PDF whose page fails to extract is still
  yielded, but the shorter body produces fewer chunks, so the tail chunk ids
  vanish and the content those pages occupied would be pruned. `IngestResult`
  gains `files_failed` and `pruning_performed` — the latter because
  `orphans_deleted == 0` means both "nothing needed pruning" and "pruning was
  skipped". **`failed_files` is an optional connector capability, not part of
  the `Connector` protocol**: connectors defined outside this package keep
  working unchanged, but an absent attribute means "does not report failures"
  rather than "had zero failures", so such a connector is not covered by the
  gate.

### Migration notes
- **Existing databases are migrated automatically, in place, the first time
  they're opened after upgrading.** Opening a pre-upgrade DB detects a stale
  `fts_version` in `schema_meta`, rebuilds the `chunks_fts` full-text index
  from the stored chunk content (no re-embedding, no API cost), and stamps
  the new version — this happens transparently inside `ChunkStore.__init__`,
  with no separate CLI command to remember or skip. It runs once; subsequent
  opens are a no-op version check. If the process is killed mid-migration,
  the version stamp is only written after every row is reinserted, in the
  same transaction, so the migration retries cleanly on next open rather than
  leaving a half-rebuilt index.

### Security
- Bumped `transformers` 5.8.1 → 5.16.1 in `uv.lock` (GHSA-xrqw-3rrv-vx5w, path
  traversal in `save_pretrained`). Transitive via `sentence-transformers`, so it
  only affects the `[reranker]`/`[all]` extras; the published package's
  constraints already resolve to a patched release.

## [0.2.3] - 2026-08-22

### Security
- Bumped dependencies to close 30 open Dependabot alerts (some duplicate
  CVEs against the same package): `mcp` 1.27.1 → 1.29.0, `cryptography`
  48.0.1 → 50.0.0, `aiohttp` 3.14.1 → 3.14.3, `pyasn1` 0.6.3 → 0.6.4,
  `pypdf` 6.13.3 → 6.16.1, `pillow` 12.2.0 → 12.3.0, `setuptools` 81.0.0 →
  84.0.0 (transitive, via `torch`). No API or behavior changes; full test
  suite (ruff, mypy, pytest, eval-gate) verified green against the bumped
  versions.

## [0.2.2] - 2026-08-20

### Fixed
- **Voyage batches are packed by real token count, not characters.** Voyage
  enforces a hard 120,000 tokens per `/embed` request. Batches were packed
  against a 400,000-char budget annotated "~100K tokens" — the chars/4 rule of
  thumb — which does not hold, because token density is not a constant.
  Measured on a mixed corpus, density ranged from 3.66 chars/token (plain
  prose) to 1.51 (dense or structured text, and CJK is denser still): a 2.4x
  spread. At 1.51 that budget is 264K tokens, more than double the cap, and
  Voyage responds `InvalidRequestError: The max allowed tokens per submitted
  batch is 120000`. Each rejection triggers recursive halving, so a large
  ingest degrades to a fraction of the API's real throughput — observed at
  0.5 chunks/s against an achievable 146/s on a large corpus.

  Packing now uses Voyage's local tokenizer (~2,100 texts/s, no API cost) with
  a 100,000-token ceiling. When the tokenizer is unavailable it falls back to a
  deliberately pessimistic 1.3 chars/token — below the densest ratio observed,
  so the fallback can only over-split, never overflow. A tokenizer returning
  the wrong arity also falls back rather than crashing packing, and the warning
  fires once per embedder rather than per batch. `_embed_batch`'s TPM estimate
  uses the same real counts instead of chars/3.

  Note: `gemini.py` keeps its own char budget; its API limits differ and it is
  unaffected by this change.

### Added
- **Generation-quality eval via a validated LLM-as-judge.** A feature-flagged
  `answer_from_context` generator (`corpus.eval.generation`), a 3-axis
  LLM-as-judge with forced tool-schema output and position-bias control
  (`corpus.eval.judge`), and a validation study that certifies the judge against
  human faithfulness labels via Cohen's κ (`corpus.eval.validation`). New
  `corpus-judge` CLI (default generate+judge / `--validate` / `--build-fixture`,
  plus `--rerank` to route generation through the BGE cross-encoder so you can
  A/B a retrieval change's effect on answer quality — the judge scores answers
  against the retrieved context, so the loop measures retrieval levers too),
  a frozen self-contained public fixture, and an opt-in, key-gated `judge-gate`
  CI job that fails on κ regression or a missed adversarial case. Client
  construction + retry are shared with the summarizer via `corpus._anthropic`.
  See `docs/judge.md`.

- **`docs/understanding-evals.md`** — a from-scratch conceptual guide to RAG and
  AI evaluation for readers with no prior background: retrieval vs generation,
  automated metrics vs LLM-as-judge, Cohen's κ validation, baseline-vs-delta,
  the noise floor, and using `--rerank` to measure a retrieval change's effect on
  generation. Linked as the "start here" doc; complements the reference-level
  `docs/eval.md` and `docs/judge.md`.

### Changed
- **Judge/generator robustness + portability.** The `submit_answer` and
  `record_verdict` tools now use `strict: true` (+ `additionalProperties: false`)
  so forced-tool output is always schema-complete, and the parsers default
  missing fields defensively — a batch generate+judge run no longer aborts if the
  model returns an incomplete tool call (forced `tool_choice` invokes the tool but
  does not guarantee its required fields are populated). The `corpus.eval`
  generation/judge/validation modules use relative imports so they can be reused
  verbatim across separate deployments.

## [0.2.1] — 2026-07-09

### Added
- **Retrieval-quality eval harness.** A pure, dependency-free metrics module
  (`corpus.eval.metrics`: recall@K / MRR / nDCG@K), a zero-dependency
  deterministic `hash` embedder (`provider="hash"`) so the retrieval pipeline
  runs with no API key, an expanded two-source-type sample corpus, an extended
  `corpus-eval` CLI (per-source-type breakdown, `--json`, `--compare`, and a
  `--check` regression gate), and a keyless `eval-gate` CI job that fails the
  build on recall@5 / nDCG@5 regression below committed thresholds.
- `docs/eval.md` — eval methodology, metric definitions, and the keyless CI gate.

### Fixed
- `corpus-eval` / `corpus-benchmark` no longer crash on an `EVAL_QUERIES` module
  that uses `from __future__ import annotations` (the dynamically-loaded module
  is now registered in `sys.modules` before execution).

## [0.2.0] — 2026-07-08

### Added
- Python 3.13 support (CI matrix + classifier).
- **Python 3.14 support** (CI matrix + classifier). The `voyageai` pin was
  relaxed to `<0.5` so the 0.4.x line (which ships cp314 wheels) installs on
  3.14; previously `pip install` failed there with a resolver error.
- Mypy strict passes and now runs in CI.
- `py.typed` marker (PEP 561) so the annotated public API is visible to
  downstream type checkers.
- `corpus-init --quiet` — non-interactive setup that accepts all defaults
  (for CI/tests).
- `[voyage]` optional extra (see Changed) and an actionable error when an
  embedder SDK is missing (`pip install corpus-rag[voyage]` / `[gemini]`).

### Changed
- **BREAKING (install only): the Voyage embedder is now an opt-in extra.**
  `voyageai` moved out of the base dependencies into a `[voyage]` extra, so a
  bare `pip install corpus-rag` no longer pulls Voyage's large transitive tree
  (`langchain-core`, `pillow`, `ffmpeg-python`, `future`) — base install drops
  from ~70 to ~34 packages. Install an embedder explicitly:
  `pip install 'corpus-rag[voyage]'` (or `[gemini]`). No runtime API change.
- Config errors (missing file, invalid TOML, bad values) now surface as a clean
  one-line `error:` message + non-zero exit instead of a raw traceback. Added
  `ConfigError`; `CorpusConfig.load` raises it and CLIs route through
  `cli._common.load_config_or_exit`.
- `GeminiEmbedder` raises a clear `RuntimeError` when the API returns no
  embeddings instead of failing later with an opaque `TypeError`.
- The sdist no longer ships `uv.lock` / `.venv` / `dist` / `*.db`.

### Fixed
- `corpus-init` no longer hangs in an infinite loop when stdin hits EOF (piped
  input or Ctrl-D); it aborts with a clear message. Use `--quiet` for defaults.
- `corpus-init` escapes `"` and `\` when writing `corpus.toml`, so a data path
  containing those characters no longer produces an unloadable config.
- `embedder.dim` is validated as a positive integer (`> 0`).

### Security
- Ingestion no longer follows symlinks or reads outside a source's configured
  `path` (a `..` glob or a `notes.md -> ~/.ssh/id_rsa` symlink is skipped),
  preventing arbitrary local files from being ingested and surfaced to the LLM.
- MCP tool handlers sanitize unexpected exceptions (generic message to the
  client, details to stderr) instead of leaking internal state.
- MCP content-returning tools prefix results with an "untrusted retrieved
  content" banner (prompt-injection mitigation).
- `[[references]]` regexes are scanned against a bounded input length
  (`MAX_REGEX_SCAN_CHARS`) to limit ReDoS blast radius.
- Bumped `torch` to `>=2.13.0` in the `[reranker]`/`[all]` extras to clear
  GHSA-rrmf-rvhw-rf47 (CVE-2025-3000, `torch.jit.script` memory corruption).

### Removed
- Dead `tomli` dependency marker (`python_version < '3.11'` can never match
  under `requires-python >= 3.12`).
- Duplicate badge block in the README.

## [0.1.2] — 2026-05-15

### Added
- `corpus-mcp --config /absolute/path/to/corpus.toml` flag. Claude Code spawns
  the MCP server from an arbitrary CWD (usually `$HOME`), so the previous
  "reads `./corpus.toml` from CWD" behavior failed for the common case where
  the config lived elsewhere. The flag makes the path explicit.

### Changed
- README + `docs/mcp_integration.md` updated with the new recommended
  `~/.claude.json` wiring (using `--config` and an absolute path).
- The `uv --directory ... run corpus-mcp` spawn pattern is now documented as
  an alternative for uv users who'd rather keep CWD-relative behavior.

### Compatibility
- Non-breaking. Bare `corpus-mcp` still works (just keeps the old
  CWD-relative behavior).

## [0.1.1] — 2026-05-15

### Removed
- `who_did_what` MCP tool. Out-of-the-box connectors (markdown / text / pdf /
  html) populate file mtime for dates but no author signal, so the tool
  returned empty for the common case. Better to remove dead UI than ship
  something that confuses users.
- Underlying `Retriever.who_did_what` and `ChunkStore.find_by_person` methods
  removed too — they had no other callers.

### Changed
- MCP tool count: 8 → 7.
- README rewritten for the pip-installed audience:
  - Quick start leads with `pip install corpus-rag` + `corpus-init` wizard
    instead of `git clone`.
  - `~/.claude.json` snippet uses bare `corpus-mcp` (on PATH after pip
    install) instead of `uv --directory /path run ...`.
  - Eval + benchmark sections no longer reference the in-repo
    `tests/eval_queries.py` path; the `EvalQuery` template is inlined so
    users can write their queries file anywhere.
  - Added a "Develop locally" footer for contributors who want the clone
    route.
  - Added PyPI + license badges.
- `docs/mcp_integration.md`, `docs/adding_a_source.md`, and the sample
  corpus notes updated to match.

### Compatibility
- Removing `who_did_what` is technically a breaking change. The tool was
  shipped in 0.1.0 but always returned empty for the default connectors,
  so no real-world callers should be affected. Custom connectors that
  populate `metadata.author` can still wire their own MCP tool by calling
  `ChunkStore` directly.

## [0.1.0] — 2026-05-15

### Added
- Initial public release.
- Hybrid retrieval: vector search (sqlite-vec) + BM25 (FTS5) fused via
  reciprocal rank fusion with auto-tuned weights.
- Multi-hop investigation via `expand_context` (chases configurable
  `[[references]]` patterns across documents).
- 4 built-in connectors: markdown, text, pdf, html.
- 2 embedder providers: Voyage `voyage-3-large` (default) and Google Gemini
  `gemini-embedding-001`.
- Optional BGE cross-encoder re-ranker (`pip install corpus-rag[reranker]`).
- Optional per-doc summarization via Claude Haiku
  (`pip install corpus-rag[summarizer]`).
- 8 MCP tools wired to Claude Code over stdio.
- 9 console scripts: `init`, `ingest`, `query`, `eval`, `summarize`, `mcp`,
  `benchmark`, `list`, `reset`.
- Embedding-dim startup guard prevents silent corruption when swapping
  embedder models without re-ingesting.
- Thread-local SQLite connections for safe MCP concurrency under
  `asyncio.to_thread`.
- Per-file failure isolation in all connectors.
- README + 4 reference docs (configuration, mcp_integration,
  adding_a_source, troubleshooting).
- MIT license.

[Unreleased]: https://github.com/monahand1023/corpus/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/monahand1023/corpus/releases/tag/v0.2.0
[0.1.2]: https://github.com/monahand1023/corpus/releases/tag/v0.1.2
[0.1.1]: https://github.com/monahand1023/corpus/releases/tag/v0.1.1
[0.1.0]: https://github.com/monahand1023/corpus/releases/tag/v0.1.0
