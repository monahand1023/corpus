# Changelog

All notable changes to `corpus-rag` are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). This project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **`corpus-transcribe --redo-stale --transcribed-since DATE`** limits a redo to
  rows written on or after DATE, so a pipeline fix redoes only what that
  pipeline wrote.

### Changed
- **Speech regions are packed into windows of up to 30s** before transcription.
  One window per detected region sent 1-2s clips to the model; on a real archive
  41% of windows under 2s came back in the wrong language. On a sample, packing
  cut wrong-language text from 6.1% to 0.7% and decoding time by about half.
  Part of the policy fingerprint, so `--redo-stale` picks up affected files.
- **`--language` constrains decoding instead of only filtering.** A window the
  model labels as any other language is decoded again in the likeliest named
  one, keeping its speech. `TranscriberBackend.transcribe_window` takes an
  optional `languages`; a backend without it still runs unconstrained.

## [0.4.0] - 2026-09-24

### Added
- **Audacity projects are transcribed.** `corpus-transcribe` now finds `.aup3`
  files, mixes their audible tracks to mono at the project's real rate, and
  transcribes them like any recording, so what is said in them is searchable
  through a `transcripts` source.
- **`corpus.connectors.aup3_layout`** reads a project's layout (sample rate,
  tracks, clip offsets, trims, mute/solo) from its binary-XML `project.doc`.
- **`corpus.retriever.assemble_results()`** — the selection pass (source dedupe,
  the per-type diversity cap, backfill, truncation to `top_k`) is now a shared
  function rather than something each consumer reimplements.
- **`corpus.retriever.build_vector_pool()`** — candidate-pool construction,
  likewise shared.
- **`ChunkStore.sweep_orphans()`** — prunes orphans AND reports which DOCUMENTS
  stopped being found, computed from the scan it already performs.
- **`ChunkStore.vanished_documents()`** — which documents disappeared, for the one
  path with no sweep to ride on (a prune suppressed by read failures).
- **`IngestResult.vanished_detail`** — names the documents behind a warning.
- **`corpus.mcp_util.UNTRUSTED_PREFIX` and `format_chunk_block()`** — the
  data-not-instructions marker and the chunk renderer, importable by consumers
  with their own MCP server.
- **`[query_log]` config section** — opt-in, local JSONL beside the store, off by
  default.
- **Ingest reports a rise in permanently-skipped inputs.** `skipped_files` means
  "this connector will never read these", which is why — unlike `failed_files` —
  it does not suppress pruning.
- **`corpus-ingest` exits 3 when a guard reported something.** A warning nobody
  reads is theatre for an unattended run: cron and CI see an exit code, not
  stderr.
- **`music` connector** (`[music]` extra, uses `mutagen`).
- **Yield-drop warning on ingest.** Each run records what a source yielded in a
  new `source_yield` table and warns when the next one produces materially fewer
  documents.
- **Contextual-Retrieval coverage in `corpus_stats`.** The MCP tool now reports
  how much of each source has been contextualized, since an uncontextualized
  source retrieves noticeably worse on fragments and that is a property of the
  index rather than the query.
- **Contextual Retrieval (`corpus-contextualize`).** For each chunk, a cheap model
  reads the chunk together with its parent document and writes one sentence
  situating it; the sentence is stored in a new `context` column and the chunk is
  re-embedded as context + content.
- **`corpus-index` CLI.**
- **`corpus-survey` CLI.** Read-only reconnaissance for deciding what to index,
  replacing the ad-hoc shell pipelines that work was previously done with.
- **`csv` / `tsv` connectors.** Stdlib `csv` only — no pandas.
- **`pptx` connector.** Uses `python-pptx`.
- **`zip` connector.**
- **`aup3` connector.** Audacity 3 project files are SQLite databases holding raw
  audio directly in a `sampleblocks` table — nothing (not ffmpeg, not any existing
  connector) could read one before this, so their content was entirely invisible.
- **`corpus-ingest --path DIR`** — ingest whatever is in a folder.
- **Orphan-pruning blast-radius guard.**
- **`skipped_files` connector counter**, alongside the existing `failed_files`.
- **`ChunkStore(path, read_only=True)`.** Opens the store through a `mode=ro`
  SQLite URI and never runs the FTS schema migration; any mutating method call
  raises a clear `ReadOnlyStoreError` naming the cause.
- **`[performance]` SQLite memory-tuning pragmas** (`cache_size_mb`,
  `mmap_size_mb`, `temp_store_memory`), applied on every `ChunkStore` connection.
- **`ChunkStore` warns when a database path resolves inside corpus's own package
  directory or (in a source checkout) its repo root.** That location is never
  correct for a consumer's index — it means personal data is being written into a
  library's own source tree, where it survives only as long as `.gitignore` stays
  correct.
- **Guardrails against personal data landing in this repo.**
  `tests/test_repo_hygiene.py` fails the suite — locally on every `pytest` run and
  in CI — if a `*.db`/`*.db-wal`/`*.db-shm` file or a real root `corpus.toml` ever
  appears anywhere in the repository.

### Changed
- **The diversity cap now backfills instead of truncating.** The cap was ABSOLUTE
  — a chunk over it was dropped and nothing replaced it — so the real ceiling on
  any answer was (source types x cap), whatever `top_k` said.
- **The vector candidate pool is adaptive rather than a per-source fan-out.**
- **`should_contextualize` floor raised from 50 to 260 tokens.** Derived, not
  guessed: the generated blurb is a near-constant ~141 characters whatever the
  chunk size, so its SHARE of the embedded text is what decides whether it helps.

### Fixed
- **`corpus-transcribe`: a new file's deadline ignored its length.** A file with
  no stored duration got the flat 120s base, so long recordings timed out and were
  eventually settled as `repeatedly_timed_out`.
- **`corpus-contextualize`: the measured 260-token floor never applied.** The
  config kept its own default of 50; it now uses the contextualizer's constant.
- **`olm` found archives with a raw glob**, bypassing the symlink/containment
  checks and the source's `exclude`.
- **`exclude` globs such as `*.mov` matched nothing on a transcripts source**;
  only substrings did.
- **Music inside a zip was known only as `.mp3`**: `.flac` beside `.mp3` was
  indexed and counted as skipped, and an `.m4a`-only archive never reached the
  music connector.
- **`corpus-rename` left the source's ingest baseline (`source_yield`) behind.**
- **`--path` detection gave a transcript sidecar the folder, not the database
  file.**
- **The MCP server refused to start with the keyless `hash` embedder.**
- **`recent_activity` returned chunks without their contextual blurb**, unlike
  search.
- **A read-only open of a path containing `#`, `?` or `%` opened the wrong file.**
- **Anthropic calls retried errors that cannot succeed** (bad key, malformed
  request) through the full backoff; a 4xx other than 408/409/429 now fails at
  once.
- **An encrypted PDF switched orphan pruning off permanently.** `failed_files`
  means "might succeed next time" and suppresses pruning for the whole source,
  which is right for a locked file or a transient I/O error.
- **The `[contextual]` config section was parsed and silently discarded**, so
  every value in it was unconfigurable and the default silently stood.
- **Ingest warnings now name the documents involved, unconditionally.**
- **`timeline` returned nothing when recent material was not the most semantically
  central.** The date filter ran in Python after retrieval, so a topic whose
  nearest chunks all fell outside the range was filtered to empty.
- **`timeline` was silently capped at 3 results per source type.** It went through
  `query()`, which applies the diversity cap by default, so a 20-item timeline
  could never return more than 3 x (number of source types) candidates regardless
  of `top_k`.
- **A large `top_k` no longer crashes vector search.**
- **A source-filtered full-text search no longer returns nothing when the filter
  is crowded out.**
- **Using a `ChunkStore` after `close()` now says so.**
- **A bad ingest no longer becomes the new normal.**
- **The yield check now watches chunks as well as documents.**
- **An expensive FTS rebuild is no longer something a constructor does.** Opening
  a store rebuilt a stale full-text index automatically.
- **A failed `ChunkStore` construction no longer leaks its connection.** If schema
  setup or a migration raised, the connection was never closed and its open write
  transaction held SQLite's single writer lock until garbage collection — for the
  life of the process, in a long-running server.
- **A contextualized chunk no longer drops out of BM25.** `set_context` wrote the
  combined context+content into `chunks_fts` RAW, while `upsert` writes
  `normalize_for_fts(content)` and the query path searches for that normalized
  form.
- **Distinct folders no longer collapse onto one source name and overwrite each
  other.**
- **Ingest warns when a source name is reused for a different path.** The
  normalizer keeps distinct folder NAMES apart, but cannot help when two folders
  in different places genuinely share one (`~/a/Notes` and `~/b/Notes`) — the
  commonest collision of all.
- **Office lock files and renamed legacy documents no longer block orphan pruning
  forever.**
- **One malformed shape no longer costs an entire presentation.** python-pptx
  reports `has_text_frame` / `has_notes_slide` as True while the frame itself is
  None, and `.text` can raise on a malformed shape — either of which raised out of
  the slide loop and lost the whole deck.
- **Byte-order marks are honoured before the text-decoding ladder.**
- **NUL characters are stripped from chunk content.** 239 chunks in one archive
  carried them, all from PDFs whose fonts had no usable encoding.
- **`corpus-ingest` / `corpus-index` no longer bury their output in extraction
  noise.** pypdf logs one ERROR per page for an unimplemented CMap and trafilatura
  one per empty HTML fragment, neither actionable.
- **`zip` and `pptx` cost estimates re-measured against many real sources.**
- **`corpus-index`'s cost estimate overstated by up to ~400x for compressed
  formats.**
- **`corpus-index` claimed noise directories were "not ingested" — they were.**
- **The `zip` connector was silently missing `pptx`/`csv`/`tsv` members.**
- **Non-ASCII filenames inside `.zip` archives were being corrupted into mojibake,
  live in a real index.** Measured: 77.3% of one archive source's chunks and a few
  percent of another — roughly a few thousand chunks total.
- **The same real index had ~970 chunks of U+FFFD replacement-character soup from
  non-UTF-8 file content** — found while chasing the mojibake bug above; the same
  Japanese-locale archives had Shift-JIS-encoded member *content*, not just
  corrupted names.
- **One unreadable PDF no longer aborts an entire source.** `pypdf` and
  `python-docx` parse lazily: the constructor succeeds on a file that cannot
  actually be read, and the failure surfaces later on first access to `.pages` or
  `.paragraphs`.
- **A real `mypy` type error in the test suite** (`upsert_batch` was handed a
  `list[float] | None` where a `Sequence[float]` was required — the embedder's
  optional-embedding return type, not asserted away before use).

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
