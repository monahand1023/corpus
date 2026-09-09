# Changelog

All notable changes to `corpus-rag` are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). This project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **`corpus-index` CLI.** One command from "here's a folder" to "it's
  searchable": surveys the directory (reusing `corpus-survey census`),
  reports the gap (file types with no connector — printed first, since it's
  the most useful fact about a real directory) and excluded noise
  directories, detects which connectors apply (reusing
  `corpus.util.autodetect.detect_sources`), and shows a plan — per-source
  file counts, sizes, and an estimated token count (file size ÷ 4, an
  explicit ceiling estimate, not the embedder's real tokenizer count) —
  before touching anything. Nothing is written or ingested until the user
  confirms, passes `--yes`, or the run stops after `--dry-run`. Confirmed
  sources are merged into corpus.toml as `[[sources]]` blocks
  (`corpus.planner.merge_sources_into_toml`) rather than ingested
  transiently, so re-running `corpus-index` on the same directory later
  picks up new/changed files — merging is idempotent (a source already
  matching what's in corpus.toml is left alone) and refuses, rather than
  silently overwrites, when two differently-located folders would collide
  on the same namespaced source name (`--name-prefix` or a manual edit
  resolves it). `--check-overlap DB` runs `corpus-survey overlap` against
  an existing database first, to catch "this is mostly already indexed"
  before paying to re-embed it. Reuses `corpus.ingester.Ingester` for the
  actual ingest — no parallel ingestion path. See the "corpus-index: point
  it at a folder" section of the README for the full worked example and its
  one documented limitation (connectors don't yet enforce directory
  excludes at ingest time, only the plan's counts do).
- **`corpus-survey` CLI.** Read-only reconnaissance for deciding what to
  index, replacing the ad-hoc shell pipelines that work was previously done
  with. Four subcommands: `census` (file-extension counts/sizes split into
  indexable / gap-with-no-connector / known-noise, derived live from the
  connector registry so a new connector needs no update here to be picked
  up), `archives` (per-zip member count, dependency/build noise, and a noise
  ratio, reusing the zip connector's own `_is_archive_noise` /
  `_is_dependency_noise` logic rather than re-deciding it, and never
  extracting a byte — `zipfile.ZipFile` reads only the central directory),
  `media` (audio/video file counts AND estimated total hours, reservoir-
  sampled per extension and probed with `ffprobe`, extrapolated with a
  stated sample-size-vs-population caveat, plus a `--rate` processing-time
  projection; degrades to counts-only when ffmpeg isn't installed), and
  `overlap` (samples distinctive phrases from a directory and checks them
  against an existing database's `chunks.content` via FTS5 recall + literal
  substring confirmation, reporting a 95% Wilson-interval confidence range
  rather than a bare percentage). Never follows symlinks, matching corpus's
  own ingestion discovery; streams rather than accumulating file lists, so a
  tree with hundreds of thousands of files doesn't exhaust RAM; permission
  errors, broken symlinks, and corrupt archives are counted, never crash the
  run. Human-readable output by default, `--json` for scripting. See the
  "Survey: deciding what to index" section of the README.
- **`csv` / `tsv` connectors.** Stdlib `csv` only — no pandas. Deliberately
  does NOT index every row: a naive dump of a 50,000-row export would produce
  thousands of near-identical row chunks that crowd out real prose in every
  subsequent search across the whole corpus, not just within that file. A
  file with at most 100 data rows AND at most 50,000 characters of rendered
  row text is indexed in full (still chunked normally afterward); above
  either cap, the document becomes filename + inferred column names/types +
  row/column counts + a fixed 20-row sample (10 from the head, 10 from the
  tail), regardless of how large the file actually is. The sample is
  deterministic head/tail, not random, specifically because this engine
  content-hashes chunks and skips re-embedding unchanged ones — a random
  sample would change the rendered body (and its hash) on every ingest run
  even when the file itself hasn't changed. Handles: no header row (detected
  via `csv.Sniffer`, with a fallback heuristic — biased toward NOT assuming a
  header on an ambiguous single-column file, since wrongly assuming one
  discards a real value, while wrongly assuming the reverse just leaves one
  value oddly placed in the sample); a single-column file (no special-casing
  needed — it's just the N=1 case of the same column-inference/sampling
  logic); inconsistent row lengths (flagged in the body, never dropped —
  short rows are missing trailing columns, long rows' extra cells still
  appear in the data/sample); embedded newlines in quoted fields (correct by
  construction, since `csv.reader` gets the whole decoded file at once, not
  split into lines first — flattened to a space only when a row is rendered
  into the body, so one row is always one line of text); non-UTF-8 encodings
  (latin-1 fallback, which cannot itself fail to decode); a NUL byte
  anywhere (binary content under a `.csv` extension can never become valid
  CSV, so — like pptx's legacy `.ppt` case below — this is `skipped_files`,
  not `failed_files`); and an empty file (skipped like an empty docx/xlsx,
  not counted as a failure — though a file with a header row and zero data
  rows is indexed, not skipped, since the schema alone is still real
  content). `tsv` registers the same connector class under its own default
  glob and a tab fallback delimiter; either way the actual delimiter is
  auto-detected per file via `csv.Sniffer`.
- **`pptx` connector.** Uses `python-pptx`. One deck becomes one
  SourceDocument with one markdown `##` section per slide, so a retrieval
  hit can identify which slide it came from; small adjacent slides still get
  packed together by the shared markdown chunker's own coalescing, same as
  xlsx's per-sheet sections. Extracts slide text AND speaker notes — notes
  are not an afterthought here, since a deck's notes routinely carry the
  actual narrative a slide's bullet fragments only gesture at. Walks tables
  and recurses into grouped shapes (`MSO_SHAPE_TYPE.GROUP`) so text nested
  inside a group isn't silently dropped. A slide with a title but no other
  content still gets a one-line section (section-divider slides are real,
  searchable structure); a slide with nothing extractable at all is dropped,
  and a deck where every slide is like that (image-only) is skipped
  entirely, same treatment as an empty docx. Legacy binary `.ppt`
  (OLE2/CFBF) is a fundamentally different container format `python-pptx`
  can never read — and can't be told apart from a genuinely corrupt `.pptx`
  by the exception it raises alone (verified: both raise the identical
  `PackageNotFoundError`), so this connector checks the OLE2 magic bytes
  directly before ever calling into `python-pptx` and counts a match in
  `skipped_files` (permanent — the bytes won't change on a retry) rather
  than `failed_files` (possibly transient — corrupted mid-write, a
  genuinely malformed `.pptx`). The default glob is `**/*.pptx` only (like
  docx's `.doc`, a `.ppt` file is invisible unless a glob is deliberately
  widened or a file was misnamed into a `.pptx` glob).
- **`zip` connector.** Makes documents inside `.zip` archives searchable by
  extracting each archive to a temp directory, re-running the existing
  per-file-type connectors (`pdf`, `docx`, `xlsx`, `html`, `markdown`, `text`,
  `rtf`) against the extracted tree, and deleting the extracted copies —
  composition, not reimplemented parsing. The archive on disk is only ever
  opened for reading. Chunk `source_key`s encode both the archive and the
  inner path (`reports.zip::q3/summary.pdf`) so two archives containing a
  same-named file can't collide and a search hit stays traceable to its
  archive. Guards against the standard archive-extraction failure modes:
  zip-slip (path containment checked via `Path.resolve()` +
  `is_relative_to`, not string prefixes), zip bombs (declared AND actual
  uncompressed bytes capped, plus a member-count cap), encrypted members
  (detected via the standard flag bit and skipped individually — never
  prompts for a password, and one encrypted file doesn't take the rest of
  the archive down with it), nested archives (refused at depth 1, not
  recursed into), and same-path collisions during extraction (two members
  differing only by case, folded onto one file by a case-insensitive host
  filesystem, are disambiguated rather than one silently overwriting the
  other). Extraction happens synchronously per archive inside a
  `try/finally`, so a crash mid-extraction or mid-read cannot leave
  extracted content on disk. A member is recognized under any accepted
  spelling of its type, matched case-insensitively (`.htm` alongside
  `.html`, `.markdown` alongside `.md`, `.PDF`/`.DOCX`-style uppercase from
  non-Unix tooling) — `.doc`/`.xls` are deliberately not treated as
  spelling variants, since they're different container formats
  python-docx/openpyxl cannot read. Archive/OS packaging artifacts — macOS's
  `__MACOSX/` AppleDouble resource-fork tree and `.DS_Store`, Windows'
  `Thumbs.db` — are filtered out before type matching and before either
  counter, since an AppleDouble stub otherwise still matches its twin's
  extension glob, reaches the real connector, fails to parse, and lands in
  `failed_files` — which suppresses orphan pruning for the whole source,
  permanently, for files that were never documents. Vendored-dependency and
  build-output members get the same before-either-counter treatment, but are
  overridable (`exclude_dependencies`, default on): any path component that
  is `node_modules`, `site-packages`, `vendor`, `bower_components`, `.git`/
  `.svn`/`.hg`, `__pycache__`/`.tox`/`.venv`/`venv`, or `.next`/`.nuxt`, plus
  `dist`/`build`/`target` when a matching ecosystem manifest
  (`package.json`/`pyproject.toml`/`setup.py`/`Cargo.toml`/`pom.xml`) is also
  present in the archive, plus `*.min.js`/`*.min.css`/`*.map` leaf files.
  Measured on a large multi-archive sample where most members were
  dependency/build artifacts — one archive's entire 243 "documents" were
  third-party npm package READMEs that would otherwise have been chunked,
  embedded, and indexed as if they were the user's own content.
- **`aup3` connector.** Audacity 3 project files are SQLite databases holding
  raw audio directly in a `sampleblocks` table — nothing (not ffmpeg, not any
  existing connector) could read one before this, so their content was
  entirely invisible. `.aup3`'s `sampleformat` 262159 (`floatSample`) stores
  `samples` as little-endian float32 — verified by hand against a real
  project, and self-verifying in code: decoding a block and recomputing its
  min/max/RMS is checked against the block's own stored
  `summin`/`summax`/`sumrms`, and a mismatch is treated as an unsupported
  format rather than guessed at. `int16Sample`/`int24Sample` blocks are
  recognized but deliberately not decoded (their on-disk layout wasn't
  independently verified the way floatSample's was). Ships as a connector
  plus a companion `extract_audio()` API, not a connector that extracts
  automatically: `load()` yields one lightweight metadata document per
  project (duration, block count, sample-format verdict, source path) from
  cheap SQL aggregates alone, so ingest stays fast even over a large archive
  of recordings; `extract_audio()` is a separate, explicitly-called function
  that decodes every floatSample block (in `blockid` order — real projects
  have non-contiguous blockids from deleted audio) and writes a normal,
  playable file — WAV by default (16-bit PCM, stdlib only), FLAC/MP3 via
  `ffmpeg` when available — adjacent to the source by default, since the
  point is that a person can play it. The source `.aup3` is opened via a
  `mode=ro` SQLite URI and is never written, migrated, moved, or deleted —
  these are irreplaceable recordings; a test asserts the source's mtime and
  size are byte-for-byte unchanged after extraction. Sample rate and channel
  count are NOT recoverable from the project file (they live only in
  `project.doc`, Audacity's own unparseable binary-XML dialect with a
  string dictionary) — defaults to 44100 Hz mono, both overridable per
  source (`sample_rate`, `channels` in `corpus.toml`), documented as
  assumptions rather than pretended detection. Handles a corrupt/truncated
  database (`failed_files`, transient), a database that just isn't an
  Audacity project (`skipped_files`, permanent), a project with zero sample
  blocks (a valid, empty project — reported, not an error), and a
  mixed-format project (`load()` reports the split; `extract_audio` refuses
  the whole file rather than writing a WAV with an unannounced gap where the
  unsupported blocks would have been).
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
- **`[performance]` SQLite memory-tuning pragmas** (`cache_size_mb`,
  `mmap_size_mb`, `temp_store_memory`), applied on every `ChunkStore`
  connection. Isolated each pragma individually on a synthetic
  150,000-chunk/711MB store rather than shipping a bundled guess:
  `mmap_size` turned out to be responsible for effectively the entire
  effect (measured 3.18x-3.29x on vector search once its window covered
  the whole store; only 1.37x covering about a third of it — `corpus`'s
  vector index does an exhaustive per-query scan with no ANN index, so
  there's no "hot" subset for a bigger page cache to help, but
  memory-mapped I/O still turns every page touch into a direct memory
  read). `cache_size` and `temp_store` both measured no effect on this
  workload and default conservatively; `mmap_size` defaults generously
  (1GiB) since — unlike `cache_size` — it's a ceiling on a lazily-paged-in,
  evictable mapping rather than a memory reservation, so a large value is
  safe even on constrained hardware. Raise `mmap_size_mb` if your
  `corpus.db` is bigger than 1GiB to get the full benefit. See
  [`configuration.md`](docs/configuration.md#performance--sqlite-memory-tuning).
- **`ChunkStore` warns when a database path resolves inside corpus's own
  package directory or (in a source checkout) its repo root.** That location
  is never correct for a consumer's index — it means personal data is being
  written into a library's own source tree, where it survives only as long
  as `.gitignore` stays correct. Advisory only (logs a `WARNING`, never
  raises), so an existing database at such a path is never locked out of its
  own data by an upgrade. See the new "Where your data lives" README section.
- **Guardrails against personal data landing in this repo.**
  `tests/test_repo_hygiene.py` fails the suite — locally on every `pytest`
  run and in CI — if a `*.db`/`*.db-wal`/`*.db-shm` file or a real root
  `corpus.toml` ever appears anywhere in the repository. A tracked
  `.githooks/pre-commit` (wired up per-checkout via `scripts/install-hooks.sh`)
  additionally blocks committing those, plus `.env`, even via `git add -f`.

### Fixed
- **The `zip` connector was silently missing `pptx`/`csv`/`tsv` members.**
  `_MEMBER_CONNECTOR_TYPES` (which file types get extracted from inside an
  archive) was a hand-maintained tuple that hadn't been updated since the
  `pptx` and `csv`/`tsv` connectors were added — a `.pptx` deck or `.csv`/
  `.tsv` table sitting inside a `.zip` was silently invisible, with no error
  and no count (it landed in the "no matching connector" bucket, same as a
  genuinely unsupported extension). `src/corpus/survey/archives.py`'s
  archive-inspection subcommand shares the same lookup table
  (`_extension_type_map`), so it under-reported "indexable" members inside
  archives the identical way. `_MEMBER_CONNECTOR_TYPES` is now derived from
  `CONNECTOR_REGISTRY` itself (minus a small, explicit, documented denylist
  of entries that are wrong to recurse into — currently `zip`, for the
  obvious recursion reason, and `aup3`, whose extraction API's "adjacent to
  the source" only makes sense for a real, persistent file, not a disposable
  zip-extraction temp directory) instead of a second hand-maintained list, so
  a future connector addition can't drift out of sync with this the same
  way. See `tests/test_zip_connector.py::test_member_connector_types_derives_from_registry_minus_denylist`
  and `::test_pptx_and_csv_and_tsv_members_become_documents`.
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
