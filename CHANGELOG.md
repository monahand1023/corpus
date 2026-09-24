# Changelog

All notable changes to `corpus-rag` are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). This project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Audacity projects are transcribed.** `corpus-transcribe` now finds `.aup3`
  files, mixes their audible tracks to mono at the project's real rate, and
  transcribes them like any recording, so what is said in them is searchable
  through a `transcripts` source. Their duration comes from the project, so
  their deadline scales like any file's.
- **`corpus.connectors.aup3_layout`** reads a project's layout (sample rate,
  tracks, clip offsets, trims, mute/solo) from its binary-XML `project.doc`.
  The `aup3` document now reports real duration, rate and tracks instead of
  assuming 44100 Hz mono, which doubled the duration of every stereo project.
  `extract_audio` writes the same mono mix. `sample_rate` / `channels` apply
  only to a project whose layout cannot be read.
- **`corpus.retriever.assemble_results()`** — the selection pass (source
  dedupe, the per-type diversity cap, backfill, truncation to `top_k`) is now a
  shared function rather than something each consumer reimplements. Hooks cover
  the parts that genuinely differ: `reject` drops a chunk outright (domain
  noise, date or author filters), `source_key` overrides the dedupe identity
  (keying email by THREAD, so one long thread cannot fill an answer), and
  `content_key` adds a second pass collapsing text republished across document
  versions.
- **`corpus.retriever.build_vector_pool()`** — candidate-pool construction,
  likewise shared. One global k-NN followed by a targeted top-up for any source
  type holding fewer than `max_per_source_type` candidates.
- **`ChunkStore.sweep_orphans()`** — prunes orphans AND reports which DOCUMENTS
  stopped being found, computed from the scan it already performs.
  `delete_orphans()` stays as a thin wrapper for callers wanting only a count.
- **`ChunkStore.vanished_documents()`** — which documents disappeared, for the
  one path with no sweep to ride on (a prune suppressed by read failures).
- **`IngestResult.vanished_detail`** — names the documents behind a warning.
  Every other guard reports a magnitude ("19% fewer documents", "412 orphans"),
  which says something moved but not what, leaving an operator to diff
  directory listings by hand.
- **`corpus.mcp_util.UNTRUSTED_PREFIX` and `format_chunk_block()`** — the
  data-not-instructions marker and the chunk renderer, importable by consumers
  with their own MCP server. `mcp_util` exists so they need not import
  `mcp_server`, which builds a FastMCP instance and reconfigures logging at
  import time.
- **`[query_log]` config section** — opt-in, local JSONL beside the store,
  off by default. Records the query, each result's `(source_type, source_key)`
  and elapsed ms: enough to rebuild a gold set from real usage later without
  copying document text into a second place. Recording what someone searched
  for is a decision they make, not one they discover.

### Changed
- **The diversity cap now backfills instead of truncating.** The cap was
  ABSOLUTE — a chunk over it was dropped and nothing replaced it — so the real
  ceiling on any answer was (source types x cap), whatever `top_k` said. On a
  store with few source types that silently removed most of a result set. The
  cap is a PREFERENCE for spread, not a budget on the answer: it is honoured
  first, then any slots it left empty are filled from what it displaced, best
  scoring first. A short result now means the candidate pool was genuinely
  exhausted, which is the only honest reason to return fewer than were asked
  for.
- **The vector candidate pool is adaptive rather than a per-source fan-out.**
  Every chunk has a distance to the query, so a source type holding most of a
  corpus fills the pool by volume regardless of relevance — and a type absent
  from the POOL cannot be recovered downstream, because neither the cap nor its
  backfill invents candidates. Fanning out per source type on every query fixed
  that but cost roughly 2.5x on a narrow corpus to help a minority of queries.
  One global k-NN plus a top-up only for under-served types costs a fraction of
  that, and is FASTER on a corpus with many source types, where a single large
  k-NN beats dozens of small filtered ones.
- **`should_contextualize` floor raised from 50 to 260 tokens.** Derived, not
  guessed: the generated blurb is a near-constant ~141 characters whatever the
  chunk size, so its SHARE of the embedded text is what decides whether it
  helps. Measured harmful where the share reached ~28%, neutral at ~8%. Holding
  the share at or under 12% needs content of at least ~1,034 characters, which
  is ~260 tokens in the units `token_count` records.

### Fixed
- **An encrypted PDF switched orphan pruning off permanently.** `failed_files`
  means "might succeed next time" and suppresses pruning for the whole source,
  which is right for a locked file or a transient I/O error. A
  password-protected PDF is not that: no future run will have the password, so
  counting it there gated pruning off forever — a source with a handful of
  encrypted PDFs could never remove a deleted document from its index again, on
  any run. Encryption now counts as `skipped_files` (reported and visible, does
  not gate pruning). Other read failures are unchanged.
- **The `[contextual]` config section was parsed and silently discarded**, so
  every value in it was unconfigurable and the default silently stood.
- **Ingest warnings now name the documents involved, unconditionally.** Every
  guard fires on a RATIO, which leaves two shapes invisible by construction: a
  drop UNDER the yield-drop threshold, and substitution — N documents replaced
  by N others, leaving document count, chunk count and orphan ratio all
  unremarkable while content turns over. Both are now reported, and both count
  toward the CLI exit code; without that they would print and still exit 0.
  The reporting is free — it rides on the scan the orphan sweep already makes.

### Added (earlier in this cycle)
- **Ingest reports a rise in permanently-skipped inputs.** `skipped_files`
  means "this connector will never read these", which is why — unlike
  `failed_files` — it does not suppress pruning. That is right for a format
  never supported and wrong for a file read successfully last week: a parser
  regression, a permission change, a dropped optional dependency, or a new
  skip rule shipped in the engine reclassifies it, and pruning then deletes
  content still sitting on disk. A rise in the count alongside an actual
  prune is now reported. Not blocking — the blast-radius guard covers the
  catastrophic version; this is for the handful-of-files case that slips
  under it.
- **`corpus-ingest` exits 3 when a guard reported something.** A warning
  nobody reads is theatre for an unattended run: cron and CI see an exit
  code, not stderr. Kept distinct from 1 so a caller can tell "this did not
  work" from "this worked and you should look at it". `--prune-anyway`
  acknowledges the warnings and clears it, so there is a way to make the
  signal go away other than ignoring it.
- **`music` connector** (`[music]` extra, uses `mutagen`). One document per
  album, built from tags only — no transcription, no audio analysis. A song's
  audio is not searchable text, and a per-TRACK document would be a title,
  an artist and a number: too thin to retrieve on, and 4,000 of them would be
  near-identical chunks competing with each other. An album is the unit
  people actually ask about, and its track listing gives the document enough
  text to match.
  - Album identity comes from the DIRECTORY, not the tags. Music is
    near-universally laid out `Artist/Album/track.mp3`, the directory is what
    `source_key` needs anyway, and tags disagree constantly — half an album
    tagged "The Beatles" and half "Beatles, The" would otherwise split in
    two. Field values are the commonest across the album's files, so one
    mistagged track cannot rename a record.
  - The same logical field has three unrelated spellings (ID3 `TALB`, MP4
    `©alb`, Vorbis `album`), so every lookup goes through one table.
  - **A directory whose files carry no album or artist tag is skipped, not
    emitted.** A real library mixes voice memos and app exports into the same
    containers; measured on one, `.m4a` files included VoiceMemos captures
    with no music tags at all. Those are transcription material, and an
    "Unknown Album" document for them would be unsearchable AND misleading
    about what the library holds.
  - Extensions are swept one glob at a time rather than combined:
    `Path.glob` has no alternation, and any single pattern wide enough for
    `.mp3` and `.flac` also matches `.md`, which made `--path` detect a music
    source in a folder of notes.
- **Yield-drop warning on ingest.** Each run records what a source yielded
  in a new `source_yield` table and warns when the next one produces
  materially fewer documents. The orphan-prune guard already refuses a
  destructive sweep that looks too large; this covers the case it cannot
  see — when pruning is suppressed because the connector reported unreadable
  files, a collapse in yield deletes nothing, reports no failure, and looks
  exactly like a healthy run. Advisory only, never blocking, since emptying
  a folder on purpose is a normal thing to do. Reuses `[pruning]`'s existing
  ratio and floor rather than adding a second pair of knobs that could
  disagree with the first. Recorded only after a run completes, so an
  aborted ingest cannot install a low-water mark that makes the next run's
  collapse look normal.
- **Contextual-Retrieval coverage in `corpus_stats`.** The MCP tool now
  reports how much of each source has been contextualized, since an
  uncontextualized source retrieves noticeably worse on fragments and that
  is a property of the index rather than the query. Shown only where some
  coverage exists, so an install that has never run `corpus-contextualize`
  sees no extra output.
- **Contextual Retrieval (`corpus-contextualize`).** For each chunk, a cheap
  model reads the chunk together with its parent document and writes one
  sentence situating it; the sentence is stored in a new `context` column and
  the chunk is re-embedded as context + content. This targets the structural
  problem with chunking prose: a fragment reading "Yes, approved, let's go
  with option B" names no project, person, or date, so no query for any of
  them finds it. Anthropic's published result is a ~35% reduction in
  retrieval failure rate, ~49% combined with reranking (which
  `corpus.reranker` already provides).
  - Affordable through prompt caching: each request sends the parent document
    as a cached prefix billed at ~10% on re-reads, so a document pays full
    price once and its remaining chunks are ~90% cheaper. This is why
    `chunks_missing_context` orders by `(source_key, chunk_index)` and the
    batch builder windows by document — break that adjacency and cost
    multiplies by chunks-per-document.
  - Resumable: batch ids and their chunk mappings are persisted before each
    submit, so an interrupted run never re-pays for results already computed.
  - `--dry-run` prices a run before it spends anything, counting each parent
    document once rather than once per chunk.
  - `--clear` removes stored contexts and restores plain-text FTS rows. It
    cannot restore pre-context embeddings (overwritten in place) and says so.
  - A `min_tokens` floor (default 50, per-source overridable) skips chunks a
    context would harm rather than help — below it the blurb rivals the
    content and the embedding describes the blurb. Measured on a real
    archive, source code needs a far higher floor than prose: a code chunk
    already carries the identifiers people search for, and on one corpus it
    was 65% of all chunks with the least to gain.
  - `content` is never rewritten, so a bad run is reverted by clearing one
    column instead of re-ingesting.
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
  it at a folder" section of the README for the full worked example.
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
- **`timeline` returned nothing when recent material was not the most
  semantically central.** The date filter ran in Python after retrieval, so a
  topic whose nearest chunks all fell outside the range was filtered to empty.
  Measured: 200 near-but-old chunks and 10 further-but-recent ones,
  `since=` returned 0 of the 10. "What happened lately about X" asks for
  recent material, which is rarely the most central — this was the ordinary
  case for the method, not a corner one. The candidate pool now widens while
  the filter is starving it, and stops when the pool comes back short.
- **`timeline` was silently capped at 3 results per source type.** It went
  through `query()`, which applies the diversity cap by default, so a
  20-item timeline could never return more than 3 x (number of source types)
  candidates regardless of `top_k`. A timeline is one topic ordered by date;
  spreading across source types is not what it is asked for.
- **A large `top_k` no longer crashes vector search.** sqlite-vec's `vec0`
  refuses a KNN query with `k` above 4096 (`OperationalError: k value in knn
  query too large`) and nothing bounded it, so a sufficiently large request
  killed the whole retrieval instead of returning the most it could. Found by
  a widening candidate pool reaching 7,680.
- **A source-filtered full-text search no longer returns nothing when the
  filter is crowded out.** `fts_search` applied its source filter in Python
  AFTER SQLite's `LIMIT`, so if the top-ranked rows all belonged to other
  sources the filter removed everything and the caller got an empty list with
  matching content sitting in the store. Measured: 100 chunks of one source
  and 20 of another, all containing the query term, filtered to the smaller
  source returned ZERO of its 20. The fast path is unchanged and still runs
  for virtually every query; when it comes up short, a pre-filtered query
  runs instead. Benchmarked at 40k chunks across 4 source types: unfiltered
  51 ms, uncrowded filter 55 ms, crowded filter 110 ms and correct — against
  0 results before.
- **Using a `ChunkStore` after `close()` now says so.** `close()` shuts down
  every connection but cannot reach into another thread's `threading.local()`
  to clear its reference, so a thread that already had a connection failed
  with sqlite3's "Cannot operate on a closed database" from a call site
  unrelated to closing. It now raises a message naming the actual mistake,
  and deliberately does not reopen silently — use-after-close is a caller bug
  and resurrecting the store would hide it.
- **A bad ingest no longer becomes the new normal.** The yield baseline was
  recorded after every run, so one collapsed run reset the bar: the next
  equally bad run compared favourably and said nothing, and repeated losses
  just under the prune guard's ratio could walk a source down to nothing —
  each step looking healthy — until it fell below `min_chunks_for_guard` and a
  single run could delete the remainder. The baseline now advances only after
  a run with no unreadable files, no refused prune, no yield drop and no path
  change. `--prune-anyway` is the acknowledgement gesture that accepts an
  anomalous run as the new baseline.
- **The yield check now watches chunks as well as documents.** It recorded
  both and evaluated only documents, so a parser regression that still yielded
  every document while extracting a fraction of the text from each passed
  unremarked — overwriting good content with thin content at an identical
  document count.
- **An expensive FTS rebuild is no longer something a constructor does.**
  Opening a store rebuilt a stale full-text index automatically. That is right
  for a small store — a migration you must remember is one that gets skipped —
  and wrong for a large one: the rebuild is a single transaction holding
  SQLite's only writer lock from first delete to final commit, so on a
  million-chunk store it is minutes during which any concurrent ingest fails
  on its busy timeout, triggered by any script, test, or MCP server that
  happens to open the store read-write. Above
  `AUTO_FTS_MIGRATION_MAX_CHUNKS` (200,000) the store now reports the stale
  index loudly and leaves it alone; `corpus-migrate-fts` performs it
  deliberately. The warning names the hazard SQLite cannot catch: deploying
  the migration before deploying the code means an older writer appends
  unnormalized rows to a store stamped as current, reintroducing the defect
  silently for new content.
- **A failed `ChunkStore` construction no longer leaks its connection.** If
  schema setup or a migration raised, the connection was never closed and its
  open write transaction held SQLite's single writer lock until garbage
  collection — for the life of the process, in a long-running server. Found
  by a test that crashed an FTS rebuild mid-way and then could not reopen the
  database at all.
- **A contextualized chunk no longer drops out of BM25.** `set_context` wrote
  the combined context+content into `chunks_fts` RAW, while `upsert` writes
  `normalize_for_fts(content)` and the query path searches for that normalized
  form. Since CJK runs are indexed as overlapping bigrams, a raw row is
  unreachable by any CJK query — so a Japanese chunk silently vanished from
  full-text search the moment it gained a context. ASCII normalizes to itself,
  which is why every English test passed. `clear_context` had the same defect
  on its restore path.
- **Distinct folders no longer collapse onto one source name and overwrite
  each other.** `normalize_source_name` stripped every leading non-letter, so
  `2023 Taxes` and `2024 Taxes` both became `taxes`; and it fell back to a
  constant `"folder"` for any name with no ASCII letters, so every
  Japanese- or Chinese-named folder became `folder`. Chunk ids derive from
  `(source_type, source_key, kind, index)`, so two such folders each holding a
  `notes.txt` produced identical chunk ids and the second ingest silently
  OVERWROTE the first's content — nothing pruned, nothing deleted, both runs
  reporting one document and looking entirely healthy. Neither the
  blast-radius guard nor the yield-drop warning can see an overwrite.
  Demonstrated end to end, then fixed: a leading digit is prefixed rather than
  stripped, and a name with no usable ASCII gets a digest of the original.
- **Ingest warns when a source name is reused for a different path.** The
  normalizer keeps distinct folder NAMES apart, but cannot help when two
  folders in different places genuinely share one (`~/a/Notes` and
  `~/b/Notes`) — the commonest collision of all. The store now records each
  source's resolved path and the next run reports a change. A warning rather
  than a refusal: moving a folder is legitimate and produces the identical
  signal.
- **Office lock files and renamed legacy documents no longer block orphan
  pruning forever.** A file a connector cannot read is reported as
  `failed_files`, which suppresses pruning — correct for a momentarily-locked
  file, wrong for one that will never be readable, because it recurs
  identically on every future run and the source can then never prune a
  genuinely deleted document. Measured on a real archive, the `.docx` files a
  source could not open included a Word owner-lock file (`~$name.docx`, not a
  document at all and invisible in Finder) and three legacy `.doc` files
  renamed rather than converted. New `corpus.util.ooxml` classifies both as
  permanent, shared by the docx, pptx and xlsx connectors so they cannot
  drift on what "permanently unreadable" means. `pptx.py`'s own OLE2 check is
  now that shared one rather than a second copy.
  - The bar is deliberately high and asymmetric: "empty" and "not a zip" stay
    TRANSIENT, because a large file mid-copy is not a valid zip yet and an
    interrupted copy leaves a zero-byte file. A stale suppressed prune leaves
    an out-of-date index and is recoverable with `--prune-anyway`; a wrong
    permanence call lets the next prune delete a real document's chunks.
- **One malformed shape no longer costs an entire presentation.** python-pptx
  reports `has_text_frame` / `has_notes_slide` as True while the frame itself
  is None, and `.text` can raise on a malformed shape — either of which
  raised out of the slide loop and lost the whole deck. Measured: a real
  40-slide presentation yielding nothing because of a single placeholder. A
  bad shape now costs only that shape.
- **Byte-order marks are honoured before the text-decoding ladder.** A
  BOM-carrying UTF-16 file previously fell through UTF-8 to CP932 or
  latin-1, both of which "succeed" on UTF-16 bytes and return one NUL per
  ASCII character — worse than the `errors="replace"` behaviour the ladder
  replaced, because U+FFFD announces the damage while mojibake looks like
  text to a search index. Measured over 66 mis-decoded files in one
  archive: 1,811 stray NUL characters across 8 files became 65 in 1 (a
  1995 telnet capture with genuinely embedded NULs). Also strips UTF-8
  BOMs that were riding into the first chunk of a document as U+FEFF.
- **NUL characters are stripped from chunk content.** 239 chunks in one
  archive carried them, all from PDFs whose fonts had no usable encoding.
  FTS5 indexing and terminal display both truncate at the first NUL,
  silently hiding the rest of an otherwise-fine chunk. Applied in
  `MarkdownChunker`, which every connector routes through, before the
  content hash is taken.
- **`corpus-ingest` / `corpus-index` no longer bury their output in
  extraction noise.** pypdf logs one ERROR per page for an unimplemented
  CMap and trafilatura one per empty HTML fragment, neither actionable.
  One archive of Japanese PDFs emitted ~2,000 such lines, scrolling every
  per-source summary off the screen; the same run now prints 12. Raised to
  CRITICAL rather than silenced, and `--verbose` restores them in full.
- **`zip` and `pptx` cost estimates re-measured against many real sources.**
  `zip` was 0.0001 chars/byte, derived from a single mostly-binary archive
  and 107x below the aggregate across four real zip sources — an
  UNDER-estimate, the direction `text_yield`'s own docstring calls the
  worse error, since the user only finds out after being billed. `pptx`
  was 0.0287, inherited from `docx` because both are OOXML containers; the
  container is shared but the content is not, and one 176 MB source
  estimated at 1.26M tokens actually cost 15,487.
- **`corpus-index`'s cost estimate overstated by up to ~400x for compressed
  formats.** The estimate was a flat `file_size / 4`, i.e. "roughly one
  character of text per byte of file" — true for plain text/markdown, wildly
  false for PDF/DOCX/XLSX/ZIP/etc., which pack images, fonts, and XML/binary
  scaffolding into their file size alongside (or instead of) extractable
  text. Measured against a real, mixed large mixed corpus: PDFs yield about
  0.16 characters of text per 100 bytes of file, not ~100. Concretely, a
  large PDF-heavy source was reported as an alarmingly largetoken estimate (an operation to think twice about) when its actual extracted
  text was a tiny fraction of that (a few cents). An estimate that scary
  makes people decline runs they should just do — the opposite of what a
  cost estimate is for. New `corpus.util.text_yield` module holds a
  per-connector-type text-yield ratio (chars extracted per byte of source
  file) from that measurement, reused by both `corpus.planner` (the
  `corpus-index` plan) and `corpus-survey census`'s indexable-bucket table
  (now shows an `est. tokens` column and JSON field, same calibration, so
  the two tools agree). Deliberately biased against understating: an
  unmeasured connector type defaults to the old ~1:1 assumption (the safe,
  conservative direction) rather than a guessed-low ratio, EXCEPT where
  that would reproduce this exact bug for a different reason — `aup3`
  project files are mostly binary audio sample data with a small,
  roughly-fixed-size generated description, so defaulting to ~1:1 there
  would "estimate" a large project at hundreds of millions of tokens for a
  few hundred words of actual output; it gets a low, reasoned default
  instead (see the module docstring). Final token counts always round up,
  never down. `corpus-index`'s and `corpus-survey census`'s printed output
  now state the method and its uncertainty explicitly (a scanned PDF with
  no text layer yields close to nothing until OCR'd, for instance) rather
  than presenting a single number as exact.
- **`corpus-index` claimed noise directories were "not ingested" — they
  were.** The plan (`corpus.survey.census`, via `corpus.survey.walk`)
  already excluded `node_modules`, `.git`, build caches, `.photoslibrary`
  bundles, etc. from its counts, but `corpus.connectors.discovery.discover_files`
  — which every file connector uses to find its files — had no directory-exclude
  mechanism at all, so the actual ingest walked straight into those same
  directories anyway. On real trees this wasn't cosmetic: measured noise
  ratios as high as ~80% dependency/build output, and one archive's entire
  document yield was third-party library READMEs. `discover_files` now
  applies the identical default exclusion (shared with `corpus.survey.walk`
  via the new `corpus.util.exclude` module, so the two can't drift back out
  of sync), including the same `dist`/`build`/`target`
  corroborated-by-manifest judgment call `corpus.connectors.zip` already
  made for archive members — `dist`/`build`/`target` are ordinary English
  words too, so they're excluded only when a `package.json`/`pyproject.toml`/
  `setup.py`/`Cargo.toml`/`pom.xml` confirms it's really build output, not a
  personal folder that happens to share the name. On by default for every
  connector, overridable per call via `discover_files(..., use_default_excludes=False)`
  (not yet exposed per-source in `corpus.toml` — see `corpus/cli/index.py`'s
  module docstring). **Upgrade note:** if an existing source previously
  picked up files inside what's now an excluded directory, the next
  `corpus-index`/`corpus-ingest` run against it will prune those chunks as
  orphans — expected, and caught by the existing orphan-pruning
  blast-radius guard (refuses a drop over `[pruning].max_orphan_ratio`,
  default 20%, rather than deleting it silently) if it's a large fraction
  of the source.
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
- **Non-ASCII filenames inside `.zip` archives were being corrupted into
  mojibake, live in a real index.** Measured: 77.3% of one archive source's
  chunks and a few percent of another — roughly a few thousand chunks total.
  Root cause: many real-world zip tools write non-ASCII
  filenames as raw UTF-8 (or, for older Japanese-locale tools, Shift-JIS/
  CP932) bytes WITHOUT setting the standard "filename is UTF-8" flag bit
  (0x800) in the member header — `zipfile` then decodes the name as CP437
  per the ZIP spec, producing garbage. Since the member name becomes both
  the extraction path and (via `_load_extracted`) the chunk `source_key` and
  document title, this meant the affected documents were unfindable by their
  real names and would re-ingest under a different key if the misdecoding
  ever changed. Fixed by `_repair_filename_encoding`: every member name is
  checked once, up front — if the UTF-8 flag isn't set, the CP437-decoded
  name is re-encoded back to its original bytes (CP437 round-trips any byte
  0-255 losslessly) and a UTF-8 or CP932 (Shift-JIS) strict decode of those
  bytes is accepted ONLY if it both succeeds AND differs from the original,
  so an archive whose non-flagged name genuinely is CP437/ASCII (the common
  case) is never touched. Every downstream use of the name (zip-slip
  containment, noise/dependency filtering, extraction, `source_key`) sees
  the repaired name automatically, since the repair runs once against the
  archive's own `ZipInfo` objects before anything else reads them.
- **The same real index had ~970 chunks of U+FFFD replacement-character
  soup from non-UTF-8 file content** — found while chasing the mojibake bug
  above; the same Japanese-locale archives had Shift-JIS-encoded member
  *content*, not just corrupted names. `markdown`/`text`/`html`/`rtf` all
  read every file as `path.read_text(encoding="utf-8", errors="replace")`,
  silently substituting U+FFFD for any byte sequence that wasn't valid
  UTF-8 — not a zip-specific bug, since the zip connector reuses these same
  connectors on extracted content (any real `.txt`/`.md`/`.html`/`.rtf` file
  with non-UTF-8 encoding on disk was affected too). New shared
  `corpus.util.encoding.read_text_with_fallback` tries UTF-8, then CP932
  (the same second tier as the filename repair above), then latin-1 as a
  final backstop that cannot itself fail to decode — deliberately not
  `chardet`/`charset-normalizer`, since this fixes the two encodings
  actually measured in a real archive rather than adding a dependency for
  every encoding that has ever existed. All four connectors now call it
  instead of duplicating the old `errors="replace"` pattern.
- **One unreadable PDF no longer aborts an entire source.** `pypdf` and
  `python-docx` parse lazily: the constructor succeeds on a file that cannot
  actually be read, and the failure surfaces later on first access to `.pages`
  or `.paragraphs`. Both accesses sat outside the per-file guard, so an
  encrypted PDF took down the whole source — and since
  `pypdf.errors.FileNotDecryptedError` derives from `Exception` rather than
  `ValueError`/`OSError`, it escaped the CLI's per-source handler too and would
  abort an entire `--all` run. Found by ingesting a real archive of PDFs, where
  a single encrypted file left the source with zero chunks; after the fix,
  thousands of documents and tens of thousands of chunks.
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
