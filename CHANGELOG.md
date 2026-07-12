# Changelog

All notable changes to `corpus-rag` are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). This project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
