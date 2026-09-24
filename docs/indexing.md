# corpus-index: point it at a folder

`corpus-index` is the one-command path from "here's a messy folder" to
"it's searchable" — the layer above `corpus-survey` and `corpus-ingest
--path` that does not require knowing either exists:

```bash
corpus-index ~/Downloads/export
```

It surveys the directory (reusing `corpus-survey census`), then prints,
**before touching anything**:

- **the gap** — file types corpus has no connector for, with counts and
  sizes. This is deliberately the first thing printed: it's the single most
  useful fact about a real directory, and a user should never have to ask
  for it separately.
- **noise** — how many directories (`node_modules`, `.git`, caches,
  `.photoslibrary` bundles, ...) were excluded by default, plus any
  loose-file noise (`.DS_Store`, minified JS, ...) found alongside real
  content.
- **the plan** — exactly which connectors will run over which files, with
  per-source file counts, sizes, and an estimated token count. Calibrated
  per format from a real, measured corpus rather than a flat file-size/4
  guess — a compressed container format like PDF or DOCX extracts to a
  small fraction of its file size (packed with images, fonts, XML
  scaffolding), so a flat guess overstated real PDF-heavy sources by
  roughly 400x in practice. Still an estimate, not the embedder's real
  tokenizer count, and rounded up rather than down when uncertain — meant
  to catch a real surprise before it happens without inventing a fake one.
- optionally, with `--check-overlap corpus.db`, an estimate of how much of
  the directory is already indexed somewhere else (reusing `corpus-survey
  overlap`) — so you can skip paying to re-embed content you already have.

Nothing is written or ingested until you confirm (`[y/N]`), pass `--yes`, or
until you decide `--dry-run` is enough and stop there:

```
corpus-index: /Users/you/Downloads/export

Gap — no connector (report this first; it's the whole point)
  extension                 count         size
  .csv                         913        41.2 MB
  .pptx                        259        88.0 MB
  -> the file(s) and megabytes that corpus cannot index today and will NOT
     be searchable after this run.

Noise (excluded from the plan, not ingested)
  4 directories excluded by default (node_modules, .git, caches, ...)

Plan — sources that would be written to corpus.toml and ingested
  name                     type      files       size   est. tokens
  export_markdown          markdown    340     6.1 MB      1,600,000
  export_pdf                pdf         52    18.4 MB          7,400
  TOTAL                                392    24.5 MB      1,607,400

  Estimated tokens = raw bytes × a per-format text-yield ratio measured
  against a real corpus (pdf ~0.2%, docx ~2.9%, html ~35.3%, text ~99.5% —
  see corpus.util.text_yield), ÷ 4. Not the embedder's real tokenizer
  count; rounded up rather than down when uncertain.

Write these sources to corpus.toml and ingest? [y/N]
```

(Illustrative numbers — run it against your own directory. Note how little
the 18.4 MB of PDFs actually costs to embed compared to the markdown, even
though it's a bigger source by file size — that gap is exactly what a flat
file-size/4 estimate used to hide.)

Confirmed sources are **merged into corpus.toml** (`[[sources]]` blocks are
appended, existing ones are never touched), not ingested transiently — a
one-off ingest nobody can repeat is a trap, since re-running the same
command is how you pick up files added or changed later. Re-running
`corpus-index` on a directory you already indexed is a no-op on the config
(reported as "already configured") and just re-ingests, picking up changes.

Source names follow the same folder-basename namespacing as `corpus-ingest
--path` (see below) — `export_pdf`, not `pdf` — but because `corpus-index`
*persists* sources across runs, two differently-located folders sharing a
basename (`~/Work/Inbox` and `~/Personal/Inbox`) can now actually collide in
one corpus.toml, which the transient `--path` mode never had to worry
about. `corpus-index` refuses that merge outright rather than guessing which
folder should win — deleting the wrong folder's chunks via `source_type`-scoped
orphan pruning is a real data-loss footgun — and tells you to pass
`--name-prefix` or edit corpus.toml by hand.

The noise directories excluded from the plan's counts above are excluded
from the real ingest too — every file connector applies the identical
default exclusion when it actually reads a source's files, so "excluded
from the plan, not ingested" is a real guarantee. **Known limitation:**
`--no-default-excludes`/`--exclude PATTERN` change only this preview — there
is currently no per-source way to turn off default exclusion at ingest time
from `corpus.toml`. If you genuinely need a vendored/build tree indexed,
point a source's `path` directly at that subdirectory (exclusion only ever
prunes a directory encountered *during* a walk, never the configured root
itself). If you're upgrading from a version where this wasn't yet enforced
and a source previously picked up files inside what's now an excluded
directory, expect those chunks to be pruned as orphans on the next
`corpus-index`/`corpus-ingest` run — the existing blast-radius guard
(`[pruning]` in corpus.toml) refuses a drop over 20% of a source rather than
silently deleting it, so a large prune will ask you to confirm with
`--prune-anyway` instead of happening invisibly.
