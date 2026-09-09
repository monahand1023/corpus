---
title: "The ingest lifecycle end to end"
id: ingest-lifecycle
---

# The ingest lifecycle end to end

This note walks a single ingest run from the moment `corpus-ingest` is invoked to the moment the last chunk is committed, naming the guard rails it passes through on the way. It is deliberately the longest document in the sample corpus: every other note here fits inside a single chunk, so without one document that genuinely exceeds the size cap the eval could not exercise section splitting, size splitting, coalescing, or overlap at all — and this corpus is what `docs/eval.md` points at when it says to re-run the eval after changing chunking.

## Resolving configuration and credentials

A run begins by locating `corpus.toml`. The `--config` flag wins if given; otherwise the file is read from the current working directory. This matters more than it sounds, because the MCP server is spawned from an arbitrary directory by the calling client, which is why wiring it up is documented with an absolute path.

Credentials are resolved next, in a fixed precedence: an environment variable that is already set wins, then a `.env` file sitting beside the resolved config, then a `.env` in the current working directory. The library never reads secrets out of its own source tree. A checkout of the engine holding a live API key would be a trap laid for the next person who clones it.

## Enumerating the source

Each configured source names a connector type, which the registry maps to a connector and a chunker. The connector walks its root with `discover_files`, which refuses symlinks, refuses any path that resolves outside the source root, and by default skips well-known noise directories such as dependency trees, version control metadata, and build caches.

The single most important property of a connector is that it enumerates completely. Reaching the end of `load()` is taken as proof that every document the source contains was yielded, and everything previously indexed but absent this time is therefore genuinely gone. A connector that cannot enumerate its source must raise rather than yield a short list, because a partial enumeration is indistinguishable from a bulk deletion.

Individual files are a softer case. A connector may skip a file it could not read without raising, and reports those through two separate counters. `failed_files` means the file might succeed on a later run: a lock, a transient read error, a volume that was briefly unavailable. Any non-zero count suppresses orphan pruning for the whole source, because deleting a file's chunks merely because it was busy is data loss. `skipped_files` means the connector has permanently given up: an unsupported format, a lock file that is not a document at all. Those do not suppress pruning, because their absence is not a surprise and counting them as failures would suppress pruning forever.

## Chunking and normalization

Each document is handed to a chunker, which splits the body at heading boundaries, size-splits any section that exceeds the cap, and coalesces small adjacent fragments so a heading-heavy document does not explode into a flood of one-line chunks. Splits never land inside a fenced code block.

Consecutive chunks overlap by a bounded amount so that a sentence spanning a split survives intact in the second piece. Without that overlap, text straddling a boundary is cut in half and matches neither chunk well. The step back is bounded by a fraction of the piece rather than a flat character count, because a paragraph break often lands well short of the cap and a fixed step back off a short piece would be a far larger proportional overlap than intended.

Every chunk then passes through one normalization point shared by every connector. Null characters are stripped, because they are never meaningful document content and truncate both full-text indexing and terminal display at the first occurrence. Credential patterns are redacted before the content hash is computed, so that dedup does not see two copies of the same document as different merely because their secrets differ.

## Embedding and upsert

Chunks are batched. For each batch the store is asked which chunk ids it already holds and at what content hash; anything whose hash is unchanged is skipped without being sent to the embedder at all, which is what makes a re-ingest of a mostly-unchanged source nearly free. The remainder are embedded and upserted, updating the vector index and the full-text index together inside one transaction.

## Pruning, and the guards around it

After enumeration completes, the set of chunk ids seen during the run is compared against what the store holds for that source. Anything absent is an orphan and would be deleted.

Two guards stand in front of that deletion. The first is the failure check already described: if the connector reported any transient failures, pruning does not run at all. The second is a blast-radius guard. If the orphans to be deleted exceed a configured fraction of the source's existing chunks, the prune is refused outright and reported as an error rather than a note, because a deletion that large is far more often a connector bug than a genuine bulk removal. Both can be overridden deliberately for a source whose unreadable files have actually been reviewed.

A third check is advisory rather than blocking. Each completed run records how many documents the source yielded, and the next run warns if that number falls sharply. This covers the case the blast-radius guard structurally cannot see: when pruning is suppressed, a collapse in yield deletes nothing, reports no failure, and produces output indistinguishable from a healthy run, so the index quietly stops matching reality.

## What the run reports

The result carries document and chunk counts, how many chunks were upserted against how many were unchanged, how many files failed and how many were skipped, whether pruning ran and how many orphans it deleted, the tokens billed, and the elapsed time. Anything that suppressed or refused a prune is surfaced explicitly, because the entire point of those guards is that the operator finds out.
