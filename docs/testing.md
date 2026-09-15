# Testing a corpus install: what each layer actually proves

Four layers, each blind to what the one below it catches. The order matters —
every layer here was added after a defect slipped through the layer above it.

| Layer | Command | Answers | Blind to |
|---|---|---|---|
| Unit tests | `pytest` | Does the logic do what we said? | Whether any of it is wired up |
| Smoke test | `corpus-smoke` | Does the server start, and answer? | Whether the answers are the right ones |
| Retrieval eval | `corpus-eval` | Can you find what you indexed? | Whether the reply reads well |
| Judge | `corpus-judge` | Is the generated answer any good? | Cost; it is the only layer that bills |
| Index scan | `corpus-survey index-quality` | Did junk get indexed? | Anything that is not a transcription artefact |
| Gold audit | (runs inside `corpus-eval`) | Is the answer key itself broken? | Whether the queries are the right ones |

Run the first two on every change. Run the third when retrieval, chunking, or
the embedder changes. Run the fourth when tuning, not as a gate.

---

## Layer 1 — unit tests

They prove the filtering, chunking and normalisation logic behaves. They
cannot prove the system runs, because they call handlers directly with a
patched store: the MCP handler tests pass whether or not the server can be
launched at all.

This is not a criticism of them; it is the reason the next layer exists.

## Layer 2 — `corpus-smoke`: is it wired up?

```sh
corpus-smoke --config corpus.toml               # this repo's server
corpus-smoke --claude-config ~/.claude.json     # every archive Claude launches
```

Spawns the real command from the real config and talks the real protocol:
handshake, `tools/list`, `corpus_stats`, a search, and a clean exit on stdin
EOF. Exit status is 0 only if every server passes.

What lives between a handler and the user — and is untested by layer 1 —
is the entry point resolving, the config parsing, the DB opening at the path
the config names, the credentials loading, and the JSON-RPC handshake. **Every
one of those fails silently.** A server pointed at a wrong `db_path` starts,
handshakes, lists its tools, and answers every question with nothing found,
which from inside Claude is indistinguishable from an archive that genuinely
lacks the answer.

Three things this command learned the hard way:

- **It pins the working directory.** Credentials load from a `.env` beside the
  config, so a server spawned in the wrong cwd dies during `initialize` with a
  credentials error. Claude launches these with `uv --directory <repo>`; the
  smoke test has to match, or it reports failures the real client never sees.
- **It prefers the script beside the running interpreter over `PATH`.** A
  `uv tool install` puts a `corpus-mcp` in `~/.local/bin` that shadows an
  editable checkout. It is a *different build* — the first run of this command
  reported an error whose text did not match the current source, because a
  stale copy was answering.
- **It treats a thin response as a failure, not a pass.** See
  `_looks_empty`. A server with an empty database says "no results" politely;
  so does a healthy server asked something its archive cannot answer. The two
  are indistinguishable from one response, so the check is conservative and
  **you** pick a probe the archive should be able to answer (`--probe`). A
  live (non-indexed) source especially needs one — a generic probe against a
  live mailbox legitimately returns nothing.

By default the sweep covers only servers backed by this engine or its
archives. A Claude config is shared with every other MCP server installed —
design tools, storefronts, auditors. Reporting on software this project does
not own makes the summary line meaningless, so those are skipped unless you
pass `--all`.

A pass means *retrieval ran and returned something shaped like results*. It
never means the results were good. That is the next layer.

## Layer 3 — `corpus-eval`: can you find what you indexed?

See [eval.md](eval.md) for methodology and [understanding-evals.md](understanding-evals.md)
for the concepts. What belongs here is how to build the gold set, because that
is where the mistakes are:

**Label from the content, never from the filename.** A gold query written from
a filename asserts what you *believe* is in a file. If the belief is wrong the
eval reports a retrieval failure that is really a mislabelled answer — the
expensive kind of eval bug, because it sends you tuning a retriever that is
working correctly. Read what the retriever actually sees, then write the query.

**Paraphrase away from the target's wording.** A query sharing keywords with
its answer tests full-text search, not semantic retrieval. If the transcript
says `撮影`, ask about `写真館`.

**Cover every language in the archive.** A retriever that only works in English
scores well on an English-only gold set and fails in practice.

**Keep the failures in the set.** A gold set with no failures in it measures
nothing. Deleting an inconvenient query is how a baseline becomes decoration.

**Measure at the `top_k` the server actually serves.** The eval defaults are
not the config's `[retriever] top_k`. Measuring at 15 when users get 5 reports
a number nobody experiences.

These are checked automatically. `corpus-eval` audits the gold set before
scoring and refuses to run when the key is broken beyond doubt -- expected
keys that are not in the index at all, which can only ever score zero. Softer
smells are reported and do not block: an answer set that matched nothing, a
round size repeated across sets (a `LIMIT` in the labelling code), a duplicate
query, and a query sharing a three-word run with its own answer. That last one
found a real defect in a shipped gold set on its first run.

The rules behind those checks, and why each one exists:

**Make the answer key COMPLETE.** An answer set is an OR-set: the query hits
if any key in it ranks. That makes an incomplete set actively misleading --
the retriever returns a perfectly good answer, it is missing from the key, and
the eval reports a miss against software that is working. This has now bitten
twice, in two different disguises:

- *Capping the set.* A photo archive's largest trip has 1,456 photos; capping
  answer sets at 400 turned a real 0.625 into a false 0.500 and made the
  largest, easiest query look like a failure.
- *Matching only the first message.* Labelling a mail thread by subject prefix
  missed every `Re:` in it, reducing a 93-message thread to one message and
  another thread to zero.

Both look exactly like retrieval failures. Before tuning anything, confirm the
key is right -- read the source text, and check the size of each answer set.

**Prefer labels from outside the retriever.** The place a photo library
recorded, the subject a sender wrote: these are ground truth produced before
any of this existed. Hand-picking documents you already know the retriever
finds measures your own memory.

**Check whether the eval path matches the served path.** The eval reranks only
with `--rerank`. If the MCP server does not rerank, neither should the eval —
and vice versa, or the number describes a system nobody is running.

### What the numbers look like in practice

Three personal archives, each measured at the `top_k` its own MCP server
serves, hybrid retrieval, no reranker, eight queries each:

| Archive | recall@k | MRR | nDCG@k |
|---|---|---|---|
| transcripts (k=5) | 0.625 | 0.442 | 0.486 |
| photos (k=8) | 0.625 | 0.479 | 0.256 |
| mail (k=8) | 0.875 | 0.342 | 0.216 |

The useful signal is the SHAPE, not the absolute values. Mail has the best
recall and the worst MRR: the right thread is nearly always in the top 8, but
it lands third to sixth. That is a ranking problem, not a retrieval one, and
it is what a reranker is for. The photo archive's low nDCG is partly by
construction -- its ideal ranking fills all 8 slots from the right trip, a
much harder bar than "find one".

Eight queries is a small set. It is enough to catch a regression and to tell
these shapes apart; it is not enough to detect a small improvement, and a
change of a few points between runs is noise, not progress.

## After ingest — `corpus-survey index-quality`

```sh
corpus-survey index-quality --db data/corpus.db
corpus-survey index-quality --db data/corpus.db --source-type transcripts --json
```

The rest of `corpus-survey` asks "what is out there, should we index it?".
This asks the question that only exists afterwards: we indexed it, is any of
it junk? It separates two defects because they are fixed differently.

A chunk that is ENTIRELY a caption sign-off should never have been indexed --
the connector is not filtering, and the command exits non-zero. A chunk with a
sign-off glued to the END of real speech must NOT be dropped, because
discarding those deletes real recordings; the tail is cut and the speech kept,
so this reports without failing.

Both are invisible from outside. The index reports a successful build, search
returns results, and some of those results are text no person ever said. Run
it after importing anything transcribed. The fix is always at ingest -- apply
`corpus.transcripts.strip_caption_tail` in the connector or chunker and
re-ingest, which re-embeds only the chunks whose content actually changed.

## Layer 4 — `corpus-judge`: is the answer good?

An LLM grades generated answers against retrieved context. The only layer that
costs money per run, and the only one whose grader can itself be wrong — see
the validation study and Cohen's κ in
[understanding-evals.md](understanding-evals.md). Use it for tuning decisions,
not as a routine gate.

---

## A worked example of layer 3 catching a layer-3 bug

An eight-query gold set over a personal transcript archive scored
recall@5 = 0.625 — three failures. Investigating each *before* touching the
retriever:

- One was a **bad label**. The query had been written from the filename, which
  named the recording device and implied the subject. The recording was from
  that device, but its contents were nothing like what the name suggested.
  Retrieval was fine; the answer key was wrong.
- Two were **genuine vocabulary misses**, confirmed by reading the source text
  and checking the FTS index directly: one document ranked 90th of 110
  containing the query's strongest term.

Only the second kind is a retriever problem. Had the first been taken at face
value, the fix would have been tuning applied to a system that was working.

The same investigation turned up something no unit test could see: 618 indexed
chunks across 389 documents contained subtitle boilerplate *embedded inside*
otherwise-real speech (the figure `corpus-survey index-quality` now reports;
hand-written `LIKE` patterns had estimated 596, which is exactly why the
measurement belongs in a command rather than in someone's shell history). The whole-text boilerplate filter is working exactly as
designed — it drops text that is *entirely* boilerplate, and deliberately does
not drop a transcript merely for containing some, because that measurably
deletes real content (see [transcript_quality.md](transcript_quality.md)).
Contaminated tails are a separate, unhandled case. Unit tests confirmed the
rule; only querying the built index showed the gap.
