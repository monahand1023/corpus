# Plan: make corpus index audio and video

## Where this starts from

`corpus-survey census` on a folder of mixed media today:

```
Gap — no connector     .heic .jpg .m4a .mov .mp4 .png .sketch
Indexable              .zip .md .txt .csv .mp3(tags only)
```

The capability exists and works — 381 hours transcribed, 45,092 chunks as of
2026-09-11, a
gold set, measured retrieval — but it lives in a private consumer (`a document consumer`)
as a 1,033-line script plus a 291-line connector. Nothing about it is secret;
it was simply never packaged.

**What is already in corpus.** All the judgement. `corpus.transcripts.quality`
holds the caption-boilerplate lists, `repeat_share`, `only_unspoken_languages`,
`impossible_speech_rate`, `judge_transcript` and `strip_caption_tail` — every
rule that decides whether transcribed text is real, each derived from a
measured failure on a real archive. `transcribe.py` already delegates to it:
its local `repeat_share`/`impossible_speech_rate`/`only_unspoken_languages` are
three-line wrappers binding archive config onto the shared implementations.

**What is not.** The pipeline: audio decode, voice-activity segmentation,
windowing, the killable worker, the sidecar database, and the connector that
turns it into chunks.

## The one hard constraint

`mlx-whisper` is **Apple-Silicon only**. corpus is a public package tested on
Python 3.12–3.14 and cannot make an Apple-only dependency part of its core.
This shapes the whole plan: the *interface* must be transcriber-agnostic, and
MLX must be one implementation behind it rather than the design.

## Stage 1 — the sidecar schema and connector (no heavy deps)

Move the part that has no dependencies at all: the shape of a transcript
database and the connector that indexes one.

- `corpus.transcripts.store` — the `transcripts` / `no_text` / `failures` /
  `dropped_windows` schema, its migrations, and typed read/write helpers.
- `corpus.connectors.transcripts` — one document per media file, one chunk per
  window, language and timestamps in metadata, `strip_caption_tail` applied at
  chunk time. This is `docs_rag/transcripts.py` with its archive-specific
  exclusions lifted out into config.
- Register `transcripts` in `CONNECTOR_REGISTRY`.

**Why first:** it is pure Python, testable without a GPU, and it immediately
lets anyone with a transcript database from *any* tool index it. It also makes
a document consumer a thin consumer instead of a fork, which is the stated architecture.

**Acceptance:** a document consumer deletes its local connector, imports corpus's, and its
gold set still scores recall@5 0.625 / MRR 0.442 / nDCG 0.486. Any change to
that number means the move was not faithful.

## Stage 2 — the transcriber interface, and MLX behind it

- `corpus.transcripts.backends.Transcriber` — a protocol: given 16 kHz mono
  float32 samples and a window, return text plus per-window language and
  confidence. Nothing in it mentions MLX.
- `corpus.transcripts.pipeline` — decode (one `ffmpeg` call, `-vn` before
  `-i`), VAD segmentation, windowing with overlap, overlap-trimming join, the
  killable worker with its orphan watchdog, resumption from the sidecar DB,
  and the policy fingerprint that invalidates cached rejections when a rule
  changes. All backend-agnostic.
- `corpus.transcripts.backends.mlx` — the MLX implementation, behind a
  `corpus-rag[transcribe-mlx]` extra, importing `mlx_whisper` lazily and
  failing with the same "install this extra" message shape the reranker uses.

**Why an interface rather than shipping MLX directly:** a public package whose
headline feature only runs on one vendor's laptops is a bad package, and the
backend boundary is where a `faster-whisper` or API-based implementation slots
in later without touching the pipeline.

**Acceptance:** every pure function tested without a model — windowing,
overlap-join, timeout scaling, resumption, the fingerprint. The backend itself
is exercised by one opt-in integration test, skipped by default.

## Stage 3 — the run, and `corpus-index` pointing at it — SHIPPED

Shipped as `corpus-transcribe` (`corpus.transcripts.run`,
`corpus.cli.transcribe`): a resumable directory runner, a dry run that prices
the job, and a media offer printed by `corpus-index` after its gap report.

**Deviation from the plan above, and why.** The plan said `corpus-index`
should transcribe on confirmation, in the same run. It does not. It reports
which gap extensions hold speech and names the command:

```
Of those, .m4a, .mov hold SPEECH that can be transcribed and indexed.
Run `corpus-transcribe <path>` first, then re-run this command
```

Everything else `corpus-index` does is seconds of I/O. Transcription is hours
of local compute, and burying it as a sub-step of an indexing command means a
`y` at an indexing prompt starts an overnight job. It gets its own command,
its own dry run, and its own confirmation. The offer keeps the discovery
property the plan actually wanted — you find out from the tool, not from the
docs — without the commitment.

**Acceptance, met:** `corpus-transcribe` on a folder of mixed media wrote a
sidecar; `corpus-ingest` ingested it as a `transcripts` source; `corpus-query
"what did the children do at the water"` returned the sentence spoken in an
`.m4a` that was never text. The silent `.mov` in the same folder was recorded
as holding no speech rather than indexed.

**Two defects this stage surfaced by being run rather than reasoned about:**

- `default_backend()` checked the platform only, so on Apple Silicon without
  the extra it returned a backend that could not work — and the failure
  surfaced on the first window of every file. A 7,000-file archive would have
  recorded 7,000 identical ImportErrors before anyone learned which package to
  install. Fixed with `preflight()`, which also runs in the dry run, so the
  cheapest possible moment tells you.
- The dry run counted the `transcripts` table only, so files judged to hold no
  speech were reported as outstanding work. On a real archive most files
  produce no text — the count understated the skip set in exactly the
  direction that gets trusted.

## Stage 4 — images, deferred (not cancelled)

Deferred for four reasons, in descending order of weight.

**1. "Images" is three separable features, not one.** `a media consumer` conflates
generic OCR of image files, model captioning of photo content, and an Apple
Photos library reader. They have different dependencies, different platforms
and different value. Porting them as a unit is how a 600-line feature becomes
a 3,300-line one.

**2. There is no quality filter for captions, and there is for speech.**
Stages 1–3 were portable largely because `corpus.transcripts.quality` already
existed and was derived from a real real archive — the filters are
the hard-won part, and the pipeline around them is comparatively simple. No
equivalent exists for captions. A captioner invents confidently too, and
shipping captioning without knowing what its failure modes look like would
repeat the mistake that stage 1's filters were written to fix. That work has
not been done, and inventing it in the abstract is exactly what the
hallucination filters prove doesn't work.

**3. Size — a weaker argument than it first looked.** Measured: the image
features total 3,350 lines (`caption.py` 2,367, plus OCR, the Photos reader,
place lookup, LZFSE decoding and privacy filtering) against 2,395 for
everything shipped in stages 1–3. That is 1.4x, not the order of magnitude
this bullet originally claimed — the estimate was made before counting and was
wrong in the direction that flattered the decision. It is listed third because
it carries the least weight, not the most. What does survive is the shape:
`caption.py` alone is larger than the entire transcription pipeline, and
`mlx-vlm` is Apple-only, so stage 2's backend-protocol exercise has to be
repeated in full before any of it can ship publicly.

**4. Demand is unmeasured, and measurable.** `a media consumer` has served **0**
real logged queries. Stage 4's precondition is not a date — it is evidence
that anyone asks image questions of an archive. `corpus-doctor`'s query-log
audit is what answers that, and it currently says no one has.

Meanwhile `corpus-survey` keeps reporting images as a gap, which is honest and
costs nothing.

## Risks, with the mitigation named

| Risk | Mitigation |
|---|---|
| **Platform lock.** MLX is Apple-only. | The backend protocol in stage 2 exists for exactly this. Core stays dependency-free. |
| **~2 GB of deps.** torch + silero + mlx. | An optional extra, as `[reranker]` already is, with the same error message when missing. |
| **Untestable in CI.** No GPU, no 3 GB model. | The pipeline is pure functions around one backend call; stage 2's acceptance is that all of them test without a model. |
| **Multi-hour runs.** | Resumption already exists — the sidecar DB plus the `no_text` memo that saved 889 files on a crash recovery. Move it intact. |
| **A faithless move.** Behaviour drifts during the port. | a document consumer's gold set is the regression test. Stage 1 acceptance is that the number does not move. |
| **Scope creep into photos.** | Stage 4 is explicitly deferred, with a stated precondition. |

## What this is not

Not a rewrite. The pipeline works and its constants are documented with the
measurements behind them (`WINDOW_S`, `OVERLAP_S`, `NO_SPEECH_MAX`,
`VAD_SPEECH_PROB`, `DEFAULT_MAX_CHARS_PER_SECOND`). The job is to move it behind
a clean boundary without losing the reasons.
