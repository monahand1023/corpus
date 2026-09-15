# Plan: make corpus index audio and video

## Where this starts from

`corpus-survey census` on a folder of mixed media today:

```
Gap — no connector     .heic .jpg .m4a .mov .mp4 .png .sketch
Indexable              .zip .md .txt .csv .mp3(tags only)
```

The capability exists and works — 381 hours transcribed, 45,092 chunks, a
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

## Stage 3 — make `corpus-index` do it automatically

Today `corpus-index` reports media as a gap and moves on. It should offer to
handle it.

- `corpus-survey media` already estimates hours and projects runtime; wire
  that into the plan so the confirmation prompt says "4.2 h of audio, ~17 min
  at 15x realtime" before anything starts.
- On confirmation: transcribe to the sidecar DB, then ingest it as a
  `transcripts` source in the same run.
- Without the extra installed, keep reporting the gap and name the extra.

**Acceptance:** `corpus-init && corpus-index ~/some/folder` on a directory of
documents *and* audio produces one database searchable across both, with the
plan having priced it first.

## Stage 4 — images, only if it earns it

`a media consumer` has captioning, OCR and an Apple Photos reader (~150 KB). Apple
Photos is macOS-specific and the captioning is a second heavy model. Defer
until stages 1–3 have shipped and been used; `corpus-survey` will keep
reporting images as a gap in the meantime, which is honest.

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
`VAD_SPEECH_PROB`, the 25 chars/second ceiling). The job is to move it behind
a clean boundary without losing the reasons.
