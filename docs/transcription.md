# Transcription: speech into the index

Operational guide to `corpus-transcribe`. Why the quality filters work the way
they do is in [transcript_quality.md](transcript_quality.md).

`corpus-index` reports audio and video as a gap and skips them. `corpus-transcribe`
is the step that closes it — point it at a folder of recordings and it writes a
sidecar database that is then a source like any other:

```bash
corpus-transcribe ~/Videos --dry-run     # how much audio, how long, what's left
corpus-transcribe ~/Videos               # do it (safe to interrupt)
corpus-transcribe ~/Videos --limit 20    # sample the quality first
corpus-transcribe ~/Videos --min-seconds 15   # skip the Live Photo clips
```

**`--min-seconds` is worth knowing about before you point this at a phone's
video folder.** Measured on a real photo library:
**81% are under four seconds** — the clip Apple stores beside each Live
Photo. Transcribing them is four fifths of the library and tens of hours of room tone, and it
floods the index with near-empty text that dilutes every search. The plain
dry run cannot warn you, because total hours cannot show that four fifths of
them are four seconds long.

A file whose duration `ffprobe` cannot read is KEPT, never skipped: a failed
probe is not evidence that a recording is short.

## After a quality threshold changes

Every rejection setting is hashed into a POLICY FINGERPRINT, so changing one
invalidates the verdicts it produced and they are redone. On an established
archive that is a lot of GPU time for what is usually a change to a regex, and
two flags exist so you pay only for what actually has to be recomputed:

```bash
corpus-transcribe --db data/transcripts.db --refilter .     # no audio decoded
corpus-transcribe --db data/transcripts.db --redo-stale .   # only what changed
```

**`--refilter`** re-applies the current TEXT rules to the per-window text
already in the sidecar. On one archive a full re-transcribe was 61.7
GPU-hours by its own recorded timings; the re-filter did a whole archive in
seconds, stripping loop windows from the affected recordings.

It re-judges **every** row, not only rows stamped with an older policy. The
policy hash is derived from SETTINGS, so a fix to the filter *code* changes
what survives while every threshold — and therefore the hash — stays
identical. Scoping to "stale" rows made the one command whose job is applying
a filter change blind to the most common reason to run it: after a decode-loop
fix shipped and a 7-hour re-transcribe ran with the old code, it reported
`0 re-filtered` while three recordings still held a window of 138 chars/s.
Re-judging is text-only and idempotent, so examining everything costs a pass
over strings.

It is equivalent only while the DECODE is unchanged — `window_s`,
`overlap_s`, the VAD threshold and the model decide which audio becomes which
window, and none of that can be re-derived from text. A row from another
model, a row with no stored windows, and a row whose windows are wider than
`window_s` are each skipped and **counted**, never silently restamped with a
policy that was not applied.

**`--redo-stale`** re-transcribes exactly the files the current policy
invalidated, taking the work list from the sidecar instead of walking the
media roots. After a threshold change that is the right operation: a re-walk
rediscovers everything the archive deliberately excluded, and those rules
live in the archive, not in corpus. On one archive whose roots hold a photo library of
photo-library videos — 81% of them the sub-4-second clip Apple stores beside
each Live Photo — a blind re-walk would have queued ~46,800 near-empty clips
for ~33 hours of room tone.

Rejections are included: a `no_text` row is a verdict too, and redoing only
the transcripts leaves every rejection frozen under rules that no longer
apply. Paths not on disk are counted and skipped rather than tried, so an
unmounted external drive does not become thousands of recorded failures.

**Why it is a separate command and not part of `corpus-index`.** Everything
`corpus-index` does is seconds of I/O. This is hours of local compute, so it
sits behind its own confirmation and its own dry run rather than happening
because you pointed the indexer at a folder that happened to contain an
`.mp4`. `corpus-index` tells you it's available and gets out of the way:

```
Gap — no connector (report this first; it's the whole point)
  .m4a      18 files
  .mov       4 files

Of those, .m4a, .mov hold SPEECH that can be transcribed and indexed.
Run `corpus-transcribe <path>` first, then re-run this command
```

The dry run is the intended first step — it is the only warning before the
hours start:

```
corpus-transcribe: ~/Videos
  media files found : 22
  already done      : 6 of these (skipped; includes files found to hold no speech)
  audio left to do  : ~6.8 h across 16 file(s)
  estimated runtime : ~27 min at 15x realtime (local compute; no API spend)
```

The skip count comes FIRST on purpose, and the runtime prices only what is
left. Printing the estimate above it meant quoting the hours for every file
that cleared the duration floor, already-transcribed ones included — on one
archive, `estimated runtime : ~14.4 h` sat directly above `already done:
5,407 of these`, when 363 files actually needed transcribing. Overstating a
run is not the safe direction it looks like: it talks you out of a job that
would have taken twenty minutes.

Durations are shown in a unit that still carries information: once the skip
set is subtracted the remaining work is often minutes, and `~0.0 h` reads as
"nothing to do" rather than "two minutes".

Then wire the sidecar in and ingest — the command prints this block for you:

```toml
[[sources]]
name = "recordings"
type = "transcripts"
path = "data/transcripts.db"
```

```bash
corpus-ingest --source recordings
corpus-query "what was decided about the roof"
```

## What it does about hallucination

A speech model trained on audio paired with scraped subtitles learned that
silence maps to caption boilerplate, and reproduces it whenever handed audio
without speech. **Confidence cannot catch this**: measured on generated
silence, one model returned `"Thank you."` at `no_speech=0.782` and
`avg_logprob=-0.24`. It is confidently wrong, so no threshold on its own
scores separates invention from speech.

So the filters run on the TEXT, not on the model's scores, and they are the
part of this that was derived from a real transcript archive rather than
designed in the abstract (`corpus.transcripts.quality`):

- **Caption boilerplate**, matched at the TAIL, not anywhere in the text.
  Dropping every transcript merely *containing* a sign-off deleted 12.3% of
  that archive — 732 transcripts, including an 84-minute talk that ended with
  someone genuinely saying "thank you very much".
- **Degenerate repetition**, as two separate signals, because one shape hides
  from the other. One catches a unit hammered ("okay okay okay okay"); the
  other catches the transcriber *looping* — a whole phrase repeated to fill the
  window, which is the commonest degenerate output there is. On 200 real
  recordings the first scored at most 0.250 against its 0.9 threshold while six
  transcripts were unmistakable loops, so on real data it was doing nothing.
  The loop signal only applies once there is enough text for a repeat to be
  unambiguous — below that, a repeat is a child saying a word four times, and
  that is a recording to keep.
- **Impossible speech rate** — more characters than a human mouth produces in
  the window's duration.
- **Unexpected language**, when you name the languages you actually speak
  (`--language en --language ja`). Silence gets labelled as languages nobody
  in the recording speaks. **This one is lossy and opt-in for a reason**: the
  label is least reliable exactly when the audio is hard, so short real
  utterances get mislabelled too. On a 200-clip test it removed 11 more files,
  of which several were genuine English mislabelled as Norwegian. It applies
  only to short text for that reason. Leave it off unless you have looked at
  what it removes.

Voice-activity detection is used only to decide where to SPEND time, and
never to decide whether a recording is worth keeping. On that same archive a
hallucinated sign-off peaked at 0.145 speech probability and a genuine
recording of a parent calling a child's name peaked at 0.144 — and of nine
detector rejections audited by hand, four were real family recordings that
were quiet, distant or reverberant. If the detector finds nothing but the
audio is not silent, the file is transcribed in full and the text is judged.

## Interrupting it is fine

Every outcome is written as it happens, including the negative ones, so a
second run skips what the first already answered. That matters more than it
sounds: files that produce NO usable text are exactly the ones a naive
restart re-does, because they leave nothing behind to find. One interrupted
pass re-decoded 889 already-examined silent clips before those rows existed.

Every stored verdict carries a fingerprint of the rules that produced it —
model, window size, thresholds, and the boilerplate phrase lists. Change any
of them and the affected files are retried rather than inheriting a verdict
made under different rules.

That applies to transcripts you already have, not only to files that were
rejected. A stored transcript is equally a verdict — *this text is real* — so
when the rules change, stored transcripts are re-judged against the new ones
and demoted if they no longer pass. This costs no model time, because judging
text does not need the audio; their text is kept in `no_text` so a rule that
proves too aggressive can be reversed against real evidence. Without it a
filter never reaches the material already indexed under the older rules —
measured: adding the loop signal correctly re-examined all 126 rejected files
in a test archive and left the six looping transcripts it was written to catch
sitting in the index.

## Requirements

Transcription needs `ffmpeg` on PATH plus two extras:

```bash
uv add 'corpus-rag[transcribe]'      # voice-activity detection
uv add 'corpus-rag[transcribe-mlx]'  # the shipped model — Apple Silicon only
```

The shipped backend is `mlx-whisper`, which runs on Apple Silicon only. A
public package cannot make one vendor's hardware a requirement of a headline
feature, so the pipeline talks to a protocol
(`corpus.transcripts.backends.TranscriberBackend`) and the Apple-specific part
sits behind it. Supplying your own takes two members — `model_name` and
`transcribe_window(samples) -> WindowResult` — and every quality rule above
then applies to its output unchanged.

## Audacity projects

`.aup3` files are transcribed like any recording. An Audacity project is a
SQLite database rather than a stream ffmpeg can read, so its layout (sample
rate, tracks, clip positions, trims, mute) is read from the project itself
(`corpus.connectors.aup3_layout`), its audible tracks are mixed to mono, and
the mix is resampled through ffmpeg. A project whose layout cannot be read
is refused rather than guessed at: a wrong rate or channel layout produces
audio that transcribes to nonsense. Mixing needs numpy (in `[transcribe]`).
