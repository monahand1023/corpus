# Transcribed audio: what goes wrong, and what to do about it

Speech-to-text models invent text. This is not an edge case — on one real
real archive, roughly 6% of transcripts contained nothing a person
said. If you index that output without filtering it, your search results fill
with sentences nobody spoke.

`corpus.transcripts` provides the filtering. This page explains why it works
the way it does, because several of the obvious approaches are actively
harmful.

## Why it happens

Whisper and similar models are trained on audio paired with scraped subtitles.
In that training data, silence and ambience were frequently paired with caption
boilerplate — "Thanks for watching", "Subtitles by …". The model learned the
mapping and reproduces it faithfully whenever handed audio with no speech.

It is not a decoding bug, and **no confidence threshold fixes it**. Measured:
twenty seconds of generated silence produced `"Thank you."` at
`no_speech_prob=0.782` with `avg_logprob=-0.24`. The model is *confidently*
wrong. Confidence cannot separate invention from speech, so anything built on
`no_speech_prob` alone has a ceiling.

## Do not use voice-activity detection as the quality gate

The intuitive fix is to run a VAD and discard whatever it calls speechless.
Measured on real material, that deletes people's recordings.

On the reference archive, a hallucinated Norwegian sign-off peaked at **0.145**
speech probability; a genuine recording of someone calling out a name peaked at
**0.144**. No threshold separates those. Auditing nine VAD rejections by hand,
**four were real** — quiet, distant or reverberant audio that a model trained
on clean speech does not hear.

Personal archives are full of exactly that material, and it is the part people
care about. A filter that deletes a toddler's first words because they were
recorded across a room has failed, however good its aggregate numbers look.

**Use VAD for cost, not for truth.** Skipping near-silence avoids transcribing
it — worth roughly 13× the cost of the transcription it replaces. When VAD
finds no speech in audio that is *not* silent, transcribe it anyway and judge
the text.

## The three signals that do work

### 1. Caption boilerplate

The whole transcript normalising to a known sign-off. `subtitle_boilerplate()`
ships the strings in the languages models drift to. A hit is near-proof of
invention, because these are the literal strings subtitle files end with.

Match the **whole** transcript, never a substring — real talks say "thank you"
too.

#### Matching it is harder than writing it

A phrase list is easy to write and easy to get silently wrong. A review of an
early version found **22 of 45 entries could never match anything**, for three
separate reasons — and the tests passed, because they asserted against the
list rather than against real model output.

**Credits end in a name.** `"Субтитры создавал"` was in the list; what the
model emits is `"Субтитры создавал DimaTorzok"`. Whole-string equality cannot
match a line that ends in an arbitrary studio, community or person. Those
entries need prefix matching.

**But prefix matching over-reaches.** A fabricated credit frequently sits in
*front* of real audio, and matching the prefix alone discards the recording to
remove the noise. `subtitle_boilerplate()` therefore requires what follows the
prefix to look like a name: short, and not in a script the prefix's language
does not use. That last clause matters both ways — a CJK credit's tail is
*always* CJK, so applying the script check blindly made every CJK prefix
unreachable. Judge the tail against the script the *prefix* is written in.

**Normalisation has to match what transcribers actually emit.** Three traps,
each of which silently disabled entries:

- Accents and interior punctuation vary by rendering: `"Amara.org"` against
  `"amara org"`, `"réalisés"` against `"realises"`.
- Decomposed Unicode. macOS emits NFD routinely, so folding that inspects each
  character leaves a standalone combining mark untouched and `"vídeo"` cannot
  match `"video"`. Normalise to NFC *before* folding.
- Fold accents on **Latin letters only**. NFKD decomposes Japanese dakuten —
  `ご` becomes `こ` — so blanket decomposition destroys every CJK entry.
- Include `U+2019`, the apostrophe transcribers default to, and the opening
  `¡` and `¿`.

The general lesson: **test a phrase list against strings the model really
produced**, not against the list's own contents. Every defect above passed a
green test suite.

### 2. A language nobody speaks

If every detected language is one the archive's people do not speak, the text
is almost certainly invented. This is the single highest-yield signal: on the
reference archive it found hundreds of clips tagged Norwegian in a household
with no Norwegian speakers.

Two constraints, both learned by breaking them:

- **Every** detected language must be foreign, never merely one. Transcribers
  tag a single real recording with several languages — one genuine hour-long
  talk came back tagged Welsh, English and Romanian.
- **Cap it by length.** Transcribers mislabel real speech: two minutes of
  English conversation came back tagged Hawaiian and Portuguese. Restricting
  the rule to short text keeps it to the sign-offs it is meant for.

`expected_languages` is yours to supply. An archive's languages are a property
of its people, not of this library.

### 3. Physically impossible speech rate

Characters of transcript per second of audio. A decoder stuck in a loop emits
text faster than anyone speaks: over a thousand characters from an
eleven-second clip is 106 characters/second.

Real content on the reference archive topped out at **14.7** c/s with a 99th
percentile of **17.7**, so the default ceiling of 25 is a wide margin. Scripts
that pack more meaning into fewer characters, such as Japanese, produce *lower*
rates, so one ceiling serves every language. This signal needs no language
knowledge at all.

## Repetition, and why the threshold is high

`repeat_share()` measures the largest share of a transcript taken by one
repeated unit. The default rejection point is **0.9** — deliberately
permissive.

Lower thresholds delete real recordings. At 0.6, the reference archive lost a
clip of a child repeating one word. That is noise to a filter and the reason
its owner keeps the archive.

Repetition is measured with character n-grams for scripts without word spaces.
Splitting on whitespace makes a wall of identical Khmer or Japanese syllables a
single token, scoring zero repetition — the failure mode is invisible unless
you look for it.

## Usage

```python
from corpus.transcripts import judge_transcript

verdict = judge_transcript(
    text,
    duration_s=clip_seconds,
    languages=detected_languages,      # what the transcriber reported
    expected_languages={"en", "ja"},   # who actually speaks here
)
if not verdict:
    log.info("dropped %s: %s", path, verdict.reason)
```

`verdict.reason` names the signal that fired, which is what makes the filter
auditable rather than a black box.

## Keep the evidence

Persist what you discard. A filter that stores only a count of its rejections
cannot be measured: you can estimate leakage from surviving text, but
precision — how much of what it threw away was real — becomes unanswerable.

Store the discarded text with whatever scores the decision used. On the
reference archive that turned an open question into a number: across 1,642
discarded windows, **0.06%** were real speech, and the single false positive
was identifiable and fixable.

Records of a rejection must also carry a fingerprint of the rules that made it.
Change a threshold and old verdicts stop matching, so affected files are
re-examined rather than silently inheriting a judgement made under rules that
no longer apply.
