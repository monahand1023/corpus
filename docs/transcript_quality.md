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

## The four signals that do work

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

**A sign-off names its object, and anchoring to the end misses that.** The
tail patterns matched only when the phrase *ended* the text, so every
real-world variant that names what it is signing off from survived — English
included:

```
"Thanks for watching this video."      survived
"Thank you for watching my video."     survived
"Gracias por ver este video."          survived
"Merci d'avoir regardé cette vidéo."   survived
```

Coverage depended on whether someone had happened to write that exact variant
into the list.

The obvious repair — allow a bounded remainder after the phrase, the way a
credit line already bounds a trailing name by length — is **wrong here**, and
the difference is worth internalising. A credit line is followed by a *name*,
unknowable in advance, so length is the only test available. A sign-off's
object is a small **closed set**, and a length bound strips
`"Thanks for watching the kids."` — two words, and exactly the family recording
the filter exists to protect. So the remainder is matched against a
*vocabulary*: every word after the phrase must be a determiner or a word for
"video". `"the video"` goes, `"the kids"` stays, and the first ordinary noun
disqualifies the match. The same-shaped problem in two places did not have the
same-shaped solution; what differed was whether the trailing text was drawn
from a closed set or an open one.

**Strip to a fixed point.** A transcriber that emits a sign-off often emits it
twice. Removing what is at the end, once, turned
`"Please subscribe. Please subscribe."` into `"Please subscribe."` — still a
sign-off, now indexed.

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

### 4. Repetition — which is TWO signals, because one shape hides from the other

`repeat_share()` measures the largest share of a transcript taken by **one**
repeated unit, rejecting at **0.9** — deliberately permissive. Lower
thresholds delete real recordings: at 0.6 the reference archive lost a clip of
a child repeating one word, which is noise to a filter and the reason its
owner keeps the archive.

That permissiveness was not the problem. **The metric was.**

Transcribing 200 real recordings and reading the survivors, six were the model
stuck in a loop — `"I am going to draw a small map."` repeated to fill the
window. Every one passed. Across all 74 kept transcripts the metric's *highest*
score was **0.250**, against a threshold of 0.9. On real data the check had
never once fired.

The reason is structural, not a tuning mistake. `repeat_share` divides one
trigram's count by the *total* number of trigrams, so the denominator grows
with the text while the numerator tracks a single trigram. A seven-word phrase
looped twenty times scores about **0.145**. It can only approach 1.0 for
single-unit hammering — `"okay okay okay okay"` — which is exactly the shape a
synthetic test reaches for. **The tests and the metric shared a blind spot, so
the tests could never reveal it.**

`looping_share()` counts **distinct** units instead, so the score rises with
how much of the text is duplicated however long the repeating phrase is.
Measured on those 74 transcripts: median **0.000**, the six loops
**0.712–0.886**, next-highest real text **0.379**. A clean gap, so the
threshold (**0.6**) is a measurement rather than a guess.

It can afford to be strict where `repeat_share` cannot, because it only applies
once there is enough text for a repeat to be unambiguous. Without that floor it
deletes the recordings the archive exists for: `"Papa! Papa! Papa! Papa!"`
scores **0.667**. Genuine short repeats measured 18–21 units and every real
loop sat at 33–111, so the floor is **24 units**.

Keep both. Neither subsumes the other: one unit hammered, versus a phrase
looped.

Repetition is measured with character n-grams for scripts without word spaces.
Splitting on whitespace makes a wall of identical Khmer or Japanese syllables a
single token, scoring zero repetition — the failure mode is invisible unless
you look for it.

### Where the checks have to run

Per-**window**, per-**transcript**, and on the rescue path, because junk
reaches an index by whichever one you skip. Measured on a live 45,092-chunk
index: 716 chunks were loops, but only 553 belonged to transcripts a
whole-transcript judgement rejects. The other 163 were single looping windows
inside genuine recordings — real material either side, so no per-file verdict
can reach them. A further 12 arrived through the *rescue* that restores the
longest window when every window was filtered, because that path tested
boilerplate only.

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

**Apply that to what you KEPT, too.** A stored transcript is equally a verdict
— *this text is real* — and it is the one people forget to expire. Adding the
loop signal correctly re-examined all 126 rejected files in a test archive and
left the six looping transcripts it was written to catch sitting in the index,
because the skip set matched transcripts on path alone. Re-judging costs no
model time: the text is already stored, and judging text does not need the
audio. Caching the negative result and not the positive one is a recognisable
bug class, and it hides because the positive path looks like data rather than
like a decision.
