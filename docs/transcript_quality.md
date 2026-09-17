# Transcribed audio: what goes wrong, and what to do about it

Speech-to-text models invent text. This is not an edge case — on one real
archive of transcripts, roughly 6% of transcripts contained nothing a person
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
percentile of **17.7**, so the ceiling was first set at 25 — a wide margin at
the time; see the re-measurement below. Scripts
that pack more meaning into fewer characters, such as Japanese, produce *lower*
rates, so one ceiling serves every language. This signal needs no language
knowledge at all.


#### Re-measure it; the data moves even when the number does not

The ceiling was set at **25.0 c/s** when the fastest genuine content in the
reference archive was **14.7** — 70% of headroom, comfortable. Re-measured
across **a full real archive of transcripts** on 2026-09-16, the archive had grown and
real speech reached **24.85 c/s**: a French speaker mid-conversation,
surviving the filter by **0.6%**. One slightly faster talker and it would have
deleted a real recording, which is the failure every other threshold here is
deliberately tuned to avoid.

The threshold never moved. **The data did.** A number validated once against a
sample is not validated forever against a growing one.

It is now **40.0**, sitting between the two things actually measured:

| | c/s | |
|---|---|---|
| fastest real speech | 24.85 | a full real archive |
| **ceiling** | **40.00** | 61% above real speech |
| documented decode loop | 59.91 | still caught, by 50% |

Raising it cost no detection. Across **879** stored rejections that kept their
text, not one exceeded even the old 25.0 — loops repeat a phrase at
conversational pace, so the repetition signals catch them and the rate check
never sees them. That makes this independent insurance against a different
shape of junk, not a redundant second opinion.

**How it surfaced:** the dormancy check reported that this filter had never
fired. That is the signature of a dead knob — and it was not one. `repeat_share`
sat 3.6× from its threshold and was structurally unreachable; this one is
reachable, correctly aimed, and had simply never been crossed. Worth knowing
that "never fired" has at least two causes, and telling them apart needs a
measurement, not an inference.

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
**0.712–0.886**, next-highest real text **0.379**. An apparently clean gap,
so the threshold was set at **0.6**.

#### That threshold was wrong, and the sample is why

Applied to the same project's **full archive of transcripts**, 0.6 removed
**241 transcripts**, and real material turned out to reach **0.5984** — a
margin of 0.3%, not the comfortable gap the sample showed. Sampling what it
rejected, at every band:

| score | what it actually was |
|---|---|
| 0.607 | `"お誕生日おめでとう"` ×8 — Happy Birthday, sung |
| 0.703 | `"don't hit it hard…"` ×3 — a parent talking to a child |
| 0.803 | `"just one cup of rice…"` — real cooking instruction |
| 0.805 | `"¿Aquello es tuyo? … Sean."` — real multilingual family speech |
| 0.986 | `"そのため、"` ×40 — a genuine decode loop |

The English birthday video survived at **0.5955** while the Japanese one was
deleted at **0.607** — same family, same event, separated by 0.012.

What the 74-transcript sample could not show is that **real family speech is
genuinely repetitive**: songs, chants, a parent repeating an instruction. The
threshold is now **0.85**, which keeps the 92 clearest loops, restores the 149
deleted recordings, and leaves 42% of margin over real material. The asymmetry
is the whole argument: an indexed loop is recoverable noise, a deleted family
recording is not.

This repeats a lesson this page already carried two paragraphs up — at 0.6,
`repeat_share` lost a clip of a child repeating one word. **A threshold
validated on a sample is not validated on the population.**

It can afford to be strict where `repeat_share` cannot, because it only applies
once there is enough text for a repeat to be unambiguous. Without that floor it
deletes the recordings the archive exists for: `"Papa! Papa! Papa! Papa!"`
scores **0.667**. Genuine short repeats measured 18–21 units and every real
loop sat at 33–111, so the floor is **24 units**.

Keep both. Neither subsumes the other: one unit hammered, versus a phrase
looped.

#### One threshold cannot serve both scopes

The same measure runs per-**window** and per-**transcript**, and the two want
opposite settings:

| scope | wants | because a rejection costs |
|---|---|---|
| window (~30s) | **strict** | one chunk. The rest of the recording survives. |
| whole transcript | **permissive** | the entire recording, real speech included. |

For a while one value served both. Raising it from 0.6 to 0.85 to stop whole
family recordings being deleted therefore loosened the window filter by
exactly the same amount — silently, because nothing in the change named the
window scope at all.

Measured on the live archive afterwards: **25 transcripts scoring 0.80–0.8444
are pure Whisper decode loops** — one Japanese sentence thirteen times over a
video of a child in a playroom — sitting in the index as searchable text,
because at 0.85 neither check fires on them. The `corpus-doctor` margin
report made it look like the opposite problem: "looping share: 0.7% headroom,
kept material reaches 0.844444" invites raising the ceiling, when the 0.844444
*was the junk*.

**A threshold that is too permissive always looks tight**, because its own
failures are in the sample it is measured against. That is why the margin
check now prints the nearest recording's text beside the number.

So `DEFAULT_MAX_WINDOW_LOOPING_SHARE` (0.6) is now separate from
`DEFAULT_MAX_LOOPING_SHARE` (0.85).

#### …and "one chunk" is false when there is only one window

The argument above — a dropped window costs one chunk, the rest of the
recording survives — has a precondition nobody stated: **there has to be a
rest.** A short recording is a single window, so dropping it *is* deleting the
recording, arriving at the outcome the permissive ceiling exists to prevent
through the side door.

Simulated against the live archive before anything was written: the strict
window rule would have emptied **176 recordings, 174 of them single-window**,
including

```
"a birthday line, then another.."        0.647
"an exclamation, three times"
"a greeting, then a chant"
"an objection, three times"
```

— the same recordings a 0.6 *whole-transcript* ceiling had deleted earlier
the same day, reached by a different route. **The lesson did not transfer
because the new code did not look like the old mistake.**

`filter_windows` therefore never lets per-window dropping empty a transcript:
if nothing would survive and any drop was a REPETITION, those windows are
reinstated and the permissive `judge_transcript` decides. Boilerplate is not
reinstated — "Thank you. Thank you." is the model filling silence, not a song,
and putting it back returns an empty result to search.

Measured on the live archive with the fallback in place:

| | |
|---|---|
| recordings deleted | **2** (both pure "Thank you.") |
| recordings with loop windows stripped | 1,108 |
| windows dropped | 1,424 |

Repetition is measured with character n-grams for scripts without word spaces.
Splitting on whitespace makes a wall of identical Khmer or Japanese syllables a
single token, scoring zero repetition — the failure mode is invisible unless
you look for it.

#### The floor and the fallback have to agree, or they open a hole

`_repetition_units` prefers word trigrams and falls back to character 6-grams;
`looping_share` returns 0.0 below a floor of 24 units. Those two rules were
written separately, and between them sat a window where **neither applied**.

Below four word-trigrams the fallback fires and characters give plenty of
units. Between four and twenty-three trigrams it returned *trigrams* — under
the floor — so the score was 0.0 and the text walked straight through. **A
pure repetition of 8 to 25 words scored exactly zero**, which is precisely the
shape this signal exists to catch, and it meant MORE repetition scored LOWER
than less.

The fix is to retry on characters before giving up, rather than to lower the
floor. The floor still protects genuinely short text, because a four-word
utterance has too few character 6-grams to clear it either.

The general shape is worth naming: when a metric has both a *fallback* and a
*minimum*, check what happens in the band where the fallback has not triggered
and the minimum has not been met. A signal that returns "nothing here" for an
input it cannot measure is indistinguishable from one that measured and found
nothing.

### Where the checks have to run

Per-**window**, per-**transcript**, and on the rescue path, because junk
reaches an index by whichever one you skip. Measured 2026-09-11 on a live
index (it has grown since; the ratios are the point, not the
total): 716 chunks were loops, but only 553 belonged to transcripts a
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
