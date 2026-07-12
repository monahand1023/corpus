# Understanding Evals — RAG and Evaluation From Scratch

> **New to all this?** Start here. This is the conceptual "why" behind
> `corpus-eval` and `corpus-judge`. It assumes **no prior knowledge** of retrieval,
> RAG, or AI evaluation, and builds up from zero using `corpus`'s own shipped
> example — the sample corpus, the eval query set, and the public judge fixture.
> Once the ideas click, the two reference docs go deeper on mechanics:
> [`docs/eval.md`](eval.md) (retrieval metrics) and [`docs/judge.md`](judge.md)
> (the LLM-as-judge).

---

## Part 1 — What is an "eval"?

An **eval** (short for *evaluation*) is a **test for an AI system**. That's the
core idea. Just as you'd write unit tests for normal code, an eval measures whether
an AI system does what you want — but instead of a clean pass/fail, it usually
produces a **score**, because AI outputs are rarely exactly-right-or-exactly-wrong.

A useful image: an eval is an **exam you give your system.** You write the
questions, decide what a good answer looks like, grade the answers, and get a score.
Change the system — a new model, a new prompt, a different retrieval setting — give
it the *same exam*, and you can see whether it got better or worse.

### Why bother? Because "it seems good" doesn't scale and doesn't survive change.

When you first build something, you test it by hand: type a few questions, eyeball
the answers, think "looks right." This **vibes-based evaluation** has three fatal
flaws:

1. **It doesn't scale.** You can eyeball 5 answers, not 5,000.
2. **It's not comparable.** You tweak a setting, try 5 *different* questions, and
   now you can't tell if the change helped or if you just asked easier questions.
3. **It drifts.** Your memory of "how good it was last week" is unreliable.

Evals fix all three with a **fixed set of questions**, a **consistent grader**, and
a **number** you can compare across versions. The single most valuable thing an eval
gives you: the ability to answer *"I changed X — did it actually help, or did I just
convince myself?"*

### The core loop

```
1. Decide what "good" means           (define your metrics)
2. Build a fixed set of test cases    (your eval set / "golden set")
3. Run the system, grade the outputs  (get a BASELINE number)
4. Change ONE thing                    (a "lever": model, prompt, retrieval, …)
5. Run again, grade again              (get a new number)
6. Compare. Keep the change if the number improved; otherwise revert.
7. Repeat.
```

The "before" number in step 3 is your **baseline**. Everything after is a **delta**
(change) from it — and you care far more about the delta than the absolute number
(Part 6 explains why).

---

## Part 2 — Why AI evals are harder than normal tests

Normal software is **deterministic**: `add(2, 2)` returns `4` forever. You assert
`add(2, 2) == 4` and you're done — exactly right or exactly wrong.

AI breaks both assumptions:

- **It's probabilistic.** Ask a model the same question twice and you can get two
  different (both valid) answers. There's often no single "correct string."
- **"Correct" is fuzzy.** "Summarize this" has countless acceptable answers and
  countless bad ones, separated by degree, not a yes/no line.

So AI evals can't lean on exact-match assertions. They use **graded metrics** and
often need a **judge** — human or AI — to assign the grade. Grading free-form output
at scale, consistently and cheaply, is the central problem of the field. Most of the
craft is about solving that one problem well.

---

## Part 3 — A map of the eval landscape

There isn't one kind of eval; there's a family. The axes that matter (and where
`corpus` sits on each):

**By what you grade:**
- **Retrieval eval** — did the system *find* the right information? → `corpus-eval`
- **Generation eval** — given that information, did it *write* a good answer? → `corpus-judge`

**By how you grade:**
- **Automated metric** — a formula (recall, MRR, nDCG). Cheap, fast, objective, but
  blind to meaning. → `corpus-eval`'s retrieval metrics
- **Human eval** — a person reads and rates. Gold standard for quality; slow,
  costly, inconsistent between people.
- **LLM-as-judge** — a *stronger AI* imitates a human grader. The modern workhorse:
  near-human quality at machine speed. → `corpus-judge`

**By when you run it:**
- **Offline eval** — on a fixed test set, in development, before shipping. → both
  corpus tools
- **Online / production monitoring** — measuring live user traffic. A different
  discipline (see FAQ).

---

## Part 4 — What `corpus` is, and what RAG means

`corpus` is a **RAG** system: **Retrieval-Augmented Generation.** The idea: an LLM
on its own only knows its training data — it doesn't know *your* documents. RAG
gives the model relevant snippets of *your* data at question time.

The librarian analogy: you ask a librarian a question. They don't answer from
memory — they walk to the shelves, pull the most relevant pages (**retrieval**),
read them, and write an answer *based on those pages* (**generation**). RAG is
exactly that: **retrieve, then generate.**

The `corpus` pipeline, stage by stage:

```
your documents
  → chunk        (split into small passages)
  → embed        (turn each chunk into a vector = numbers capturing its meaning)
  → store        (SQLite: a vector index + a keyword index)
  ───────────────  (that was "ingest"; below runs per question)  ────────────
  → retrieve     (find candidates two ways and fuse them):
                    • vector search  = "semantically similar" (meaning)
                    • BM25 / keyword = "literally contains the words"
                    • hybrid fusion  = combine both rankings (reciprocal rank fusion)
  → rerank       (optional: a smarter model re-sorts the top candidates)
  → generate     (feed the top chunks to the LLM; it writes an answer + citations)
```

Two very different things can go wrong here, which is why there are **two evals**:

1. **Retrieval** can fail — the right chunk never gets pulled off the shelf.
2. **Generation** can fail — the right chunk *was* pulled, but the answer is wrong,
   invented, or cites the wrong source.

Keeping these separate is one of the most important ideas in this document. A system
can be great at one and bad at the other, and the fix is completely different in each
case. `corpus` evaluates them separately for exactly this reason.

---

## Part 5 — The two evals

### 5.1 Retrieval eval (`corpus-eval`) — the tractable half

Retrieval is easy to grade because a **known-good answer exists**: for a test
question, you know which document *should* be found. So you use automated metrics
(no AI, no API cost, fully deterministic):

- **recall@k** — "for how many questions did the right document show up in the top
  *k*?" `corpus` reports **recall@5**: is the target in the top 5?
- **MRR (Mean Reciprocal Rank)** — rewards ranking the right answer *higher* (rank 1
  → 1.0, rank 2 → 0.5, rank 5 → 0.2).
- **nDCG@k** — a refined version that handles multiple relevant results and discounts
  ones found lower in the list.

`corpus` ships a runnable example so you can see this with **no API key**: a
**20-document sample corpus** scored against a **30-query set**, using the
zero-dependency `hash` embedder. A real run:

```
=== Aggregate (n=30) ===
  recall@5: 1.000    mrr: 0.865    ndcg@5: 0.898
```

And here's a first taste of *eval-driven decisions*. Running the same 30 queries
under two retrieval configs (`corpus-eval --compare`) produces:

| config | recall@5 | MRR | nDCG@5 |
|---|---|---|---|
| **hybrid** (BM25 + vector) | 1.000 | 0.865 | 0.898 |
| vector-only | 0.933 | 0.838 | 0.861 |

**Finding:** hybrid beats vector-only on all three metrics. That's not a hunch — it's
a measured delta, which is why `corpus` fuses keyword and vector search by default.
Full mechanics and metric formulas: [`docs/eval.md`](eval.md).

> **On absolute numbers:** the sample corpus uses the `hash` embedder, a deterministic
> *lexical* stand-in — not a semantic model. So recall@5 = 1.000 here is a
> **reproducibility substrate, not a quality claim**. The point of the example is the
> *comparison* (hybrid vs vector-only), not the absolute 1.000. Real quality is
> measured on your own corpus with a real embedder (`voyage`/`gemini`).

### 5.2 Generation eval (`corpus-judge`) — the hard half

Now the hard part. The system produced a free-text answer with citations. How do you
grade *that*, automatically, for thousands of answers? There's no single gold string
to match. This is where **LLM-as-judge** comes in: hand the answer (plus the sources
it was supposed to use) to a **separate, stronger LLM** and have it act as examiner.

`corpus-judge` grades every answer on **three axes**. Understanding these precisely
is the key to reading any generation result:

| Axis | The question it asks | Failure looks like |
|---|---|---|
| **Faithfulness** | Is every claim actually *supported by the retrieved sources*? | The model **hallucinated** — stated something the sources don't say. |
| **Answer relevance** | Does the answer *address the question that was asked*? | A true, grounded answer to a *different* question. |
| **Citation correctness** | Do the cited `source_key`s actually *contain* the claims attached to them? | The answer is right, but it cites the wrong source. |

These come apart in important ways:

- **Faithful but irrelevant:** everything is grounded, but it didn't answer the question.
- **Relevant but unfaithful:** on-topic, but it invented a detail.
- **Faithful and relevant but mis-cited:** correct and on-topic, but pointing at the
  wrong source. This is why **citation is its own axis** — it's the strictest.

Two deliberate design choices in `corpus-judge`:

- **The judge is a stronger, *different* model than the generator.** If a model
  grades its own output, it's biased toward being kind to itself — **self-preference**.
  A separate, more capable judge is a fairer examiner. (`corpus` defaults to a stronger
  judge model than generator model.)
- **The judge returns a structured verdict, not prose** — a forced pass/fail plus a
  short rationale per axis, so verdicts are machine-countable and consistent.

There are no sample-corpus generation *numbers* shipped, because meaningful
generation quality is a property of *your* documents and questions — run
`corpus-judge --queries your_queries.py --config your.toml` to measure your own.

---

## Part 6 — Can you even trust the judge? (Validation)

The obvious objection: **"You're using an AI to grade an AI — isn't that circular?"**

The honest answer: **an unvalidated judge is untrustworthy, so you validate it.** You
don't take the judge's word for it — you first *prove* it agrees with human judgment,
and only then trust its verdicts at scale. This is the step most people skip, and
it's what separates a real eval from a vibes-generator with extra steps.

### The validation study

`corpus` ships a **frozen, hand-labeled fixture** ([`tests/judge_fixture.py`](../tests/judge_fixture.py)):
answers a *human* labeled faithful-or-not by hand, including **adversarial** cases
(answers deliberately written to be subtly wrong). Then the judge grades those same
answers, and you ask: **how often does the judge agree with the human?**

```sh
corpus-judge --validate --fixture tests/judge_fixture.py
```

### Cohen's kappa (κ) — measuring agreement *honestly*

You might think "just measure the % of times they agree." But raw agreement lies,
because **some agreement happens by luck.** If 90% of answers are faithful, a judge
that blindly says "faithful" every time agrees 90% of the time — while being useless.

**Cohen's kappa (κ)** corrects for chance. It answers: "how much do they agree
*beyond what random guessing would produce*?"

- **κ = 1.0** — perfect agreement.
- **κ = 0.0** — chance-level. Worthless judge.
- **κ < 0** — worse than chance.
- Rules of thumb: **> 0.8** excellent, 0.6–0.8 good, below ~0.4 weak.

`corpus`'s public fixture reports a high κ, and the CI gate (`judge-gate`) fails the
build if κ drops below a committed floor (0.8) or any adversarial case is missed — so
the judge stays calibrated over time. Full mechanics: [`docs/judge.md`](judge.md).

> **A caveat worth internalizing:** if your fixture labels are *deliberately
> unambiguous* (clear-cut faithful/not), a high κ partly reflects an *easy* exam —
> easy cases are easy to agree on. A harder validation (a human independently
> re-grading messy, real answers) is more discriminating. Read a high κ as "the judge
> is not obviously broken and agrees on clear cases," not "the judge is flawless on
> hard cases." **Being honest about what a metric does and doesn't prove is itself a
> core eval skill.**

---

## Part 7 — Reading results: baseline, delta, and NOISE

Once you have numbers, three ideas keep you from fooling yourself.

### 7.1 Baseline vs verdict

A **baseline** is "the score before you change anything." Its absolute value is only
meaningful *relative to this specific query set and config*. A well-built eval set is
often *deliberately hard* (it includes tricky and even unanswerable questions), so a
"50%" is not "the system is right half the time in normal use" — it's "on a hard,
strict exam, it passed half." **A single absolute number is a baseline, not a
verdict.** Treat it as a starting line, not a report-card grade.

### 7.2 You care about the delta

The whole point is: *did this change help?* That's a **delta** against a fixed
baseline, on the *same* eval set. An absolute score in isolation tells you little; a
+X against last week's identical run tells you a lot.

### 7.3 The single most important concept: NOISE

Suppose a score goes from 70% to 73% after a change. **Is that an improvement?**
Often, **no — it's noise.** Here's the math, and it's the most useful thing to
internalize:

> With **n questions, each question is worth 1/n of the score.**
> On `corpus`'s 30-query set, one question = **1/30 ≈ 3.3%**.
> So a "70% → 73%" swing is *literally one more question passing* — and a one- or
> two-question move on a small set is **within the random wobble** you'd get just
> from re-running a probabilistic model.

A swing of one or two questions is **not signal.** If your eval set is small, your
**noise floor** is high — you *cannot* detect small real improvements, because they're
drowned out by the per-question granularity. **Always ask: how many questions is this
delta worth?** If the answer is "one or two," don't celebrate and don't panic. (This
is also why bigger eval sets are better: 500 questions → each is worth 0.2%, so you
can resolve much finer differences.)

### 7.4 A metric is a proxy, not the truth

Faithfulness/relevance/citation are *proxies* for "is this a good answer." They're
useful precisely because they're specific and countable — but don't overfit to them.
If you find yourself gaming a number rather than improving the product, step back.

---

## Part 8 — Retrieval levers, and a worked lesson: don't assume, measure

`corpus-judge` scores the answer against the *retrieved* context. That makes it a way
to measure not just generation, but **whether a retrieval change actually improves
generation.** Run the loop with and without a change and read the delta. `corpus`
exposes two cheap levers on the judge CLI: `--rerank` and `--top-k`.

**The reranker (`--rerank`).** A reranker is a slower, smarter model (a cross-encoder)
that re-reads the retrieved candidates and **re-sorts** them by true relevance, before
the top few feed the generator. Rerankers are well-known to improve *retrieval* — so
the natural assumption is "reranking will make the answers better too, especially
citation."

**Here's the lesson: test that assumption; don't ship it on faith.** A reranker that
improves *retrieval recall* (getting the right chunk *into* the top results) does not
necessarily improve *generation quality* — because when the supporting chunk is
**already** in the top *k* (the common case), *re-ordering* those chunks doesn't
change what the generator reads, so the answer doesn't change. Reranking rescues the
rarer case where the right chunk was ranked just below the cutoff; if it was already
above the cutoff, reranking is a no-op for the generator. **Retrieval recall and
generation quality are different measurements, and a lever that moves one may not move
the other.**

So don't reach for a reranker to fix a weak *citation* axis on faith — **measure it on
your corpus:**

```sh
corpus-judge --queries your_queries.py --config your.toml            # baseline
corpus-judge --queries your_queries.py --config your.toml --rerank   # +reranker
# then compare the two aggregates — is the delta bigger than your noise floor?
```

This is exactly what evals are *for*: they turn a plausible, confident assumption into
either evidence or a caught mistake — before it costs you latency, money, or a
regression.

---

## Part 9 — The lessons, distilled

1. **Measure deltas, not vibes.** The value is "did this change help?", answered with
   a number against a fixed baseline.
2. **Know your noise floor.** delta ÷ (1/n) = "how many questions is this worth?" If
   that's 1–2, it's noise. Small eval sets can't detect small wins.
3. **Validate the judge before trusting it** (Cohen's κ). An LLM-as-judge is only as
   good as its agreement with humans.
4. **A metric is a proxy, not the truth.** A hard-exam score is a baseline to improve,
   not a verdict on the product.
5. **Separate retrieval from generation.** They fail differently and are fixed
   differently; a lever that helps one may do nothing for the other.
6. **One config + one query set = a baseline, not a verdict.** Re-run after changes;
   compare configs and corpora carefully.
7. **Be honest about what a result proves.** A high κ on easy labels isn't a high κ on
   hard ones. Writing down the caveat *is* the skill.

---

## Part 10 — FAQ

**Q: Isn't using an AI to grade an AI hopelessly circular?**
Only if you skip validation. You first *prove* the judge agrees with human graders on
a labeled set (Cohen's κ), and you use a *stronger, different* model as judge than as
generator. A validated judge is a calibrated instrument, not a circular argument. An
*unvalidated* judge genuinely is worthless — which is why `corpus` ships a fixture and
a κ gate.

**Q: Why not just have humans grade everything?**
Humans are the gold standard for quality but are slow, expensive, and surprisingly
*inconsistent*. Grading thousands of answers per code change is infeasible — you'd
never iterate. The modern practice is hybrid: humans label a *small* set to validate
the judge, then the judge scales to everything.

**Q: Is 50% a bad score? What's "good"?**
There's no universal "good." A score is only meaningful against its own query set and
axes, and good eval sets are deliberately hard. The only score that matters is the
*next* one compared to this one. Chasing an absolute target usually means you've
started gaming the metric.

**Q: Faithfulness vs answer relevance — what's the difference?**
Faithfulness = "true to the sources" (no hallucination). Relevance = "answers the
question asked" (on-target). An answer can nail one and fail the other: a grounded
summary of the *wrong* topic is faithful-but-irrelevant; a spot-on answer with one
invented figure is relevant-but-unfaithful.

**Q: What does "within noise" actually mean?**
The change is small enough to be random run-to-run wobble rather than a real effect.
With *n* questions, the finest change you can even represent is 1/*n*. A delta of one
or two questions is at the resolution limit — indistinguishable from randomness. To
detect small real improvements, use a bigger eval set or a change big enough to move
many questions at once.

**Q: How many test questions do I need?**
More is better, with diminishing returns. Tens of questions catch *big* effects but
are too coarse for *subtle* ones (noise floor ≈ a few % per question). Rule of thumb:
your eval set should be large enough that the smallest improvement you *care about* is
worth several questions, not one.

**Q: Retrieval recall can be high while generation quality is mediocre — how?**
They measure different stages. High recall says "the right chunk was *available* in
the top results." Mediocre citation says "the model didn't correctly *point at* it
when writing the answer." The info was on the desk; the model cited the wrong page.
Separating the two evals localizes the problem to the right stage.

**Q: Why not set temperature to 0 for deterministic grading?**
Newer models can *reject* the `temperature` parameter outright. `corpus` gets
consistency a different way — disabling "thinking" mode and forcing structured tool
output, which constrains the response into a fixed shape. Determinism by construction,
not by temperature.

**Q: Why does `corpus-judge` force the model to fill out a tool/form?**
So verdicts are structured and countable rather than free prose. One subtlety learned
the hard way: forcing a tool guarantees it's *called* but not that every field is
*filled*, so the schemas use `strict: true` (+ `additionalProperties: false`) to
guarantee complete, valid output, with defensive parser defaults as a backstop. See
[`docs/judge.md`](judge.md).

**Q: How much does a generation eval cost?**
Roughly one generation call plus one (stronger) judge call per question — a few
dollars for a few-dozen-question set, input-dominated. Cheap enough to run after every
meaningful change. The *retrieval* eval is free and keyless (it runs in CI on every
commit).

**Q: Can I trust a single run?**
Treat one run as one data point. Probabilistic models wobble; a one-question
difference in a single run isn't reliable. Big, consistent deltas across runs are
trustworthy — run twice, or use a larger eval set, when a change is close to your
noise floor.

**Q: Offline evals vs "monitoring in production" — what's the difference?**
Offline evals (what `corpus` does) run a *fixed* set of questions in development to
compare versions before shipping — a controlled experiment. Production monitoring
watches *live* traffic for problems (latency, errors, thumbs-down, sampled quality).
You need both: offline to choose what to ship, online to catch what the lab didn't
predict.

**Q: Both generator and judge are the same model family — isn't that biased?**
Somewhat; the cleanest setup uses a judge from a *different* family to rule out
family-level self-preference. `corpus` mitigates the worst of it by using a
*different, stronger* model as judge and by validating against human labels (the κ
study). Cross-family judging is a reasonable further hardening step.

---

## Part 11 — Glossary

- **Eval** — a test/measurement of an AI system's quality, usually producing a score.
- **Baseline** — the score before a change; your reference point for deltas.
- **Delta** — the change in score after a modification (the thing you actually care about).
- **Noise / noise floor** — random run-to-run wobble; on *n* questions the floor is ≈ 1/*n* per question. Deltas below it aren't signal.
- **Eval set / golden set** — the fixed collection of test questions (with known-good targets for retrieval).
- **RAG** — Retrieval-Augmented Generation: retrieve relevant documents, then have the LLM answer from them.
- **Retrieval** — the "find the right documents" stage. **Generation** — the "write the answer" stage.
- **Chunk** — a small passage a document is split into for retrieval.
- **Embedding / vector** — a document turned into numbers capturing its meaning, for semantic search.
- **BM25** — classic keyword-overlap retrieval. **Hybrid** — fusing vector + BM25 (reciprocal rank fusion).
- **Reranker / cross-encoder** — a slower, smarter model that re-sorts retrieved candidates by true relevance.
- **top_k** — how many retrieved chunks the generator is given.
- **recall@k / MRR / nDCG@k** — automated retrieval metrics (found? how highly? weighted by rank?).
- **LLM-as-judge** — using a stronger LLM to grade generated answers at scale.
- **Faithfulness / Answer relevance / Citation correctness** — the three generation-quality axes (grounded / on-target / correctly attributed).
- **Self-preference** — a model's bias toward rating its own outputs highly (why judge ≠ generator).
- **Cohen's kappa (κ)** — agreement between two graders, corrected for chance. 1.0 = perfect, 0 = chance-level.
- **Adversarial case** — a deliberately-tricky test item used to check the judge catches subtle errors.
- **Offline eval** — fixed-set evaluation in development. **Online / monitoring** — measuring live production traffic.
- **Hallucination** — a confident claim the sources don't support.

---

*Next: [`docs/eval.md`](eval.md) for retrieval-metric mechanics, and
[`docs/judge.md`](judge.md) for the LLM-as-judge and its validation study.*
