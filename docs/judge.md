# Generation quality — validated LLM-as-judge

The retrieval eval (`corpus-eval`, see [eval.md](eval.md)) measures whether the
right chunks come back. This layer measures whether the system produces a
**grounded, correct answer** from those chunks — scored by an LLM-as-judge that
is itself **validated** against human labels.

## Components

- **`answer_from_context(query, context)`** (`corpus.eval.generation`) — a
  feature-flagged answerer. `context` is a list of `(source_key, text)` tuples,
  so it never touches the retriever or DB. Uses forced tool output so the cited
  source_keys come back structured. Default model: `claude-sonnet-5`.
- **`judge_answer(query, answer, cited_keys, context)`** (`corpus.eval.judge`) —
  a 3-axis judge (faithfulness, answer relevance, citation correctness) with
  forced tool-schema output, disabled thinking, and randomized context order
  (position-bias control). Default model: `claude-opus-4-8` — a **separate,
  stronger** model than the generator, to mitigate self-preference bias.
- **`cohens_kappa` + `run_validation_study`** (`corpus.eval.validation`) —
  certify the judge: run it over a frozen, hand-labeled fixture and report
  Cohen's κ (judge faithfulness vs human) plus an adversarial-catch check.

No `temperature` is set anywhere: the current judge/generator models reject it.
Determinism comes from disabled thinking + forced structured output.

## Robustness: complete forced-tool output

Forced `tool_choice` guarantees the model *calls* the tool, but **not** that it
populates every `required` field. On small inputs this never surfaces; on larger
real-world archives it does — the generator occasionally returned `submit_answer`
with no `answer` field, and the judge occasionally returned `record_verdict`
missing a verdict field, enough to abort a batch run. Two layers prevent that:

- **`strict: true` + `additionalProperties: false`** on both tool schemas, so the
  API guarantees a schema-complete response (all required fields, correct types).
- **Defensive defaults** in the parsers (`data.get(...)`) as belt-and-suspenders,
  so a malformed response degrades to an empty answer / a conservative "fail"
  verdict instead of raising — a batch run over many queries never aborts on one
  bad response.

The eval modules (`_anthropic`, `generation`, `judge`, `validation`) use relative
imports, so they are reusable verbatim across separate deployments (a `diff` is
the drift check).

## The validation study

`tests/judge_fixture.py` holds frozen `JUDGE_CASES` — each carries its own
context, so validation needs **no database**, only an API key. Cases are
self-contained public content. Adversarial cases are hand-authored: plausible
answers the context does not support; a correct judge must mark them unfaithful.

```bash
export ANTHROPIC_API_KEY=...
uv run corpus-judge --validate --fixture tests/judge_fixture.py
```

reports Cohen's κ, raw agreement, and the adversarial-catch count. Only
faithfulness is human-validated in v1; relevance and citation-correctness are
produced and reported but not yet κ-calibrated.

## Running the full generate+judge loop

```bash
uv run corpus-judge --queries tests/eval_queries.py \
  --config examples/sample_corpus/corpus.toml
```

retrieves, answers, and judges each query, printing a 3-axis aggregate.
`--build-fixture --out PATH` writes a draft fixture (with `human_faithful=None`)
for hand-labeling.

## CI gate

The opt-in `judge-gate` CI job is **gated on the `ANTHROPIC_API_KEY` secret**.
When the secret is unset (forks, or before configuration) it skips cleanly. When
set, it runs `corpus-judge --validate --check
examples/sample_corpus/judge_thresholds.json`, failing the build if Cohen's κ
drops below the committed floor or any adversarial case is missed. The keyless
`eval-gate` is untouched.

## Privacy

The fixture, tests, CI, and this document use only public content. Running
generation or judging over a private archive stays strictly local and is never
committed or pushed.
