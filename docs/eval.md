# Eval methodology

`corpus-eval` runs a hand-written, known-answer query set against a live corpus and reports retrieval-quality metrics: recall@K, MRR, and nDCG@K, plus an aggregate table, a per-source-type breakdown, and `--json`. It's a regression signal — run it after changing chunking, switching embedders, tweaking retrieval fusion, or flipping the reranker on.

## What the eval measures

Relevance is **binary**: for a given query, a retrieved chunk's `source_key` either counts (it's a member of that query's `expected_keys`) or it doesn't — no graded 0–3 relevance scale. `expected_keys` is an **OR-set**: if a query has two valid answer docs, either one showing up counts as a hit, not both.

Three metrics, computed by the pure functions in [`src/corpus/eval/metrics.py`](../src/corpus/eval/metrics.py):

| Metric | Exact formula | One-liner |
|---|---|---|
| `recall@K` | `1.0` if any `expected_keys` member is in the top-K ranked `source_key`s, else `0.0` | Binary hit/miss at a cutoff |
| `MRR` | `1 / rank` (1-indexed) of the **first** expected key found, searching the **whole ranked list** (no `K` cutoff); `0.0` if none found | How far down the full ranking you had to look |
| `nDCG@K` | binary-gain DCG = `Σ 1/log2(rank+1)` over hits in the top-K, divided by the ideal DCG — the DCG of an ordering that places `min(#expected_keys, K)` hits at ranks `1..that count` | Rewards hits ranked *higher*, not merely present |

Each query scores a `QueryScore(recall, rr, ndcg)`. `aggregate()` macro-averages those into a `MetricSummary(recall_at_k, mrr, ndcg_at_k, n)` across every **scored** query. Negative queries (empty `expected_keys` — see "Writing queries" below) are excluded from the average; they still print, informationally, so you can eyeball what a query with no right answer actually retrieves.

`metrics.py` takes only `(expected_keys, ranked_found_keys, k)` — plain strings and an int, no `Retriever` / `ChunkStore` / config imports anywhere in the module. That purity is deliberate; it's what lets the module be reused unchanged across deployments (see "Portability" below).

## Running it with no API key

Both real embedders (`voyage`, `gemini`) are cloud services, and `Retriever.query` always calls `embedder.embed_query` before it can search — so ordinarily there's no way to run retrieval, let alone eval it, without an API key on file. `corpus` ships a zero-dependency `hash` embedder (`provider="hash"`, [`src/corpus/embedder/hash.py`](../src/corpus/embedder/hash.py)) plus a committed sample corpus and config so you — and CI — can run the full hybrid BM25+vector pipeline with nothing installed and no key:

```sh
uv run corpus-ingest --config examples/sample_corpus/corpus.toml --all
uv run corpus-eval   --config examples/sample_corpus/corpus.toml
```

`examples/sample_corpus/` is 20 markdown docs across two source types — 12 `note` pages (conceptual docs about corpus's own architecture: chunking, hybrid search, reranking, dedup, etc.) and 8 `faq` pages (short Q&A) — ingested with the committed `corpus.toml`'s `provider = "hash"`, `dim = 256`, then scored against the 30-query set in [`tests/eval_queries.py`](../tests/eval_queries.py).

**The hash embedder is a reproducibility substrate, not a semantic-quality model.** It hashes token features into a fixed-dim vector (signed feature hashing over blake2b digests, so it's stable across processes and immune to `PYTHONHASHSEED` salting) and L2-normalizes the result — cosine similarity ends up approximating *weighted lexical overlap*, nowhere close to what an actual embedding model captures. Its entire job is making "run the eval with no key" and the CI gate deterministic and free. **Absolute retrieval quality is measured on real corpora with the `voyage` or `gemini` providers** — see [`configuration.md`](configuration.md#choosing-a-provider) for that comparison.

One direct consequence: the shipped eval queries deliberately share distinctive keywords with their target doc (e.g. *"content-hash dedup skips unchanged chunks on re-ingest"* → `deduplication`), because the hash embedder has nothing else to key on — it can't recognize a paraphrase it's never seen the tokens for. On a real embedder, paraphrase harder; see the caveat below.

## Writing queries

An eval query set is any Python file exporting `EVAL_QUERIES`, loaded via `--queries` (default `tests/eval_queries.py`):

```python
# my_queries.py
from dataclasses import dataclass, field

@dataclass(frozen=True)
class EvalQuery:
    query: str
    expected_keys: list[str] = field(default_factory=list)
    source_filter: list[str] | None = None
    source_type: str | None = None
    note: str = ""

EVAL_QUERIES = [
    EvalQuery(
        query="how does the payment flow work?",
        expected_keys=["payment-design-doc"],
        source_type="doc",
        note="paraphrased to stress semantic retrieval",
    ),
    EvalQuery(
        query="reciprocal rank fusion",
        expected_keys=["hybrid-search"],
        source_filter=["note"],
        source_type="note",
        note="filter restricts the search to the note source type",
    ),
    EvalQuery(
        query="what is the capital of france",
        note="negative — off-topic; expected_keys empty, excluded from scoring",
    ),
    # add more...
]
```

- **`expected_keys`** is the OR-set: list every `source_key` that would count as a correct answer. Most queries have exactly one; a few valid duplicate/overlapping docs can share a query.
- **`source_filter`** narrows the retriever's search to specific source types (same as `corpus-query --source`) — use it to test a filtered-search path, not to make a query artificially easier.
- **`source_type`** is bucket-only: it tags which row of the per-source-type breakdown a query's score rolls into. `metrics.py` never sees it — it's a CLI/reporting-layer concept, not a scoring concept.
- **Negative queries** (empty `expected_keys`) mark topics the corpus should legitimately have nothing for. They're excluded from every aggregate but still print under `[INFO]` so you can sanity-check the corpus isn't confidently returning garbage for absent topics.
- **Lexical overlap vs. paraphrasing**: under the `hash` embedder, keep queries close to their target's vocabulary (see above — it's a lexical proxy, not semantic). Once you point `--config` at a `corpus.toml` using `voyage` or `gemini` against your own real corpus, paraphrase away from the doc's exact wording — that's what stresses whether the embedder is actually capturing meaning rather than keyword overlap.

## The reports

Default (human-readable) output prints one block per query, then an aggregate table, then a per-source-type breakdown (shown only when queries carry 2+ distinct `source_type` values). Real output from the sample-corpus run above:

```
[PASS] why keep configuration in corpus.toml instead of hardcoding it in python, since it changes across deployments
       expected in ['configuration'], got: ['configuration', 'faq-multi-corpus', 'faq-cost', 'benchmarking', 'connectors']
       recall@5=1 rr=1.000 ndcg@5=1.000

[PASS] corpus calls itself a straightforward single-user local RAG framework
       expected in ['architecture'], got: ['faq-sources', 'welcome', 'faq-multi-corpus', 'faq-cost', 'architecture']
       recall@5=1 rr=0.200 ndcg@5=0.387

[INFO] what is the capital of france
       top-5 (informational): ['scrubbing', 'faq-updating', 'faq-privacy', 'faq-cost', 'deduplication']
       note: negative — off-topic

=== Aggregate (n=30) ===
  recall@5: 1.000
  MRR:       0.865
  nDCG@5:   0.898

=== By source_type ===
  source_type         n   recall      mrr     ndcg
  faq                11    1.000    0.955    0.966
  note               19    1.000    0.813    0.859
```

The second `[PASS]` line is worth reading closely: recall@5 is a hit (the right doc is somewhere in the top 5), but `rr=0.200` (it landed at rank 5) drags `ndcg@5` down to 0.387 — this is exactly the case recall@K is blind to and MRR/nDCG surface: "found it" vs. "found it near the top."

### `--json`

`--json` emits the same data as a single structured document for tooling and dashboards (the CI gate itself uses `--check`, not JSON parsing — see [CI gate](#ci-gate-phase-3)):

```json
{
  "config": { "top_k": 5, "hybrid": true, "rerank": false },
  "aggregate": { "recall_at_k": 1.0, "mrr": 0.865, "ndcg_at_k": 0.898281970629313, "n": 30 },
  "by_source_type": {
    "faq":  { "recall_at_k": 1.0, "mrr": 0.9545454545454546, "ndcg_at_k": 0.966448159415587, "n": 11 },
    "note": { "recall_at_k": 1.0, "mrr": 0.8131578947368421, "ndcg_at_k": 0.858817335016207, "n": 19 }
  },
  "queries": [
    {
      "query": "corpus calls itself a straightforward single-user local RAG framework",
      "expected_keys": ["architecture"],
      "found_keys": ["faq-sources", "welcome", "faq-multi-corpus", "faq-cost", "architecture"],
      "source_type": "note",
      "is_negative": false,
      "recall": 1.0,
      "rr": 0.2,
      "ndcg": 0.38685280723454163
    }
  ]
}
```

`recall`/`rr`/`ndcg` are `null` for negative (`is_negative: true`) queries instead of `0.0`, so a JSON consumer can distinguish "scored and missed" from "not scored."

### `--compare`

`--compare` reuses one store/embedder and re-runs the whole query set under several retrieval configs, printing a metric × config table — hybrid on/off always, plus a `hybrid+rerank` row when you also pass `--rerank` (and the reranker extra is installed):

```
=== Config comparison (top_k=5) ===
  config             recall      mrr     ndcg
  hybrid              1.000    0.865    0.898
  vector-only         0.933    0.838    0.861
```

On the sample corpus, hybrid (BM25 + vector, fused) beats vector-only on every metric — expected, since several eval queries lean on exact-keyword overlap (`hybrid-search`, `reciprocal rank fusion`) that BM25 is well-suited to and pure vector similarity can miss.

### CLI flags at a glance

| Flag | Default | Effect |
|---|---|---|
| `--queries PATH` | `tests/eval_queries.py` | Python module exporting `EVAL_QUERIES` |
| `--config PATH` | your `corpus.toml` | Which corpus to run against (`load_config_or_exit`) |
| `--top-k N` | `5` | Cutoff for `recall@K` / `nDCG@K` (MRR ignores it) |
| `--rerank` | off | Adds the BGE cross-encoder rerank stage |
| `--no-hybrid` | off (hybrid on) | Vector-only baseline, no BM25 fusion |
| `--compare` | off | Metric × config comparison table (see above) instead of the single-config report |
| `--json` | off | Emit the structured result instead of human-readable tables |
| `--check PATH` | off | JSON thresholds file; gate the exit code on the aggregate meeting each floor (single-config only, see [CI gate](#ci-gate-phase-3)) |

`corpus-eval` is **report-only by default**: without `--check`, it always exits `0` and does not fail on a low score. Regression gating lives in the pytest smoke floor (`tests/test_eval_queries_smoke.py`) and in `--check PATH` (a JSON thresholds file — see ["CI gate (Phase 3)"](#ci-gate-phase-3) below), which the `eval-gate` CI job runs on every push/PR.

## Results

### Sample corpus (`examples/sample_corpus/`, keyless `hash` embedder)

| | Aggregate (n=30) | `faq` (n=11) | `note` (n=19) |
|---|---|---|---|
| recall@5 | 1.000 | 1.000 | 1.000 |
| MRR | 0.865 | 0.955 | 0.813 |
| nDCG@5 | 0.898 | 0.966 | 0.859 |

Read this plainly: **`recall@5 = 1.000` here is a keyless hash-embedder reproducibility baseline, not a semantic-quality result.** The `hash` embedder (see ["Running it with no API key"](#running-it-with-no-api-key) above) approximates lexical overlap, not meaning, and the shipped queries deliberately share distinctive keywords with their target docs. A perfect recall score means "the hybrid pipeline reliably surfaces the doc that shares the query's keywords," not "corpus retrieves perfectly" — and it says nothing about semantic retrieval quality. That's measured on your own corpus with the `voyage`/`gemini` providers.

### The finding: hybrid fusion earns its place

Because the absolute numbers above are a reproducibility substrate rather than a quality claim, the result worth trusting is the **ablation delta** between retrieval configs (`--compare`, same store/embedder/queries — only fusion changes):

| config | recall@5 | MRR | nDCG@5 |
|---|---|---|---|
| hybrid | 1.000 | 0.865 | 0.898 |
| vector-only | 0.933 | 0.838 | 0.861 |

Hybrid (BM25 + vector, fused via reciprocal rank fusion) beats vector-only on **all three metrics** — recall 1.000 vs. 0.933, MRR 0.865 vs. 0.838, nDCG@5 0.898 vs. 0.861. The takeaway: even on a purely lexical `hash` embedder, where vector similarity has no semantic signal to lose in the first place, fusing BM25 with vectors still measurably wins. That's the honest reading of this corpus's numbers: not "retrieval is perfect," but "hybrid fusion earns its place."

### Portability: the metric module travels

**`metrics.py` is dependency-free by design so it can be reused byte-identical across separate RAG deployments.** It's copied, not imported — its zero-dependency, corpus-agnostic surface (this doc's first section) is what makes that copy safe. The extended CLI (`cli/eval.py`) ports the same way, wired to whatever `Retriever` a given deployment uses.

## CI gate (Phase 3)

Phase 1 (this doc) plus the eval harness itself run with no API key against `examples/sample_corpus` — that's what makes a CI gate possible without embedding secrets into the pipeline. Phase 3 wires that up as the `eval-gate` job in [`.github/workflows/ci.yml`](../.github/workflows/ci.yml), separate from the matrix `test` job:

```sh
uv sync --dev   # base install only — no --all-extras, no API key
uv run corpus-ingest --config examples/sample_corpus/corpus.toml --all
uv run corpus-eval   --config examples/sample_corpus/corpus.toml --check examples/sample_corpus/thresholds.json
```

`--check PATH` (added alongside this job) loads a JSON thresholds file — floor per metric, checked against the run's aggregate — prints one `[gate] metric = value  floor floor  PASS|FAIL` line per checked metric to stderr, and exits `1` if any metric is below its floor (`0` otherwise). The committed [`examples/sample_corpus/thresholds.json`](../examples/sample_corpus/thresholds.json):

```json
{"recall_at_k": 0.95, "ndcg_at_k": 0.85}
```

Floors sit just under the measured baseline (recall@5 = 1.000, nDCG@5 = 0.898 — see [Results](#results)) so the gate catches real regressions in chunking, hybrid fusion, or config changes without being so tight that it flakes on noise. The `hash` embedder is deterministic (no `PYTHONHASHSEED` sensitivity, see above), so there's no run-to-run variance to buffer against — a drop below floor means something actually changed.

**Updating the floors:** when a legitimate retrieval improvement moves the sample-corpus numbers (new chunking strategy, better fusion, etc.), re-run the command above, update `examples/sample_corpus/thresholds.json` to sit just under the new baseline, and update the [Results](#results) table in the same commit — don't raise the floor to exactly the new number, leave a little headroom. Never lower a floor to make a real regression pass; if the gate fails, treat it as a bug to fix, not a threshold to relax.

**No API key needed:** `eval-gate` runs `uv sync --dev` (no `--all-extras`), sets no `ANTHROPIC_API_KEY` or provider secret, and still exercises the full hybrid BM25+vector pipeline end to end via `provider="hash"` — proving the keyless path stays green on every push/PR to `main`.

**Deferred to Phase 2:** the LLM-as-judge / generation-quality gate (an opt-in job gated on `ANTHROPIC_API_KEY`) doesn't exist yet — it's out of scope for retrieval eval and belongs to Phase 2's generation work, not this gate.
