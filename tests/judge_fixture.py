"""Frozen, self-contained judge validation fixture.

Each JudgeCase carries its own context (source_key, text) pairs, so
`corpus-judge --validate` needs no database — only an API key. All content here
is self-authored PUBLIC text: generic facts about the corpus project and
everyday topics. It contains ZERO private-corpus data.

The adversarial cases are hand-authored: plausible-sounding answers whose
context does not support them. A correct faithfulness judge must mark every
adversarial case unfaithful.
"""

from __future__ import annotations

from corpus.eval.validation import JudgeCase

JUDGE_CASES: list[JudgeCase] = [
    # --- Faithful cases -----------------------------------------------------
    JudgeCase(
        query="What database does corpus store chunks in?",
        answer="corpus stores chunks in SQLite, using sqlite-vec for vectors and FTS5 for BM25.",
        cited_keys=["arch-1"],
        context=[
            ("arch-1", "corpus keeps all data in a single SQLite database. Vector search "
                       "uses the sqlite-vec extension; keyword search uses SQLite FTS5 (BM25)."),
        ],
        human_faithful=True,
        note="fully supported",
    ),
    JudgeCase(
        query="How does corpus combine keyword and vector results?",
        answer="It fuses the two ranked lists with reciprocal rank fusion.",
        cited_keys=["arch-2"],
        context=[
            ("arch-2", "Hybrid retrieval fuses the vector and BM25 result lists using "
                       "reciprocal rank fusion (RRF) with auto-tuned weights."),
            ("arch-9", "The optional reranker is a BGE cross-encoder."),
        ],
        human_faithful=True,
        note="supported; extra context present but unused",
    ),
    JudgeCase(
        query="Is a network connection required to run the retrieval eval?",
        answer="No. The eval can run with the zero-dependency 'hash' embedder, "
        "which needs no API key.",
        cited_keys=["eval-1"],
        context=[
            ("eval-1", "A deterministic 'hash' embedder lets the retrieval pipeline run with "
                       "no API key, so the eval gate is keyless in CI."),
        ],
        human_faithful=True,
        note="supported",
    ),
    # --- Adversarial cases (answer NOT supported by context) ----------------
    JudgeCase(
        query="What license is corpus released under?",
        answer="corpus is released under the Apache 2.0 license.",
        cited_keys=["meta-1"],
        context=[
            ("meta-1", "corpus is a single-user RAG framework for personal archives. "
                       "It is local-first and MCP-native."),
        ],
        human_faithful=False,
        adversarial=True,
        note="context never states a license; 'Apache 2.0' is unsupported (it is in fact MIT)",
    ),
    JudgeCase(
        query="How many MCP tools does corpus expose?",
        answer="corpus exposes exactly twelve MCP tools.",
        cited_keys=["mcp-1"],
        context=[
            ("mcp-1", "corpus wires a set of MCP tools to Claude Code over stdio, including "
                      "search, get_document, and expand_context."),
        ],
        human_faithful=False,
        adversarial=True,
        note="context gives no count; 'twelve' is fabricated",
    ),
    JudgeCase(
        query="What embedding model does the Voyage provider use?",
        answer="The Voyage provider uses the voyage-2-lite model by default.",
        cited_keys=["emb-1"],
        context=[
            ("emb-1", "corpus supports pluggable embedding providers. The provider is chosen "
                      "in corpus.toml under the [embedder] section."),
        ],
        human_faithful=False,
        adversarial=True,
        note="context names no specific model; 'voyage-2-lite' is unsupported",
    ),
    JudgeCase(
        query="When was corpus first released?",
        answer="corpus was first released in January 2024.",
        cited_keys=["meta-2"],
        context=[
            ("meta-2", "corpus is distributed on PyPI as the corpus-rag package and is "
                       "installable with pip."),
        ],
        human_faithful=False,
        adversarial=True,
        note="context gives no date; 'January 2024' is fabricated",
    ),
]
