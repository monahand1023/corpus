"""Frozen, self-contained judge validation fixture.

Each JudgeCase carries its own context (source_key, text) pairs, so
`corpus-judge --validate` needs no database — only an API key. All content here
is self-authored PUBLIC text: generic facts about the corpus project. It
contains ZERO private-corpus data.

Labels are by construction: a *faithful* case's answer is directly supported by
its own context; an *adversarial* case's answer asserts a claim the context does
not contain (a correct faithfulness judge must mark every adversarial case
unfaithful). 20 cases, balanced 10 faithful / 10 adversarial so Cohen's kappa is
not dominated by skewed marginals.
"""

from __future__ import annotations

from corpus.eval.validation import JudgeCase

JUDGE_CASES: list[JudgeCase] = [
    # ------------------------------------------------------------------ #
    # Faithful cases (answer fully supported by its own context)          #
    # ------------------------------------------------------------------ #
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
        query="Does running the retrieval eval require a network connection?",
        answer="No. The eval can run with the zero-dependency 'hash' embedder, "
        "which needs no API key and no network.",
        cited_keys=["eval-1"],
        context=[
            ("eval-1", "A deterministic 'hash' embedder lets the retrieval pipeline run with "
                       "no API key and no network connection, so the eval gate is keyless in CI."),
        ],
        human_faithful=True,
        note="supported (context explicitly states no network)",
    ),
    JudgeCase(
        query="What built-in connectors does corpus ship with?",
        answer="corpus ships four built-in connectors: markdown, text, PDF, and HTML.",
        cited_keys=["conn-1"],
        context=[
            ("conn-1", "Out of the box, corpus provides four connectors for ingesting "
                       "documents: markdown, plain text, PDF, and HTML."),
        ],
        human_faithful=True,
        note="count and list both stated in context",
    ),
    JudgeCase(
        query="How does corpus follow links between documents?",
        answer="It uses expand_context to chase configurable [[references]] patterns across "
        "documents.",
        cited_keys=["hop-1"],
        context=[
            ("hop-1", "Multi-hop investigation is handled by expand_context, which chases "
                      "configurable [[references]] patterns to related documents across the archive."),
        ],
        human_faithful=True,
        note="supported",
    ),
    JudgeCase(
        query="What is the minimum Python version for corpus?",
        answer="corpus requires Python 3.12 or newer.",
        cited_keys=["req-1"],
        context=[
            ("req-1", "corpus requires Python 3.12+."),
        ],
        human_faithful=True,
        note="supported",
    ),
    JudgeCase(
        query="Does corpus offer a reranking step?",
        answer="Yes — an optional BGE cross-encoder reranker can rescore the fused results.",
        cited_keys=["rank-1"],
        context=[
            ("rank-1", "corpus includes an optional BGE cross-encoder reranker that rescores "
                       "the fused candidate list to improve ordering."),
        ],
        human_faithful=True,
        note="supported",
    ),
    JudgeCase(
        query="How does corpus expose its tools to Claude Code?",
        answer="Over the Model Context Protocol (MCP), using a stdio transport.",
        cited_keys=["mcp-2"],
        context=[
            ("mcp-2", "corpus exposes its retrieval tools to Claude Code over the Model "
                      "Context Protocol (MCP) using a stdio transport."),
        ],
        human_faithful=True,
        note="supported",
    ),
    JudgeCase(
        query="Where does a corpus instance keep its configuration?",
        answer="In a corpus.toml file at the instance root.",
        cited_keys=["cfg-1"],
        context=[
            ("cfg-1", "Each corpus instance is configured through a corpus.toml file at its root."),
        ],
        human_faithful=True,
        note="supported",
    ),
    JudgeCase(
        query="What does corpus's embedding-dimension startup guard protect against?",
        answer="It prevents silent corruption when an embedder model is swapped without "
        "re-ingesting, by checking the configured dimension against the stored vectors.",
        cited_keys=["guard-1"],
        context=[
            ("guard-1", "A startup guard compares the configured embedding dimension against "
                        "the dimension of already-stored vectors, preventing silent corruption "
                        "when an embedder model is swapped without re-ingesting."),
        ],
        human_faithful=True,
        note="supported",
    ),
    # ------------------------------------------------------------------ #
    # Adversarial cases (answer asserts a claim absent from the context) #
    # ------------------------------------------------------------------ #
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
    JudgeCase(
        query="How many document connectors does corpus include?",
        answer="corpus includes nine document connectors.",
        cited_keys=["conn-2"],
        context=[
            ("conn-2", "corpus can ingest several common document formats through its "
                       "connector system."),
        ],
        human_faithful=False,
        adversarial=True,
        note="context gives no count; 'nine' is fabricated",
    ),
    JudgeCase(
        query="Which reranking model does corpus use?",
        answer="corpus uses Cohere's rerank-3 model for reranking.",
        cited_keys=["rank-2"],
        context=[
            ("rank-2", "corpus supports an optional reranking step to improve the ordering "
                       "of retrieved results."),
        ],
        human_faithful=False,
        adversarial=True,
        note="context names no specific reranker; 'Cohere rerank-3' is unsupported",
    ),
    JudgeCase(
        query="What programming language is corpus implemented in?",
        answer="corpus is implemented in Rust for maximum performance.",
        cited_keys=["impl-1"],
        context=[
            ("impl-1", "corpus is a local-first RAG framework you install from PyPI with pip."),
        ],
        human_faithful=False,
        adversarial=True,
        note="context states no language; 'Rust' is fabricated (corpus is Python)",
    ),
    JudgeCase(
        query="What is the maximum archive size corpus supports?",
        answer="corpus supports archives of up to 10 million documents.",
        cited_keys=["scale-1"],
        context=[
            ("scale-1", "corpus is designed for a single user's personal archive."),
        ],
        human_faithful=False,
        adversarial=True,
        note="context states no size limit; '10 million' is fabricated",
    ),
    JudgeCase(
        query="Does corpus need a GPU to run?",
        answer="Yes, corpus requires a CUDA-capable GPU.",
        cited_keys=["hw-1"],
        context=[
            ("hw-1", "corpus runs hybrid retrieval over a local SQLite database on an "
                     "ordinary machine."),
        ],
        human_faithful=False,
        adversarial=True,
        note="context says nothing about a GPU requirement; fabricated (and false)",
    ),
    JudgeCase(
        query="Who maintains the corpus project?",
        answer="corpus is maintained by Microsoft Research.",
        cited_keys=["who-1"],
        context=[
            ("who-1", "corpus is an open-source project released under a permissive license."),
        ],
        human_faithful=False,
        adversarial=True,
        note="context names no maintainer; 'Microsoft Research' is fabricated",
    ),
]
