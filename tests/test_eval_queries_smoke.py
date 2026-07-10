from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from corpus.cli.eval import _run_query_set, _scored
from corpus.config import CorpusConfig
from corpus.db.sqlite import ChunkStore
from corpus.embedder.hash import HashEmbedder
from corpus.eval.metrics import aggregate
from corpus.ingester import Ingester
from corpus.retriever import Retriever

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_eval_queries() -> list:
    path = REPO_ROOT / "tests" / "eval_queries.py"
    spec = importlib.util.spec_from_file_location("eval_queries", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules before exec: eval_queries.py uses `from
    # __future__ import annotations`, so its frozen dataclass needs
    # sys.modules[cls.__module__] to resolve string annotations at class
    # creation time (AttributeError otherwise).
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return list(mod.EVAL_QUERIES)


def test_sample_corpus_recall_clears_floor(tmp_path: Path) -> None:
    db = tmp_path / "sample.db"
    config = CorpusConfig.load(REPO_ROOT / "examples" / "sample_corpus" / "corpus.toml")
    # redirect the DB to a tmp path so the test never touches a committed file
    config = config.model_copy(update={"db_path": db})
    ingester = Ingester(config)
    try:
        for s in config.sources:
            ingester.ingest(s.name)
    finally:
        ingester.close()

    store = ChunkStore(db, embedding_dim=config.embedder.dim)
    embedder = HashEmbedder(dim=config.embedder.dim)
    retriever = Retriever(store=store, embedder=embedder)
    try:
        records = _run_query_set(retriever, _load_eval_queries(), top_k=5, hybrid=True, rerank=False)
    finally:
        retriever.close()
    summary = aggregate(_scored(records))
    # Lexical hash embedder + keyword-sharing queries should retrieve most targets.
    # Observed baseline is a perfect 1.0 (30/30) — floor is tightened to
    # observed - 0.05 so it stays a meaningful regression signal without
    # being brittle to a single query's ranking shifting by one slot.
    assert summary.n >= 25
    assert summary.recall_at_k >= 0.95, f"recall dropped to {summary.recall_at_k:.3f}"
