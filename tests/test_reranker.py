from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from corpus.reranker.local import BGEReranker


def fake_chunk(content: str) -> SimpleNamespace:
    return SimpleNamespace(content=content, source_key="doc-1", source_type="notes")


def make_reranker_with_mock_model() -> tuple[BGEReranker, MagicMock]:
    """Return a BGEReranker with a pre-injected mock model, bypassing lazy load."""
    reranker = BGEReranker()
    mock_model = MagicMock()
    reranker._model = mock_model
    return reranker, mock_model


# ---------------------------------------------------------------------------
# 1. Empty input
# ---------------------------------------------------------------------------

def test_rerank_empty_returns_empty() -> None:
    reranker = BGEReranker()
    # _model is None; the empty-guard should return before touching it
    result = reranker.rerank("any query", [])
    assert result == []
    assert reranker._model is None  # model was never loaded


# ---------------------------------------------------------------------------
# 2. Sorting by descending score
# ---------------------------------------------------------------------------

def test_rerank_sorts_by_descending_score() -> None:
    reranker, mock_model = make_reranker_with_mock_model()
    chunks = [fake_chunk("low"), fake_chunk("high"), fake_chunk("mid")]
    mock_model.predict.return_value = [0.1, 0.9, 0.5]

    result = reranker.rerank("query", chunks)

    assert result[0].content == "high"
    assert result[1].content == "mid"
    assert result[2].content == "low"


# ---------------------------------------------------------------------------
# 3. top_n trims the result
# ---------------------------------------------------------------------------

def test_rerank_top_n_trims_result() -> None:
    reranker, mock_model = make_reranker_with_mock_model()
    chunks = [fake_chunk("a"), fake_chunk("b"), fake_chunk("c")]
    mock_model.predict.return_value = [0.3, 0.8, 0.5]

    result = reranker.rerank("query", chunks, top_n=2)

    assert len(result) == 2
    # Highest-scored chunk should be first
    assert result[0].content == "b"


# ---------------------------------------------------------------------------
# 4. top_n=None returns all chunks
# ---------------------------------------------------------------------------

def test_rerank_top_n_none_returns_all() -> None:
    reranker, mock_model = make_reranker_with_mock_model()
    chunks = [fake_chunk(f"chunk-{i}") for i in range(5)]
    mock_model.predict.return_value = [0.5, 0.4, 0.3, 0.2, 0.1]

    result = reranker.rerank("query", chunks, top_n=None)

    assert len(result) == 5


# ---------------------------------------------------------------------------
# 5. Lazy load is skipped when model is already set
# ---------------------------------------------------------------------------

def test_rerank_lazy_load_skipped_if_model_already_set() -> None:
    reranker, mock_model = make_reranker_with_mock_model()
    chunks = [fake_chunk("x"), fake_chunk("y")]
    mock_model.predict.return_value = [0.2, 0.8]

    with patch.object(reranker, "_ensure_loaded", wraps=reranker._ensure_loaded) as spy:
        reranker.rerank("query", chunks)
        spy.assert_called_once()

    # predict was called, meaning the pre-injected mock was used as-is
    mock_model.predict.assert_called_once()
    # The model reference was not replaced
    assert reranker._model is mock_model


# ---------------------------------------------------------------------------
# 6. Lazy load calls CrossEncoder on first use
# ---------------------------------------------------------------------------

def test_rerank_lazy_load_calls_cross_encoder() -> None:
    mock_st = MagicMock()
    mock_ce_class = MagicMock()
    mock_ce_instance = MagicMock()
    mock_ce_class.return_value = mock_ce_instance
    mock_st.CrossEncoder = mock_ce_class

    with patch.dict(sys.modules, {"sentence_transformers": mock_st}):
        reranker = BGEReranker(model_name="BAAI/bge-reranker-v2-m3")
        assert reranker._model is None  # not loaded yet

        reranker._ensure_loaded()

    mock_ce_class.assert_called_once_with("BAAI/bge-reranker-v2-m3")
    assert reranker._model is mock_ce_instance
