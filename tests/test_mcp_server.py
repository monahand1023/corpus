"""Unit tests for corpus.mcp_server tool handlers.

All seven async tool handlers are tested in isolation by patching
``corpus.mcp_server._init`` to return a tuple of MagicMock objects so no
real DB, embedder, or config is touched.

Because pytest-asyncio is not installed the tests drive async functions with
``asyncio.run()``.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import corpus.mcp_server as _mod
from corpus.mcp_server import (
    corpus_stats,
    expand_context,
    get_doc,
    get_summary,
    recent_activity,
    search_knowledge,
    timeline,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stored_chunk(
    source_type: str = "notes",
    source_key: str = "doc-1",
    content: str = "chunk content",
    title: str | None = "My Title",
    url: str | None = None,
    distance: float | None = None,
    metadata: dict | None = None,
) -> SimpleNamespace:
    """Return a minimal fake StoredChunk compatible with _format_chunk_block."""
    return SimpleNamespace(
        source_type=source_type,
        source_key=source_key,
        content=content,
        title=title,
        url=url,
        distance=distance,
        metadata=metadata or {},
    )


def _make_mocks() -> tuple[MagicMock, MagicMock, MagicMock, MagicMock]:
    """Return (store, embedder, retriever, config) mocks."""
    store = MagicMock()
    embedder = MagicMock()
    retriever = MagicMock()
    config = MagicMock()
    return store, embedder, retriever, config


def _patch_init(store, embedder, retriever, config):
    """Context manager: patch _init and reset module singletons."""
    return patch.object(_mod, "_init", return_value=(store, embedder, retriever, config))


def _reset_singletons() -> None:
    """Reset module-level singletons so tests don't bleed state."""
    _mod._store = None
    _mod._embedder = None
    _mod._retriever = None
    _mod._config = None


# Autouse fixture — reset singletons before every test.
@pytest.fixture(autouse=True)
def reset_singletons():
    _reset_singletons()
    yield
    _reset_singletons()


# ---------------------------------------------------------------------------
# search_knowledge
# ---------------------------------------------------------------------------

class TestSearchKnowledge:
    def test_happy_path_returns_formatted_string(self):
        store, embedder, retriever, config = _make_mocks()
        chunk = _make_stored_chunk(content="hello world", title="Doc A", distance=0.123)
        result_obj = MagicMock()
        result_obj.chunks = [chunk]
        retriever.query.return_value = result_obj

        with _patch_init(store, embedder, retriever, config):
            out = asyncio.run(search_knowledge(query="hello"))

        assert "[1] notes:doc-1 d=0.1230" in out
        assert "Title: Doc A" in out
        assert "hello world" in out
        retriever.query.assert_called_once_with("hello", 5, None)

    def test_empty_results_message(self):
        store, embedder, retriever, config = _make_mocks()
        result_obj = MagicMock()
        result_obj.chunks = []
        retriever.query.return_value = result_obj

        with _patch_init(store, embedder, retriever, config):
            out = asyncio.run(search_knowledge(query="nothing here"))

        assert out == "No results for: nothing here"

    def test_source_types_string_coerced_to_list(self):
        store, embedder, retriever, config = _make_mocks()
        result_obj = MagicMock()
        result_obj.chunks = []
        retriever.query.return_value = result_obj

        with _patch_init(store, embedder, retriever, config):
            asyncio.run(search_knowledge(query="q", source_types="notes"))

        # retriever.query should have been called with a list, not a string
        _, _, called_filter = retriever.query.call_args.args
        assert called_filter == ["notes"]

    def test_source_types_list_passes_through(self):
        store, embedder, retriever, config = _make_mocks()
        result_obj = MagicMock()
        result_obj.chunks = []
        retriever.query.return_value = result_obj

        with _patch_init(store, embedder, retriever, config):
            asyncio.run(search_knowledge(query="q", source_types=["notes", "papers"]))

        _, _, called_filter = retriever.query.call_args.args
        assert called_filter == ["notes", "papers"]

    def test_source_types_none_passes_none(self):
        store, embedder, retriever, config = _make_mocks()
        result_obj = MagicMock()
        result_obj.chunks = []
        retriever.query.return_value = result_obj

        with _patch_init(store, embedder, retriever, config):
            asyncio.run(search_knowledge(query="q", source_types=None))

        _, _, called_filter = retriever.query.call_args.args
        assert called_filter is None

    def test_no_distance_omits_distance_str(self):
        store, embedder, retriever, config = _make_mocks()
        chunk = _make_stored_chunk(distance=None, title=None)
        result_obj = MagicMock()
        result_obj.chunks = [chunk]
        retriever.query.return_value = result_obj

        with _patch_init(store, embedder, retriever, config):
            out = asyncio.run(search_knowledge(query="q"))

        assert " d=" not in out
        assert "Title:" not in out

    def test_url_included_in_output(self):
        store, embedder, retriever, config = _make_mocks()
        chunk = _make_stored_chunk(url="https://example.com/doc")
        result_obj = MagicMock()
        result_obj.chunks = [chunk]
        retriever.query.return_value = result_obj

        with _patch_init(store, embedder, retriever, config):
            out = asyncio.run(search_knowledge(query="q"))

        assert "URL: https://example.com/doc" in out

    def test_multiple_chunks_separated_by_divider(self):
        store, embedder, retriever, config = _make_mocks()
        chunks = [_make_stored_chunk(content=f"content {i}") for i in range(3)]
        result_obj = MagicMock()
        result_obj.chunks = chunks
        retriever.query.return_value = result_obj

        with _patch_init(store, embedder, retriever, config):
            out = asyncio.run(search_knowledge(query="q"))

        assert out.count("\n\n---\n\n") == 2


# ---------------------------------------------------------------------------
# get_doc
# ---------------------------------------------------------------------------

class TestGetDoc:
    def test_happy_path(self):
        store, embedder, retriever, config = _make_mocks()
        chunks = [_make_stored_chunk(content="first"), _make_stored_chunk(content="second")]
        store.get_by_source_key.return_value = chunks

        with _patch_init(store, embedder, retriever, config):
            out = asyncio.run(get_doc(source_type="notes", source_key="doc-1"))

        assert "[1]" in out
        assert "[2]" in out
        assert "first" in out
        assert "second" in out
        store.get_by_source_key.assert_called_once_with("notes", "doc-1")

    def test_no_chunks_returns_message(self):
        store, embedder, retriever, config = _make_mocks()
        store.get_by_source_key.return_value = []

        with _patch_init(store, embedder, retriever, config):
            out = asyncio.run(get_doc(source_type="notes", source_key="missing"))

        assert out == "No chunks found for notes:missing"


# ---------------------------------------------------------------------------
# expand_context
# ---------------------------------------------------------------------------

class TestExpandContext:
    def test_include_none_defaults_to_all_three_types(self):
        store, embedder, retriever, config = _make_mocks()
        chunk = _make_stored_chunk()
        retriever.expand_context.return_value = [chunk]

        with _patch_init(store, embedder, retriever, config):
            asyncio.run(expand_context(chunk_id="abc123", include=None))

        _, called_include, _ = retriever.expand_context.call_args.args
        assert called_include == ["siblings", "references", "parent"]

    def test_include_string_coerced_to_list(self):
        store, embedder, retriever, config = _make_mocks()
        retriever.expand_context.return_value = []

        with _patch_init(store, embedder, retriever, config):
            asyncio.run(expand_context(chunk_id="abc123", include="siblings"))

        _, called_include, _ = retriever.expand_context.call_args.args
        assert called_include == ["siblings"]

    def test_include_list_passes_through(self):
        store, embedder, retriever, config = _make_mocks()
        retriever.expand_context.return_value = []

        with _patch_init(store, embedder, retriever, config):
            asyncio.run(expand_context(chunk_id="abc123", include=["siblings", "references"]))

        _, called_include, _ = retriever.expand_context.call_args.args
        assert called_include == ["siblings", "references"]

    def test_no_chunks_returns_message(self):
        store, embedder, retriever, config = _make_mocks()
        retriever.expand_context.return_value = []

        with _patch_init(store, embedder, retriever, config):
            out = asyncio.run(expand_context(chunk_id="xyz999"))

        assert out == "No related chunks found for xyz999."

    def test_happy_path_formats_chunks(self):
        store, embedder, retriever, config = _make_mocks()
        chunk = _make_stored_chunk(content="sibling content", title="Sib Doc")
        retriever.expand_context.return_value = [chunk]

        with _patch_init(store, embedder, retriever, config):
            out = asyncio.run(expand_context(chunk_id="abc123"))

        assert "[1]" in out
        assert "sibling content" in out

    def test_max_results_passed_to_retriever(self):
        store, embedder, retriever, config = _make_mocks()
        retriever.expand_context.return_value = []

        with _patch_init(store, embedder, retriever, config):
            asyncio.run(expand_context(chunk_id="abc123", max_results=20))

        _, _, called_max = retriever.expand_context.call_args.args
        assert called_max == 20


# ---------------------------------------------------------------------------
# timeline
# ---------------------------------------------------------------------------

class TestTimeline:
    def test_no_chunks_returns_message(self):
        store, embedder, retriever, config = _make_mocks()
        retriever.timeline.return_value = []

        with _patch_init(store, embedder, retriever, config):
            out = asyncio.run(timeline(topic="payments"))

        assert out == "No timeline events for: payments"

    def test_happy_path_with_metadata(self):
        store, embedder, retriever, config = _make_mocks()
        chunk = _make_stored_chunk(
            source_type="notes",
            source_key="note-1",
            content="important event happened here",
            title="Note Title",
            metadata={"updated_at": "2024-03-15T10:00:00Z"},
        )
        retriever.timeline.return_value = [chunk]

        with _patch_init(store, embedder, retriever, config):
            out = asyncio.run(timeline(topic="payments"))

        assert "[1]" in out
        assert "2024-03-15" in out
        assert "notes:note-1" in out
        assert "Note Title" in out
        assert "important event happened here" in out

    def test_falls_back_to_created_at_when_no_updated_at(self):
        store, embedder, retriever, config = _make_mocks()
        chunk = _make_stored_chunk(
            metadata={"created_at": "2023-01-01T00:00:00Z"},
        )
        retriever.timeline.return_value = [chunk]

        with _patch_init(store, embedder, retriever, config):
            out = asyncio.run(timeline(topic="topic"))

        assert "2023-01-01" in out

    def test_missing_timestamp_shows_question_mark(self):
        store, embedder, retriever, config = _make_mocks()
        chunk = _make_stored_chunk(metadata={})
        retriever.timeline.return_value = [chunk]

        with _patch_init(store, embedder, retriever, config):
            out = asyncio.run(timeline(topic="topic"))

        assert "[1] ?" in out

    def test_content_truncated_at_600_chars(self):
        store, embedder, retriever, config = _make_mocks()
        long_content = "x" * 1000
        chunk = _make_stored_chunk(content=long_content, metadata={"updated_at": "2024-01-01"})
        retriever.timeline.return_value = [chunk]

        with _patch_init(store, embedder, retriever, config):
            out = asyncio.run(timeline(topic="topic"))

        # The output should contain at most 600 x's (plus surrounding text)
        assert "x" * 601 not in out
        assert "x" * 600 in out

    def test_passes_since_until_to_retriever(self):
        store, embedder, retriever, config = _make_mocks()
        retriever.timeline.return_value = []

        with _patch_init(store, embedder, retriever, config):
            asyncio.run(timeline(topic="topic", top_k=10, since="2024-01-01", until="2024-12-31"))

        retriever.timeline.assert_called_once_with("topic", 10, "2024-01-01", "2024-12-31", None)

    def test_multiple_chunks_separated_by_divider(self):
        store, embedder, retriever, config = _make_mocks()
        chunks = [
            _make_stored_chunk(metadata={"updated_at": f"2024-0{i+1}-01"})
            for i in range(3)
        ]
        retriever.timeline.return_value = chunks

        with _patch_init(store, embedder, retriever, config):
            out = asyncio.run(timeline(topic="topic"))

        assert out.count("\n\n---\n\n") == 2


# ---------------------------------------------------------------------------
# recent_activity
# ---------------------------------------------------------------------------

class TestRecentActivity:
    def test_happy_path(self):
        store, embedder, retriever, config = _make_mocks()
        chunk = _make_stored_chunk(content="recent work")
        retriever.recent_activity.return_value = [chunk]

        with _patch_init(store, embedder, retriever, config):
            out = asyncio.run(recent_activity(days=7))

        assert "[1]" in out
        assert "recent work" in out
        retriever.recent_activity.assert_called_once_with(7, None, 15)

    def test_empty_returns_message(self):
        store, embedder, retriever, config = _make_mocks()
        retriever.recent_activity.return_value = []

        with _patch_init(store, embedder, retriever, config):
            out = asyncio.run(recent_activity(days=30))

        assert out == "No activity in the last 30 days"

    def test_source_types_string_coerced(self):
        store, embedder, retriever, config = _make_mocks()
        retriever.recent_activity.return_value = []

        with _patch_init(store, embedder, retriever, config):
            asyncio.run(recent_activity(days=7, source_types="papers"))

        _, called_filter, _ = retriever.recent_activity.call_args.args
        assert called_filter == ["papers"]

    def test_source_types_list_passes_through(self):
        store, embedder, retriever, config = _make_mocks()
        retriever.recent_activity.return_value = []

        with _patch_init(store, embedder, retriever, config):
            asyncio.run(recent_activity(days=7, source_types=["notes", "papers"]))

        _, called_filter, _ = retriever.recent_activity.call_args.args
        assert called_filter == ["notes", "papers"]

    def test_source_types_none_passes_none(self):
        store, embedder, retriever, config = _make_mocks()
        retriever.recent_activity.return_value = []

        with _patch_init(store, embedder, retriever, config):
            asyncio.run(recent_activity(days=7, source_types=None))

        _, called_filter, _ = retriever.recent_activity.call_args.args
        assert called_filter is None

    def test_top_k_forwarded(self):
        store, embedder, retriever, config = _make_mocks()
        retriever.recent_activity.return_value = []

        with _patch_init(store, embedder, retriever, config):
            asyncio.run(recent_activity(days=14, top_k=25))

        retriever.recent_activity.assert_called_once_with(14, None, 25)


# ---------------------------------------------------------------------------
# get_summary
# ---------------------------------------------------------------------------

class TestGetSummary:
    def test_summary_found_returns_formatted_string(self):
        store, embedder, retriever, config = _make_mocks()
        store.get_summary.return_value = {
            "model": "claude-3-haiku-20240307",
            "generated_at": "2024-06-01T12:00:00Z",
            "summary": "This document covers the payment processing flow.",
        }

        with _patch_init(store, embedder, retriever, config):
            out = asyncio.run(get_summary(source_type="notes", source_key="pay-flow"))

        assert "notes:pay-flow" in out
        assert "model=claude-3-haiku-20240307" in out
        assert "generated=2024-06-01T12:00:00Z" in out
        assert "This document covers the payment processing flow." in out
        store.get_summary.assert_called_once_with("notes", "pay-flow")

    def test_no_summary_returns_instructions_message(self):
        store, embedder, retriever, config = _make_mocks()
        store.get_summary.return_value = None

        with _patch_init(store, embedder, retriever, config):
            out = asyncio.run(get_summary(source_type="notes", source_key="missing-doc"))

        assert "No summary cached for notes:missing-doc" in out
        assert "corpus-summarize --source notes" in out

    def test_no_summary_falsy_empty_dict(self):
        """get_summary treats any falsy return (empty dict, None) as no summary."""
        store, embedder, retriever, config = _make_mocks()
        store.get_summary.return_value = {}

        with _patch_init(store, embedder, retriever, config):
            out = asyncio.run(get_summary(source_type="notes", source_key="doc"))

        assert "No summary cached for" in out


# ---------------------------------------------------------------------------
# corpus_stats
# ---------------------------------------------------------------------------

class TestCorpusStats:
    def test_formats_total_and_per_source(self):
        store, embedder, retriever, config = _make_mocks()
        store.stats.return_value = {
            "total": 1234,
            "by_source": {"notes": 800, "papers": 434},
        }

        with _patch_init(store, embedder, retriever, config):
            out = asyncio.run(corpus_stats())

        assert "Total chunks: 1,234" in out
        assert "notes: 800" in out
        assert "papers: 434" in out

    def test_per_source_sorted_alphabetically(self):
        store, embedder, retriever, config = _make_mocks()
        store.stats.return_value = {
            "total": 3,
            "by_source": {"zebra": 1, "alpha": 2},
        }

        with _patch_init(store, embedder, retriever, config):
            out = asyncio.run(corpus_stats())

        lines = out.splitlines()
        # alpha should appear before zebra
        alpha_idx = next(i for i, line in enumerate(lines) if "alpha" in line)
        zebra_idx = next(i for i, line in enumerate(lines) if "zebra" in line)
        assert alpha_idx < zebra_idx

    def test_empty_corpus(self):
        store, embedder, retriever, config = _make_mocks()
        store.stats.return_value = {"total": 0, "by_source": {}}

        with _patch_init(store, embedder, retriever, config):
            out = asyncio.run(corpus_stats())

        assert "Total chunks: 0" in out
        assert out.strip() == "Total chunks: 0"

    def test_large_numbers_formatted_with_commas(self):
        store, embedder, retriever, config = _make_mocks()
        store.stats.return_value = {
            "total": 1_000_000,
            "by_source": {"notes": 1_000_000},
        }

        with _patch_init(store, embedder, retriever, config):
            out = asyncio.run(corpus_stats())

        assert "1,000,000" in out
