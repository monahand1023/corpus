"""Unit tests for AnthropicSummarizer and doc_hash."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# A stable mock for the `anthropic` module — used by patch.dict so that the
# lazy `from anthropic import Anthropic` inside __init__ never hits the real SDK.
_mock_anthropic_module = MagicMock()
_mock_anthropic_module.Anthropic.return_value = MagicMock()


def fake_response(
    text: str,
    input_tokens: int = 10,
    output_tokens: int = 5,
    cached: int | None = 0,
) -> SimpleNamespace:
    """Build a minimal object that looks like an Anthropic Messages response."""
    block = SimpleNamespace(text=text)
    usage = SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cached,
    )
    return SimpleNamespace(content=[block], usage=usage)


@pytest.fixture()
def summarizer(monkeypatch):
    """Return an AnthropicSummarizer with a fresh mock _client."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch.dict(sys.modules, {"anthropic": _mock_anthropic_module}):
        # Import inside the patch so __init__'s `from anthropic import Anthropic` succeeds.
        from corpus.summarizer.anthropic_summarizer import AnthropicSummarizer

        s = AnthropicSummarizer()
        # Replace with a brand-new mock so tests don't share state.
        s._client = MagicMock()
        return s


# ---------------------------------------------------------------------------
# 1. Missing API key raises
# ---------------------------------------------------------------------------


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with patch.dict(sys.modules, {"anthropic": _mock_anthropic_module}):
        from corpus.summarizer.anthropic_summarizer import AnthropicSummarizer

        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            AnthropicSummarizer(api_key=None)


# ---------------------------------------------------------------------------
# 2. summarize returns summary text
# ---------------------------------------------------------------------------


def test_summarize_returns_summary_text(summarizer):
    summarizer._client.messages.create.return_value = fake_response("The summary text")
    result = summarizer.summarize("notes", "My Doc", "Some content")
    assert result.summary == "The summary text"


# ---------------------------------------------------------------------------
# 3. summarize truncates long content
# ---------------------------------------------------------------------------


def test_summarize_truncates_long_content(summarizer):
    from corpus.summarizer.anthropic_summarizer import MAX_INPUT_CHARS

    # Use distinct characters so we can detect the boundary unambiguously.
    long_content = "A" * MAX_INPUT_CHARS + "B" * 1000
    summarizer._client.messages.create.return_value = fake_response("ok")
    summarizer.summarize("notes", "Title", long_content)

    call_kwargs = summarizer._client.messages.create.call_args
    user_msg = call_kwargs.kwargs["messages"][0]["content"]
    # The truncated portion (all A's) must appear in the user message.
    assert "A" * MAX_INPUT_CHARS in user_msg
    # The overflow characters (B's) must NOT appear — they were cut off.
    assert "B" not in user_msg


# ---------------------------------------------------------------------------
# 4. Token counts are populated
# ---------------------------------------------------------------------------


def test_summarize_token_counts_populated(summarizer):
    summarizer._client.messages.create.return_value = fake_response(
        "summary", input_tokens=100, output_tokens=50, cached=25
    )
    result = summarizer.summarize("pr", "PR #42", "diff content")
    assert result.input_tokens == 100
    assert result.output_tokens == 50
    assert result.cached_input_tokens == 25


# ---------------------------------------------------------------------------
# 5. cache_read_input_tokens=None becomes 0
# ---------------------------------------------------------------------------


def test_summarize_cache_read_tokens_none_becomes_zero(summarizer):
    summarizer._client.messages.create.return_value = fake_response(
        "summary", cached=None
    )
    result = summarizer.summarize("pr", "PR #1", "content")
    assert result.cached_input_tokens == 0


# ---------------------------------------------------------------------------
# 6. source_type appears in the guidance system block
# ---------------------------------------------------------------------------


def test_summarize_source_type_in_guidance(summarizer):
    summarizer._client.messages.create.return_value = fake_response("ok")
    summarizer.summarize("tickets", "TICKET-123", "description text")

    call_kwargs = summarizer._client.messages.create.call_args
    system_blocks = call_kwargs.kwargs["system"]
    # Second block is the per-source-type guidance
    guidance_text = system_blocks[1]["text"]
    assert "tickets" in guidance_text


# ---------------------------------------------------------------------------
# 7. Custom model is forwarded to messages.create
# ---------------------------------------------------------------------------


def test_summarize_model_passed_to_create(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch.dict(sys.modules, {"anthropic": _mock_anthropic_module}):
        from corpus.summarizer.anthropic_summarizer import AnthropicSummarizer

        s = AnthropicSummarizer(model="custom-model")
        s._client = MagicMock()
        s._client.messages.create.return_value = fake_response("ok")
        s.summarize("notes", "title", "content")

    s._client.messages.create.assert_called_once()
    assert s._client.messages.create.call_args.kwargs["model"] == "custom-model"


# ---------------------------------------------------------------------------
# 8. doc_hash is stable and 32 hex chars
# ---------------------------------------------------------------------------


def test_doc_hash_stable_and_truncated():
    from corpus.summarizer.anthropic_summarizer import doc_hash

    h = doc_hash("hello")
    assert h == doc_hash("hello")
    assert len(h) == 32
    assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# 9. doc_hash differs for different inputs
# ---------------------------------------------------------------------------


def test_doc_hash_differs_for_different_input():
    from corpus.summarizer.anthropic_summarizer import doc_hash

    assert doc_hash("a") != doc_hash("b")


# ---------------------------------------------------------------------------
# 10. Non-text blocks in response.content are skipped
# ---------------------------------------------------------------------------


def test_summarize_retries_transient_failure(summarizer, monkeypatch):
    """A transient error on messages.create is retried, not fatal."""
    monkeypatch.setattr("corpus._anthropic.time.sleep", lambda *_: None)  # no real backoff wait
    summarizer._client.messages.create.side_effect = [
        RuntimeError("503 overloaded"),
        fake_response("recovered summary"),
    ]
    result = summarizer.summarize("notes", "doc", "body")
    assert result.summary == "recovered summary"
    assert summarizer._client.messages.create.call_count == 2


def test_summarize_raises_after_exhausting_retries(summarizer, monkeypatch):
    """Persistent failures eventually surface (after RETRY_ATTEMPTS tries)."""
    import corpus.summarizer.anthropic_summarizer as mod

    monkeypatch.setattr("corpus._anthropic.time.sleep", lambda *_: None)
    summarizer._client.messages.create.side_effect = RuntimeError("429 always")
    with pytest.raises(RuntimeError, match="429"):
        summarizer.summarize("notes", "doc", "body")
    assert summarizer._client.messages.create.call_count == mod.RETRY_ATTEMPTS


def test_summarize_response_with_non_text_blocks_skipped(summarizer):
    no_text_block = SimpleNamespace()  # no `.text` attribute
    text_block = SimpleNamespace(text="real summary")
    usage = SimpleNamespace(
        input_tokens=5,
        output_tokens=3,
        cache_read_input_tokens=0,
    )
    response = SimpleNamespace(content=[no_text_block, text_block], usage=usage)
    summarizer._client.messages.create.return_value = response

    result = summarizer.summarize("notes", "doc", "body")
    assert result.summary == "real summary"
