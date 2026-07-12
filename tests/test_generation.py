from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from corpus.eval.generation import GENERATOR_DEFAULT_MODEL, Answer, answer_from_context


def _fake_tool_response(tool_name, tool_input, in_tok=11, out_tok=7):
    block = SimpleNamespace(type="tool_use", name=tool_name, input=tool_input)
    usage = SimpleNamespace(input_tokens=in_tok, output_tokens=out_tok)
    return SimpleNamespace(content=[block], usage=usage)


def _mock_client(response):
    return SimpleNamespace(messages=SimpleNamespace(create=MagicMock(return_value=response)))


def test_answer_from_context_parses_tool_output():
    resp = _fake_tool_response("submit_answer", {"answer": "The sky is blue.", "cited_keys": ["k1"]})
    client = _mock_client(resp)
    result = answer_from_context("What color is the sky?", [("k1", "The sky is blue.")], client=client)
    assert isinstance(result, Answer)
    assert result.text == "The sky is blue."
    assert result.cited_keys == ["k1"]
    assert result.model == GENERATOR_DEFAULT_MODEL
    assert result.input_tokens == 11
    assert result.output_tokens == 7


def test_answer_from_context_missing_answer_field_is_empty():
    # Forced tool_choice guarantees submit_answer is called, but without strict
    # mode it does NOT guarantee required fields are populated. A tool call that
    # omits "answer" must yield an empty answer, not crash the run (found live on
    # real data).
    resp = _fake_tool_response("submit_answer", {"cited_keys": ["k1"]})
    client = _mock_client(resp)
    result = answer_from_context("Q?", [("k1", "ctx")], client=client)
    assert result.text == ""
    assert result.cited_keys == ["k1"]


def test_answer_from_context_request_shape():
    resp = _fake_tool_response("submit_answer", {"answer": "x", "cited_keys": []})
    client = _mock_client(resp)
    answer_from_context("Q?", [("k1", "ctx-one"), ("k2", "ctx-two")], client=client)
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["model"] == GENERATOR_DEFAULT_MODEL
    assert kwargs["thinking"] == {"type": "disabled"}
    assert kwargs["tool_choice"] == {"type": "tool", "name": "submit_answer"}
    assert kwargs["tools"][0]["name"] == "submit_answer"
    assert "temperature" not in kwargs  # Sonnet 5 rejects it
    prompt = kwargs["messages"][0]["content"]
    assert "Q?" in prompt and "ctx-one" in prompt and "k1" in prompt


def test_answer_from_context_no_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        answer_from_context("Q?", [("k1", "ctx")])
