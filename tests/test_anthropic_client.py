from __future__ import annotations

from types import SimpleNamespace

import pytest

import corpus._anthropic as mod
from corpus._anthropic import extract_tool_input, make_client, retry


def test_make_client_no_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        make_client()


def test_make_client_with_key_returns_client(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = make_client(api_key="test-key")
    assert hasattr(client, "messages")


def test_retry_recovers_after_transient_failure(monkeypatch):
    monkeypatch.setattr(mod.time, "sleep", lambda *_: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("503 overloaded")
        return "ok"

    assert retry(flaky, "flaky") == "ok"
    assert calls["n"] == 3


def test_retry_reraises_after_exhausting_attempts(monkeypatch):
    monkeypatch.setattr(mod.time, "sleep", lambda *_: None)
    calls = {"n": 0}

    def always_fail():
        calls["n"] += 1
        raise RuntimeError("429 always")

    with pytest.raises(RuntimeError, match="429"):
        retry(always_fail, "always")
    assert calls["n"] == mod.RETRY_ATTEMPTS


def test_extract_tool_input_returns_input():
    block = SimpleNamespace(type="tool_use", name="record_verdict", input={"x": 1})
    resp = SimpleNamespace(content=[block])
    assert extract_tool_input(resp, "record_verdict") == {"x": 1}


def test_extract_tool_input_missing_tool_raises():
    text = SimpleNamespace(type="text", text="hi")
    resp = SimpleNamespace(content=[text])
    with pytest.raises(RuntimeError, match="record_verdict"):
        extract_tool_input(resp, "record_verdict")
