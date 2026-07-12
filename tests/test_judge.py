from __future__ import annotations

import random
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from corpus.eval.judge import (
    JUDGE_DEFAULT_MODEL,
    AxisVerdict,
    JudgeVerdict,
    aggregate_verdicts,
    judge_answer,
)

_VERDICT_INPUT = {
    "faithfulness_rationale": "every claim is in the context",
    "faithfulness_passed": True,
    "answer_relevance_rationale": "addresses the question",
    "answer_relevance_passed": True,
    "citation_correctness_rationale": "cited keys support the claims",
    "citation_correctness_passed": False,
}


def _fake_verdict_response(tool_input):
    block = SimpleNamespace(type="tool_use", name="record_verdict", input=tool_input)
    return SimpleNamespace(content=[block], usage=SimpleNamespace(input_tokens=9, output_tokens=4))


def _mock_client(response):
    return SimpleNamespace(messages=SimpleNamespace(create=MagicMock(return_value=response)))


def test_judge_answer_missing_verdict_field_defaults():
    # Forced tool_choice guarantees record_verdict is called but not that every
    # required field is populated; a partial verdict must default (missing -> a
    # conservative fail), not crash the run (found live on real data).
    partial = dict(_VERDICT_INPUT)
    del partial["citation_correctness_passed"]
    del partial["faithfulness_rationale"]
    client = _mock_client(_fake_verdict_response(partial))
    verdict = judge_answer("Q?", "A.", ["k1"], [("k1", "ctx")], client=client)
    assert verdict.citation_correctness.passed is False  # missing -> default fail
    assert verdict.faithfulness.rationale == ""  # missing -> default empty
    assert verdict.answer_relevance.passed is True  # present field unaffected


def test_judge_answer_parses_three_axes():
    client = _mock_client(_fake_verdict_response(_VERDICT_INPUT))
    verdict = judge_answer("Q?", "A.", ["k1"], [("k1", "ctx")], client=client)
    assert isinstance(verdict, JudgeVerdict)
    assert verdict.faithfulness == AxisVerdict(passed=True, rationale="every claim is in the context")
    assert verdict.answer_relevance.passed is True
    assert verdict.citation_correctness.passed is False
    assert verdict.model == JUDGE_DEFAULT_MODEL


def test_judge_answer_request_shape_and_no_temperature():
    client = _mock_client(_fake_verdict_response(_VERDICT_INPUT))
    judge_answer("Q?", "A.", ["k1"], [("k1", "ctx-a"), ("k2", "ctx-b")], client=client)
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["model"] == JUDGE_DEFAULT_MODEL
    assert kwargs["thinking"] == {"type": "disabled"}
    assert kwargs["tool_choice"] == {"type": "tool", "name": "record_verdict"}
    assert kwargs["tools"][0]["name"] == "record_verdict"
    assert "temperature" not in kwargs


def test_judge_answer_shuffles_context_deterministically():
    client = _mock_client(_fake_verdict_response(_VERDICT_INPUT))
    ctx = [("k1", "aaa"), ("k2", "bbb"), ("k3", "ccc")]
    judge_answer("Q?", "A.", [], ctx, client=client, rng=random.Random(0))
    prompt = client.messages.create.call_args.kwargs["messages"][0]["content"]
    expected = list(ctx)
    random.Random(0).shuffle(expected)
    positions = [prompt.index(text) for _, text in expected]
    assert positions == sorted(positions)  # presented in the shuffled order


def test_judge_answer_no_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        judge_answer("Q?", "A.", [], [("k1", "ctx")])


def test_aggregate_verdicts_rates():
    def v(f, r, c):
        return JudgeVerdict(
            AxisVerdict(f, ""), AxisVerdict(r, ""), AxisVerdict(c, ""), "m"
        )

    verdicts = [v(True, True, False), v(True, False, False)]
    agg = aggregate_verdicts(verdicts)
    assert agg["n"] == 2
    assert agg["faithfulness"] == 1.0
    assert agg["answer_relevance"] == 0.5
    assert agg["citation_correctness"] == 0.0


def test_aggregate_verdicts_empty():
    agg = aggregate_verdicts([])
    assert agg["n"] == 0
    assert agg["faithfulness"] == 0.0
