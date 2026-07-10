"""3-axis LLM-as-judge for generated answers.

Axes: faithfulness (every claim supported by the context), answer_relevance
(addresses the query), citation_correctness (cited source_keys actually support
the answer). Forced `record_verdict` tool output with rationale-before-boolean
per axis (structured reasoning). Disabled thinking + no `temperature` (Opus 4.8
rejects it). Context order is shuffled deterministically per query to control
position bias. DB-free: takes `(source_key, text)` tuples, same as generation.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from corpus._anthropic import extract_tool_input, make_client, retry

if TYPE_CHECKING:
    from anthropic import Anthropic
    from anthropic.types import MessageParam, ThinkingConfigParam, ToolChoiceToolParam, ToolParam

JUDGE_DEFAULT_MODEL = "claude-opus-4-8"
_MAX_TOKENS = 1024
_AXES = ("faithfulness", "answer_relevance", "citation_correctness")

_THINKING_OFF: ThinkingConfigParam = {"type": "disabled"}
_RECORD_VERDICT_TOOL: ToolParam = {
    "name": "record_verdict",
    "description": (
        "Record the three-axis judgment of the answer. For each axis, write the "
        "rationale FIRST, then the boolean verdict."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "faithfulness_rationale": {
                "type": "string",
                "description": "Is every claim in the answer supported by the context? Explain.",
            },
            "faithfulness_passed": {"type": "boolean"},
            "answer_relevance_rationale": {
                "type": "string",
                "description": "Does the answer address the question? Explain.",
            },
            "answer_relevance_passed": {"type": "boolean"},
            "citation_correctness_rationale": {
                "type": "string",
                "description": "Do the cited source_keys actually contain the support? Explain.",
            },
            "citation_correctness_passed": {"type": "boolean"},
        },
        "required": [
            "faithfulness_rationale",
            "faithfulness_passed",
            "answer_relevance_rationale",
            "answer_relevance_passed",
            "citation_correctness_rationale",
            "citation_correctness_passed",
        ],
    },
}
_RECORD_VERDICT_CHOICE: ToolChoiceToolParam = {"type": "tool", "name": "record_verdict"}


@dataclass(frozen=True)
class AxisVerdict:
    passed: bool
    rationale: str


@dataclass(frozen=True)
class JudgeVerdict:
    faithfulness: AxisVerdict
    answer_relevance: AxisVerdict
    citation_correctness: AxisVerdict
    model: str


def _seed_from(query: str) -> int:
    """Stable per-query seed (blake2b, NOT builtin hash() which is salted)."""
    return int.from_bytes(hashlib.blake2b(query.encode("utf-8"), digest_size=8).digest(), "big")


def _format_context(context: Sequence[tuple[str, str]]) -> str:
    return "\n\n".join(f"[source_key: {key}]\n{text}" for key, text in context)


def judge_answer(
    query: str,
    answer: str,
    cited_keys: Sequence[str],
    context: Sequence[tuple[str, str]],
    *,
    model: str = JUDGE_DEFAULT_MODEL,
    client: Anthropic | None = None,
    rng: random.Random | None = None,
) -> JudgeVerdict:
    """Judge `answer` to `query` against `context` on three axes.

    Feature-flagged (clean RuntimeError with no key). `rng` defaults to a
    per-query-seeded RNG so context order is shuffled (position-bias control)
    but reproducible.
    """
    client = client or make_client()
    shuffled = list(context)
    (rng or random.Random(_seed_from(query))).shuffle(shuffled)
    prompt = (
        "You are grading an answer produced from retrieved context. Judge it on "
        "three axes; for each, reason first, then give a boolean.\n\n"
        f"Question:\n{query}\n\n"
        f"Answer:\n{answer}\n\n"
        f"Cited source_keys: {list(cited_keys)}\n\n"
        f"Context (order randomized):\n{_format_context(shuffled)}"
    )
    messages: list[MessageParam] = [{"role": "user", "content": prompt}]
    response: Any = retry(
        lambda: client.messages.create(
            model=model,
            max_tokens=_MAX_TOKENS,
            thinking=_THINKING_OFF,
            tools=[_RECORD_VERDICT_TOOL],
            tool_choice=_RECORD_VERDICT_CHOICE,
            messages=messages,
        ),
        f"judge_answer({query[:40]!r})",
    )
    data = extract_tool_input(response, "record_verdict")
    for axis_name in _AXES:
        for suffix in ("_passed", "_rationale"):
            if f"{axis_name}{suffix}" not in data:
                raise RuntimeError(f"judge verdict missing field: {axis_name}{suffix}")

    def axis(name: str) -> AxisVerdict:
        return AxisVerdict(
            passed=bool(data[f"{name}_passed"]), rationale=str(data[f"{name}_rationale"])
        )

    return JudgeVerdict(
        faithfulness=axis("faithfulness"),
        answer_relevance=axis("answer_relevance"),
        citation_correctness=axis("citation_correctness"),
        model=model,
    )


def aggregate_verdicts(verdicts: Sequence[JudgeVerdict]) -> dict[str, float]:
    """Per-axis pass-rate over a set of verdicts, plus `n`."""
    n = len(verdicts)
    if n == 0:
        return {"n": 0.0, "faithfulness": 0.0, "answer_relevance": 0.0, "citation_correctness": 0.0}
    return {
        "n": float(n),
        "faithfulness": sum(1 for v in verdicts if v.faithfulness.passed) / n,
        "answer_relevance": sum(1 for v in verdicts if v.answer_relevance.passed) / n,
        "citation_correctness": sum(1 for v in verdicts if v.citation_correctness.passed) / n,
    }
