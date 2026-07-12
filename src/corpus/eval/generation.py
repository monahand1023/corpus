"""Feature-flagged answer generation from retrieved context.

Kept OUT of the retriever: `answer_from_context` takes `(source_key, text)`
tuples, so this module never imports the DB or retriever. Uses forced tool
output (`submit_answer`) so the cited source_keys come back structured rather
than parsed out of prose. No `temperature` (Sonnet 5 rejects it); determinism
comes from disabled thinking + the forced tool.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .._anthropic import extract_tool_input, make_client, retry

if TYPE_CHECKING:
    from anthropic import Anthropic
    from anthropic.types import MessageParam, ThinkingConfigParam, ToolChoiceToolParam, ToolParam

GENERATOR_DEFAULT_MODEL = "claude-sonnet-5"
_MAX_TOKENS = 1024

_THINKING_OFF: ThinkingConfigParam = {"type": "disabled"}
_SUBMIT_ANSWER_TOOL: ToolParam = {
    "name": "submit_answer",
    "description": (
        "Return the answer to the user's question, grounded ONLY in the provided "
        "context, together with the source_keys you actually used."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "The answer, using only the provided context.",
            },
            "cited_keys": {
                "type": "array",
                "items": {"type": "string"},
                "description": "source_keys from the context that support the answer.",
            },
        },
        "required": ["answer", "cited_keys"],
        "additionalProperties": False,
    },
    "strict": True,
}
_SUBMIT_ANSWER_CHOICE: ToolChoiceToolParam = {"type": "tool", "name": "submit_answer"}


@dataclass(frozen=True)
class Answer:
    text: str
    cited_keys: list[str]
    model: str
    input_tokens: int
    output_tokens: int


def _format_context(context: Sequence[tuple[str, str]]) -> str:
    return "\n\n".join(f"[source_key: {key}]\n{text}" for key, text in context)


def answer_from_context(
    query: str,
    context: Sequence[tuple[str, str]],
    *,
    model: str = GENERATOR_DEFAULT_MODEL,
    client: Anthropic | None = None,
) -> Answer:
    """Answer `query` using only `context` (list of (source_key, text)).

    Feature-flagged: with no `client` and no `ANTHROPIC_API_KEY`, `make_client`
    raises a clean RuntimeError.
    """
    client = client or make_client()
    prompt = (
        "Answer the question using ONLY the provided context. If the context does "
        "not contain the answer, say so plainly. Cite the source_keys you used.\n\n"
        f"Question: {query}\n\n"
        f"Context:\n{_format_context(context)}"
    )
    messages: list[MessageParam] = [{"role": "user", "content": prompt}]
    response: Any = retry(
        lambda: client.messages.create(
            model=model,
            max_tokens=_MAX_TOKENS,
            thinking=_THINKING_OFF,
            tools=[_SUBMIT_ANSWER_TOOL],
            tool_choice=_SUBMIT_ANSWER_CHOICE,
            messages=messages,
        ),
        f"answer_from_context({query[:40]!r})",
    )
    data = extract_tool_input(response, "submit_answer")
    return Answer(
        text=str(data.get("answer", "")),
        cited_keys=[str(k) for k in data.get("cited_keys", [])],
        model=model,
        input_tokens=getattr(response.usage, "input_tokens", 0),
        output_tokens=getattr(response.usage, "output_tokens", 0),
    )
