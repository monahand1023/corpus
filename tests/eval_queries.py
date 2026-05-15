"""Template eval query set. **REPLACE THESE** with queries that target real
content in your corpus before relying on the recall@K number.

See docs/eval.md for the methodology.

Each EvalQuery has:
  - query: the natural-language question or search terms
  - expected_keys: source_keys that should appear in the top-K result.
    Listing multiple = any one counts (logical OR).
  - source_filter: optional list[str] to restrict the search to specific source types
  - note: free-text annotation (printed during eval; not used in scoring)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvalQuery:
    query: str
    expected_keys: list[str] = field(default_factory=list)
    source_filter: list[str] | None = None
    note: str = ""


# Replace these placeholders with queries that exercise your corpus.
EVAL_QUERIES: list[EvalQuery] = [
    EvalQuery(
        query="placeholder — replace me",
        expected_keys=[],
        note="See docs/eval.md for how to write good eval queries",
    ),
]
