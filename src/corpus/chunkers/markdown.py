"""Markdown chunker — section-split + size-bounded with fence-aware boundaries.

Splits at H2/H3 boundaries (ignoring headings inside fenced code blocks),
then size-splits long sections at paragraph breaks. NEVER lands a split
inside a fenced code block.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from corpus.util.tokens import MAX_CHUNK_TOKENS

MAX_CHUNK_CHARS = MAX_CHUNK_TOKENS * 4

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


@dataclass(frozen=True)
class MarkdownDoc:
    frontmatter: dict[str, str]
    body: str


def parse_markdown(text: str) -> MarkdownDoc:
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return MarkdownDoc(frontmatter={}, body=text)
    fm_text = match.group(1)
    body = text[match.end():]
    fm: dict[str, str] = {}
    for line in fm_text.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        fm[key.strip()] = value.strip().strip('"').strip("'")
    return MarkdownDoc(frontmatter=fm, body=body)


def _find_code_fences(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(text):
        start = text.find("```", i)
        if start == -1:
            break
        end = text.find("```", start + 3)
        if end == -1:
            spans.append((start, len(text)))
            break
        end += 3
        spans.append((start, end))
        i = end
    return spans


def _in_any_fence(pos: int, spans: list[tuple[int, int]]) -> bool:
    return any(s <= pos < e for s, e in spans)


def _enclosing_fence_end(pos: int, spans: list[tuple[int, int]]) -> int | None:
    for s, e in spans:
        if s <= pos < e:
            return e
    return None


def _split_into_sections(body: str) -> list[str]:
    fences = _find_code_fences(body)
    matches = [
        m for m in _HEADING_RE.finditer(body)
        if len(m.group(1)) in (2, 3) and not _in_any_fence(m.start(), fences)
    ]
    if not matches:
        return [body]

    sections: list[str] = []
    if matches[0].start() > 0:
        preamble = body[: matches[0].start()].strip()
        if preamble:
            sections.append(preamble)
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        section = body[start:end].strip()
        if section:
            sections.append(section)
    return sections


def _split_long_section(section: str, max_chars: int) -> list[str]:
    if len(section) <= max_chars:
        return [section]
    fences = _find_code_fences(section)
    pieces: list[str] = []
    cursor = 0
    while cursor < len(section):
        target = min(cursor + max_chars, len(section))
        if target == len(section):
            pieces.append(section[cursor:target].strip())
            break

        fence_end = _enclosing_fence_end(target, fences)
        if fence_end is not None:
            target = fence_end

        boundary = -1
        search_end = target
        while True:
            candidate = section.rfind("\n\n", cursor, search_end)
            if candidate == -1:
                break
            if not _in_any_fence(candidate, fences):
                boundary = candidate
                break
            search_end = candidate

        if boundary == -1 or boundary <= cursor:
            piece = section[cursor:target].strip()
            if piece:
                pieces.append(piece)
            # Hard guard: ensure forward progress even on pathological input
            # (single-line content with no paragraph breaks AND no fence we
            # could snap to). Without max(...), `cursor = target` could equal
            # `cursor` and loop forever emitting empty pieces.
            cursor = max(target, cursor + 1)
        else:
            piece = section[cursor:boundary].strip()
            if piece:
                pieces.append(piece)
            cursor = boundary + 2

    return pieces


def chunk_markdown_body(body: str) -> list[str]:
    sections = _split_into_sections(body)
    chunks: list[str] = []
    for section in sections:
        chunks.extend(_split_long_section(section, MAX_CHUNK_CHARS))
    return [c for c in chunks if c.strip()]
