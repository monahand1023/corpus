"""Every relative link between markdown files must point at something real.

A cross-document link is a claim: "the thing I just described is explained
over there." Nothing in the repo checked that claim, so a renamed heading or
a moved file left behind a link that looks authoritative and goes nowhere --
and a reader who follows it lands on GitHub's 404, or worse, silently at the
top of the right file having missed the section. Two such links were sitting
in the repo when this test was written.

Anchors are the fragile half. A heading is prose; it gets reworded. The
anchor derived from it changes with it, and every link to the old spelling
breaks without a single file moving.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# [text](target) and [text](target "title"), skipping images (![...]).
_LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+?)(?:\s+\"[^\"]*\")?\)")
_HEADING = re.compile(r"\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
_HTML_ANCHOR = re.compile(r'<a\s+[^>]*(?:name|id)="([^"]+)"')

_SKIP_SCHEMES = ("http://", "https://", "mailto:", "tel:", "#!")


def _slug(heading: str) -> str:
    """GitHub's heading -> anchor rule.

    Lowercase, drop inline HTML, drop every character that is not
    alphanumeric / space / hyphen / underscore (this is what removes
    backticks, parentheses, periods and em dashes), then spaces to hyphens.
    An em dash surrounded by spaces therefore leaves a DOUBLE hyphen, which
    is a real and easy-to-get-wrong case in this repo's headings.
    """
    text = re.sub(r"<[^>]+>", "", heading.strip().lower())
    kept = "".join(c for c in text if c.isalnum() or c in " -_")
    return kept.replace(" ", "-")


def _anchors(path: Path) -> set[str]:
    found: set[str] = set()
    in_fence = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            # A '#' inside a code block is a comment or a shell prompt, not a
            # heading. Counting those would invent anchors that do not exist.
            continue
        heading = _HEADING.match(line)
        if heading:
            found.add(_slug(heading.group(2)))
        found.update(_HTML_ANCHOR.findall(line))
    return found


def _markdown_files() -> list[Path]:
    return sorted(
        p
        for p in REPO_ROOT.rglob("*.md")
        if not {".venv", ".git", "node_modules", "__pycache__"} & set(p.parts)
    )


def _broken_links(files: list[Path]) -> list[str]:
    anchor_cache: dict[Path, set[str]] = {}
    broken: list[str] = []
    for path in files:
        rel = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for target in _LINK.findall(line):
                if target.startswith(_SKIP_SCHEMES):
                    continue
                if target.startswith("#"):
                    dest, fragment = path, target[1:]
                else:
                    file_part, _, fragment = target.partition("#")
                    dest = (path.parent / file_part).resolve()
                    if not dest.exists():
                        broken.append(f"{rel}:{line_no}: no such file: {target}")
                        continue
                    if dest.suffix != ".md" or not fragment:
                        continue
                if fragment:
                    if dest not in anchor_cache:
                        anchor_cache[dest] = _anchors(dest)
                    if fragment.lower() not in anchor_cache[dest]:
                        broken.append(f"{rel}:{line_no}: no such anchor: {target}")
    return broken


def test_every_relative_markdown_link_resolves() -> None:
    files = _markdown_files()
    assert files, "found no markdown to check -- the walk itself is broken"
    broken = _broken_links(files)
    assert not broken, "broken documentation links:\n  " + "\n  ".join(broken)


@pytest.mark.parametrize(
    ("body", "why"),
    [
        ("[x](does-not-exist.md)", "a missing file"),
        ("[x](neighbour.md#no-such-heading)", "a missing anchor in another file"),
        ("[x](#no-such-heading-here)", "a missing anchor in the same file"),
    ],
)
def test_the_link_checker_actually_catches_a_break(tmp_path: Path, body: str, why: str) -> None:
    """Positive control: a checker that has never caught anything proves nothing.

    Without this, a regex that silently stopped matching links would report a
    clean sweep over every file in the repo -- the exact failure this suite
    exists to make impossible.
    """
    (tmp_path / "neighbour.md").write_text("# Real Heading\n", encoding="utf-8")
    page = tmp_path / "page.md"
    page.write_text(f"# Page\n\n{body}\n", encoding="utf-8")

    broken = _broken_links([page])

    assert broken, f"checker missed {why}: {body}"


def test_a_heading_with_an_em_dash_resolves() -> None:
    """The repo's headings really do contain ' -- ' style em dashes, and the
    double hyphen they leave behind is the anchor mistake most likely to be
    made by hand."""
    assert _slug("5.1 Retrieval eval (`corpus-eval`) — the tractable half") == (
        "51-retrieval-eval-corpus-eval--the-tractable-half"
    )


def test_code_fences_do_not_invent_anchors(tmp_path: Path) -> None:
    page = tmp_path / "page.md"
    page.write_text("# Real\n\n```sh\n# not a heading\n```\n", encoding="utf-8")
    assert _anchors(page) == {"real"}
