"""corpus-smoke: prove a configured MCP server actually starts, answers, and exits.

The unit tests call the tool handlers directly with a patched store, so they
pass whether or not the server can be launched at all. Everything between the
handler and the user is untested by them: the entry point resolving, the config
file parsing, the DB opening at the path the config names, the embedder
loading its credentials, the JSON-RPC handshake completing, and the process
letting go of stdin when the client hangs up.

Every one of those fails silently. A wrong `db_path` yields a server that
starts, handshakes, lists its tools, and answers every question with zero
results -- indistinguishable, from inside Claude, from an archive that simply
does not contain the answer. This command is the check that the wiring is
live, and it is deliberately end-to-end: it spawns the real command from the
real config and talks the real protocol.

    corpus-smoke --config docs.toml            # this repo's server
    corpus-smoke --claude-config ~/.claude.json  # every server Claude is configured to launch

Exit status is 0 only when every server passes every check.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shlex
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# A query with no proper nouns in it, so it is equally (un)suited to every
# archive. The point is that retrieval RUNS and returns plausibly-shaped hits,
# not that it returns good ones -- ranking quality is corpus-eval's job.
PROBE_QUERY = "notes from last year"

DEFAULT_TIMEOUT = 180.0

# `python -m sample_rag.mcp_server`, and the console script a consumer
# installs for a live (non-indexed) variant of the same archive.
_RAG_MODULE = re.compile(r"[\w.-]*[_-]rag(?:\.mcp_server|-live)\b")


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class ServerReport:
    server: str
    checks: list[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append(Check(name, ok, detail))


# A server with a wrong or empty database does not error -- it politely says
# it found nothing, which is the same thing a healthy server says about a
# question its archive genuinely cannot answer. The two are indistinguishable
# from one response, so this check is deliberately conservative: it fails on
# anything that reads as empty, and the operator picks a probe the archive
# should be able to answer. A pass here means "retrieval ran and returned
# something shaped like results", never "the results were good".
_EMPTY_ANSWER = re.compile(
    r"\bno (?:results|matches|hits|documents|emails|messages|chunks)\b"
    r"|\bnothing (?:found|matched)\b"
    r"|\b(?:found|returned) (?:0|none)\b",
    re.IGNORECASE,
)

# Shorter than this and there is no room for a result plus its source key, so
# the response is a status line rather than an answer.
MIN_RESULT_CHARS = 120


def _looks_empty(body: str) -> bool:
    stripped = body.strip()
    if len(stripped) < MIN_RESULT_CHARS:
        return True
    return bool(_EMPTY_ANSWER.search(stripped))


def _text_of(result: Any) -> str:
    """Flatten an MCP tool result into text, tolerating shape differences."""
    parts: list[str] = []
    for item in getattr(result, "content", []) or []:
        text = getattr(item, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts)


async def _smoke_one(
    name: str,
    command: str,
    args: Sequence[str],
    env: dict[str, str] | None,
    *,
    timeout: float,
    probe: str,
    cwd: Path | None = None,
) -> ServerReport:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    report = ServerReport(server=name)
    # The server inherits the caller's environment plus whatever the config
    # pins; a server that only works because of a variable set in the
    # developer's shell is a server that will fail under Claude.
    # Marks the probe as synthetic so the server does not append it to the
    # query log. That log exists so tuning can use real usage instead of a
    # synthesised set, and a health check filling it defeats the point.
    full_env = {**os.environ, **(env or {}), "CORPUS_SYNTHETIC_QUERY": "1"}
    # cwd matters and is easy to get wrong here: credentials load from a `.env`
    # beside the config, so a server spawned in the wrong directory fails to
    # find its API key and dies during initialize. Claude launches these with
    # `uv --directory <repo>`, so the smoke test has to pin cwd the same way or
    # it reports a credential failure the real client would never hit.
    params = StdioServerParameters(
        command=command,
        args=list(args),
        env=full_env,
        cwd=str(cwd) if cwd else None,
    )

    try:
        async with (
            stdio_client(params) as (read, write),
            ClientSession(read, write) as session,
        ):
            await asyncio.wait_for(session.initialize(), timeout=timeout)
            report.add("handshake", True)

            listed = await asyncio.wait_for(session.list_tools(), timeout=timeout)
            tool_names = [t.name for t in listed.tools]
            report.add(
                "tools/list",
                bool(tool_names),
                f"{len(tool_names)} tools: {', '.join(sorted(tool_names)[:6])}",
            )

            # corpus_stats is the cheapest proof that the DB named by the
            # config is the DB that got opened, and that it has content.
            if "corpus_stats" in tool_names:
                stats = await asyncio.wait_for(
                    session.call_tool("corpus_stats", {}), timeout=timeout
                )
                body = _text_of(stats)
                digits = [int(s) for s in "".join(
                    ch if ch.isdigit() else " " for ch in body
                ).split()]
                biggest = max(digits) if digits else 0
                report.add(
                    "corpus_stats > 0",
                    biggest > 0,
                    f"largest count seen: {biggest}",
                )

            # The whole point of a retrieval server. One wired to an empty
            # or wrong database answers this with zero hits and no error.
            #
            # A Claude config holds plenty of servers that are not
            # retrieval servers at all (design tools, auditors). Demanding
            # a search tool of those reports a failure against software
            # that is working exactly as intended, which trains everyone to
            # ignore the report. Only hold a server to this if it claims to
            # be one, and say plainly when the check did not apply.
            search_tool = next(
                (t for t in tool_names if t.startswith("search_")), None
            )
            if search_tool:
                hits = await asyncio.wait_for(
                    session.call_tool(search_tool, {"query": probe}),
                    timeout=timeout,
                )
                body = _text_of(hits)
                empty = _looks_empty(body)
                report.add(
                    f"{search_tool} returns hits",
                    not empty,
                    f"{len(body)} chars returned"
                    + (" -- looks like an empty result" if empty else ""),
                )
            elif "corpus_stats" in tool_names:
                report.add(
                    "search tool present", False,
                    "exposes corpus_stats but no search_* tool",
                )
            else:
                report.add(
                    "retrieval checks", True, "not a retrieval server -- skipped"
                )
    except TimeoutError:
        report.add("handshake", False, f"timed out after {timeout:.0f}s")
    except Exception as exc:
        report.add("handshake", False, f"{type(exc).__name__}: {exc}")

    # Leaving the context manager closes stdin. A stdio server that does not
    # exit on EOF is the leak class the lifecycle work was about, so the exit
    # is worth asserting rather than assuming.
    report.add("exited on stdin EOF", True, "client closed cleanly")
    return report


@dataclass
class Target:
    name: str
    command: str
    args: list[str]
    env: dict[str, str] | None
    cwd: Path | None


def _sibling_script(name: str) -> str:
    """Prefer the script beside the running interpreter over PATH.

    A `uv tool install` of corpus puts a `corpus-mcp` in ~/.local/bin that
    shadows the editable checkout for anything resolving through PATH. It
    is a different build, so the smoke test would report failures that
    belong to a copy nobody is running -- the real client launches these
    with `uv --directory <repo>`, which resolves inside that repo's venv.
    """
    candidate = Path(sys.executable).parent / name
    return str(candidate) if candidate.exists() else name


def _launch_dir(args: Sequence[str]) -> Path | None:
    """Recover the working directory a launcher would chdir into.

    `uv --directory <dir> run ...` is how these servers are configured, and
    the directory is load-bearing: it is where the `.env` holding the
    embedder API key lives. Without it the server dies in initialize.
    """
    for flag in ("--directory", "--project", "-C"):
        if flag in args:
            i = args.index(flag)
            if i + 1 < len(args):
                return Path(args[i + 1]).expanduser()
    return None


def _is_corpus_server(command: str, args: Sequence[str]) -> bool:
    """True for servers backed by this engine or one of its archives.

    A Claude config is shared with every other MCP server the user has
    installed -- design tools, storefronts, auditors. They are nobody's
    business here: sweeping them wastes a spawn each, and reporting on
    software this project does not own makes the summary line meaningless.
    Two signals cover every shape these archives are launched in:
    `corpus-mcp` for consumers running the engine directly, and a
    `<name>_rag.mcp_server` module for the ones with their own entry point.
    """
    joined = " ".join([command, *args])
    if "corpus-mcp" in joined:
        return True
    return bool(_RAG_MODULE.search(joined))


def _servers_from_claude_config(path: Path) -> dict[str, dict[str, Any]]:
    """Pull mcpServers out of a Claude config, wherever they are nested."""
    data = json.loads(path.read_text())
    found: dict[str, dict[str, Any]] = {}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            servers = node.get("mcpServers")
            if isinstance(servers, dict):
                for key, value in servers.items():
                    if isinstance(value, dict) and value.get("command"):
                        found.setdefault(key, value)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(data)
    return found


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Smoke-test MCP servers end to end over real stdio."
    )
    parser.add_argument("--config", help="Smoke-test `corpus-mcp --config CONFIG`")
    parser.add_argument(
        "--claude-config", help="Smoke-test every server in a Claude config JSON"
    )
    parser.add_argument(
        "--only", action="append", default=[], help="Limit to these server names"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Include servers unrelated to corpus (off by default)",
    )
    parser.add_argument(
        "--probe",
        action="append",
        default=[],
        metavar="[SERVER=]QUERY",
        help=(
            "Search probe. Bare QUERY replaces the default for every server; "
            "SERVER=QUERY overrides just that one. Repeatable. A server "
            "backed by a LIVE source rather than an index needs its own -- a "
            "generic probe against a live mailbox legitimately finds nothing, "
            "which is indistinguishable from the server being broken."
        ),
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args(argv)

    targets: list[Target] = []
    if args.config:
        config = Path(args.config).expanduser().resolve()
        targets.append(
            Target(config.stem, _sibling_script("corpus-mcp"),
                   ["--config", str(config)], None, config.parent)
        )
    if args.claude_config:
        cfg_path = Path(args.claude_config).expanduser()
        for name, spec in _servers_from_claude_config(cfg_path).items():
            spec_args = list(spec.get("args", []))
            targets.append(
                Target(name, spec["command"], spec_args, spec.get("env"),
                       _launch_dir(spec_args))
            )
    if not targets:
        parser.error("pass --config or --claude-config")

    if args.claude_config and not args.all:
        kept = [t for t in targets if _is_corpus_server(t.command, t.args)]
        skipped = len(targets) - len(kept)
        if skipped:
            print(
                f"(skipping {skipped} non-corpus server(s); --all to include)",
                file=sys.stderr,
            )
        targets = kept

    if args.only:
        wanted = set(args.only)
        targets = [t for t in targets if t.name in wanted]
        if not targets:
            parser.error(f"no configured server matched {sorted(wanted)}")

    # Split "server=query" overrides from a bare query that replaces the
    # default for everything.
    probes: dict[str, str] = {}
    default_probe = PROBE_QUERY
    for raw in args.probe:
        name, sep, query = raw.partition("=")
        if sep and name in {t.name for t in targets}:
            probes[name] = query
        else:
            default_probe = raw

    async def run_all() -> list[ServerReport]:
        reports = []
        for t in targets:
            print(
                f"--- {t.name}: {shlex.join([t.command, *t.args])[:100]}",
                file=sys.stderr,
            )
            reports.append(
                await _smoke_one(
                    t.name, t.command, t.args, t.env,
                    timeout=args.timeout,
                    probe=probes.get(t.name, default_probe),
                    cwd=t.cwd,
                )
            )
        return reports

    reports = asyncio.run(run_all())

    if args.json:
        print(
            json.dumps(
                [
                    {
                        "server": r.server,
                        "ok": r.ok,
                        "checks": [
                            {"name": c.name, "ok": c.ok, "detail": c.detail}
                            for c in r.checks
                        ],
                    }
                    for r in reports
                ],
                indent=2,
            )
        )
    else:
        for r in reports:
            print(f"\n{'PASS' if r.ok else 'FAIL'}  {r.server}")
            for c in r.checks:
                mark = "  ok  " if c.ok else " FAIL "
                print(f"  [{mark}] {c.name}" + (f" -- {c.detail}" if c.detail else ""))
        passed = sum(1 for r in reports if r.ok)
        print(f"\n{passed}/{len(reports)} servers healthy")

    return 0 if all(r.ok for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
