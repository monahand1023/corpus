"""corpus-judge: generation-quality eval via a validated LLM-as-judge.

Three modes, one entrypoint (the keyless corpus-eval is untouched):
  default          retrieve + answer_from_context + judge_answer over a query
                   set, print a 3-axis aggregate. Needs DB + API key.
  --validate       run the judge over a frozen fixture, report Cohen's kappa
                   (judge faithfulness vs human) + the adversarial-catch check,
                   optionally gate on a kappa floor. Needs API key only (no DB).
  --build-fixture  generate candidate answers over a query set and write a draft
                   fixture with human_faithful=None for hand-labeling.

Feature-flagged: requires ANTHROPIC_API_KEY (and the [summarizer] extra for the
anthropic SDK). Never runs over a private corpus in CI — see docs/judge.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from corpus._anthropic import make_client
from corpus.cli._common import load_config_or_exit, load_python_export
from corpus.eval.generation import GENERATOR_DEFAULT_MODEL, answer_from_context
from corpus.eval.judge import JUDGE_DEFAULT_MODEL, aggregate_verdicts, judge_answer
from corpus.eval.validation import JudgeCase, run_validation_study

load_dotenv()


def _load_kappa_floor(path: Path) -> float:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    data: Any = json.loads(path.read_text())
    if not isinstance(data, dict) or "kappa" not in data:
        raise ValueError('thresholds file must be a JSON object with a "kappa" key')
    value = data["kappa"]
    if not isinstance(value, int | float):
        raise ValueError('"kappa" must be a number')
    return float(value)


def _run_validate(args: argparse.Namespace) -> int:
    fixture_path = Path(args.fixture)
    if not fixture_path.is_file():
        print(f"Fixture not found: {fixture_path}", file=sys.stderr)
        return 2
    cases = list(load_python_export(fixture_path, "JUDGE_CASES"))
    client = make_client()
    report = run_validation_study(cases, model=args.judge_model, client=client)
    print("=== Judge validation ===")
    print(f"  labeled cases:     {report.n}")
    print(f"  Cohen's kappa:     {report.kappa:.3f}")
    print(f"  raw agreement:     {report.raw_agreement:.3f}")
    print(f"  adversarial caught {report.adversarial_caught}/{report.adversarial_total}")

    passed = report.adversarial_ok
    if not report.adversarial_ok:
        print(
            f"[gate] adversarial = {report.adversarial_caught}/{report.adversarial_total}  FAIL",
            file=sys.stderr,
        )
    if args.check:
        try:
            floor = _load_kappa_floor(Path(args.check))
        except FileNotFoundError:
            print(f"Thresholds file not found: {args.check}", file=sys.stderr)
            return 2
        except ValueError as e:
            print(f"Invalid thresholds file {args.check}: {e}", file=sys.stderr)
            return 2
        ok = report.kappa >= floor
        status = "PASS" if ok else "FAIL"
        print(f"[gate] kappa = {report.kappa:.3f}  floor {floor:.3f}  {status}", file=sys.stderr)
        passed = passed and ok
    return 0 if passed else 1


def _retriever_from_config(config_path: str | None, rerank: bool = False) -> Any:
    from corpus.db.sqlite import ChunkStore
    from corpus.embedder.factory import make_embedder
    from corpus.retriever import Retriever

    config = load_config_or_exit(config_path)
    store = ChunkStore(config.db_path, embedding_dim=config.embedder.dim)
    embedder = make_embedder(
        provider=config.embedder.provider,
        model=config.embedder.model,
        dim=config.embedder.dim,
    )
    reranker = None
    if rerank:
        from corpus.reranker.local import BGEReranker

        reranker = BGEReranker()
    return Retriever(
        store=store,
        embedder=embedder,
        reranker=reranker,
        reference_patterns=config.compiled_references(),
    )


def _context_for(
    retriever: Any, query: str, top_k: int, rerank: bool = False
) -> list[tuple[str, str]]:
    result = retriever.query(query, top_k=top_k, hybrid=True, rerank=rerank)
    return [(c.source_key, c.content) for c in result.chunks]


def _run_default(args: argparse.Namespace) -> int:
    queries_path = Path(args.queries)
    if not queries_path.is_file():
        print(f"Eval queries file not found: {queries_path}", file=sys.stderr)
        return 2
    queries = list(load_python_export(queries_path, "EVAL_QUERIES"))
    client = make_client()
    retriever = _retriever_from_config(args.config, args.rerank)
    try:
        verdicts = []
        rows = []
        for q in queries:
            context = _context_for(retriever, q.query, args.top_k, rerank=args.rerank)
            answer = answer_from_context(
                q.query, context, model=args.generator_model, client=client
            )
            verdict = judge_answer(
                q.query, answer.text, answer.cited_keys, context,
                model=args.judge_model, client=client,
            )
            verdicts.append(verdict)
            rows.append((q.query, verdict))
        agg = aggregate_verdicts(verdicts)
        if args.as_json:
            print(json.dumps({"aggregate": agg, "n": int(agg["n"])}, indent=2))
        else:
            for query_text, verdict in rows:
                print(f"[Q] {query_text}")
                print(
                    f"    faithfulness={verdict.faithfulness.passed} "
                    f"relevance={verdict.answer_relevance.passed} "
                    f"citation={verdict.citation_correctness.passed}"
                )
            print(f"=== Aggregate (n={int(agg['n'])}) ===")
            print(f"  faithfulness:        {agg['faithfulness']:.3f}")
            print(f"  answer_relevance:    {agg['answer_relevance']:.3f}")
            print(f"  citation_correctness:{agg['citation_correctness']:.3f}")
        return 0
    finally:
        retriever.close()


def _run_build_fixture(args: argparse.Namespace) -> int:
    queries_path = Path(args.queries)
    if not queries_path.is_file():
        print(f"Eval queries file not found: {queries_path}", file=sys.stderr)
        return 2
    if not args.out:
        print("--build-fixture requires --out PATH", file=sys.stderr)
        return 2
    queries = list(load_python_export(queries_path, "EVAL_QUERIES"))
    client = make_client()
    retriever = _retriever_from_config(args.config, args.rerank)
    try:
        cases: list[JudgeCase] = []
        for q in queries:
            context = _context_for(retriever, q.query, args.top_k, rerank=args.rerank)
            answer = answer_from_context(
                q.query, context, model=args.generator_model, client=client
            )
            cases.append(
                JudgeCase(
                    query=q.query,
                    answer=answer.text,
                    cited_keys=answer.cited_keys,
                    context=context,
                    human_faithful=None,
                    adversarial=False,
                    note=getattr(q, "note", ""),
                )
            )
        _write_fixture(Path(args.out), cases)
        print(f"Wrote {len(cases)} draft cases to {args.out} — LABEL human_faithful by hand.")
        return 0
    finally:
        retriever.close()


def _write_fixture(path: Path, cases: list[JudgeCase]) -> None:
    lines = [
        '"""Draft judge fixture — LABEL human_faithful by hand, then commit."""',
        "from __future__ import annotations",
        "",
        "from corpus.eval.validation import JudgeCase",
        "",
        "JUDGE_CASES: list[JudgeCase] = [",
    ]
    for c in cases:
        lines.append("    JudgeCase(")
        lines.append(f"        query={c.query!r},")
        lines.append(f"        answer={c.answer!r},")
        lines.append(f"        cited_keys={c.cited_keys!r},")
        lines.append(f"        context={c.context!r},")
        lines.append("        human_faithful=None,  # TODO: label True/False by hand")
        lines.append("        adversarial=False,")
        lines.append(f"        note={c.note!r},")
        lines.append("    ),")
    lines.append("]")
    path.write_text("\n".join(lines) + "\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generation-quality eval via a validated LLM-as-judge")
    parser.add_argument("--validate", action="store_true", help="Run the judge over the fixture; report Cohen's kappa")
    parser.add_argument("--build-fixture", dest="build_fixture", action="store_true", help="Write a draft fixture to --out")
    parser.add_argument("--queries", default="tests/eval_queries.py", help="Python module exporting EVAL_QUERIES")
    parser.add_argument("--fixture", default="tests/judge_fixture.py", help="Python module exporting JUDGE_CASES")
    parser.add_argument("--config", default=None)
    parser.add_argument("--generator-model", default=GENERATOR_DEFAULT_MODEL)
    parser.add_argument("--judge-model", default=JUDGE_DEFAULT_MODEL)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rerank", action="store_true", help="Enable the local BGE cross-encoder re-ranker (needs the [reranker] extra) over the retrieved pool before generation")
    parser.add_argument("--out", default=None, help="Output path for --build-fixture")
    parser.add_argument("--check", default=None, metavar="PATH", help='JSON kappa floor, e.g. {"kappa": 0.6}')
    parser.add_argument("--json", action="store_true", dest="as_json", help="Emit the aggregate as JSON")
    return parser


def main_argv(argv: list[str]) -> int:
    args = _build_parser().parse_args(argv)
    if args.validate:
        return _run_validate(args)
    if args.build_fixture:
        return _run_build_fixture(args)
    return _run_default(args)


def main() -> int:
    return main_argv(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
