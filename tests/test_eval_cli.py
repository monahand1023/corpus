from __future__ import annotations

import json
from pathlib import Path

from corpus.cli import eval as eval_cli
from corpus.db.sqlite import ChunkStore
from corpus.embedder.hash import HashEmbedder
from corpus.types import Chunk, ChunkKind, ChunkMetadata
from corpus.util.hash import chunk_id, sha256

DIM = 256


def _make_chunk(stype: str, key: str, text: str) -> Chunk:
    return Chunk(
        id=chunk_id(stype, key, ChunkKind.SECTION, 0),
        content=text,
        content_hash=sha256(text),
        metadata=ChunkMetadata(
            source_type=stype, source_key=key, chunk_kind=ChunkKind.SECTION,
            chunk_index=0, title=key,
        ),
    )


def _build_store(path: Path) -> None:
    store = ChunkStore(path, embedding_dim=DIM)
    e = HashEmbedder(dim=DIM)
    docs = [
        ("note", "alpha", "reciprocal rank fusion combines vector and keyword search"),
        ("note", "beta", "the bge cross encoder reranks the candidate pool"),
        ("faq", "faq-keys", "you need an api key for voyage but not the hash embedder"),
    ]
    pairs = []
    for stype, key, text in docs:
        emb = e.embed_query(text)
        pairs.append((_make_chunk(stype, key, text), emb))
    store.upsert_batch(pairs)
    store.close()


def _write_config(cfg_path: Path, db_path: Path) -> None:
    cfg_path.write_text(
        f'[corpus]\ndb_path = "{db_path.as_posix()}"\n'
        '[embedder]\nprovider = "hash"\nmodel = "hash-v1"\ndim = 256\n'
        '[retriever]\ntop_k = 5\n'
    )


def _write_queries(q_path: Path) -> None:
    q_path.write_text(
        "from dataclasses import dataclass, field\n"
        "@dataclass\n"
        "class EvalQuery:\n"
        "    query: str\n"
        "    expected_keys: list = field(default_factory=list)\n"
        "    source_filter: list | None = None\n"
        "    source_type: str | None = None\n"
        "    note: str = ''\n"
        "EVAL_QUERIES = [\n"
        "    EvalQuery('reciprocal rank fusion', ['alpha'], None, 'note'),\n"
        "    EvalQuery('api key voyage hash embedder', ['faq-keys'], None, 'faq'),\n"
        "    EvalQuery('unrelated off topic', [], None, 'note', 'negative'),\n"
        "]\n"
    )


def _run(cfg: Path, q: Path, extra: list[str]) -> int:
    import sys

    argv = sys.argv
    sys.argv = ["corpus-eval", "--config", str(cfg), "--queries", str(q), *extra]
    try:
        return eval_cli.main()
    finally:
        sys.argv = argv


def test_json_output_has_aggregate_and_breakdown(tmp_path: Path, capsys) -> None:
    db = tmp_path / "c.db"
    _build_store(db)
    cfg = tmp_path / "corpus.toml"
    _write_config(cfg, db)
    q = tmp_path / "q.py"
    _write_queries(q)

    rc = _run(cfg, q, ["--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["aggregate"]["n"] == 2  # negative excluded from scoring
    assert set(payload["by_source_type"]) == {"note", "faq"}
    assert payload["aggregate"]["recall_at_k"] == 1.0


def test_compare_runs_all_configs(tmp_path: Path, capsys) -> None:
    db = tmp_path / "c.db"
    _build_store(db)
    cfg = tmp_path / "corpus.toml"
    _write_config(cfg, db)
    q = tmp_path / "q.py"
    _write_queries(q)
    rc = _run(cfg, q, ["--compare"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "hybrid" in out and "vector-only" in out


def test_check_passes_when_floor_is_met(tmp_path: Path, capsys) -> None:
    db = tmp_path / "c.db"
    _build_store(db)
    cfg = tmp_path / "corpus.toml"
    _write_config(cfg, db)
    q = tmp_path / "q.py"
    _write_queries(q)
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text(json.dumps({"recall_at_k": 0.95}))

    rc = _run(cfg, q, ["--check", str(thresholds)])
    err = capsys.readouterr().err
    assert rc == 0
    assert "[gate]" in err
    assert "PASS" in err


def test_check_fails_when_floor_is_impossible(tmp_path: Path, capsys) -> None:
    db = tmp_path / "c.db"
    _build_store(db)
    cfg = tmp_path / "corpus.toml"
    _write_config(cfg, db)
    q = tmp_path / "q.py"
    _write_queries(q)
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text(json.dumps({"recall_at_k": 1.1}))

    rc = _run(cfg, q, ["--check", str(thresholds)])
    err = capsys.readouterr().err
    assert rc == 1
    assert "[gate]" in err
    assert "FAIL" in err


def test_check_with_json_still_parses_and_reflects_gate(tmp_path: Path, capsys) -> None:
    db = tmp_path / "c.db"
    _build_store(db)
    cfg = tmp_path / "corpus.toml"
    _write_config(cfg, db)
    q = tmp_path / "q.py"
    _write_queries(q)
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text(json.dumps({"recall_at_k": 1.1}))

    rc = _run(cfg, q, ["--check", str(thresholds), "--json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert rc == 1
    assert payload["aggregate"]["n"] == 2
    assert "[gate]" in captured.err
    assert "FAIL" in captured.err


def test_check_missing_file_returns_2(tmp_path: Path, capsys) -> None:
    db = tmp_path / "c.db"
    _build_store(db)
    cfg = tmp_path / "corpus.toml"
    _write_config(cfg, db)
    q = tmp_path / "q.py"
    _write_queries(q)
    thresholds = tmp_path / "does-not-exist.json"

    rc = _run(cfg, q, ["--check", str(thresholds)])
    err = capsys.readouterr().err
    assert rc == 2
    assert err.strip() != ""
