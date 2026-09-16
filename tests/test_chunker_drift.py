"""Stored chunks can silently stop matching what the chunker would produce.

A source is ingested, the chunker changes, and nothing re-ingests that
source. Its stored chunks are then whatever an older version made, and
NOTHING SAYS SO: search still works, `corpus-doctor` is happy, and the drift
is invisible until someone re-ingests and sees the bill.

Measured on a live archive: 24% of one source's chunks no longer
matched what the current chunker produces. The re-ingest re-embedded 18,080
chunks for 8.4M tokens, and the only warning anyone got was the invoice. The
content was not wrong, exactly -- it was two chunker versions old, which
means the boundaries it was embedded at are not the boundaries search is
tuned for.

Detection is cheap: chunk a SAMPLE of documents and compare hashes against
the store. It is a sample on purpose, because chunking a million-chunk
archive to answer "is it current?" costs more than the answer is worth.
"""

from __future__ import annotations

import math
from pathlib import Path

from corpus.db.sqlite import ChunkStore
from corpus.survey.drift import chunker_drift
from corpus.types import Chunk, ChunkKind, ChunkMetadata, SourceDocument
from corpus.util.hash import chunk_id, sha256

DIM = 8


def _store(tmp_path: Path, bodies: dict[str, str]) -> Path:
    db = tmp_path / "c.db"
    store = ChunkStore(db, embedding_dim=DIM)
    store.upsert_batch([
        (
            Chunk(
                id=chunk_id("notes", key, ChunkKind.BODY, 0),
                content=body,
                content_hash=sha256(body),
                metadata=ChunkMetadata(
                    source_type="notes", source_key=key,
                    chunk_kind=ChunkKind.BODY, chunk_index=0, title=key,
                ),
            ),
            [math.sin((i + 1) * 0.1) for i in range(DIM)],
        )
        for i, (key, body) in enumerate(bodies.items())
    ])
    store.close()
    return db


class _Chunker:
    """Produces whatever `transform` says, keyed like the stored chunks."""

    source_type = "notes"

    def __init__(self, transform):
        self._transform = transform

    def chunk(self, doc: SourceDocument) -> list[Chunk]:
        content = self._transform(doc.raw["body"])
        return [
            Chunk(
                id=chunk_id("notes", doc.source_key, ChunkKind.BODY, 0),
                content=content,
                content_hash=sha256(content),
                metadata=ChunkMetadata(
                    source_type="notes", source_key=doc.source_key,
                    chunk_kind=ChunkKind.BODY, chunk_index=0, title=doc.source_key,
                ),
            )
        ]


def _docs(bodies):
    return [
        SourceDocument(
            source_type="notes", source_key=k, title=k, url=None, raw={"body": v}
        )
        for k, v in bodies.items()
    ]


def test_a_current_source_reports_no_drift(tmp_path):
    """The positive control. Without it a broken comparison reads as clean."""
    bodies = {f"d{i}": f"body number {i}" for i in range(10)}
    db = _store(tmp_path, bodies)
    report = chunker_drift(db, "notes", _docs(bodies), _Chunker(lambda b: b))
    assert report.examined == 10
    assert report.drifted == 0
    assert report.is_current


def test_a_changed_chunker_is_detected(tmp_path):
    bodies = {f"d{i}": f"body number {i}" for i in range(10)}
    db = _store(tmp_path, bodies)
    report = chunker_drift(
        db, "notes", _docs(bodies), _Chunker(lambda b: b.upper())
    )
    assert report.drifted == 10
    assert not report.is_current
    assert "100" in report.describe()


def test_partial_drift_is_reported_as_a_share(tmp_path):
    bodies = {f"d{i}": f"body number {i}" for i in range(10)}
    db = _store(tmp_path, bodies)
    report = chunker_drift(
        db, "notes", _docs(bodies),
        _Chunker(lambda b: b.upper() if b.endswith(("0", "1", "2")) else b),
    )
    assert report.drifted == 3
    assert report.percent == 30.0


def test_a_document_the_store_has_never_seen_is_not_drift(tmp_path):
    """New content is not stale content. Counting it as drift would make
    every growing archive look permanently out of date."""
    db = _store(tmp_path, {"old": "body"})
    report = chunker_drift(
        db, "notes", _docs({"old": "body", "brand_new": "body"}), _Chunker(lambda b: b)
    )
    assert report.drifted == 0
    assert report.unseen == 1


def test_an_empty_source_is_vacuous_not_clean(tmp_path):
    """"I examined nothing" and "I examined everything and it was fine" are
    the two facts this codebase keeps having to keep apart."""
    db = _store(tmp_path, {"a": "body"})
    report = chunker_drift(db, "notes", [], _Chunker(lambda b: b))
    assert report.examined == 0
    assert not report.is_current
    assert "nothing" in report.describe().lower()


def test_the_sample_is_bounded(tmp_path):
    """Chunking a million-chunk archive to answer "is it current?" costs more
    than the answer is worth."""
    bodies = {f"d{i}": f"body {i}" for i in range(200)}
    db = _store(tmp_path, bodies)
    report = chunker_drift(db, "notes", _docs(bodies), _Chunker(lambda b: b), sample=25)
    assert report.examined == 25


# --- the check must say WHY it could not look ---------------------------------


def test_the_doctor_names_why_a_source_could_not_be_sampled(tmp_path, capsys, monkeypatch):
    """"SKIPPED (no source could be sampled)" is the failure mode this whole
    session has been about: a check that did not run, reported in words that
    read like a check that found nothing.

    Live: every source in one archive used a connector type registered by the
    CONSUMER at runtime, which the doctor does not load. The answer is
    `--load`, and the output has to say so.
    """
    import corpus.cli.doctor as mod
    from corpus.config import CorpusConfig, EmbedderConfig, SourceConfig

    cfg = CorpusConfig(
        db_path=tmp_path / "c.db",
        embedder=EmbedderConfig(provider="hash", dim=8),
        sources=[SourceConfig(name="mail", type="markdown", path=str(tmp_path))],
    )
    (tmp_path / "c.db").write_bytes(b"")
    # Imported inside the function, so patch it at its source module.
    monkeypatch.setattr(CorpusConfig, "load", classmethod(lambda cls, p: cfg))

    import corpus.connectors.registry as reg

    def boom(_src):
        raise ValueError("Source 'mail' uses type='email', which is not registered.")

    monkeypatch.setattr(reg, "build_pipeline", boom)
    mod._check_chunker_drift(str(tmp_path / "corpus.toml"), str(tmp_path / "c.db"))

    out = capsys.readouterr().out
    assert "not registered" in out, out
    assert "--load" in out, "the fix was not named"


def test_load_applies_to_every_check_not_just_the_shadow_one(tmp_path, capsys):
    """`--load` imports a consumer's module and calls its `register()`. It was
    wired into the shadowed-components check ONLY, so a consumer's own
    connector types stayed unregistered for every other check -- including
    the drift one, which then reported "no source could be sampled" and
    advised passing the flag that was already passed.

    A consumer's connectors are a property of the deployment, not of one
    check.
    """
    import sys
    import types

    import corpus.cli.doctor as mod
    from corpus.connectors.registry import CONNECTOR_REGISTRY

    called = {"n": 0}
    fake = types.ModuleType("fake_consumer_reg")

    def register():
        called["n"] += 1

    fake.register = register  # type: ignore[attr-defined]
    sys.modules["fake_consumer_reg"] = fake
    try:
        mod._apply_load("fake_consumer_reg")
    finally:
        del sys.modules["fake_consumer_reg"]
    assert called["n"] == 1, "register() was never called"
    assert CONNECTOR_REGISTRY  # sanity


def test_a_load_that_fails_says_so_rather_than_passing_silently(capsys):
    import corpus.cli.doctor as mod

    ok = mod._apply_load("module.that.does.not.exist")
    assert ok is False
    assert "could not load" in capsys.readouterr().out.lower()
