"""The zip connector must not build the transcripts connector on a member tree.

`_MEMBER_CONNECTOR_TYPES` is derived from `CONNECTOR_REGISTRY` minus a
hand-maintained exclusion set, and that derivation is deliberate: a
hand-duplicated list silently missed `pptx`, `csv` and `tsv` for a release
cycle each. But it means a registry entry that must NOT be recursed into is
included the moment it is registered, and `transcripts` is one.

`TranscriptConnector` does not read FILES. It reads a sidecar DATABASE, and
its `DEFAULT_GLOBS` entry is marked "detection pattern only" in the registry
for exactly that reason. Handed a zip's extraction directory it raises

    transcript database not found: /.../corpus-zip-uch9iq37.
    Run a transcription pass first.

which `Ingester.ingest` treats as a source that cannot be enumerated: the
WHOLE source is skipped, with its index left intact and nothing indexed.
Found on a live archive whose `archives` source had silently been failing
that way -- reported once as an ERROR line and then never again, because the
next run failed the same way.

The exclusion set is the right place for it, alongside `zip` (recursion) and
`aup3` (its audio extraction has nowhere to write inside a temp dir).
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from corpus.connectors.zip import _MEMBER_CONNECTOR_TYPES, ZipConnector


def test_transcripts_is_not_a_member_connector_type():
    assert "transcripts" not in _MEMBER_CONNECTOR_TYPES, (
        "the zip connector will build TranscriptConnector on an extraction "
        "directory, which raises and takes the whole source with it"
    )


def test_the_exclusions_that_were_already_there_stay_excluded():
    """Negative control: this must not become a list that drifts the other
    way."""
    assert "zip" not in _MEMBER_CONNECTOR_TYPES
    assert "aup3" not in _MEMBER_CONNECTOR_TYPES


def test_ordinary_member_types_are_still_built():
    """The derivation exists so adding a connector is enough. Excluding one
    must not turn it back into a hand-maintained allowlist."""
    for expected in ("markdown", "pdf", "docx", "xlsx", "pptx", "csv", "html"):
        assert expected in _MEMBER_CONNECTOR_TYPES


def test_a_zip_of_ordinary_documents_ingests(tmp_path: Path):
    """The end-to-end shape of the live failure: before the fix this raised
    FileNotFoundError from the transcripts connector and yielded nothing."""
    archive = tmp_path / "docs.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("notes/one.md", "# One\n\nThe first document's body text.\n")
        z.writestr("notes/two.md", "# Two\n\nA second, different body.\n")

    connector = ZipConnector(source_type="archives", path=str(tmp_path))
    docs = list(connector.load())
    assert len(docs) == 2, f"expected both members, got {[d.source_key for d in docs]}"
