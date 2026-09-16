"""Every shipped backend actually matches the Protocol it claims to.

`corpus.embedder.base` and `corpus.connectors.base` are pure Protocol
declarations, which made them look like there was nothing to test. There is,
and it is the shape this project keeps finding: **`@runtime_checkable`
`isinstance` only checks method NAMES.** A backend whose `embed_query` takes
different parameters, or returns the wrong thing, passes `isinstance` exactly
as a correct one does. The check that looks like it verifies the contract
verifies the spelling.

So this compares SIGNATURES. Renaming a parameter, adding a required one, or
dropping a method is caught here rather than at the call site of whichever
command happened to reach the new backend first.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Protocol

import pytest

import corpus.connectors
import corpus.embedder
from corpus.connectors.base import Chunker, Connector
from corpus.embedder.base import Embedder


def _protocol_methods(proto: type) -> dict[str, inspect.Signature]:
    return {
        name: inspect.signature(getattr(proto, name))
        for name in proto.__protocol_attrs__  # type: ignore[attr-defined]
        if callable(getattr(proto, name, None))
    }


def _implementations(package, protocol: type) -> list[type]:
    """Every concrete class in `package` that declares the protocol's methods."""
    wanted = set(_protocol_methods(protocol))
    found: list[type] = []
    for info in pkgutil.iter_modules(package.__path__):
        if info.name in ("base", "factory", "registry", "discovery"):
            continue
        module = importlib.import_module(f"{package.__name__}.{info.name}")
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ != module.__name__ or issubclass(obj, Protocol):
                continue
            if wanted and wanted.issubset(set(dir(obj))):
                found.append(obj)
    return found


def _params(sig: inspect.Signature) -> list[tuple[str, object]]:
    return [
        (p.name, p.kind)
        for p in sig.parameters.values()
        if p.name != "self" and p.default is inspect.Parameter.empty
    ]


# --- the positive control -----------------------------------------------------


def test_the_sweep_actually_finds_implementations():
    """Without this, an import error or a renamed module would silently make
    every test below vacuous -- zero implementations, all green."""
    assert len(_implementations(corpus.embedder, Embedder)) >= 2
    assert len(_implementations(corpus.connectors, Connector)) >= 8


def test_isinstance_alone_would_not_have_caught_a_wrong_signature():
    """The reason this file compares signatures instead. Documents the gap
    rather than assuming the reader knows it."""

    class Impostor:
        total_tokens_used = 0
        counts_tokens = True

        def embed_documents(self, wrong_name, extra_required):
            return []

        def embed_query(self):  # takes no text at all
            return []

    assert isinstance(Impostor(), Embedder), "protocol semantics changed"
    assert _params(inspect.signature(Impostor.embed_query)) != _params(
        inspect.signature(Embedder.embed_query)
    )


# --- conformance --------------------------------------------------------------


@pytest.mark.parametrize(
    "impl", _implementations(corpus.embedder, Embedder), ids=lambda c: c.__name__
)
def test_every_embedder_matches_the_protocol_signature(impl):
    for name, expected in _protocol_methods(Embedder).items():
        actual = inspect.signature(getattr(impl, name))
        assert _params(actual) == _params(expected), (
            f"{impl.__name__}.{name}{actual} does not match "
            f"Embedder.{name}{expected}"
        )


@pytest.mark.parametrize(
    "impl", _implementations(corpus.connectors, Connector), ids=lambda c: c.__name__
)
def test_every_connector_matches_the_protocol_signature(impl):
    for name, expected in _protocol_methods(Connector).items():
        actual = inspect.signature(getattr(impl, name))
        assert _params(actual) == _params(expected), (
            f"{impl.__name__}.{name}{actual} does not match "
            f"Connector.{name}{expected}"
        )


@pytest.mark.parametrize(
    "impl", _implementations(corpus.connectors, Chunker), ids=lambda c: c.__name__
)
def test_every_chunker_matches_the_protocol_signature(impl):
    for name, expected in _protocol_methods(Chunker).items():
        actual = inspect.signature(getattr(impl, name))
        assert _params(actual) == _params(expected), (
            f"{impl.__name__}.{name}{actual} does not match "
            f"Chunker.{name}{expected}"
        )


def test_every_embedder_initialises_the_token_counter_the_pipeline_reads():
    """`total_tokens_used` is how a run reports what it spent, and the
    pipeline reads it unconditionally. An embedder that never set it would
    raise at the END of an expensive run, not the start.

    Checked by reading `__init__`, not with `hasattr`: it is an instance
    attribute, so the class does not carry it, and the providers cannot be
    constructed here without credentials.
    """
    import ast

    for impl in _implementations(corpus.embedder, Embedder):
        init = getattr(impl, "__init__", None)
        assert init is not None, f"{impl.__name__} has no __init__"
        tree = ast.parse(inspect.getsource(init).lstrip())
        assigned = {
            t.attr
            for node in ast.walk(tree)
            for t in getattr(node, "targets", []) + (
                [node.target] if isinstance(node, ast.AugAssign) else []
            )
            if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
            and t.value.id == "self"
        }
        assert "total_tokens_used" in assigned, (
            f"{impl.__name__}.__init__ never sets total_tokens_used"
        )
