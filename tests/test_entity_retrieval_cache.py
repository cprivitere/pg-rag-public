"""Tests entity_retrieval's cached document loading (_load_docs): it rebuilds
the cache when the source file's mtime changes and serves the cached list
between calls."""

import json
import types

import pytest

from pgrag.rag import entity_retrieval as er


def _mk_doc(pid="item_1", dtype="item"):
    return {"id": pid, "text": f"{pid} text", "metadata": {"name": "Item", "type": dtype}}


class _FakePath:
    def __init__(self, state):
        self._state = state

    def stat(self):
        return types.SimpleNamespace(st_mtime=self._state["mtime"])

    def read_text(self, encoding="utf-8"):
        return json.dumps(self._state["docs"])


@pytest.fixture
def clean_docs_path(monkeypatch):
    state = {"mtime": 1.0, "docs": [_mk_doc("item_1")]}
    monkeypatch.setattr(er, "Path", lambda *a: _FakePath(state))
    er._DOCS_CACHE = None
    yield state
    er._DOCS_CACHE = None


def test_docs_cache_respects_file_mtime(clean_docs_path):
    first = er._load_docs()
    assert first[0]["id"] == "item_1"

    clean_docs_path["docs"] = [_mk_doc("item_2")]
    clean_docs_path["mtime"] = 2.0

    second = er._load_docs()
    assert second[0]["id"] == "item_2"


def test_docs_cache_serves_without_rebuild(clean_docs_path):
    first = er._load_docs()
    second = er._load_docs()
    assert first is second