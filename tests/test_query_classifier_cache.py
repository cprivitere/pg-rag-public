"""Tests query_classifier's entity-index cache: it invalidates when the
underlying index file's mtime changes and serves cached lookups between
calls."""

import json
import types

import pytest

from pgrag.rag import query_classifier as qc


def _docs(name, base_id):
    return [
        {"id": f"{base_id}_chunk_0",
         "metadata": {"name": name, "type": "skill"}}
    ]


class _FakePath:
    def __init__(self, state):
        self._state = state

    def exists(self):
        return True

    def stat(self):
        return types.SimpleNamespace(st_mtime=self._state["mtime"])

    def read_text(self, encoding="utf-8"):
        return json.dumps(self._state["docs"])


@pytest.fixture
def clean_index(monkeypatch):
    state = {"mtime": 1.0, "docs": _docs("Dungcrafting", "skill_Pooping")}
    monkeypatch.setattr(qc, "Path", lambda *a: _FakePath(state))
    qc._ENTITY_INDEX = None
    yield state
    qc._ENTITY_INDEX = None


def test_index_invalidates_on_rebuild(clean_index):
    assert qc.find_entity("what is dungcrafting") == ("skillprofile_Pooping", "skill")

    clean_index["docs"] = _docs("Gardening", "skill_Gardening")
    clean_index["mtime"] = 2.0

    assert qc.find_entity("what is gardening") == ("skillprofile_Gardening", "skill")
    assert qc.find_entity("what is dungcrafting") == (None, None)


def test_index_cached_between_calls(clean_index):
    first = qc.find_entity("what is dungcrafting")
    second = qc.find_entity("dungcrafting")
    assert first == second == ("skillprofile_Pooping", "skill")