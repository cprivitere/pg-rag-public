"""Tests that pgrag.rag.retriever.retrieve corrects typos in query text before
embedding, while leaving already-correct queries unaltered."""

import types

import pytest

from pgrag.rag import retriever


class _FakeCollection:
    def query(self, **kw):
        self.query_kwargs = kw
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}


class _FakeClient:
    def __init__(self, *a, **k):
        self.collection = _FakeCollection()

    def get_collection(self, *a, **k):
        return self.collection


def test_retrieve_corrects_typo_before_embedding(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        retriever,
        "chromadb",
        types.SimpleNamespace(PersistentClient=_FakeClient),
    )

    def fake_embed(text):
        seen["embed"] = text
        return [0.0] * 1024

    monkeypatch.setattr(retriever, "embed_text", fake_embed)

    retriever.retrieve(
        "Whast is this highest level msurhoom?",
        count=3,
        hybrid=False,
        rerank=False,
    )

    assert "mushroom" in seen["embed"]


def test_retrieve_leaves_correct_query_alone(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        retriever,
        "chromadb",
        types.SimpleNamespace(PersistentClient=_FakeClient),
    )

    def fake_embed(text):
        seen["embed"] = text
        return [0.0] * 1024

    monkeypatch.setattr(retriever, "embed_text", fake_embed)

    q = "What is the highest level mushroom?"
    retriever.retrieve(q, count=3, hybrid=False, rerank=False)
    assert seen["embed"] == "what is the highest level mushroom"