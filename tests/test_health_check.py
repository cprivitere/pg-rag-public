"""Tests pgrag.vectorstore.health_check.health_check exit codes: a healthy index
exits 0, while count mismatches, orphaned/missing docs, embedding/metadata hash
corruption, and dimension mismatches are reported as 1."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import chromadb
import pytest

from pgrag.config import EMBEDDING_DIM
from pgrag.vectorstore.build_index import build_index
from pgrag.vectorstore.health_check import health_check


def fake_embed_batch(texts):
    return [[0.1] * EMBEDDING_DIM for _ in texts]


def _build_test_index(docs, chroma_path):
    with patch("pgrag.vectorstore.build_index.embed_batch", side_effect=fake_embed_batch):
        build_index(documents=docs, chroma_path=chroma_path)


META = {"source": "cdn", "table": "items", "type": "item"}


def test_healthy_index_exits_0():
    docs = [
        {"id": "a", "type": "item", "text": "alpha", "metadata": dict(META)},
        {"id": "b", "type": "item", "text": "beta", "metadata": dict(META)},
    ]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        chroma_path = str(Path(tmp) / "chroma")
        doc_path = str(Path(tmp) / "documents.json")
        with open(doc_path, "w") as f:
            json.dump(docs, f)
        _build_test_index(docs, chroma_path)
        assert health_check(chroma_path=chroma_path, documents_path=doc_path) == 0


def test_count_mismatch_reported():
    docs = [
        {"id": "a", "type": "item", "text": "alpha", "metadata": dict(META)},
    ]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        chroma_path = str(Path(tmp) / "chroma")
        doc_path = str(Path(tmp) / "documents.json")
        with open(doc_path, "w") as f:
            json.dump(docs, f)
        _build_test_index(docs, chroma_path)
        extra_docs = [
            {"id": "a", "type": "item", "text": "alpha", "metadata": dict(META)},
            {"id": "c", "type": "item", "text": "extra", "metadata": dict(META)},
        ]
        with open(doc_path, "w") as f:
            json.dump(extra_docs, f)
        assert health_check(chroma_path=chroma_path, documents_path=doc_path) == 1


def test_orphaned_docs_reported():
    docs_first = [
        {"id": "a", "type": "item", "text": "alpha", "metadata": dict(META)},
        {"id": "b", "type": "item", "text": "beta", "metadata": dict(META)},
    ]
    docs_second = [
        {"id": "a", "type": "item", "text": "alpha", "metadata": dict(META)},
    ]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        chroma_path = str(Path(tmp) / "chroma")
        doc_path = str(Path(tmp) / "documents.json")
        with open(doc_path, "w") as f:
            json.dump(docs_first, f)
        _build_test_index(docs_first, chroma_path)
        with open(doc_path, "w") as f:
            json.dump(docs_second, f)
        assert health_check(chroma_path=chroma_path, documents_path=doc_path) == 1


def test_missing_docs_reported():
    docs_indexed = [
        {"id": "a", "type": "item", "text": "alpha", "metadata": dict(META)},
    ]
    docs_expected = [
        {"id": "a", "type": "item", "text": "alpha", "metadata": dict(META)},
        {"id": "b", "type": "item", "text": "beta", "metadata": dict(META)},
    ]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        chroma_path = str(Path(tmp) / "chroma")
        doc_path = str(Path(tmp) / "documents.json")
        with open(doc_path, "w") as f:
            json.dump(docs_expected, f)
        _build_test_index(docs_indexed, chroma_path)
        assert health_check(chroma_path=chroma_path, documents_path=doc_path) == 1


def test_embedding_hash_corruption_detected():
    docs = [
        {"id": "a", "type": "item", "text": "alpha", "metadata": dict(META)},
    ]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        chroma_path = str(Path(tmp) / "chroma")
        doc_path = str(Path(tmp) / "documents.json")
        with open(doc_path, "w") as f:
            json.dump(docs, f)
        _build_test_index(docs, chroma_path)
        client = chromadb.PersistentClient(path=chroma_path)
        coll = client.get_collection("project_gorgon")
        meta = coll.get()["metadatas"][0]
        meta["embedding_hash"] = "corrupt"
        coll.update(ids=["a"], metadatas=[meta])
        assert health_check(chroma_path=chroma_path, documents_path=doc_path) == 1


def test_v23_dimension_mismatch_reported():
    docs = [
        {"id": "a", "type": "item", "text": "alpha", "metadata": dict(META)},
    ]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        chroma_path = str(Path(tmp) / "chroma")
        doc_path = str(Path(tmp) / "documents.json")
        with open(doc_path, "w") as f:
            json.dump(docs, f)
        _build_test_index(docs, chroma_path)
        with patch("pgrag.vectorstore.health_check.EMBEDDING_DIM", 999):
            assert health_check(chroma_path=chroma_path, documents_path=doc_path) == 1


def test_metadata_hash_corruption_detected():
    docs = [
        {"id": "a", "type": "item", "text": "alpha", "metadata": dict(META)},
    ]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        chroma_path = str(Path(tmp) / "chroma")
        doc_path = str(Path(tmp) / "documents.json")
        with open(doc_path, "w") as f:
            json.dump(docs, f)
        _build_test_index(docs, chroma_path)
        client = chromadb.PersistentClient(path=chroma_path)
        coll = client.get_collection("project_gorgon")
        meta = coll.get()["metadatas"][0]
        meta["metadata_hash"] = "corrupt"
        coll.update(ids=["a"], metadatas=[meta])
        assert health_check(chroma_path=chroma_path, documents_path=doc_path) == 1


def test_large_collection_no_sqlite_limit_crash():
    docs = [
        {"id": f"doc_{i}", "type": "item", "text": f"text {i}", "metadata": dict(META)}
        for i in range(5500)
    ]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        chroma_path = str(Path(tmp) / "chroma")
        doc_path = str(Path(tmp) / "documents.json")
        with open(doc_path, "w") as f:
            json.dump(docs, f)
        _build_test_index(docs, chroma_path)
        assert health_check(chroma_path=chroma_path, documents_path=doc_path) == 0
