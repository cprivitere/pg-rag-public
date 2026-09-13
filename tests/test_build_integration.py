"""Tests pgrag.vectorstore.build_index.build_index end-to-end against a temp
Chroma collection: correct embedding dims, purging deleted docs, metadata-only
re-image of reembeds, dimension-mismatch aborts, per-batch interleaving,
interruption resume, and partial --source rebuilds."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import chromadb
import pytest
from chromadb.api.models.Collection import Collection

from pgrag.config import EMBEDDING_DIM
from pgrag.vectorstore.build_index import build_index


def fake_embed_batch(texts):
    return [[0.1] * EMBEDDING_DIM for _ in texts]


TMP_KW = {"ignore_cleanup_errors": True}


@patch("pgrag.vectorstore.build_index.embed_batch", side_effect=fake_embed_batch)
def test_build_upserts_docs_with_correct_dim(mock_embed):
    docs = [
        {
            "id": "a",
            "type": "item",
            "text": "alpha",
            "metadata": {"source": "cdn", "table": "items"},
        },
        {
            "id": "b",
            "type": "item",
            "text": "beta",
            "metadata": {"source": "cdn", "table": "items"},
        },
    ]
    with tempfile.TemporaryDirectory(**TMP_KW) as tmp:
        chroma_path = str(Path(tmp) / "chroma")
        build_index(documents=docs, chroma_path=chroma_path)
        client = chromadb.PersistentClient(path=chroma_path)
        coll = client.get_collection("project_gorgon")
        assert coll.count() == 2
        result = coll.get(include=["embeddings", "metadatas"])
        assert set(result["ids"]) == {"a", "b"}
        assert len(result["embeddings"][0]) == EMBEDDING_DIM


@patch("pgrag.vectorstore.build_index.embed_batch", side_effect=fake_embed_batch)
def test_build_deleted_doc_purged(mock_embed):
    docs_a = [
        {
            "id": "a",
            "type": "item",
            "text": "alpha",
            "metadata": {"source": "cdn", "table": "items"},
        },
        {
            "id": "b",
            "type": "item",
            "text": "beta",
            "metadata": {"source": "cdn", "table": "items"},
        },
    ]
    docs_b = [
        {
            "id": "a",
            "type": "item",
            "text": "alpha",
            "metadata": {"source": "cdn", "table": "items"},
        },
    ]
    with tempfile.TemporaryDirectory(**TMP_KW) as tmp:
        chroma_path = str(Path(tmp) / "chroma")
        build_index(documents=docs_a, chroma_path=chroma_path)
        build_index(documents=docs_b, chroma_path=chroma_path)
        client = chromadb.PersistentClient(path=chroma_path)
        coll = client.get_collection("project_gorgon")
        assert coll.count() == 1
        assert coll.get()["ids"] == ["a"]


@patch("pgrag.vectorstore.build_index.embed_batch", side_effect=fake_embed_batch)
def test_build_metadata_only_skips_reembed(mock_embed):
    docs_first = [
        {
            "id": "x",
            "type": "item",
            "text": "same",
            "metadata": {"source": "cdn", "table": "items", "name": "old"},
        },
    ]
    docs_second = [
        {
            "id": "x",
            "type": "item",
            "text": "same",
            "metadata": {"source": "cdn", "table": "items", "name": "new"},
        },
    ]
    with tempfile.TemporaryDirectory(**TMP_KW) as tmp:
        chroma_path = str(Path(tmp) / "chroma")
        build_index(documents=docs_first, chroma_path=chroma_path)
        build_index(documents=docs_second, chroma_path=chroma_path)
        client = chromadb.PersistentClient(path=chroma_path)
        coll = client.get_collection("project_gorgon")
        result = coll.get(include=["metadatas"])
        assert result["metadatas"][0]["name"] == "new"
        assert coll.count() == 1


def test_build_dimension_mismatch_aborts():
    docs_first = [
        {
            "id": "a",
            "type": "item",
            "text": "alpha",
            "metadata": {"source": "cdn", "table": "items"},
        },
    ]
    docs_second = [
        {
            "id": "b",
            "type": "item",
            "text": "beta",
            "metadata": {"source": "cdn", "table": "items"},
        },
    ]
    with tempfile.TemporaryDirectory(**TMP_KW) as tmp:
        chroma_path = str(Path(tmp) / "chroma")
        with (
            patch("pgrag.vectorstore.build_index.embed_batch", return_value=[[0.1, 0.2, 0.3, 0.4]]),
            patch("pgrag.vectorstore.build_index.EMBEDDING_DIM", 4),
        ):
            build_index(documents=docs_first, chroma_path=chroma_path)
        with (
            patch("pgrag.vectorstore.build_index.embed_batch", return_value=[[0.1, 0.2]]),
            patch("pgrag.vectorstore.build_index.EMBEDDING_DIM", 4),
            pytest.raises(Exception, match="expected 4"),
        ):
            build_index(documents=docs_second, chroma_path=chroma_path)


def test_v23_build_start_aborts_on_dim_mismatch():
    docs_first = [
        {
            "id": "a",
            "type": "item",
            "text": "alpha",
            "metadata": {"source": "cdn", "table": "items"},
        },
    ]
    docs_second = [
        {
            "id": "b",
            "type": "item",
            "text": "beta",
            "metadata": {"source": "cdn", "table": "items"},
        },
    ]
    with tempfile.TemporaryDirectory(**TMP_KW) as tmp:
        chroma_path = str(Path(tmp) / "chroma")
        with (
            patch("pgrag.vectorstore.build_index.embed_batch", return_value=[[0.1, 0.2, 0.3, 0.4]]),
            patch("pgrag.vectorstore.build_index.EMBEDDING_DIM", 4),
        ):
            build_index(documents=docs_first, chroma_path=chroma_path)
        with (
            patch(
                "pgrag.vectorstore.build_index.embed_batch",
                side_effect=AssertionError("must not embed"),
            ),
            pytest.raises(ValueError, match="EMBEDDING_DIM"),
        ):
            build_index(documents=docs_second, chroma_path=chroma_path)


def test_build_interleaves_embed_and_upsert_per_batch():
    docs = [
        {"id": f"d{i}", "type": "item", "text": f"text {i}", "metadata": {"source": "cdn"}}
        for i in range(9)
    ]
    real_upsert = Collection.upsert
    events = []

    def tracked_embed(texts):
        events.append(("embed", len(texts)))
        return [[0.1] * EMBEDDING_DIM for _ in texts]

    def tracked_upsert(self, ids=None, embeddings=None, metadatas=None, documents=None, **kw):
        events.append(("upsert", len(ids)))
        return real_upsert(
            self, ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents, **kw
        )

    with tempfile.TemporaryDirectory(**TMP_KW) as tmp:
        chroma_path = str(Path(tmp) / "chroma")
        with (
            patch("pgrag.vectorstore.build_index.embed_batch", side_effect=tracked_embed),
            patch("pgrag.vectorstore.build_index.EMBED_BATCH_SIZE", 4),
            patch("pgrag.vectorstore.build_index.BATCH_SIZE", 2),
            patch.object(Collection, "upsert", side_effect=tracked_upsert, autospec=True),
        ):
            build_index(documents=docs, chroma_path=chroma_path)

    # each embed batch must be upserted before the next embed batch starts
    assert events == [
        ("embed", 4),
        ("upsert", 2),
        ("upsert", 2),
        ("embed", 4),
        ("upsert", 2),
        ("upsert", 2),
        ("embed", 1),
        ("upsert", 1),
    ]


def test_build_interruption_persists_completed_batches_and_resumes():
    docs = [
        {"id": f"d{i}", "type": "item", "text": f"text {i}", "metadata": {"source": "cdn"}}
        for i in range(8)
    ]
    embed_calls = {"n": 0}

    def failing_embed(texts):
        embed_calls["n"] += 1
        if embed_calls["n"] > 1:
            raise RuntimeError("simulated crash")
        return [[0.1] * EMBEDDING_DIM for _ in texts]

    with tempfile.TemporaryDirectory(**TMP_KW) as tmp:
        chroma_path = str(Path(tmp) / "chroma")
        with (
            patch("pgrag.vectorstore.build_index.embed_batch", side_effect=failing_embed),
            patch("pgrag.vectorstore.build_index.EMBED_BATCH_SIZE", 4),
            patch("pgrag.vectorstore.build_index.BATCH_SIZE", 4),
            pytest.raises(RuntimeError, match="simulated crash"),
        ):
            build_index(documents=docs, chroma_path=chroma_path)

        client = chromadb.PersistentClient(path=chroma_path)
        coll = client.get_collection("project_gorgon")
        # first batch persisted despite the crash
        assert coll.count() == 4

        # resume: only the remaining docs are embedded, not re-embedding persisted ones
        embedded_texts = []

        def record_embed(texts):
            embedded_texts.extend(texts)
            return [[0.1] * EMBEDDING_DIM for _ in texts]

        with (
            patch("pgrag.vectorstore.build_index.embed_batch", side_effect=record_embed),
            patch("pgrag.vectorstore.build_index.EMBED_BATCH_SIZE", 4),
            patch("pgrag.vectorstore.build_index.BATCH_SIZE", 4),
        ):
            build_index(documents=docs, chroma_path=chroma_path)

        assert embedded_texts == ["text 4", "text 5", "text 6", "text 7"]
        assert coll.count() == 8


@patch("pgrag.vectorstore.build_index.embed_batch", side_effect=fake_embed_batch)
def test_partial_rebuild_source_keeps_other_sources(mock_embed):
    """--source wiki rebuilds only wiki docs; CDN vectors stay untouched."""
    docs_all = [
        {
            "id": "c1",
            "type": "item",
            "text": "cdn one",
            "metadata": {"source": "cdn", "table": "items"},
        },
        {
            "id": "c2",
            "type": "item",
            "text": "cdn two",
            "metadata": {"source": "cdn", "table": "items"},
        },
        {"id": "w1", "type": "wiki", "text": "wiki one", "metadata": {"source": "wiki"}},
    ]
    docs_wiki_changed = [
        {
            "id": "c1",
            "type": "item",
            "text": "cdn one",
            "metadata": {"source": "cdn", "table": "items"},
        },
        {
            "id": "c2",
            "type": "item",
            "text": "cdn two",
            "metadata": {"source": "cdn", "table": "items"},
        },
        {"id": "w1", "type": "wiki", "text": "wiki one EDITED", "metadata": {"source": "wiki"}},
        {"id": "w2", "type": "wiki", "text": "wiki two", "metadata": {"source": "wiki"}},
    ]
    embedded = []

    def record_embed(texts):
        embedded.extend(texts)
        return [[0.1] * EMBEDDING_DIM for _ in texts]

    with tempfile.TemporaryDirectory(**TMP_KW) as tmp:
        chroma_path = str(Path(tmp) / "chroma")
        build_index(documents=docs_all, chroma_path=chroma_path)

        with patch("pgrag.vectorstore.build_index.embed_batch", side_effect=record_embed):
            build_index(documents=docs_wiki_changed, chroma_path=chroma_path, source="wiki")

        client = chromadb.PersistentClient(path=chroma_path)
        coll = client.get_collection("project_gorgon")
        result = coll.get(include=["documents"])
        assert set(result["ids"]) == {"c1", "c2", "w1", "w2"}
        assert sorted(e for e in embedded) == ["wiki one EDITED", "wiki two"]
        texts = dict(zip(result["ids"], result["documents"], strict=False))
        assert texts["w1"] == "wiki one EDITED"


@patch("pgrag.vectorstore.build_index.embed_batch", side_effect=fake_embed_batch)
def test_partial_rebuild_source_deletes_only_that_source(mock_embed):
    """Stale docs of the scoped source are purged; other sources are not touched."""
    docs_all = [
        {
            "id": "c1",
            "type": "item",
            "text": "cdn one",
            "metadata": {"source": "cdn", "table": "items"},
        },
        {
            "id": "c2",
            "type": "item",
            "text": "cdn two",
            "metadata": {"source": "cdn", "table": "items"},
        },
        {"id": "w1", "type": "wiki", "text": "wiki one", "metadata": {"source": "wiki"}},
        {"id": "w2", "type": "wiki", "text": "wiki two", "metadata": {"source": "wiki"}},
    ]
    # w1 is dropped from documents; cdn docs are absent too (out of scope)
    docs_wiki_only = [
        {"id": "w2", "type": "wiki", "text": "wiki two", "metadata": {"source": "wiki"}},
    ]
    with tempfile.TemporaryDirectory(**TMP_KW) as tmp:
        chroma_path = str(Path(tmp) / "chroma")
        build_index(documents=docs_all, chroma_path=chroma_path)

        embedded = []
        with patch(
            "pgrag.vectorstore.build_index.embed_batch",
            side_effect=lambda texts: (
                embedded.extend(texts) or [[0.1] * EMBEDDING_DIM for _ in texts]
            ),
        ):
            build_index(documents=docs_wiki_only, chroma_path=chroma_path, source="wiki")

        client = chromadb.PersistentClient(path=chroma_path)
        coll = client.get_collection("project_gorgon")
        assert set(coll.get()["ids"]) == {"c1", "c2", "w2"}, (
            "only stale wiki docs should be purged; cdn docs must survive"
        )
        assert embedded == []


@patch("pgrag.vectorstore.build_index.embed_batch", side_effect=fake_embed_batch)
def test_partial_rebuild_unknown_source_aborts(mock_embed):
    docs = [
        {"id": "c1", "type": "item", "text": "cdn one", "metadata": {"source": "cdn"}},
    ]
    with tempfile.TemporaryDirectory(**TMP_KW) as tmp:
        chroma_path = str(Path(tmp) / "chroma")
        with pytest.raises(ValueError, match="No documents with source='bogus'"):
            build_index(documents=docs, chroma_path=chroma_path, source="bogus")
