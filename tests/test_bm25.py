"""Tests BM25 ranking, the hybrid dense+BM25 fusion (_hybrid_fuse) and
routing, and the HYBRID_MULTIPLIER/RRF_K constants and result-count choices in
pgrag.rag.retriever.retrieve."""

from unittest.mock import patch, MagicMock, ANY

from pgrag.rag.bm25 import BM25
from pgrag.rag.retriever import _hybrid_fuse, retrieve, HYBRID_MULTIPLIER, RRF_K


def test_bm25_rank_known_doc_highest():
    model = BM25()
    model.index([
        "potion recipe for healing wounds",
        "sword sharpening with whetstone",
        "potion brewing tips for beginners",
    ])
    indices, scores = model.search("potion", k=3)
    assert indices[0] == 0 or indices[0] == 2


def test_bm25_empty_query():
    model = BM25()
    model.index(["doc a", "doc b"])
    indices, scores = model.search("", k=2)
    assert len(indices) == 0


def test_bm25_no_match():
    model = BM25()
    model.index(["potion recipe", "sword guide"])
    indices, scores = model.search("baking", k=2)
    assert len(indices) == 0 or scores[0] == 0.0


def test_bm25_case_insensitive():
    model = BM25()
    model.index(["Potion Recipe For Healing"])
    indices, scores = model.search("potion", k=1)
    assert len(indices) == 1


def test_bm25_k_respected():
    model = BM25()
    model.index([
        "doc a content here",
        "doc b content here",
        "doc c content here",
        "doc d content here",
        "doc e content here",
    ])
    indices, scores = model.search("content", k=3)
    assert len(indices) == 3


def test_hybrid_fuse_intersection():
    dense_ids = ["a", "b", "c", "d"]
    dense_texts = ["text a", "text b", "text c", "text d"]
    dense_metas = [{"source": "cdn"}] * 4
    dense_dists = [0.1, 0.2, 0.3, 0.4]
    bm25_ids = ["b", "d", "a"]
    all_docs = [
        {"id": "a", "text": "text a", "metadata": {"source": "cdn"}},
        {"id": "b", "text": "text b", "metadata": {"source": "cdn"}},
        {"id": "c", "text": "text c", "metadata": {"source": "cdn"}},
        {"id": "d", "text": "text d", "metadata": {"source": "cdn"}},
    ]

    ids, texts, metas, dists = _hybrid_fuse(
        dense_ids, dense_texts, dense_metas, dense_dists,
        bm25_ids, all_docs, 2,
    )

    assert len(ids) == 2


def test_hybrid_fuse_bm25_only_doc():
    dense_ids = ["a", "b"]
    dense_texts = ["text a", "text b"]
    dense_metas = [{"source": "cdn"}] * 2
    dense_dists = [0.1, 0.2]
    bm25_ids = ["b", "c"]
    all_docs = [
        {"id": "a", "text": "text a", "metadata": {"source": "cdn"}},
        {"id": "b", "text": "text b", "metadata": {"source": "cdn"}},
        {"id": "c", "text": "text c from bm25", "metadata": {"source": "wiki"}},
    ]

    ids, texts, metas, dists = _hybrid_fuse(
        dense_ids, dense_texts, dense_metas, dense_dists,
        bm25_ids, all_docs, 3,
    )

    assert "c" in ids
    c_idx = ids.index("c")
    assert texts[c_idx] == "text c from bm25"
    assert dists[c_idx] == 0.0


def test_hybrid_multiplier_constant():
    assert HYBRID_MULTIPLIER == 3


def test_hybrid_default_disabled():
    from pgrag.rag.retriever import retrieve
    import inspect
    src = inspect.signature(retrieve)
    assert "hybrid" in src.parameters
    assert src.parameters["hybrid"].default is False


@patch("pgrag.rag.retriever.embed_text")
@patch("pgrag.rag.retriever.chromadb.PersistentClient")
@patch("pgrag.rag.bm25.load_bm25_index")
def test_retrieve_hybrid_routes_to_bm25(mock_load, mock_client, mock_embed):
    mock_embed.return_value = [0.1] * 128
    mock_col = MagicMock()
    mock_client.return_value.get_collection.return_value = mock_col
    mock_col.query.return_value = {
        "ids": [["a", "b", "c", "d"]],
        "documents": [["ta", "tb", "tc", "td"]],
        "metadatas": [[{"s": "c"}] * 4],
        "distances": [[0.1, 0.2, 0.3, 0.4]],
    }

    bm25_model = MagicMock()
    bm25_model.search.return_value = ([0, 1], [2.5, 1.5])
    all_docs = [
        {"id": "a", "text": "ta", "metadata": {"s": "c"}},
        {"id": "b", "text": "tb", "metadata": {"s": "c"}},
        {"id": "c", "text": "tc", "metadata": {"s": "c"}},
        {"id": "d", "text": "td", "metadata": {"s": "c"}},
    ]
    mock_load.return_value = (bm25_model, all_docs)

    results = retrieve("test", count=3, hybrid=True)

    bm25_model.search.assert_called_once_with("test", k=3 * HYBRID_MULTIPLIER)
    assert len(results["ids"][0]) == 3


@patch("pgrag.rag.retriever.embed_text")
@patch("pgrag.rag.retriever.chromadb.PersistentClient")
def test_retrieve_comparison_uses_higher_count(mock_client, mock_embed):
    mock_embed.return_value = [0.1] * 128
    mock_col = MagicMock()
    mock_client.return_value.get_collection.return_value = mock_col
    mock_col.query.return_value = {
        "ids": [["a"] * 20],
        "documents": [["t"] * 20],
        "metadatas": [[{"s": "c"}] * 20],
        "distances": [[0.1] * 20],
    }

    results = retrieve("highest level cheese", count=3, query_type="comparison", rerank=False)

    call_kwargs = mock_col.query.call_args[1]
    assert call_kwargs["n_results"] == 20


@patch("pgrag.rag.retriever.embed_text")
@patch("pgrag.rag.retriever.chromadb.PersistentClient")
def test_retrieve_general_uses_default_count(mock_client, mock_embed):
    mock_embed.return_value = [0.1] * 128
    mock_col = MagicMock()
    mock_client.return_value.get_collection.return_value = mock_col
    mock_col.query.return_value = {
        "ids": [["a", "b", "c"]],
        "documents": [["t1", "t2", "t3"]],
        "metadatas": [[{"s": "c"}] * 3],
        "distances": [[0.1, 0.2, 0.3]],
    }

    results = retrieve("tell me about cheese", count=3, query_type="general", rerank=False)

    call_kwargs = mock_col.query.call_args[1]
    assert call_kwargs["n_results"] == 3
