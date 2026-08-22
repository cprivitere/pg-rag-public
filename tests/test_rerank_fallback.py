"""Tests the pgrag.rag.retriever._rerank_or_cross_encoder fallback: reranker
server failures fall back to lexical score order and are recorded in stats,
while the cross-encoder path reorders results by reranker scores."""

from pgrag.rag.retriever import _rerank_or_cross_encoder, _term_overlap
from pgrag.rag import reranker_client


IDS = ["a", "b", "c"]
DOCS = ["cheese wheel", "aged cheddar cheese", "stale bread"]
METAS = [{"type": "item"}, {"type": "item"}, {"type": "item"}]
DISTS = [0.9, 0.8, 0.7]


def test_fallback_when_server_down(monkeypatch, tmp_path):
    client = reranker_client
    monkeypatch.setattr(client, "STATS_FILE", tmp_path / "stats.json")

    def boom(query, documents, top_n):
        raise reranker_client.RerankError("conn refused")

    monkeypatch.setattr(client, "rerank_documents", boom)

    ids, docs, metas, dists, used = _rerank_or_cross_encoder(
        "cheese", IDS[:], DOCS[:], METAS[:], DISTS[:], 2
    )

    assert used is False
    assert ids[0] == "a"  # lexical score order preserved
    stats = reranker_client.load_stats()
    assert stats["failures"] == 1


def test_fallback_records_even_if_stats_write_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(
        reranker_client,
        "STATS_FILE",
        tmp_path / "missing" / "dir" / "stats.json",
    )
    monkeypatch.setattr(
        reranker_client,
        "rerank_documents",
        lambda *a, **k: (_ for _ in ()).throw(reranker_client.RerankError("x")),
    )
    ids, docs, metas, dists, used = _rerank_or_cross_encoder(
        "cheese", IDS, DOCS, METAS, DISTS, 2
    )
    assert used is False
    assert ids == ["a", "b"]  # lexical order preserved, trimmed to count


def test_cross_encoder_path_uses_overlap(monkeypatch, tmp_path):
    monkeypatch.setattr(reranker_client, "STATS_FILE", tmp_path / "stats.json")

    def fake_rerank(query, documents, top_n):
        return [1, 0, 2]  # b over a

    monkeypatch.setattr(reranker_client, "rerank_documents", fake_rerank)

    ids, docs, metas, dists, used = _rerank_or_cross_encoder(
        "cheese", IDS, DOCS, METAS, DISTS, 2
    )

    assert used is True
    assert ids[0] == "b"
    stats = reranker_client.load_stats()
    assert stats["failures"] == 0
    assert stats["last_success"] is not None


def test_skip_call_when_pool_at_or_below_count(monkeypatch):
    # retriever only invokes reranker when len(ids) > count — simulate no-op
    called = []

    def fake_rerank(*a, **k):
        called.append(True)
        return list(range(len(a[1])))

    monkeypatch.setattr(reranker_client, "rerank_documents", fake_rerank)
    ids, docs, metas, dists, used = _rerank_or_cross_encoder(
        "cheese", ["a"], ["cheese"], [{}], [0.5], 3
    )
    assert used is True
    assert ids == ["a"]