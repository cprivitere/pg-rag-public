"""Opt-in retrieval trace: retrieve() records per-stage ids; ask() fills
top-level trace keys without changing the returned payload."""

from pgrag.rag import retriever


class _FakeCollection:
    def query(self, **kwargs):
        return {
            "ids": [["d1", "d2", "d3", "d4", "d5"]],
            "documents": [["a", "b", "c", "d", "e"]],
            "metadatas": [
                [
                    {"type": "recipe", "name": "a"},
                    {"type": "recipe", "name": "b"},
                    {"type": "recipe", "name": "c"},
                    {"type": "recipe", "name": "d"},
                    {"type": "recipe", "name": "e"},
                ]
            ],
            "distances": [[0.1, 0.2, 0.3, 0.4, 0.5]],
        }


class _FakeClient:
    def get_collection(self, name):
        return _FakeCollection()


class _FakeChroma:
    class PersistentClient(_FakeClient):
        def __init__(self, path=None):
            pass


def _install_retrieve_harness(monkeypatch):
    monkeypatch.setattr(retriever, "correct_query", lambda q: q)
    monkeypatch.setattr(retriever, "embed_text", lambda q: [0.0] * 384)
    monkeypatch.setattr(retriever, "chromadb", _FakeChroma)

    def _load_bm25():
        class _Model:
            def search(self, q, k):
                return [0, 1, 2, 3, 4], None

        all_docs = [{"id": f"d{i + 1}"} for i in range(5)]
        return _Model(), all_docs

    # retrieve() imports load_bm25_index locally from pgrag.rag.bm25
    from pgrag.rag import bm25

    monkeypatch.setattr(bm25, "load_bm25_index", _load_bm25)

    def _rerank(q, ids, docs, metas, dists, count, name_query=None):
        return ids, docs, metas, dists, True

    monkeypatch.setattr(retriever, "_rerank_or_cross_encoder", _rerank)


def test_retrieve_trace_captures_every_stage(monkeypatch):
    _install_retrieve_harness(monkeypatch)
    trace = {}
    retriever.retrieve(
        "silk recipe",
        count=3,
        metadata_filter={"type": "recipe"},
        rerank=True,
        hybrid=True,
        trace=trace,
    )

    assert len(trace["retrieval_calls"]) == 1
    rec = trace["retrieval_calls"][0]
    for key in (
        "query",
        "hybrid",
        "metadata_filter",
        "dense_ids",
        "dense_dists",
        "bm25_ids",
        "rrf_ids",
        "post_filter_ids",
        "reranked_ids",
        "rerank_used",
    ):
        assert key in rec, f"missing trace key: {key}"
    assert rec["query"] == "silk recipe"
    assert rec["hybrid"] is True
    assert rec["metadata_filter"] == {"type": "recipe"}
    assert rec["rerank_used"] is True


def test_ask_trace_does_not_change_payload(monkeypatch):
    from pgrag.rag import pipeline

    monkeypatch.setattr("pgrag.rag.pipeline.classify_query", lambda q: "general")
    monkeypatch.setattr("pgrag.rag.pipeline.should_synthesize", lambda *a, **k: False)

    def _fake_retrieve(question, **kwargs):
        return {
            "ids": [["s1"]],
            "documents": [["First snippet."]],
            "metadatas": [[{"name": "N"}]],
            "distances": [[0.3]],
            "rerank_used": False,
        }

    monkeypatch.setattr("pgrag.rag.pipeline.retrieve", _fake_retrieve)
    monkeypatch.setattr("pgrag.rag.pipeline.generate", lambda prompt: "answer")

    plain = pipeline.ask("silk recipe")
    traced = {}
    traced_result = pipeline.ask("silk recipe", trace=traced)

    assert traced_result["answer"] == plain["answer"]
    assert traced_result["documents"] == plain["documents"]
    assert traced["query"] == "silk recipe"
    assert traced["classifier"] == "general"
    assert traced["generation"] == {}
    assert traced["corrected_query"] == "silk recipe"
