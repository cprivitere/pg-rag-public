"""Wiki page context expansion (Step 5 of metadata-bm25-eval; moved to
`pgrag.rag.resolve` as part of the deferred "request more" build).

General queries retrieve only the top chunks of a wiki page; the answer often
sits in a sibling chunk. `expand_parents` pulls sibling chunks via
`parent_id` — bounded (2 pages, ~16k chars), id-deduped, non-wiki untouched.
"""

from unittest.mock import patch

from pgrag.rag import resolve
from pgrag.rag.pipeline import _prepare_general


def _docs(ids, parent="wiki_Mushroom Farming"):
    return [
        {
            "id": i,
            "type": "wiki",
            "text": f"text {i} " + ("x" * 200),
            "metadata": {"source": "wiki", "parent_id": parent},
        }
        for i in ids
    ]


def _patch_index(monkeypatch, docs):
    index = {}
    for d in docs:
        index.setdefault(d["metadata"].get("parent_id"), []).append(d)
    monkeypatch.setattr(
        "pgrag.rag.resolve.load_parent_index",
        lambda: ({d["id"]: d for d in docs}, index),
    )


def test_expansion_pulls_siblings(monkeypatch):
    docs = _docs(
        [
            "wiki_Mushroom Farming_Mechanics_chunk_0",
            "wiki_Mushroom Farming_Mechanics_chunk_1",
            "wiki_Mushroom Farming_Mechanics_chunk_3",
        ]
    )
    _patch_index(monkeypatch, docs)
    r = docs[0]
    ids, texts, metas, dists = resolve.expand_parents(
        [r["id"]],
        [r["text"]],
        [r["metadata"]],
        [0.5],
    )
    assert len(ids) == 3
    assert "wiki_Mushroom Farming_Mechanics_chunk_1" in ids
    assert "wiki_Mushroom Farming_Mechanics_chunk_3" in ids
    # appended docs get a placeholder distance, aligned by index
    assert len(ids) == len(texts) == len(metas) == len(dists)
    assert dists[0] == 0.5
    assert all(d == 1.0 for d in dists[1:])


def test_expansion_dedupes_retrieved_ids(monkeypatch):
    docs = _docs(["wiki_Page_A", "wiki_Page_B", "wiki_Page_C"], parent="P")
    _patch_index(monkeypatch, docs)
    retrieved = docs[:2]
    ids, _, _, _ = resolve.expand_parents(
        [r["id"] for r in retrieved],
        [r["text"] for r in retrieved],
        [r["metadata"] for r in retrieved],
        [0.1, 0.2],
    )
    # sibling C is spliced right after its parent A, then B
    assert ids == ["wiki_Page_A", "wiki_Page_C", "wiki_Page_B"]


def test_expansion_bounded_to_two_pages(monkeypatch):
    retrieved = [
        {"id": f"id_{i}", "type": "wiki", "text": "t", "metadata": {"parent_id": f"wiki_Page_{i}"}}
        for i in range(4)
    ]
    siblings = [
        {
            "id": f"id_{i}_sib",
            "type": "wiki",
            "text": f"s{i}",
            "metadata": {"parent_id": f"wiki_Page_{i}"},
        }
        for i in range(4)
    ]
    _patch_index(monkeypatch, retrieved + siblings)
    ids, _, _, _ = resolve.expand_parents(
        [r["id"] for r in retrieved],
        [r["text"] for r in retrieved],
        [r["metadata"] for r in retrieved],
        [0.0] * 4,
    )
    # only first 2 parents are expanded
    assert "id_0_sib" in ids
    assert "id_1_sib" in ids
    assert "id_2_sib" not in ids
    assert "id_3_sib" not in ids


def test_expansion_respects_char_cap(monkeypatch):
    big = {
        "id": "wiki_Big_chunk",
        "type": "wiki",
        "text": "z" * 9000,
        "metadata": {"parent_id": "wiki_Other"},
    }
    _patch_index(monkeypatch, [big])
    ids, texts, _, _ = resolve.expand_parents(
        ["wiki_Other"],
        ["other"],
        [{"parent_id": "wiki_Other"}],
        [0.0],
        max_chars=100,
    )
    # 9000 > budget -> dropped
    assert ids == ["wiki_Other"]
    assert sum(len(t) for t in texts) < 16000


def test_expansion_non_wiki_untouched(monkeypatch):
    _patch_index(monkeypatch, [])
    ids, texts, metas, dists = resolve.expand_parents(
        ["item_1"],
        ["item text"],
        [{"source": "cdn", "table": "items"}],
        [0.3],
    )
    assert ids == ["item_1"]
    assert texts == ["item text"]
    assert dists == [0.3]


@patch("pgrag.rag.pipeline.expand_parents")
@patch("pgrag.rag.pipeline.should_synthesize", return_value=False)
@patch("pgrag.rag.pipeline.retrieve")
def test_prepare_general_expands_wiki(mock_retrieve, mock_synth, mock_expand):
    meta = [{"source": "wiki", "parent_id": "wiki_Mushroom Farming"}]
    mock_retrieve.return_value = {
        "ids": [["wiki_chunk_0"]],
        "documents": [["chunk text"]],
        "metadatas": [meta],
        "distances": [[0.4]],
        "rerank_used": False,
    }
    mock_expand.return_value = (
        ["wiki_chunk_0", "wiki_chunk_3"],
        ["chunk text", "sibling"],
        [meta[0], {"source": "wiki", "parent_id": "wiki_Mushroom Farming"}],
        [0.4, 1.0],
    )

    _, ids, docs, _, dists, _ = _prepare_general("How do I grow mushrooms?", "general")
    assert ids == ["wiki_chunk_0", "wiki_chunk_3"]
    assert docs == ["chunk text", "sibling"]
    mock_expand.assert_called_once()


@patch("pgrag.rag.pipeline.expand_parents")
@patch("pgrag.rag.pipeline.should_synthesize", return_value=False)
@patch("pgrag.rag.pipeline.retrieve")
def test_prepare_lookup_uses_wide_retrieval(mock_retrieve, mock_synth, mock_expand):
    """'Where can I find X?' classifies as lookup; it must get hybrid recall
    + wiki expansion like general, not a 3-doc dense-only window."""
    meta = [{"source": "wiki", "parent_id": "wiki_Field Mushroom"}]
    mock_retrieve.return_value = {
        "ids": [["wiki_chunk_0"]],
        "documents": [["chunk text"]],
        "metadatas": [meta],
        "distances": [[0.4]],
        "rerank_used": False,
    }
    mock_expand.return_value = (
        ["wiki_chunk_0"],
        ["chunk text"],
        meta,
        [0.4],
    )

    _prepare_general("Where can I find Field Mushrooms?", "lookup")
    kw = mock_retrieve.call_args.kwargs
    assert kw["hybrid"] is True
    # Wide (lookup/general) queries pull a Top-K of 40 for hybrid RRF recall,
    # not a 3-doc dense-only window (bumped by the "Raise context budget/Top-K"
    # change; expansion then widens wiki pages within it).
    assert kw["count"] == 40
    mock_expand.assert_called_once()


def test_expansion_skips_when_no_wiki_parent(monkeypatch):
    _patch_index(monkeypatch, [])
    ids, texts, metas, dists = resolve.expand_parents(
        ["skill_Alchemy"],
        ["alchemy text"],
        [{"source": "cdn", "table": "skills"}],
        [0.2],
    )
    assert ids == ["skill_Alchemy"]
    assert texts == ["alchemy text"]
