"""Post-fusion metadata filter (Step 3 of metadata-bm25-eval).

The dense Chroma path supports `$eq/$ne/$gt/$gte/$lt/$lte` natively, but BM25
fused hits bypass the `where` clause, so retrieve() re-filters afterwards.
That post-fusion pass must:
- evaluate operator dicts instead of blind equality (operator dicts were
  previously compared with `==` and matched nothing)
- match delimited (`" | "`-joined) string fields by token membership
- pass operator dicts through to Chroma untouched
"""
from unittest.mock import MagicMock, patch

from pgrag.rag.retriever import _where_matches, _scalar_eq, retrieve


# --- predicate unit tests ---


def test_where_eq_match():
    assert _where_matches({"skill": "Alchemy"}, {"skill": "Alchemy"})
    assert not _where_matches({"skill": "Alchemy"}, {"skill": "Mycology"})
    assert not _where_matches({"skill": "Alchemy"}, {"skill": None})


def test_where_missing_key_fails():
    assert not _where_matches(
        {"skill": "Alchemy"}, {"skill_level_req": 25}
    )


def test_where_lte_gte():
    meta = {"skill_level_req": 25}
    assert _where_matches(meta, {"skill_level_req": {"$lte": 25}})
    assert _where_matches(meta, {"skill_level_req": {"$lte": 30}})
    assert not _where_matches(meta, {"skill_level_req": {"$lte": 24}})
    assert _where_matches(meta, {"skill_level_req": {"$gte": 25}})
    assert not _where_matches(meta, {"skill_level_req": {"$gte": 26}})


def test_where_gt_lt_ne():
    meta = {"skill_level_req": 25}
    assert _where_matches(meta, {"skill_level_req": {"$gt": 24}})
    assert not _where_matches(meta, {"skill_level_req": {"$gt": 25}})
    assert _where_matches(meta, {"skill_level_req": {"$lt": 26}})
    assert not _where_matches(meta, {"skill_level_req": {"$lt": 25}})
    assert _where_matches(meta, {"skill_level_req": {"$ne": 30}})
    assert not _where_matches(meta, {"skill_level_req": {"$ne": 25}})


def test_where_delimited_membership():
    meta = {"ingredients": "Spider Silk | Toadstool Cap"}
    assert _where_matches(meta, {"ingredients": "Spider Silk"})
    assert _where_matches(meta, {"ingredients": "Toadstool Cap"})
    assert not _where_matches(meta, {"ingredients": "Banana"})
    assert not _where_matches(meta, {"ingredients": "Silk"})


def test_scalar_eq_handles_numeric_strings():
    assert _scalar_eq(25, "25")
    assert not _scalar_eq(25, "26")
    assert not _scalar_eq("not-a-number", "5")


def test_where_compound_and_or():
    meta = {"skill": "Alchemy", "skill_level_req": 25}
    assert _where_matches(meta, {
        "$and": [
            {"skill": "Alchemy"},
            {"skill_level_req": {"$gte": 20}},
        ]
    })
    assert not _where_matches(meta, {
        "$and": [{"skill": "Alchemy"}, {"skill_level_req": {"$gte": 30}}]
    })
    assert _where_matches(meta, {
        "$or": [{"skill": "Baking"}, {"skill_level_req": {"$gte": 20}}]
    })
    assert not _where_matches(meta, {
        "$or": [{"skill": "Baking"}, {"skill_level_req": {"$gte": 30}}]
    })


# --- retrieve() pass-through + fused re-filter ---


@patch("pgrag.rag.retriever.embed_text")
@patch("pgrag.rag.retriever.chromadb.PersistentClient")
def test_retrieve_passes_operator_filter_to_chroma(mock_client, mock_embed):
    mock_embed.return_value = [0.1] * 384
    inst = mock_client.return_value
    col = inst.get_collection.return_value
    col.query.return_value = {
        "ids": [["a"]],
        "documents": [["recipe"]],
        "metadatas": [[{"skill_level_req": 25}]],
        "distances": [[0.1]],
    }

    retrieve(
        "crafting question",
        count=3,
        metadata_filter={"skill_level_req": {"$lte": 25}},
    )
    kwargs = col.query.call_args.kwargs
    assert kwargs["where"] == {"skill_level_req": {"$lte": 25}}


@patch("pgrag.rag.retriever.embed_text")
@patch("pgrag.rag.retriever.chromadb.PersistentClient")
@patch("pgrag.rag.bm25.load_bm25_index")
def test_retrieve_token_filter_post_fusion_only(
    mock_load, mock_client, mock_embed
):
    """A delimited-token filter (ingredient) cannot go to Chroma $contains;
    it only narrows the fused results, and Chroma where stays unset."""
    mock_embed.return_value = [0.1] * 384
    inst = mock_client.return_value
    col = inst.get_collection.return_value
    col.query.return_value = {
        "ids": [["dense_1", "dense_2"]],
        "documents": [["recipe A", "recipe B"]],
        "metadatas": [[
            {"type": "recipe", "ingredients": "Spider Silk | Mushroom"},
            {"type": "recipe", "ingredients": "Flax Cloth"},
        ]],
        "distances": [[0.2, 0.3]],
    }

    bm25_docs = [
        {"id": "dense_1", "text": "a",
         "metadata": {"type": "recipe", "ingredients": "Spider Silk | Mushroom"}},
        {"id": "dense_2", "text": "b",
         "metadata": {"type": "recipe", "ingredients": "Flax Cloth"}},
    ]
    fake_model = MagicMock()
    fake_model.search.return_value = ([1, 0], [0.9, 0.8])
    mock_load.return_value = (fake_model, bm25_docs)

    results = retrieve(
        "recipes using mushrooms",
        count=3,
        token_filter={"ingredients": "Mushroom"},
        hybrid=True,
    )
    # No scalar native filter -> Chroma where untouched.
    assert "where" not in col.query.call_args.kwargs
    kept = results["metadatas"][0]
    kept_ingredients = [m.get("ingredients") for m in kept]
    assert "Spider Silk | Mushroom" in kept_ingredients
    assert "Flax Cloth" not in kept_ingredients  # no Mushroom -> dropped


@patch("pgrag.rag.retriever.embed_text")
@patch("pgrag.rag.retriever.chromadb.PersistentClient")
@patch("pgrag.rag.bm25.load_bm25_index")
def test_retrieve_native_and_token_filters_agree(
    mock_load, mock_client, mock_embed
):
    """native (scalar, Chroma where) + token (post-fusion) combine: a doc
    must satisfy both."""
    mock_embed.return_value = [0.1] * 384
    inst = mock_client.return_value
    col = inst.get_collection.return_value
    col.query.return_value = {
        "ids": [["dense_1", "dense_2"]],
        "documents": [["a", "b"]],
        "metadatas": [[
            {"type": "recipe", "skill": "Alchemy",
             "ingredients": "Mushroom"},
            {"type": "recipe", "skill": "Cooking",
             "ingredients": "Mushroom"},
        ]],
        "distances": [[0.2, 0.3]],
    }
    bm25_docs = [{"id": d, "text": t, "metadata": m} for d, t, m in
                 zip(col.query.return_value["ids"][0],
                     col.query.return_value["documents"][0],
                     col.query.return_value["metadatas"][0])]
    fake_model = MagicMock()
    fake_model.search.return_value = ([1, 0], [0.9, 0.8])
    mock_load.return_value = (fake_model, bm25_docs)

    results = retrieve(
        "alchemy recipes using mushrooms",
        count=3,
        metadata_filter={"type": "recipe", "skill": "Alchemy"},
        token_filter={"ingredients": "Mushroom"},
        hybrid=True,
    )
    assert col.query.call_args.kwargs["where"] == {
        "type": "recipe", "skill": "Alchemy",
    }
    kept = results["metadatas"][0]
    assert kept == [{"type": "recipe", "skill": "Alchemy",
                     "ingredients": "Mushroom"}]
    assert {"type": "recipe", "skill": "Cooking", "ingredients": "Mushroom"} \
        not in kept


@patch("pgrag.rag.retriever.embed_text")
@patch("pgrag.rag.retriever.chromadb.PersistentClient")
@patch("pgrag.rag.bm25.load_bm25_index")
def test_retrieve_hybrid_filters_fused_docs_by_operator(
    mock_load, mock_client, mock_embed
):
    """BM25-only hits dodge Chroma's where; the post-fusion pass must apply
    the operator clause so `$lte` doesn't silently drop them via `==`."""
    mock_embed.return_value = [0.1] * 384
    inst = mock_client.return_value
    col = inst.get_collection.return_value
    col.query.return_value = {
        "ids": [["dense_1"]],
        "documents": [["dense doc"]],
        "metadatas": [[{"skill_level_req": 40}]],
        "distances": [[0.2]],
    }

    bm25_docs = [
        {"id": "bm25_1", "text": "low-level recipe",
         "metadata": {"skill_level_req": 10}},
        {"id": "bm25_2", "text": "high-level recipe",
         "metadata": {"skill_level_req": 50}},
    ]
    fake_model = MagicMock()
    fake_model.search.return_value = ([1, 0], [0.9, 0.8])  # bm25 hits both
    mock_load.return_value = (fake_model, bm25_docs)

    results = retrieve(
        "recipe question",
        count=3,
        metadata_filter={"skill_level_req": {"$lte": 25}},
        hybrid=True,
    )
    kept = results["metadatas"][0]
    assert {"skill_level_req": 10} in kept  # 10 <= 25 -> kept
    assert {"skill_level_req": 40} not in kept  # dense 40 -> dropped
    assert {"skill_level_req": 50} not in kept  # bm25 50 -> dropped

@patch("pgrag.rag.retriever.embed_text")
@patch("pgrag.rag.retriever.chromadb.PersistentClient")
@patch("pgrag.rag.bm25.load_bm25_index")
def test_token_filter_fallback_on_empty(
    mock_load, mock_client, mock_embed
):
    """When token_filter alone would drop all results, fall back to
    metadata_filter only so the query isn't emptied by an over-narrowing
    ingredient name mismatch (e.g. "Animal Feces" vs "Animal Poop")."""
    mock_embed.return_value = [0.1] * 384
    inst = mock_client.return_value
    col = inst.get_collection.return_value
    # Two docs: one matches metadata_filter (type="recipe"), one does NOT (type="other").
    # Only the one with type="recipe" should survive the fallback.
    col.query.return_value = {
        "ids": [["dense_1", "dense_2"]],
        "documents": [["recipe with Animal Poop", "not a recipe"]],
        "metadatas": [[
            {"type": "recipe", "ingredients": "Animal Poop"},
            {"type": "other", "ingredients": "Spider Silk"},
        ]],
        "distances": [[0.2, 0.3]],
    }

    bm25_docs = [
        {"id": "dense_1", "text": "a",
         "metadata": {"type": "recipe", "ingredients": "Animal Poop"}},
        {"id": "dense_2", "text": "b",
         "metadata": {"type": "other", "ingredients": "Spider Silk"}},
    ]
    fake_model = MagicMock()
    fake_model.search.return_value = ([1, 0], [0.9, 0.8])
    mock_load.return_value = (fake_model, bm25_docs)

    trace: dict = {}
    results = retrieve(
        "recipes using Animal Feces",
        count=3,
        metadata_filter={"type": "recipe"},
        token_filter={"ingredients": "Animal Feces"},
        hybrid=True,
        trace=trace,
    )

    # token filter alone would empty the pool (neither doc has "Animal Feces"
    # in its delimited ingredients field), so fallback fires using metadata_filter only.
    # Only dense_1 (type="recipe") should survive; dense_2 (type="other") should be dropped.
    assert len(results["ids"][0]) == 1, f"Expected 1 result after fallback, got {len(results['ids'][0])}"
    assert results["metadatas"][0][0].get("ingredients") == "Animal Poop", \
        f"Expected 'Animal Poop' in metadatas, got {results['metadatas'][0]}"

    # trace should record the fallback fired
    rec = trace["retrieval_calls"][0]
    assert rec.get("token_filter_fallback") is True, \
        f"token_filter_fallback not recorded: {rec.get('token_filter_fallback')}"
    # fallback survivors land in post_filter_ids, never an emptied token-only set
    assert rec.get("post_filter_ids") == ["dense_1"], rec.get("post_filter_ids")
