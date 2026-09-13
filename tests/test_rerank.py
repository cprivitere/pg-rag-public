"""Tests pgrag.rag.retriever hybrid-stage helpers: _term_overlap scoring,
_rerank count/term-match promotion, _hybrid_fuse tsys chunk-cluster and
distinct-base capping, _tsys_base_id/_tsys_origin_id, _entity_name_match,
_name_injection_ids longest-span injection, _is_fragment_id, and
_apply_name_promotion."""

from pgrag.rag.retriever import (
    MAX_TSYS_BASES,
    MAX_TSYS_CHUNK_MEMBERS,
    RERANK_MULTIPLIER,
    _apply_name_promotion,
    _entity_name_match,
    _hybrid_fuse,
    _is_fragment_id,
    _name_injection_ids,
    _rerank,
    _term_overlap,
    _tsys_base_id,
    _tsys_origin_id,
)


def test_term_overlap_full_match():
    score = _term_overlap("how to make potion", "how to make potion recipe")
    assert score == 1.0


def test_term_overlap_partial():
    score = _term_overlap("how to make potion", "potion recipe")
    assert score == 0.25


def test_term_overlap_no_match():
    score = _term_overlap("how to make potion", "sword fighting")
    assert score == 0.0


def test_term_overlap_empty_query():
    score = _term_overlap("", "some document text")
    assert score == 0.0


def test_term_overlap_case_insensitive():
    score = _term_overlap("POTION", "potion recipe")
    assert score == 1.0


def test_rerank_returns_correct_count():
    query = "potion making"
    ids = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
    docs = [
        "potion recipe",
        "sword sharpening",
        "potion brewing tips",
        "shield block",
        "potion effects guide",
        "arrow fletching",
        "potion history book",
        "helmet polishing",
        "potion ingredients list",
    ]
    metas = [{}] * len(docs)
    dists = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

    result_ids, result_docs, result_metas, result_dists = _rerank(query, ids, docs, metas, dists, 3)

    assert len(result_ids) == 3
    assert len(result_docs) == 3
    assert len(result_metas) == 3
    assert len(result_dists) == 3


def test_rerank_promotes_term_match():
    query = "potion"
    ids = ["no_match", "has_match"]
    docs = ["sword fighting guide", "potion brewing guide"]
    metas = [{}, {}]
    dists = [0.1, 0.3]

    result_ids, _, _, _ = _rerank(query, ids, docs, metas, dists, 2)

    assert result_ids[0] == "has_match", "doc with term match should rank first"


def test_rerank_with_scored_inputs():
    query = "potion fighting"
    ids = ["match_both", "match_one", "match_none"]
    docs = ["potion fighting guide", "potion brewing guide", "sword shield guide"]
    metas = [{}, {}, {}]
    dists = [0.3, 0.2, 0.1]

    result_ids, _, _, _ = _rerank(query, ids, docs, metas, dists, 3)

    assert result_ids[0] == "match_both", "doc matching both terms ranks first"


def test_rerank_multiplier_constant():
    assert RERANK_MULTIPLIER == 3


def test_tsys_base_id_honors_chunk_fragments():
    assert _tsys_base_id("tsys_power_2005_chunk_3") == "tsys_power_2005"
    assert _tsys_base_id("tsys_power_2005_chunk_0") == "tsys_power_2005"
    # chunked base doc (no _chunk suffix) is not itself a fragment
    assert _tsys_base_id("tsys_power_2005") is None
    # non-tsys docs are never collapsed
    assert _tsys_base_id("ability_ability_3504") is None
    assert _tsys_base_id("wiki_Fire_Magic_Abilities_table_0_row_1") is None


def test_hybrid_fuse_caps_tsys_chunk_cluster():
    # Five fragments of the same tsys base + a distinct target doc. The
    # cluster must collapse to MAX_TSYS_CHUNK_MEMBERS so the distinct target
    # survives; wiki rows (distinctly relevant) must NOT collapse.
    dense = (
        [f"tsys_power_2001_chunk_{i}" for i in range(5)]
        + ["ability_ability_9"]
        + [f"wiki_X_table_0_row_{i}" for i in range(3)]
    )
    texts = [f"text {i}" for i in range(len(dense))]
    metas = [{}] * len(dense)
    dists = [0.1] * len(dense)

    ids, _, _, _ = _hybrid_fuse(
        dense, texts, metas, dists, list(dense), [{"id": d} for d in dense], len(dense)
    )

    tsys_frags = [i for i in ids if i.startswith("tsys_power_2001_chunk_")]
    assert len(tsys_frags) == MAX_TSYS_CHUNK_MEMBERS
    assert "ability_ability_9" in ids, "distinct non-tsys target must survive"
    # wiki table rows / coverage are NOT collapsed (they are distinct content)
    assert any(i.startswith("wiki_X_table_0_row_") for i in ids)


def test_hybrid_fuse_preserves_distinct_docs_without_tsys():
    # No tsys fragments present: window must keep every distinct doc.
    dense = ["a", "b", "c"]
    texts = ["ta", "tb", "tc"]
    metas = [{}, {}, {}]
    dists = [0.3, 0.2, 0.1]
    ids, *_ = _hybrid_fuse(
        dense, texts, metas, dists, ["c", "b", "a"], [{"id": d} for d in dense], 3
    )
    assert set(ids) == {"a", "b", "c"}


def test_hybrid_fuse_caps_distinct_tsys_bases():
    # 6 distinct bare tsys_power_* bases (the pollution form: un-chunked
    # treasure powers) + a distinct target. The window must cap distinct tsys
    # bases to MAX_TSYS_BASES so the real target survives.
    dense = [f"tsys_power_{2000 + i}" for i in range(6)] + ["ability_ability_9"]
    texts = [f"text {i}" for i in range(len(dense))]
    metas = [{}] * len(dense)
    dists = [0.05] * len(dense)

    ids, _, _, _ = _hybrid_fuse(
        dense, texts, metas, dists, list(dense), [{"id": d} for d in dense], len(dense)
    )

    tsys_bases = {d for d in ids if d.startswith("tsys_power_")}
    assert len(tsys_bases) == MAX_TSYS_BASES, (
        f"distinct tsys bases must cap at {MAX_TSYS_BASES}, got {len(tsys_bases)}"
    )
    assert "ability_ability_9" in ids, "distinct non-tsys target must survive"


def test_hybrid_fuse_tsys_origin_matches_both_forms():
    # chunk fragments and bare docs of the same base count as one origin
    # against the distinct-base cap.
    assert _tsys_origin_id("tsys_power_2005") == "tsys_power_2005"
    assert _tsys_origin_id("tsys_power_2005_chunk_3") == "tsys_power_2005"
    assert _tsys_origin_id("ability_ability_9") is None


def test_entity_name_match_contiguous_multitoken():
    assert _entity_name_match("What level is Fireball 3 unlocked at?", "Fireball 3")
    # punctuation shouldn't break the contiguous match
    assert _entity_name_match("What level should I be for Gazluk Keep?", "Gazluk Keep")
    # single-token (common-word) names must not match
    assert not _entity_name_match("What level is fireball unlocked at?", "fireball")
    # non-contiguous / absent names
    assert not _entity_name_match("What level is Fireball 3 unlocked at?", "Fireball unlocked")


def test_name_injection_ids_prefers_longest_span_and_skips_fragments(monkeypatch):
    fake_idx = {
        "healing potion omega": ["item_1", "recipe_2", "wiki_X_table_0_row_1"],
        "healing potion": ["item_3"],
    }
    monkeypatch.setattr("pgrag.rag.retriever._NAME_INDEX", fake_idx)
    res = _name_injection_ids("What is the Healing Potion Omega recipe?")
    # longest span "healing potion omega" wins, fragment row excluded
    assert "item_1" in res
    assert "recipe_2" in res
    assert "wiki_X_table_0_row_1" not in res
    assert "item_3" not in res, "shorter span must not be injected"


def test_fragment_id_detection():
    assert _is_fragment_id("wiki_X_table_0_row_3")
    assert _is_fragment_id("wiki_X_table_0_coverage")
    assert _is_fragment_id("tsys_power_2005_chunk_1")
    assert not _is_fragment_id("item_18033")
    assert not _is_fragment_id("wiki_Healing Potion Omega_How_to_Obtain")


def test_apply_name_promotion_floats_entity_and_skips_fragments():
    query = "What is the Healing Potion Omega recipe?"
    pool_ids = ["item_1", "recipe_2", "wiki_X_table_0_row_1", "other"]
    pool_docs = ["a", "b", "c", "d"]
    pool_metas = [
        {"name": "Healing Potion Omega"},
        {"name": "Healing Potion Omega"},
        {"name": "Healing Potion Omega"},  # fragment row: same name, must NOT float
        {"name": "Something Else"},
    ]
    pool_dists = [0.1] * 4
    # cross-encoder ranked: other, item_1, recipe_2, row (worst)
    ranked_ids = ["other", "item_1", "recipe_2", "wiki_X_table_0_row_1"]
    ranked_docs = ["d", "a", "b", "c"]
    ranked_metas = [pool_metas[3], pool_metas[0], pool_metas[1], pool_metas[2]]
    ranked_dists = [0.1] * 4

    ids, *_ = _apply_name_promotion(
        query,
        pool_ids,
        pool_docs,
        pool_metas,
        pool_dists,
        ranked_ids,
        ranked_docs,
        ranked_metas,
        ranked_dists,
        4,
    )
    # the two substantive gold items float ahead of "other"; the fragment row
    # keeps its relative position (after them), "other" pushed down
    assert ids[0] in {"item_1", "recipe_2"}
    assert ids[1] in {"item_1", "recipe_2"}
    # fragment row must NOT be floated above the substantive entity docs
    assert ids.index("wiki_X_table_0_row_1") > ids.index("item_1")
    assert ids.index("wiki_X_table_0_row_1") > ids.index("recipe_2")


def test_apply_name_promotion_splices_matched_doc_missing_from_topn():
    query = "What level should I be for Gazluk Keep?"
    pool_ids = ["area_AreaGazlukKeep", "a", "b", "c"]
    pool_docs = ["d0", "d1", "d2", "d3"]
    pool_metas = [{"name": "Gazluk Keep"}, {"name": "X"}, {"name": "Y"}, {"name": "Z"}]
    pool_dists = [0.1] * 4
    # cross-encoder dropped the area doc entirely
    ranked_ids = ["a", "b", "c"]
    ranked_docs = ["d1", "d2", "d3"]
    ranked_metas = [pool_metas[1], pool_metas[2], pool_metas[3]]
    ranked_dists = [0.1] * 3
    ids, *_ = _apply_name_promotion(
        query,
        pool_ids,
        pool_docs,
        pool_metas,
        pool_dists,
        ranked_ids,
        ranked_docs,
        ranked_metas,
        ranked_dists,
        3,
    )
    assert ids[0] == "area_AreaGazlukKeep"


def test_apply_name_promotion_noop_without_match():
    query = "some query"
    pool_ids = ["a", "b"]
    pool_docs = ["x", "y"]
    pool_metas = [{"name": "Foo Bar"}, {"name": "Baz Qux"}]
    pool_dists = [0.1, 0.2]
    ids, docs, metas, dists = _apply_name_promotion(
        query,
        pool_ids,
        pool_docs,
        pool_metas,
        pool_dists,
        ["a", "b"],
        ["x", "y"],
        pool_metas,
        [0.1, 0.2],
        2,
    )
    assert ids == ["a", "b"]
