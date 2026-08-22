"""Tests pgrag.documents.chunking: small documents pass through unmodified,
large documents split into suffixed chunks with preserved metadata, and
splitting is sentence-aware with character fallback, type-aware size limits,
and overlap handling."""

from pgrag.documents.chunking import (
    chunk_document, chunk_all_documents,
    DEFAULT_MAX_CHARS, OVERLAP_CHARS, TYPE_MAX_CHARS,
    _find_best_split, _split_sentences,
)


def test_small_doc_not_chunked():
    doc = {"id": "item_1", "type": "item", "text": "short text", "metadata": {"source": "cdn"}}
    result = chunk_document(doc)
    assert len(result) == 1
    assert result[0] is doc


def test_large_doc_split_into_chunks():
    doc = {"id": "item_1", "type": "item", "text": "para one\n\npara two\n\npara three\n\npara four\n\npara five\n\npara six\n\npara seven\n\npara eight\n\npara nine\n\npara ten", "metadata": {"source": "cdn"}}
    result = chunk_document(doc, max_chars=20)
    assert len(result) > 1


def test_chunk_ids_suffixed():
    doc = {"id": "item_1", "type": "item", "text": "a\n\nb\n\nc\n\nd\n\ne\n\nf\n\ng\n\nh", "metadata": {"source": "cdn"}}
    result = chunk_document(doc, max_chars=10)
    for i, chunk in enumerate(result):
        assert chunk["id"] == f"item_1_chunk_{i}"


def test_chunks_contain_chunk_metadata():
    doc = {"id": "item_1", "type": "item", "text": "a\n\nb\n\nc\n\nd\n\ne\n\nf", "metadata": {"source": "cdn"}}
    result = chunk_document(doc, max_chars=10)
    assert len(result) >= 2
    for chunk in result:
        assert "chunk_index" in chunk["metadata"]
        assert "chunk_count" in chunk["metadata"]
        assert chunk["metadata"]["chunk_count"] == len(result)


def test_chunk_preserves_metadata():
    doc = {"id": "item_1", "type": "item", "text": "a\n\nb\n\nc\n\nd\n\ne\n\nf", "metadata": {"source": "cdn", "table": "items", "name": "Test"}}
    result = chunk_document(doc, max_chars=10)
    for chunk in result:
        assert chunk["metadata"]["source"] == "cdn"
        assert chunk["metadata"]["table"] == "items"
        assert chunk["metadata"]["name"] == "Test"
        assert chunk["type"] == "item"


def test_chunk_text_preserved():
    doc = {"id": "item_1", "type": "item", "text": "first paragraph\n\nsecond paragraph\n\nthird paragraph\n\nfourth paragraph", "metadata": {"source": "cdn"}}
    result = chunk_document(doc, max_chars=30)
    assert len(result) >= 2
    for chunk in result:
        assert len(chunk["text"]) > 0
    assert result[0]["text"] == "first paragraph"


def test_single_paragraph_exceeds_max():
    text = "word " * 500
    doc = {"id": "item_1", "type": "item", "text": text.strip(), "metadata": {"source": "cdn"}}
    result = chunk_document(doc, max_chars=100)
    assert len(result) > 1


def test_chunk_all_documents():
    docs = [
        {"id": "a", "type": "item", "text": "short", "metadata": {"source": "cdn"}},
        {"id": "b", "type": "item", "text": "x\n\ny\n\nz\n\n1\n\n2\n\n3\n\n4", "metadata": {"source": "cdn"}},
    ]
    result = chunk_all_documents(docs, max_chars=10)
    assert len(result) > len(docs)


def test_constants():
    assert DEFAULT_MAX_CHARS == 1024
    assert OVERLAP_CHARS == 100
    assert TYPE_MAX_CHARS["item"] == 1024
    assert TYPE_MAX_CHARS["recipe"] == 1024
    assert TYPE_MAX_CHARS["wiki"] == 1024
    assert TYPE_MAX_CHARS["lorebook"] == 2048
    assert TYPE_MAX_CHARS["skillprofile"] == 2048
    assert TYPE_MAX_CHARS["summary"] == 8192
    assert TYPE_MAX_CHARS["curated"] == 8192


def test_type_aware_limits():
    item_doc = {"id": "i1", "type": "item", "text": "x\n\n" * 500, "metadata": {}}
    lore_doc = {"id": "w1", "type": "lorebook", "text": "x\n\n" * 500, "metadata": {}}

    item_chunks = chunk_document(item_doc)
    lore_chunks = chunk_document(lore_doc)

    assert len(item_chunks) > len(lore_chunks)


def test_overlap_between_chunks():
    text = "sentence one. " * 50
    doc = {"id": "w1", "type": "wiki", "text": text, "metadata": {}}
    result = chunk_document(doc, max_chars=200)
    assert len(result) >= 2

    for i in range(1, len(result)):
        prev_end = result[i - 1]["text"][-50:]
        curr_start = result[i]["text"][:50]
        assert prev_end[:20] in result[i]["text"] or result[i]["text"][:30] in prev_end


def test_find_best_split_prefers_sentence():
    text = "First sentence. Second sentence. Third sentence. Fourth sentence."
    piece, remainder = _find_best_split(text, 40)
    assert piece.endswith(".")
    assert remainder.startswith(" Second") or remainder.startswith("Third")


def test_find_best_split_falls_back_to_char():
    text = "a" * 100
    piece, remainder = _find_best_split(text, 40)
    assert len(piece) <= 40
    assert len(remainder) > 0


def test_split_sentences():
    text = "Hello world. How are you? I'm fine!"
    sentences = _split_sentences(text)
    assert len(sentences) == 3


def test_overlap_text_not_lost():
    text = "abcdefghij " * 100
    doc = {"id": "w1", "type": "wiki", "text": text.strip(), "metadata": {}}
    result = chunk_document(doc, max_chars=200)
    assert len(result) >= 2

    full_text = "".join(c["text"] for c in result)
    original_words = text.split()
    for word in original_words[:50]:
        assert word in full_text
