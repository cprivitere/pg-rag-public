"""Tests the pgrag.vectorstore.hashes embedding_hash and metadata_hash
contracts: which document fields each key, determinism, key-order
independence, and required fields."""

import pytest

from pgrag.vectorstore.hashes import embedding_hash, metadata_hash


def test_embedding_hash_uses_id_and_text_only():
    doc = {"id": "item_1", "text": "foo", "metadata": {"source": "cdn"}}
    result = embedding_hash(doc)
    assert isinstance(result, str)
    assert len(result) == 64


def test_embedding_hash_excludes_metadata():
    doc1 = {"id": "item_1", "text": "foo", "metadata": {"source": "cdn"}}
    doc2 = {"id": "item_1", "text": "foo", "metadata": {"source": "wiki"}}
    assert embedding_hash(doc1) == embedding_hash(doc2)


def test_embedding_hash_changes_with_text():
    doc1 = {"id": "item_1", "text": "foo"}
    doc2 = {"id": "item_1", "text": "bar"}
    assert embedding_hash(doc1) != embedding_hash(doc2)


def test_embedding_hash_changes_with_id():
    doc1 = {"id": "item_1", "text": "foo"}
    doc2 = {"id": "item_2", "text": "foo"}
    assert embedding_hash(doc1) != embedding_hash(doc2)


def test_metadata_hash_uses_metadata_only():
    doc1 = {"id": "item_1", "text": "foo", "metadata": {"source": "cdn"}}
    doc2 = {"id": "item_1", "text": "bar", "metadata": {"source": "cdn"}}
    assert metadata_hash(doc1) == metadata_hash(doc2)


def test_metadata_hash_changes_with_metadata():
    doc1 = {"id": "item_1", "text": "foo", "metadata": {"source": "cdn"}}
    doc2 = {"id": "item_1", "text": "foo", "metadata": {"source": "wiki"}}
    assert metadata_hash(doc1) != metadata_hash(doc2)


def test_metadata_hash_handles_empty_metadata():
    doc = {"id": "item_1", "text": "foo", "metadata": {}}
    result = metadata_hash(doc)
    assert isinstance(result, str)
    assert len(result) == 64


def test_metadata_hash_handles_missing_metadata():
    doc = {"id": "item_1", "text": "foo"}
    result = metadata_hash(doc)
    assert isinstance(result, str)
    assert len(result) == 64


def test_hashes_differ_for_same_document():
    doc = {"id": "item_1", "text": "foo", "metadata": {"source": "cdn"}}
    assert embedding_hash(doc) != metadata_hash(doc)


def test_deterministic_embedding_hash():
    doc = {"id": "item_1", "text": "foo"}
    assert embedding_hash(doc) == embedding_hash(doc)


def test_deterministic_metadata_hash():
    doc = {"id": "item_1", "text": "foo", "metadata": {"source": "cdn"}}
    assert metadata_hash(doc) == metadata_hash(doc)


def test_embedding_hash_sorted_keys():
    doc1 = {"id": "item_1", "text": "foo"}
    doc2 = {"text": "foo", "id": "item_1"}
    assert embedding_hash(doc1) == embedding_hash(doc2)


def test_metadata_hash_sorted_keys():
    doc1 = {"id": "item_1", "text": "foo", "metadata": {"b": 2, "a": 1}}
    doc2 = {"id": "item_1", "text": "foo", "metadata": {"a": 1, "b": 2}}
    assert metadata_hash(doc1) == metadata_hash(doc2)


def test_v11_document_must_have_id_and_text():
    with pytest.raises(KeyError):
        embedding_hash({"text": "foo"})
    with pytest.raises(KeyError):
        embedding_hash({"id": "item_1"})
