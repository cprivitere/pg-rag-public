"""Tests pgrag.embeddings.llama_embeddings.validate_embeddings: valid vectors
pass through, and EmbeddingValidationError is raised for empty input,
non-list vectors, empty/non-numeric entries, and dimension mismatches."""

import pytest

from pgrag.embeddings.llama_embeddings import EmbeddingValidationError, validate_embeddings


def test_valid_vectors_pass():
    vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    assert validate_embeddings(vectors) is vectors


def test_empty_list_raises():
    with pytest.raises(EmbeddingValidationError, match="Empty"):
        validate_embeddings([])


def test_non_list_vector_raises():
    with pytest.raises(EmbeddingValidationError, match="expected list"):
        validate_embeddings(["not_a_list"])


def test_empty_vector_raises():
    with pytest.raises(EmbeddingValidationError, match="empty"):
        validate_embeddings([[]])


def test_non_numeric_element_raises():
    with pytest.raises(EmbeddingValidationError, match="expected float"):
        validate_embeddings([[0.1, "bad"]])


def test_dimension_mismatch_raises():
    with pytest.raises(EmbeddingValidationError, match=r"length.*!=.*expected"):
        validate_embeddings([[0.1, 0.2]], expected_dim=3)


def test_dimension_match_passes():
    vectors = [[0.1, 0.2], [0.3, 0.4]]
    assert validate_embeddings(vectors, expected_dim=2) is vectors


def test_mixed_dimension_raises():
    with pytest.raises(EmbeddingValidationError, match=r"length.*!=.*expected"):
        validate_embeddings([[0.1, 0.2], [0.3]], expected_dim=2)
