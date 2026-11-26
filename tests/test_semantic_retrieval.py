
import json
import pytest

from unittest.mock import patch

from backend.app.services.semantic_retrieval_service import (
    cosine_similarity,
    semantic_search_chunks,
)


def test_identical_vectors_have_similarity_one():
    assert cosine_similarity(
        [1.0, 0.0],
        [1.0, 0.0],
    ) == pytest.approx(1.0)


def test_orthogonal_vectors_have_similarity_zero():
    assert cosine_similarity(
        [1.0, 0.0],
        [0.0, 1.0],
    ) == pytest.approx(0.0)


def test_opposite_vectors_have_similarity_minus_one():
    assert cosine_similarity(
        [1.0, 0.0],
        [-1.0, 0.0],
    ) == pytest.approx(-1.0)


def test_different_dimensions_raise_error():
    with pytest.raises(ValueError):
        cosine_similarity(
            [1.0, 0.0],
            [1.0, 0.0, 0.0],
        )


def test_empty_vectors_raise_error():
    with pytest.raises(ValueError):
        cosine_similarity([], [])


def test_zero_vector_returns_zero():
    assert cosine_similarity(
        [0.0, 0.0],
        [1.0, 0.0],
    ) == 0.0


@patch(
    "backend.app.services.semantic_retrieval_service.generate_embedding"
)
@patch(
    "backend.app.services.semantic_retrieval_service.get_connection"
)
def test_semantic_search_ranks_best_match_first(
    mock_get_connection,
    mock_generate_embedding,
):
    mock_connection = mock_get_connection.return_value

    mock_connection.execute.return_value.fetchall.return_value = [
        {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "page_number": 1,
            "chunk_index": 0,
            "text": "Diabetes symptoms include thirst.",
            "filename": "medical.pdf",
            "embedding": json.dumps([1.0, 0.0]),
        },
        {
            "chunk_id": "chunk-2",
            "document_id": "doc-1",
            "page_number": 2,
            "chunk_index": 0,
            "text": "Weather conditions are changing.",
            "filename": "medical.pdf",
            "embedding": json.dumps([0.0, 1.0]),
        },
    ]

    mock_generate_embedding.return_value = [1.0, 0.0]

    results = semantic_search_chunks(
        query="What are diabetes symptoms?",
        limit=2,
    )

    assert len(results) == 2
    assert results[0]["chunk_id"] == "chunk-1"
    assert results[0]["score"] == pytest.approx(1.0)
    assert results[1]["score"] == pytest.approx(0.0)

    mock_generate_embedding.assert_called_once_with(
        "What are diabetes symptoms?"
    )

    mock_connection.close.assert_called_once()


@patch(
    "backend.app.services.semantic_retrieval_service.generate_embedding"
)
@patch(
    "backend.app.services.semantic_retrieval_service.get_connection"
)
def test_semantic_search_returns_empty_when_no_embeddings(
    mock_get_connection,
    mock_generate_embedding,
):
    mock_connection = mock_get_connection.return_value
    mock_connection.execute.return_value.fetchall.return_value = []

    results = semantic_search_chunks("diabetes")

    assert results == []

    mock_generate_embedding.assert_not_called()
    mock_connection.close.assert_called_once()


def test_semantic_search_rejects_invalid_limit():
    with pytest.raises(ValueError):
        semantic_search_chunks(
            query="diabetes",
            limit=0,
        )


def test_semantic_search_returns_empty_for_blank_query():
    assert semantic_search_chunks("   ") == []
