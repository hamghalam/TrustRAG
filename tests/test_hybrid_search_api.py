
import pytest

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app import main


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_hybrid_search_success(client, monkeypatch):

    expected_results = [
        {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "filename": "medical.pdf",
            "page_number": 1,
            "chunk_index": 0,
            "text": "Diabetes symptoms include thirst.",
            "score": 0.0325,
            "rrf_score": 0.0325,
            "bm25_score": 2.45,
            "semantic_score": 0.81,
            "bm25_rank": 1,
            "semantic_rank": 2,
            "retrieval_methods": ["keyword", "semantic"],
        }
    ]

    monkeypatch.setattr(
        main,
        "hybrid_search_chunks",
        lambda query, limit: expected_results,
    )

    response = client.get(
        "/search/hybrid",
        params={"q": "diabetes symptoms"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["query"] == "diabetes symptoms"
    assert data["search_type"] == "hybrid"
    assert data["result_count"] == 1
    assert data["results"] == expected_results


def test_hybrid_search_passes_limit(client, monkeypatch):

    received_arguments = {}

    def mock_hybrid_search(query, limit):
        received_arguments["query"] = query
        received_arguments["limit"] = limit
        return []

    monkeypatch.setattr(
        main,
        "hybrid_search_chunks",
        mock_hybrid_search,
    )

    response = client.get(
        "/search/hybrid",
        params={
            "q": "diabetes",
            "limit": 3,
        },
    )

    assert response.status_code == 200
    assert received_arguments["query"] == "diabetes"
    assert received_arguments["limit"] == 3


def test_hybrid_search_returns_empty_results(client, monkeypatch):

    monkeypatch.setattr(
        main,
        "hybrid_search_chunks",
        lambda query, limit: [],
    )

    response = client.get(
        "/search/hybrid",
        params={"q": "unknown topic"},
    )

    assert response.status_code == 200
    assert response.json()["result_count"] == 0
    assert response.json()["results"] == []


def test_hybrid_search_rejects_empty_query(client):

    response = client.get(
        "/search/hybrid",
        params={"q": ""},
    )

    assert response.status_code == 422


def test_hybrid_search_rejects_invalid_limit(client):

    response = client.get(
        "/search/hybrid",
        params={
            "q": "diabetes",
            "limit": 51,
        },
    )

    assert response.status_code == 422


def test_hybrid_search_handles_internal_error(client, monkeypatch):

    def mock_hybrid_search(query, limit):
        raise RuntimeError("Test database error")

    monkeypatch.setattr(
        main,
        "hybrid_search_chunks",
        mock_hybrid_search,
    )

    response = client.get(
        "/search/hybrid",
        params={"q": "diabetes"},
    )

    assert response.status_code == 500

    assert response.json()["detail"] == (
        "Hybrid search operation failed."
    )
