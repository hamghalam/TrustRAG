
import pytest
from fastapi.testclient import TestClient

from backend.app import main
from backend.app.database import database


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Use a temporary database for API tests.
    test_db = tmp_path / "test_trustrag.db"

    monkeypatch.setattr(
        database,
        "DATABASE_PATH",
        test_db,
    )

    with TestClient(main.app) as test_client:
        yield test_client


def sample_results():
    return [
        {
            "chunk_id": "chunk-001",
            "document_id": "doc-001",
            "filename": "medical_research.pdf",
            "page_number": 1,
            "chunk_index": 0,
            "text": "Diabetes is a chronic metabolic disease.",
            "score": 1.0,
            "matched_terms": ["diabetes"],
        },
        {
            "chunk_id": "chunk-002",
            "document_id": "doc-001",
            "filename": "medical_research.pdf",
            "page_number": 2,
            "chunk_index": 0,
            "text": "Diabetes treatment includes lifestyle changes.",
            "score": 1.0,
            "matched_terms": ["diabetes"],
        },
    ]


def test_search_returns_matching_results(client, monkeypatch):
    monkeypatch.setattr(
        main,
        "search_chunks",
        lambda query, limit: sample_results(),
    )

    response = client.get(
        "/search",
        params={"q": "diabetes", "limit": 5},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["query"] == "diabetes"
    assert data["result_count"] == 2
    assert len(data["results"]) == 2

    assert data["results"][0]["filename"] == "medical_research.pdf"
    assert data["results"][0]["page_number"] == 1
    assert data["results"][0]["score"] == 1.0


def test_search_respects_result_limit(client, monkeypatch):
    def mock_search(query, limit):
        return sample_results()[:limit]

    monkeypatch.setattr(
        main,
        "search_chunks",
        mock_search,
    )

    response = client.get(
        "/search",
        params={"q": "diabetes", "limit": 1},
    )

    assert response.status_code == 200
    assert response.json()["result_count"] == 1


def test_search_returns_empty_results(client, monkeypatch):
    monkeypatch.setattr(
        main,
        "search_chunks",
        lambda query, limit: [],
    )

    response = client.get(
        "/search",
        params={"q": "unrelated keyword"},
    )

    assert response.status_code == 200
    assert response.json()["result_count"] == 0
    assert response.json()["results"] == []


def test_search_rejects_empty_query(client):
    response = client.get(
        "/search",
        params={"q": ""},
    )

    assert response.status_code == 422


def test_search_rejects_invalid_limit(client):
    response = client.get(
        "/search",
        params={"q": "diabetes", "limit": 0},
    )

    assert response.status_code == 422


def test_search_handles_internal_error(client, monkeypatch):
    def failing_search(query, limit):
        raise RuntimeError("Simulated search failure")

    monkeypatch.setattr(
        main,
        "search_chunks",
        failing_search,
    )

    response = client.get(
        "/search",
        params={"q": "diabetes"},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Search operation failed."
