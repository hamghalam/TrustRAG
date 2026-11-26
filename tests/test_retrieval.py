
import pytest

from backend.app.database import database
from backend.app.services.document_storage import save_document
from backend.app.services.retrieval_service import search_chunks


@pytest.fixture
def test_database(tmp_path, monkeypatch):

    test_db_path = tmp_path / "test_retrieval.db"

    monkeypatch.setattr(
        database,
        "DATABASE_PATH",
        test_db_path,
    )

    database.initialize_database()

    chunks = [
        {
            "chunk_id": "chunk-001",
            "page_number": 1,
            "chunk_index": 0,
            "text": (
                "Diabetes is a chronic disease "
                "that affects blood glucose levels."
            ),
        },
        {
            "chunk_id": "chunk-002",
            "page_number": 2,
            "chunk_index": 0,
            "text": (
                "Type 2 diabetes is associated "
                "with insulin resistance."
            ),
        },
        {
            "chunk_id": "chunk-003",
            "page_number": 3,
            "chunk_index": 0,
            "text": (
                "Cardiovascular diseases affect "
                "the heart and blood vessels."
            ),
        },
    ]

    save_document(
        document_id="doc-001",
        filename="medical_research.pdf",
        page_count=3,
        chunks=chunks,
    )

    return test_db_path


def test_search_returns_matching_chunks(test_database):

    results = search_chunks("diabetes")

    assert len(results) == 2

    assert all(
        "diabetes" in result["text"].lower()
        for result in results
    )



def test_search_ranks_best_match_first(test_database):

    results = search_chunks(
        "diabetes insulin resistance"
    )

    assert len(results) == 2

    # The most relevant chunk should appear first.
    assert results[0]["chunk_id"] == "chunk-002"

    # BM25 scores should be positive.
    assert results[0]["score"] > 0

    # The best result should have a higher score.
    assert results[0]["score"] > results[1]["score"]

    # All query terms should be matched in the best result.
    assert set(results[0]["matched_terms"]) == {
        "diabetes",
        "insulin",
        "resistance",
    }



def test_search_returns_source_metadata(test_database):

    results = search_chunks("cardiovascular")

    assert len(results) == 1

    assert results[0]["filename"] == "medical_research.pdf"

    assert results[0]["page_number"] == 3

    assert results[0]["document_id"] == "doc-001"


def test_search_respects_limit(test_database):

    results = search_chunks(
        "diabetes",
        limit=1,
    )

    assert len(results) == 1


def test_search_empty_query(test_database):

    results = search_chunks("")

    assert results == []


def test_search_no_matching_results(test_database):

    results = search_chunks("kidney")

    assert results == []


def test_search_invalid_limit(test_database):

    with pytest.raises(ValueError):

        search_chunks(
            "diabetes",
            limit=0,
        )
