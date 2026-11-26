
import pytest

from backend.app.database import database
from backend.app.services.document_storage import (
    save_document,
    get_all_documents,
    get_document_by_id,
    delete_document,
)


@pytest.fixture
def test_database(tmp_path, monkeypatch):
    """
    Create a separate temporary SQLite database
    for each test.
    """

    test_db_path = tmp_path / "test_trustrag.db"

    monkeypatch.setattr(
        database,
        "DATABASE_PATH",
        test_db_path,
    )

    database.initialize_database()

    return test_db_path


def sample_chunks():
    return [
        {
            "chunk_id": "chunk-001",
            "page_number": 1,
            "chunk_index": 0,
            "text": "This is the first document chunk.",
        },
        {
            "chunk_id": "chunk-002",
            "page_number": 1,
            "chunk_index": 1,
            "text": "This is the second document chunk.",
        },
        {
            "chunk_id": "chunk-003",
            "page_number": 2,
            "chunk_index": 0,
            "text": "This is the third document chunk.",
        },
    ]


def test_save_document(test_database):
    chunks = sample_chunks()

    save_document(
        document_id="doc-001",
        filename="research.pdf",
        page_count=2,
        chunks=chunks,
    )

    result = get_document_by_id("doc-001")

    assert result is not None

    assert result["document"]["filename"] == "research.pdf"

    assert result["document"]["page_count"] == 2

    assert result["document"]["chunk_count"] == 3

    assert len(result["chunks"]) == 3


def test_get_all_documents(test_database):
    save_document(
        document_id="doc-001",
        filename="research.pdf",
        page_count=2,
        chunks=sample_chunks(),
    )

    documents = get_all_documents()

    assert len(documents) == 1

    assert documents[0]["document_id"] == "doc-001"

    assert documents[0]["filename"] == "research.pdf"


def test_get_document_by_id(test_database):
    save_document(
        document_id="doc-001",
        filename="research.pdf",
        page_count=2,
        chunks=sample_chunks(),
    )

    result = get_document_by_id("doc-001")

    assert result is not None

    assert result["document"]["document_id"] == "doc-001"

    assert len(result["chunks"]) == 3

    assert result["chunks"][0]["page_number"] == 1

    assert result["chunks"][0]["chunk_index"] == 0

    assert result["chunks"][0]["text"] == (
        "This is the first document chunk."
    )


def test_delete_document_and_cascade_chunks(test_database):
    save_document(
        document_id="doc-001",
        filename="research.pdf",
        page_count=2,
        chunks=sample_chunks(),
    )

    deleted = delete_document("doc-001")

    assert deleted is True

    # Verify that the document no longer exists.
    assert get_document_by_id("doc-001") is None

    # Verify that all associated chunks were deleted.
    connection = database.get_connection()

    try:
        result = connection.execute(
            """
            SELECT COUNT(*)
            FROM chunks
            WHERE document_id = ?
            """,
            ("doc-001",),
        ).fetchone()

        assert result[0] == 0

    finally:
        connection.close()


def test_delete_nonexistent_document(test_database):
    deleted = delete_document("nonexistent-document")

    assert deleted is False
