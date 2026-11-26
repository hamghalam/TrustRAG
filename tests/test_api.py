
import pytest

from fastapi.testclient import TestClient

from backend.app import main
from backend.app.database import database


@pytest.fixture
def client(tmp_path, monkeypatch):
    """
    Start the FastAPI application with a temporary database.
    """

    test_db_path = tmp_path / "test_api.db"

    monkeypatch.setattr(
        database,
        "DATABASE_PATH",
        test_db_path,
    )

    with TestClient(main.app) as test_client:
        yield test_client


def mock_pdf_extraction(monkeypatch):
    """
    Replace PDF extraction with predictable sample text.
    """

    sample_pages = [
        {
            "page_number": 1,
            "text": "This is the first page of the test document.",
        },
        {
            "page_number": 2,
            "text": "This is the second page of the test document.",
        },
    ]

    monkeypatch.setattr(
        main,
        "extract_text_from_pdf",
        lambda pdf_bytes: sample_pages,
    )


def upload_sample_document(client, monkeypatch):
    """
    Upload a sample PDF through the API.
    """

    mock_pdf_extraction(monkeypatch)

    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "research.pdf",
                b"%PDF-1.4\nTest PDF content",
                "application/pdf",
            )
        },
    )

    return response


def test_health_endpoint(client):

    response = client.get("/health")

    assert response.status_code == 200


def test_upload_document(client, monkeypatch):

    response = upload_sample_document(
        client,
        monkeypatch,
    )

    assert response.status_code == 200

    result = response.json()

    assert result["filename"] == "research.pdf"

    assert result["page_count"] == 2

    assert result["chunk_count"] == 2

    assert result["status"] == "stored"

    assert len(result["chunks"]) == 2


def test_list_documents(client, monkeypatch):

    upload_response = upload_sample_document(
        client,
        monkeypatch,
    )

    assert upload_response.status_code == 200

    response = client.get("/documents")

    assert response.status_code == 200

    result = response.json()

    assert result["document_count"] == 1

    assert result["documents"][0]["filename"] == "research.pdf"


def test_get_document_details(client, monkeypatch):

    upload_response = upload_sample_document(
        client,
        monkeypatch,
    )

    document_id = upload_response.json()["document_id"]

    response = client.get(
        f"/documents/{document_id}"
    )

    assert response.status_code == 200

    result = response.json()

    assert result["document"]["document_id"] == document_id

    assert result["document"]["filename"] == "research.pdf"

    assert len(result["chunks"]) == 2

    assert (
        result["chunks"][0]["text"]
        == "This is the first page of the test document."
    )


def test_get_nonexistent_document(client):

    response = client.get(
        "/documents/nonexistent-document"
    )

    assert response.status_code == 404


def test_delete_document(client, monkeypatch):

    upload_response = upload_sample_document(
        client,
        monkeypatch,
    )

    document_id = upload_response.json()["document_id"]

    response = client.delete(
        f"/documents/{document_id}"
    )

    assert response.status_code == 200

    assert (
        response.json()["document_id"]
        == document_id
    )

    # Confirm the document has been deleted.
    detail_response = client.get(
        f"/documents/{document_id}"
    )

    assert detail_response.status_code == 404

    # Confirm the document library is empty.
    library_response = client.get("/documents")

    assert library_response.status_code == 200

    assert library_response.json()["document_count"] == 0


def test_delete_nonexistent_document(client):

    response = client.delete(
        "/documents/nonexistent-document"
    )

    assert response.status_code == 404
