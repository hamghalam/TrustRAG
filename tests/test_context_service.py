
import pytest

from backend.app.services.context_service import (
    assemble_context,
)


@pytest.fixture
def sample_chunks():
    return [
        {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "filename": "report.pdf",
            "page_number": 2,
            "chunk_index": 0,
            "text": "The patient received chemotherapy.",
            "score": 0.032,
            "retrieval_methods": ["keyword", "semantic"],
        },
        {
            "chunk_id": "chunk-2",
            "document_id": "doc-1",
            "filename": "report.pdf",
            "page_number": 3,
            "chunk_index": 1,
            "text": "The tumor decreased in size.",
            "score": 0.031,
            "retrieval_methods": ["semantic"],
        },
    ]


def test_assemble_context_returns_query_and_sources(sample_chunks):
    result = assemble_context(
        query="What happened to the tumor?",
        retrieved_chunks=sample_chunks,
    )

    assert result["query"] == "What happened to the tumor?"
    assert result["source_count"] == 2
    assert len(result["sources"]) == 2


def test_context_contains_citations(sample_chunks):
    result = assemble_context(
        query="What happened?",
        retrieved_chunks=sample_chunks,
    )

    assert "[1]" in result["context"]
    assert "[2]" in result["context"]


def test_sources_contain_document_metadata(sample_chunks):
    result = assemble_context(
        query="What happened?",
        retrieved_chunks=sample_chunks,
    )

    first_source = result["sources"][0]

    assert first_source["filename"] == "report.pdf"
    assert first_source["page_number"] == 2
    assert first_source["chunk_id"] == "chunk-1"


def test_duplicate_chunks_are_removed(sample_chunks):
    duplicated_chunks = sample_chunks + [sample_chunks[0]]

    result = assemble_context(
        query="What happened?",
        retrieved_chunks=duplicated_chunks,
    )

    assert result["source_count"] == 2


def test_empty_results_return_empty_context():
    result = assemble_context(
        query="What happened?",
        retrieved_chunks=[],
    )

    assert result["context"] == ""
    assert result["sources"] == []
    assert result["source_count"] == 0


def test_context_respects_character_limit(sample_chunks):
    result = assemble_context(
        query="What happened?",
        retrieved_chunks=sample_chunks,
        max_context_chars=100,
    )

    assert result["context_char_count"] <= 100


def test_empty_query_raises_error(sample_chunks):
    with pytest.raises(ValueError):
        assemble_context(
            query=" ",
            retrieved_chunks=sample_chunks,
        )


def test_invalid_context_limit_raises_error(sample_chunks):
    with pytest.raises(ValueError):
        assemble_context(
            query="What happened?",
            retrieved_chunks=sample_chunks,
            max_context_chars=0,
        )
