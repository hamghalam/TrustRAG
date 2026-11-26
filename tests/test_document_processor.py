
import pytest

from backend.app.services.document_processor import chunk_text


def test_chunk_text_returns_chunks():
    text = "A" * 2500

    chunks = chunk_text(
        text,
        chunk_size=1000,
        chunk_overlap=200,
    )

    assert len(chunks) == 3
    assert len(chunks[0]) == 1000
    assert len(chunks[1]) == 1000
    assert len(chunks[2]) == 900


def test_chunk_text_preserves_overlap():
    text = "A" * 1500

    chunks = chunk_text(text, 1000, 200)

    assert chunks[0][-200:] == chunks[1][:200]


def test_empty_text_returns_empty_list():
    assert chunk_text("") == []


def test_invalid_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("Hello", chunk_size=0)


def test_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_text("Hello", chunk_size=100, chunk_overlap=100)
