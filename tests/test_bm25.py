
import pytest

from backend.app.services.retrieval_service import (
    calculate_idf,
    calculate_bm25_score,
)


def test_rare_terms_have_higher_idf():
    total_documents = 100

    rare_term_idf = calculate_idf(
        total_documents,
        document_frequency=2,
    )

    common_term_idf = calculate_idf(
        total_documents,
        document_frequency=80,
    )

    assert rare_term_idf > common_term_idf


def test_higher_term_frequency_increases_score():
    idf = 2.0
    document_length = 100
    average_length = 100

    score_once = calculate_bm25_score(
        term_frequency=1,
        document_length=document_length,
        average_length=average_length,
        idf=idf,
    )

    score_repeated = calculate_bm25_score(
        term_frequency=3,
        document_length=document_length,
        average_length=average_length,
        idf=idf,
    )

    assert score_repeated > score_once


def test_longer_chunks_receive_length_penalty():
    idf = 2.0
    average_length = 100

    short_chunk_score = calculate_bm25_score(
        term_frequency=1,
        document_length=100,
        average_length=average_length,
        idf=idf,
    )

    long_chunk_score = calculate_bm25_score(
        term_frequency=1,
        document_length=300,
        average_length=average_length,
        idf=idf,
    )

    assert short_chunk_score > long_chunk_score


def test_bm25_score_is_positive():
    score = calculate_bm25_score(
        term_frequency=2,
        document_length=100,
        average_length=100,
        idf=1.5,
    )

    assert score > 0


def test_invalid_term_frequency_returns_zero():
    score = calculate_bm25_score(
        term_frequency=0,
        document_length=100,
        average_length=100,
        idf=1.5,
    )

    assert score == 0
