
import pytest

from backend.app.services.hybrid_retrieval_service import (
    reciprocal_rank_fusion,
)


def make_result(chunk_id, score=1.0):
    return {
        "chunk_id": chunk_id,
        "document_id": "doc-1",
        "filename": "test.pdf",
        "page_number": 1,
        "chunk_index": 0,
        "text": f"Text for {chunk_id}",
        "score": score,
    }


def test_rrf_combines_two_ranked_lists():

    keyword_results = [
        make_result("chunk-1"),
        make_result("chunk-2"),
    ]

    semantic_results = [
        make_result("chunk-2"),
        make_result("chunk-3"),
    ]

    results = reciprocal_rank_fusion(
        [keyword_results, semantic_results],
        limit=3,
    )

    assert len(results) == 3

    # chunk-2 ranks second in keyword and first in semantic.
    assert results[0]["chunk_id"] == "chunk-2"


def test_rrf_rewards_results_found_by_both_methods():

    keyword_results = [
        make_result("chunk-1"),
        make_result("chunk-2"),
    ]

    semantic_results = [
        make_result("chunk-2"),
        make_result("chunk-1"),
    ]

    results = reciprocal_rank_fusion(
        [keyword_results, semantic_results],
        limit=2,
    )

    assert results[0]["chunk_id"] in {
        "chunk-1",
        "chunk-2",
    }

    assert results[0]["rrf_score"] == pytest.approx(
        results[1]["rrf_score"]
    )


def test_rrf_removes_duplicate_chunks():

    keyword_results = [
        make_result("chunk-1"),
        make_result("chunk-2"),
    ]

    semantic_results = [
        make_result("chunk-1"),
    ]

    results = reciprocal_rank_fusion(
        [keyword_results, semantic_results],
        limit=5,
    )

    chunk_ids = [
        result["chunk_id"]
        for result in results
    ]

    assert len(chunk_ids) == len(set(chunk_ids))
    assert len(results) == 2


def test_rrf_respects_limit():

    ranked_list = [
        make_result("chunk-1"),
        make_result("chunk-2"),
        make_result("chunk-3"),
    ]

    results = reciprocal_rank_fusion(
        [ranked_list],
        limit=2,
    )

    assert len(results) == 2


def test_rrf_rejects_invalid_limit():

    with pytest.raises(ValueError):
        reciprocal_rank_fusion(
            [[make_result("chunk-1")]],
            limit=0,
        )


def test_rrf_rejects_invalid_k():

    with pytest.raises(ValueError):
        reciprocal_rank_fusion(
            [[make_result("chunk-1")]],
            limit=5,
            k=0,
        )


def test_rrf_handles_empty_results():

    results = reciprocal_rank_fusion(
        [[], []],
        limit=5,
    )

    assert results == []


def test_rrf_returns_expected_score():

    results = reciprocal_rank_fusion(
        [[make_result("chunk-1")]],
        limit=5,
        k=60,
    )

    expected_score = 1 / 61

    assert results[0]["rrf_score"] == pytest.approx(
        expected_score
    )

    assert results[0]["score"] == pytest.approx(
        expected_score
    )
