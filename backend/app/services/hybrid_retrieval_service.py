
from backend.app.services.retrieval_service import (
    search_chunks,
)

from backend.app.services.semantic_retrieval_service import (
    semantic_search_chunks,
)


RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    limit: int = 5,
    k: int = RRF_K,
) -> list[dict]:
    """
    Combine multiple ranked result lists using
    Reciprocal Rank Fusion (RRF).
    """

    if limit <= 0:
        raise ValueError("limit must be positive")

    if k <= 0:
        raise ValueError("k must be positive")

    fused_results = {}

    for ranked_list in ranked_lists:

        for rank, result in enumerate(ranked_list, start=1):

            chunk_id = result["chunk_id"]

            if chunk_id not in fused_results:

                fused_results[chunk_id] = {
                    **result,
                    "rrf_score": 0.0,
                    "retrieval_methods": [],
                }

            fused_results[chunk_id]["rrf_score"] += (
                1.0 / (k + rank)
            )

    results = list(fused_results.values())

    results.sort(
        key=lambda item: (
            -item["rrf_score"],
            item["page_number"],
            item["chunk_index"],
        )
    )

    for result in results:
        result["rrf_score"] = round(
            result["rrf_score"],
            10,
        )

        # Keep a common score field for frontend compatibility.
        result["score"] = result["rrf_score"]

    return results[:limit]


def hybrid_search_chunks(
    query: str,
    limit: int = 5,
) -> list[dict]:
    """
    Search using both BM25 and semantic retrieval,
    then combine their rankings using RRF.
    """

    if limit <= 0:
        raise ValueError("limit must be positive")

    if not query or not query.strip():
        return []

    # Retrieve a larger candidate set from each method.
    candidate_limit = max(limit * 3, 20)

    keyword_results = search_chunks(
        query=query,
        limit=candidate_limit,
    )

    semantic_results = semantic_search_chunks(
        query=query,
        limit=candidate_limit,
    )

    # Record the original scores and ranks.
    for rank, result in enumerate(
        keyword_results,
        start=1,
    ):
        result["bm25_score"] = result["score"]
        result["bm25_rank"] = rank

    for rank, result in enumerate(
        semantic_results,
        start=1,
    ):
        result["semantic_score"] = result["score"]
        result["semantic_rank"] = rank

    fused_results = reciprocal_rank_fusion(
        ranked_lists=[
            keyword_results,
            semantic_results,
        ],
        limit=limit,
    )

    # Add method-specific information to each result.
    keyword_ids = {
        result["chunk_id"]
        for result in keyword_results
    }

    semantic_ids = {
        result["chunk_id"]
        for result in semantic_results
    }

    for result in fused_results:

        chunk_id = result["chunk_id"]

        methods = []

        if chunk_id in keyword_ids:
            methods.append("keyword")

        if chunk_id in semantic_ids:
            methods.append("semantic")

        result["retrieval_methods"] = methods

    return fused_results
