
import json
import math

from backend.app.database.database import get_connection

from backend.app.services.embedding_service import (
    generate_embedding,
    MODEL_NAME,
)


def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float],
) -> float:
    """
    Calculate cosine similarity between two vectors.
    """

    if len(vector_a) != len(vector_b):
        raise ValueError(
            "Vectors must have the same dimensions."
        )

    if not vector_a:
        raise ValueError(
            "Vectors cannot be empty."
        )

    dot_product = sum(
        a * b
        for a, b in zip(vector_a, vector_b)
    )

    norm_a = math.sqrt(
        sum(a * a for a in vector_a)
    )

    norm_b = math.sqrt(
        sum(b * b for b in vector_b)
    )

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def semantic_search_chunks(
    query: str,
    limit: int = 5,
) -> list[dict]:
    """
    Search document chunks using semantic similarity.
    """

    if limit <= 0:
        raise ValueError("limit must be positive")

    if not query or not query.strip():
        return []

    connection = get_connection()

    try:
        rows = connection.execute(
            """
            SELECT
                chunks.chunk_id,
                chunks.document_id,
                chunks.page_number,
                chunks.chunk_index,
                chunks.text,
                documents.filename,
                chunk_embeddings.embedding
            FROM chunks
            JOIN documents
                ON chunks.document_id = documents.document_id
            JOIN chunk_embeddings
                ON chunks.chunk_id = chunk_embeddings.chunk_id
            WHERE chunk_embeddings.model_name = ?
            """,
            (MODEL_NAME,),
        ).fetchall()

    finally:
        connection.close()

    if not rows:
        return []

    query_embedding = generate_embedding(query)

    results = []

    for row in rows:
        chunk = dict(row)

        stored_embedding = json.loads(
            chunk.pop("embedding")
        )

        score = cosine_similarity(
            query_embedding,
            stored_embedding,
        )

        chunk["score"] = round(score, 4)

        results.append(chunk)

    results.sort(
        key=lambda item: (
            -item["score"],
            item["page_number"],
            item["chunk_index"],
        )
    )

    return results[:limit]
