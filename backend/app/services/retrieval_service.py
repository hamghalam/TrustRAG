
import re
import math
from collections import Counter

from backend.app.database.database import get_connection


def tokenize(text: str) -> list[str]:
    """Convert text into lowercase word tokens."""
    return re.findall(r"\w+", text.lower())


def calculate_idf(
    total_documents: int,
    document_frequency: int,
) -> float:
    """
    Calculate inverse document frequency (IDF).

    Rare terms receive higher weights than common terms.
    """

    return math.log(
        1 + (
            total_documents - document_frequency + 0.5
        ) / (
            document_frequency + 0.5
        )
    )


def calculate_bm25_score(
    term_frequency: int,
    document_length: int,
    average_length: float,
    idf: float,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """
    Calculate the BM25 score contribution of one query term.
    """

    if term_frequency <= 0:
        return 0.0

    if average_length <= 0:
        return 0.0

    denominator = term_frequency + k1 * (
        1 - b + b * document_length / average_length
    )

    return idf * (
        term_frequency * (k1 + 1)
    ) / denominator


def search_chunks(
    query: str,
    limit: int = 5,
) -> list[dict]:
    """
    Search document chunks using BM25 ranking.

    Results are ranked based on:
    - Term frequency (TF)
    - Inverse document frequency (IDF)
    - Chunk length normalization
    """

    if limit <= 0:
        raise ValueError("limit must be positive")

    query_terms = list(dict.fromkeys(tokenize(query)))

    if not query_terms:
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
                documents.filename
            FROM chunks
            JOIN documents
                ON chunks.document_id = documents.document_id
            """
        ).fetchall()

        if not rows:
            return []

        # Tokenize all chunks.
        documents = []

        for row in rows:
            chunk = dict(row)
            tokens = tokenize(chunk["text"])

            if not tokens:
                continue

            documents.append({
                "chunk": chunk,
                "tokens": tokens,
                "term_frequencies": Counter(tokens),
                "length": len(tokens),
            })

        total_documents = len(documents)

        if total_documents == 0:
            return []

        # Calculate average chunk length.
        average_length = (
            sum(doc["length"] for doc in documents)
            / total_documents
        )

        # Calculate document frequency for each term.
        document_frequency = Counter()

        for doc in documents:
            document_frequency.update(
                set(doc["term_frequencies"])
            )

        # BM25 parameters.
        k1 = 1.5
        b = 0.75

        # Calculate IDF for query terms.
        idf = {
            term: calculate_idf(
                total_documents,
                document_frequency.get(term, 0),
            )
            for term in query_terms
        }

        results = []

        # Calculate BM25 score for each chunk.
        for doc in documents:

            score = 0.0
            matched_terms = []

            for term in query_terms:

                tf = doc["term_frequencies"].get(term, 0)

                if tf == 0:
                    continue

                matched_terms.append(term)

                term_score = calculate_bm25_score(
                    term_frequency=tf,
                    document_length=doc["length"],
                    average_length=average_length,
                    idf=idf[term],
                    k1=k1,
                    b=b,
                )

                score += term_score

            # Only return chunks containing query terms.
            if score > 0:

                chunk = doc["chunk"]

                chunk["score"] = round(score, 4)
                chunk["matched_terms"] = matched_terms

                results.append(chunk)

        # Highest BM25 score first.
        results.sort(
            key=lambda item: (
                -item["score"],
                item["page_number"],
                item["chunk_index"],
            )
        )

        return results[:limit]

    finally:
        connection.close()
