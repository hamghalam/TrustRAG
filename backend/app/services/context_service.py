
from backend.app.services.hybrid_retrieval_service import (
    hybrid_search_chunks,
)


DEFAULT_CONTEXT_LIMIT = 5
DEFAULT_MAX_CONTEXT_CHARS = 8000


def assemble_context(
    query: str,
    retrieved_chunks: list[dict],
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
) -> dict:
    """
    Assemble retrieved document chunks into an LLM-ready context.

    Each included chunk receives a citation identifier such as [1].
    The returned sources map these identifiers to document metadata.
    """

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    if max_context_chars <= 0:
        raise ValueError("max_context_chars must be positive.")

    context_parts = []
    sources = []
    seen_chunk_ids = set()
    current_length = 0

    for chunk in retrieved_chunks:
        chunk_id = chunk.get("chunk_id")
        text = chunk.get("text", "").strip()

        if not chunk_id or not text:
            continue

        if chunk_id in seen_chunk_ids:
            continue

        citation_number = len(sources) + 1

        filename = chunk.get("filename", "Unknown document")
        page_number = chunk.get("page_number", "Unknown")

        source_header = (
            f"[{citation_number}]\n"
            f"Document: {filename}\n"
            f"Page: {page_number}\n"
            f"Content:\n"
        )

        source_text = source_header + text
        separator_length = 2 if context_parts else 0

        remaining_chars = (
            max_context_chars
            - current_length
            - separator_length
        )

        if remaining_chars <= len(source_header):
            break

        was_truncated = len(source_text) > remaining_chars

        if was_truncated:
            available_text_chars = remaining_chars - len(source_header)

            if available_text_chars <= 0:
                break

            source_text = (
                source_header
                + text[:available_text_chars]
            )

        context_parts.append(source_text)
        current_length += len(source_text) + separator_length
        seen_chunk_ids.add(chunk_id)

        sources.append({
            "citation": f"[{citation_number}]",
            "chunk_id": chunk_id,
            "document_id": chunk.get("document_id"),
            "filename": filename,
            "page_number": page_number,
            "chunk_index": chunk.get("chunk_index"),
            "retrieval_score": chunk.get("score"),
            "retrieval_methods": chunk.get("retrieval_methods", []),
            "truncated": was_truncated,
        })

        if was_truncated:
            break

    return {
        "query": query.strip(),
        "context": "\n\n".join(context_parts),
        "sources": sources,
        "source_count": len(sources),
        "context_char_count": current_length,
    }


def retrieve_and_assemble_context(
    query: str,
    limit: int = DEFAULT_CONTEXT_LIMIT,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
) -> dict:
    """
    Retrieve relevant chunks using hybrid search
    and assemble them into an LLM-ready context.
    """

    if limit <= 0:
        raise ValueError("limit must be positive.")

    retrieved_chunks = hybrid_search_chunks(
        query=query,
        limit=limit,
    )

    return assemble_context(
        query=query,
        retrieved_chunks=retrieved_chunks,
        max_context_chars=max_context_chars,
    )
