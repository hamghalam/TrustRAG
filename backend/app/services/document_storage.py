
import json

from backend.app.database.database import get_connection
from backend.app.services.embedding_service import (
    MODEL_NAME,
    EMBEDDING_DIMENSION,
)


def save_document(
    document_id: str,
    filename: str,
    page_count: int,
    chunks: list[dict],
    embeddings: list[list[float]] | None = None,
):
    """
    Save document metadata, chunks, and optional embeddings.

    All database operations are performed in one transaction.
    """

    if embeddings is not None:
        if len(embeddings) != len(chunks):
            raise ValueError(
                "Number of embeddings must match number of chunks"
            )

        if any(
            len(embedding) != EMBEDDING_DIMENSION
            for embedding in embeddings
        ):
            raise ValueError(
                f"Each embedding must have "
                f"{EMBEDDING_DIMENSION} dimensions"
            )

    connection = get_connection()

    try:
        # Save document metadata.
        connection.execute(
            """
            INSERT INTO documents (
                document_id,
                filename,
                page_count,
                chunk_count
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                document_id,
                filename,
                page_count,
                len(chunks),
            ),
        )

        # Save all document chunks.
        chunk_records = [
            (
                chunk["chunk_id"],
                document_id,
                chunk["page_number"],
                chunk["chunk_index"],
                chunk["text"],
            )
            for chunk in chunks
        ]

        connection.executemany(
            """
            INSERT INTO chunks (
                chunk_id,
                document_id,
                page_number,
                chunk_index,
                text
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            chunk_records,
        )

        # Save embeddings when provided.
        if embeddings is not None:

            embedding_records = [
                (
                    chunk["chunk_id"],
                    json.dumps(embedding),
                    MODEL_NAME,
                )
                for chunk, embedding in zip(chunks, embeddings)
            ]

            connection.executemany(
                """
                INSERT INTO chunk_embeddings (
                    chunk_id,
                    embedding,
                    model_name
                )
                VALUES (?, ?, ?)
                """,
                embedding_records,
            )

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


def get_all_documents():
    connection = get_connection()

    try:
        rows = connection.execute(
            """
            SELECT
                document_id,
                filename,
                page_count,
                chunk_count,
                uploaded_at
            FROM documents
            ORDER BY uploaded_at DESC
            """
        ).fetchall()

        return [dict(row) for row in rows]

    finally:
        connection.close()


def get_document_by_id(document_id: str):
    connection = get_connection()

    try:
        document = connection.execute(
            """
            SELECT
                document_id,
                filename,
                page_count,
                chunk_count,
                uploaded_at
            FROM documents
            WHERE document_id = ?
            """,
            (document_id,),
        ).fetchone()

        if document is None:
            return None

        chunks = connection.execute(
            """
            SELECT
                chunk_id,
                page_number,
                chunk_index,
                text
            FROM chunks
            WHERE document_id = ?
            ORDER BY page_number, chunk_index
            """,
            (document_id,),
        ).fetchall()

        return {
            "document": dict(document),
            "chunks": [dict(chunk) for chunk in chunks],
        }

    finally:
        connection.close()


def delete_document(document_id: str) -> bool:
    connection = get_connection()

    try:
        cursor = connection.execute(
            """
            DELETE FROM documents
            WHERE document_id = ?
            """,
            (document_id,),
        )

        connection.commit()

        return cursor.rowcount > 0

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()
