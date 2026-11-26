
from backend.app.database.database import get_connection


def save_document(
    document_id: str,
    filename: str,
    page_count: int,
    chunks: list[dict],
):
    connection = get_connection()

    try:
        # Save document metadata
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

        # Save all document chunks
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

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()
