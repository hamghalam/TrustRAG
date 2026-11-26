
from io import BytesIO
from pypdf import PdfReader


def extract_text_from_pdf(pdf_bytes: bytes) -> list[dict]:
    """
    Extract text from a PDF while preserving page numbers.

    Returns:
        A list containing extracted text and page metadata.
    """

    reader = PdfReader(BytesIO(pdf_bytes))

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            pages.append({
                "page_number": page_number,
                "text": text.strip()
            })

    return pages


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[str]:
    """
    Split text into overlapping character-based chunks.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be non-negative "
            "and smaller than chunk_size"
        )

    text = text.strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        chunks.append(text[start:end])

        if end >= len(text):
            break

        start = end - chunk_overlap

    return chunks
