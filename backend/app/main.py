
from uuid import uuid4

from fastapi import FastAPI, UploadFile, File, HTTPException, Query

from backend.app.database.database import initialize_database

from backend.app.services.document_processor import (
    extract_text_from_pdf,
    chunk_text,
)

from backend.app.services.document_storage import (
    save_document,
    get_all_documents,
    get_document_by_id,
    delete_document,
)

from backend.app.services.retrieval_service import search_chunks

from backend.app.services.embedding_service import (
    generate_embeddings,
    MODEL_NAME,
)


app = FastAPI(
    title="TrustRAG API",
    description=(
        "A quality, security, and evaluation "
        "platform for RAG systems."
    ),
    version="0.1.0",
)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@app.on_event("startup")
def startup_event():
    initialize_database()


@app.get("/")
def root():
    return {
        "project": "TrustRAG",
        "status": "running",
        "version": "0.1.0",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "TrustRAG API",
    }


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):

    # 1. Validate filename.
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    # 2. Read file into memory.
    pdf_bytes = await file.read()

    # 3. Validate file size.
    if len(pdf_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="PDF file exceeds the 10 MB limit.",
        )

    if not pdf_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    # 4. Check PDF signature.
    if not pdf_bytes.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid PDF.",
        )

    # 5. Extract text.
    try:
        pages = extract_text_from_pdf(pdf_bytes)

    except Exception:
        raise HTTPException(
            status_code=400,
            detail=(
                "Could not read this PDF. "
                "It may be corrupted or encrypted."
            ),
        )

    if not pages:
        raise HTTPException(
            status_code=422,
            detail="No extractable text found in this PDF.",
        )

    # 6. Generate document ID.
    document_id = str(uuid4())

    # 7. Generate chunks.
    chunks = []

    for page in pages:
        page_chunks = chunk_text(page["text"])

        for chunk_index, chunk in enumerate(page_chunks):
            chunks.append({
                "chunk_id": str(uuid4()),
                "document_id": document_id,
                "page_number": page["page_number"],
                "chunk_index": chunk_index,
                "text": chunk,
            })

    if not chunks:
        raise HTTPException(
            status_code=422,
            detail="No text chunks could be generated from this PDF.",
        )

    # 8. Generate embeddings in batches.
    try:
        chunk_texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = generate_embeddings(chunk_texts)

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Failed to generate document embeddings.",
        )

    # 9. Save document, chunks, and embeddings.
    try:
        save_document(
            document_id=document_id,
            filename=file.filename,
            page_count=len(pages),
            chunks=chunks,
            embeddings=embeddings,
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Failed to save document and embeddings in database.",
        )

    # 10. Return processing result.
    return {
        "document_id": document_id,
        "filename": file.filename,
        "page_count": len(pages),
        "chunk_count": len(chunks),
        "embedding_count": len(embeddings),
        "embedding_model": MODEL_NAME,
        "status": "stored",
        "chunks": chunks,
    }


@app.get("/documents")
def list_documents():

    try:
        documents = get_all_documents()

        return {
            "document_count": len(documents),
            "documents": documents,
        }

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve documents from database.",
        )


@app.get("/documents/{document_id}")
def retrieve_document(document_id: str):

    try:
        result = get_document_by_id(document_id)

        if result is None:
            raise HTTPException(
                status_code=404,
                detail="Document not found.",
            )

        return result

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve document.",
        )


@app.delete("/documents/{document_id}")
def remove_document(document_id: str):

    try:
        deleted = delete_document(document_id)

        if not deleted:
            raise HTTPException(
                status_code=404,
                detail="Document not found.",
            )

        return {
            "message": "Document deleted successfully.",
            "document_id": document_id,
        }

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Failed to delete document.",
        )


@app.get("/search")
def search_documents(
    q: str = Query(..., min_length=1),
    limit: int = Query(default=5, ge=1, le=50),
):

    try:
        results = search_chunks(
            query=q,
            limit=limit,
        )

        return {
            "query": q,
            "result_count": len(results),
            "results": results,
        }

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Search operation failed.",
        )
