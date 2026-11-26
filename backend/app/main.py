
from uuid import uuid4
from backend.app.services.document_storage import save_document

from backend.app.database.database import initialize_database

from fastapi import FastAPI, UploadFile, File, HTTPException

from backend.app.services.document_processor import (
    extract_text_from_pdf,
    chunk_text,
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

    # 1. Validate filename
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    # 2. Read file into memory
    pdf_bytes = await file.read()

    # 3. Validate file size
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

    # 4. Check PDF signature
    if not pdf_bytes.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid PDF.",
        )

    # 5. Extract text
    try:
        pages = extract_text_from_pdf(pdf_bytes)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not read this PDF. It may be corrupted or encrypted.",
        )

    if not pages:
        raise HTTPException(
            status_code=422,
            detail="No extractable text found in this PDF.",
        )

    # 6. Generate document ID
    document_id = str(uuid4())

    # 7. Generate chunks
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

    # 8. Save document and chunks in SQLite
    try:
        save_document(
            document_id=document_id,
            filename=file.filename,
            page_count=len(pages),
            chunks=chunks,
        )
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Failed to save document in database.",
        )

    # 9. Return processing result
    return {
        "document_id": document_id,
        "filename": file.filename,
        "page_count": len(pages),
        "chunk_count": len(chunks),
        "status": "stored",
        "chunks": chunks,
    }

