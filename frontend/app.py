
import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="TrustRAG",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 TrustRAG")
st.subheader("RAG Quality, Security & Evaluation Platform")

st.markdown(
    """
    Upload documents, build knowledge bases,
    and evaluate the reliability of RAG systems.
    """
)

st.divider()

# Backend status
st.header("System Status")

if st.button("Check Backend Status"):
    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=5,
        )
        response.raise_for_status()

        st.success("Backend is running!")
        st.json(response.json())

    except requests.RequestException as error:
        st.error(f"Backend connection failed: {error}")

st.divider()

# Document upload
st.header("📄 Document Ingestion")

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
    help="Maximum file size: 10 MB",
)

if uploaded_file is not None:

    st.write(f"Selected file: **{uploaded_file.name}**")

    if st.button("Process Document", type="primary"):

        with st.spinner("Extracting text and generating chunks..."):

            try:
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        "application/pdf",
                    )
                }

                response = requests.post(
                    f"{API_URL}/documents/upload",
                    files=files,
                    timeout=60,
                )

                response.raise_for_status()

                result = response.json()

                st.success("Document processed successfully!")

                col1, col2, col3 = st.columns(3)

                col1.metric("Pages", result["page_count"])
                col2.metric("Chunks", result["chunk_count"])
                col3.metric("Document ID", result["document_id"][:8])

                st.divider()

                st.subheader("Generated Chunks")

                for chunk in result["chunks"]:

                    with st.expander(
                        f"Page {chunk['page_number']} | "
                        f"Chunk {chunk['chunk_index'] + 1}"
                    ):
                        st.caption(
                            f"Chunk ID: {chunk['chunk_id']}"
                        )

                        st.write(chunk["text"])

            except requests.RequestException as error:
                if error.response is not None:
                    st.error(
                        f"Upload failed: {error.response.text}"
                    )
                else:
                    st.error(f"Backend connection failed: {error}")

st.divider()

# Platform modules
st.header("Platform Modules")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📄 Documents")
    st.write("Upload and manage knowledge documents.")

with col2:
    st.subheader("🔍 Retrieval")
    st.write("Search documents using semantic and keyword retrieval.")

with col3:
    st.subheader("🛡️ Evaluation")
    st.write("Evaluate retrieval quality and answer reliability.")
