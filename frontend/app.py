
import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="TrustRAG",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 TrustRAG")

st.subheader(
    "RAG Quality, Security & Evaluation Platform"
)

st.markdown(
    """
    Upload documents, build knowledge bases,
    and evaluate the reliability of RAG systems.
    """
)


# --------------------------------------------------
# Session State
# --------------------------------------------------

if "selected_document" not in st.session_state:
    st.session_state.selected_document = None


# --------------------------------------------------
# Backend Status
# --------------------------------------------------

st.divider()

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

        st.error(
            f"Backend connection failed: {error}"
        )


# --------------------------------------------------
# Document Upload
# --------------------------------------------------

st.divider()

st.header("📄 Document Ingestion")

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
    help="Maximum file size: 10 MB",
)

if uploaded_file is not None:

    st.write(
        f"Selected file: **{uploaded_file.name}**"
    )

    if st.button(
        "Process Document",
        type="primary",
    ):

        with st.spinner(
            "Extracting text and generating chunks..."
        ):

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

                st.success(
                    "Document processed successfully!"
                )

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Pages",
                    result["page_count"],
                )

                col2.metric(
                    "Chunks",
                    result["chunk_count"],
                )

                col3.metric(
                    "Document ID",
                    result["document_id"][:8],
                )

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

                    st.error(
                        f"Backend connection failed: {error}"
                    )


# --------------------------------------------------
# Platform Modules
# --------------------------------------------------

st.divider()

st.header("Platform Modules")

col1, col2, col3 = st.columns(3)

with col1:

    st.subheader("📄 Documents")

    st.write(
        "Upload and manage knowledge documents."
    )

with col2:

    st.subheader("🔍 Retrieval")

    st.write(
        "Search documents using semantic and keyword retrieval."
    )

with col3:

    st.subheader("🛡️ Evaluation")

    st.write(
        "Evaluate retrieval quality and answer reliability."
    )


# -----------------------------
# Document Search
# -----------------------------

st.divider()
st.header("🔍 Search Documents")

st.write(
    "Search across your uploaded documents using keyword matching."
)

if "search_response" not in st.session_state:
    st.session_state.search_response = None

if "search_error" not in st.session_state:
    st.session_state.search_error = None

search_query = st.text_input(
    "Enter keywords or a question",
    placeholder="e.g., diabetes treatment, medical imaging",
    key="document_search_query",
)

search_col1, search_col2 = st.columns([1, 5])

with search_col1:
    search_button = st.button(
        "Search",
        type="primary",
        use_container_width=True,
    )

if search_button:
    if not search_query.strip():
        st.session_state.search_error = "Please enter a search query."
        st.session_state.search_response = None

    else:
        try:
            response = requests.get(
                f"{API_URL}/search",
                params={
                    "q": search_query.strip(),
                    "limit": 5,
                },
                timeout=15,
            )

            response.raise_for_status()

            st.session_state.search_response = response.json()
            st.session_state.search_error = None

        except requests.exceptions.RequestException as error:
            st.session_state.search_error = (
                f"Search request failed: {error}"
            )
            st.session_state.search_response = None

        except ValueError:
            st.session_state.search_error = (
                "The API returned an invalid response."
            )
            st.session_state.search_response = None


# Display search errors
if st.session_state.search_error:
    st.error(st.session_state.search_error)


# Display search results
if st.session_state.search_response is not None:

    search_data = st.session_state.search_response

    st.subheader(
        f"Results ({search_data['result_count']})"
    )

    if search_data["result_count"] == 0:
        st.info("No matching chunks found. Try different keywords.")

    else:
        for index, result in enumerate(search_data["results"], start=1):

            st.markdown(f"**Result {index}**")

            st.markdown(
                f"**Source:** {result['filename']}  \n"
                f"**Page:** {result['page_number']}  \n"
                f"**Chunk:** {result['chunk_index']}  \n"
                f"**Relevance score:** {result['score']:.2f}"
            )

            matched_terms = result.get("matched_terms", [])

            if matched_terms:
                st.caption(
                    "Matched keywords: "
                    + ", ".join(matched_terms)
                )

            with st.expander("View matching text"):
                st.write(result["text"])

            st.divider()





# --------------------------------------------------
# Document Library
# --------------------------------------------------

st.divider()

st.header("📚 Document Library")

if st.button("Refresh Document Library"):

    try:
        response = requests.get(
            f"{API_URL}/documents",
            timeout=10,
        )

        response.raise_for_status()

        result = response.json()

        st.session_state.documents = result["documents"]

    except requests.RequestException as error:

        if error.response is not None:

            st.error(
                f"Could not retrieve documents: "
                f"{error.response.text}"
            )

        else:

            st.error(
                f"Backend connection failed: {error}"
            )


# Display documents stored in session state

if "documents" in st.session_state:

    documents = st.session_state.documents

    st.success(
        f"Found {len(documents)} documents."
    )

    if not documents:

        st.info(
            "No documents have been uploaded yet."
        )

    else:

        st.subheader("Uploaded Documents")

        for document in documents:

            with st.expander(document["filename"]):

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Pages",
                    document["page_count"],
                )

                col2.metric(
                    "Chunks",
                    document["chunk_count"],
                )

                col3.metric(
                    "Document ID",
                    document["document_id"][:8],
                )

                st.caption(
                    f"Uploaded at: {document['uploaded_at']}"
                )

                st.text("Full Document ID")

                st.code(
                    document["document_id"],
                    language="text",
                )

                # View Document Chunks

                if st.button(
                    "View Document Chunks",
                    key=f"view_{document['document_id']}",
                ):

                    st.session_state.selected_document = (
                        document["document_id"]
                    )

                    st.rerun()

                # Delete Document

                if st.button(
                    "Delete Document",
                    key=f"delete_{document['document_id']}",
                ):

                    try:
                        delete_response = requests.delete(
                            f"{API_URL}/documents/"
                            f"{document['document_id']}",
                            timeout=15,
                        )

                        delete_response.raise_for_status()

                        st.session_state.documents = [
                            item
                            for item in st.session_state.documents
                            if item["document_id"]
                            != document["document_id"]
                        ]

                        if (
                            st.session_state.selected_document
                            == document["document_id"]
                        ):

                            st.session_state.selected_document = None

                        st.success(
                            "Document deleted successfully."
                        )

                        st.rerun()

                    except requests.RequestException as error:

                        st.error(
                            f"Could not delete document: {error}"
                        )


# --------------------------------------------------
# Selected Document Details
# --------------------------------------------------

st.divider()

selected_document_id = (
    st.session_state.selected_document
)

if selected_document_id:

    st.header("📄 Document Details")

    try:
        detail_response = requests.get(
            f"{API_URL}/documents/{selected_document_id}",
            timeout=15,
        )

        detail_response.raise_for_status()

        detail_result = detail_response.json()

        document_detail = detail_result["document"]

        chunks = detail_result["chunks"]

        st.subheader(
            document_detail["filename"]
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "Pages",
            document_detail["page_count"],
        )

        col2.metric(
            "Chunks",
            len(chunks),
        )

        st.caption(
            f"Document ID: {selected_document_id}"
        )

        st.subheader("Extracted Text Chunks")

        if not chunks:

            st.warning(
                "No chunks found for this document."
            )

        else:

            for chunk in chunks:

                with st.expander(
                    f"Page {chunk['page_number']} | "
                    f"Chunk {chunk['chunk_index'] + 1}"
                ):

                    st.caption(
                        f"Chunk ID: {chunk['chunk_id']}"
                    )

                    st.write(chunk["text"])

        if st.button("Close Document Details"):

            st.session_state.selected_document = None

            st.rerun()

    except requests.RequestException as error:

        st.error(
            f"Could not retrieve document details: {error}"
        )

        if st.button("Close Document Details"):

            st.session_state.selected_document = None

            st.rerun()
