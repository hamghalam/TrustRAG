
import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT = 120


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

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
    retrieve relevant information, and generate
    answers grounded in your documents.
    """
)


# --------------------------------------------------
# Session State
# --------------------------------------------------

SESSION_DEFAULTS = {
    "selected_document": None,
    "search_response": None,
    "search_error": None,
    "ask_response": None,
    "ask_error": None,
    "documents": None,
}

for key, default_value in SESSION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


# --------------------------------------------------
# Helper Functions
# --------------------------------------------------

def get_api_error(response):
    """Extract a readable error message from an API response."""
    try:
        data = response.json()
        return data.get("detail", response.text)
    except (ValueError, requests.exceptions.JSONDecodeError):
        return response.text or "Unknown API error."


def display_request_error(error, response=None):
    """Display an HTTP or connection error."""
    if response is not None:
        st.error(
            f"Request failed ({response.status_code}): "
            f"{get_api_error(response)}"
        )
    else:
        st.error(f"Could not connect to the backend: {error}")


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
        display_request_error(error, getattr(error, "response", None))


# --------------------------------------------------
# Document Upload
# --------------------------------------------------

st.divider()
st.header("📄 Document Ingestion")

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
    help="Maximum file size: 10 MB.",
)

if uploaded_file is not None:
    st.write(f"Selected file: **{uploaded_file.name}**")

    if st.button("Process Document", type="primary"):
        with st.spinner(
            "Extracting text, generating chunks, and creating embeddings..."
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
                    timeout=REQUEST_TIMEOUT,
                )
                response.raise_for_status()
                result = response.json()

                st.success("Document processed successfully!")

                col1, col2, col3 = st.columns(3)

                col1.metric("Pages", result["page_count"])
                col2.metric("Chunks", result["chunk_count"])
                col3.metric(
                    "Document ID",
                    result["document_id"][:8],
                )

                if "embedding_count" in result:
                    st.caption(
                        f"Generated embeddings: "
                        f"{result['embedding_count']} "
                        f"using {result.get('embedding_model', 'unknown model')}"
                    )

                st.subheader("Generated Chunks")

                for chunk in result["chunks"]:
                    with st.expander(
                        f"Page {chunk['page_number']} | "
                        f"Chunk {chunk['chunk_index'] + 1}"
                    ):
                        st.caption(f"Chunk ID: {chunk['chunk_id']}")
                        st.write(chunk["text"])

                # Refresh cached document information.
                st.session_state.documents = None

                # Previous answers may no longer reflect the knowledge base.
                st.session_state.ask_response = None

            except requests.RequestException as error:
                display_request_error(
                    error,
                    getattr(error, "response", None),
                )


# --------------------------------------------------
# Platform Modules
# --------------------------------------------------

st.divider()
st.header("Platform Modules")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📄 Documents")
    st.write("Upload, process, and manage PDF knowledge documents.")

with col2:
    st.subheader("🔍 Retrieval")
    st.write(
        "Retrieve relevant information using BM25, "
        "semantic similarity, and hybrid retrieval."
    )

with col3:
    st.subheader("🛡️ Quality & Security")
    st.write(
        "Build the foundation for grounded answers, "
        "reliability evaluation, and RAG security testing."
    )


# --------------------------------------------------
# Document Search
# --------------------------------------------------

st.divider()
st.header("🔍 Search Documents")

st.write(
    "Inspect matching document chunks using keyword, "
    "semantic, or hybrid retrieval."
)

search_query = st.text_input(
    "Enter keywords or a question",
    placeholder="e.g., What are the symptoms of diabetes?",
    key="document_search_query",
)

search_mode = st.radio(
    "Retrieval method",
    options=[
        "Keyword (BM25)",
        "Semantic",
        "Hybrid (BM25 + Semantic)",
    ],
    horizontal=True,
    key="search_mode",
)

search_endpoints = {
    "Keyword (BM25)": "/search",
    "Semantic": "/search/semantic",
    "Hybrid (BM25 + Semantic)": "/search/hybrid",
}

if st.button("Search", type="primary", key="document_search_button"):
    if not search_query.strip():
        st.session_state.search_error = "Please enter a search query."
        st.session_state.search_response = None
    else:
        endpoint = search_endpoints[search_mode]

        try:
            with st.spinner(f"Searching using {search_mode}..."):
                response = requests.get(
                    f"{API_URL}{endpoint}",
                    params={
                        "q": search_query.strip(),
                        "limit": 5,
                    },
                    timeout=REQUEST_TIMEOUT,
                )
                response.raise_for_status()

                search_data = response.json()
                search_data["search_mode"] = search_mode

                st.session_state.search_response = search_data
                st.session_state.search_error = None

        except requests.RequestException as error:
            st.session_state.search_error = (
                f"Search request failed: {get_api_error(error.response)}"
                if error.response is not None
                else f"Backend connection failed: {error}"
            )
            st.session_state.search_response = None

        except ValueError:
            st.session_state.search_error = (
                "The API returned an invalid response."
            )
            st.session_state.search_response = None


if st.session_state.search_error:
    st.error(st.session_state.search_error)


# --------------------------------------------------
# Display Search Results
# --------------------------------------------------

if st.session_state.search_response is not None:
    search_data = st.session_state.search_response
    result_count = search_data["result_count"]
    result_mode = search_data.get("search_mode", "Keyword (BM25)")

    st.subheader(f"Results ({result_count})")
    st.caption(f"Retrieval method: {result_mode}")

    if result_count == 0:
        st.info(
            "No matching chunks found. "
            "Try a different query or retrieval method."
        )

    else:
        for index, result in enumerate(
            search_data["results"],
            start=1,
        ):
            st.markdown(f"**Result {index}**")

            st.markdown(
                f"**Source:** {result['filename']}  \n"
                f"**Page:** {result['page_number']}  \n"
                f"**Chunk:** {result['chunk_index'] + 1}"
            )

            if result_mode == "Semantic":
                st.metric(
                    "Cosine similarity",
                    f"{result['score']:.4f}",
                )

            elif result_mode == "Hybrid (BM25 + Semantic)":
                st.metric(
                    "RRF score",
                    f"{result['rrf_score']:.10f}",
                )

                col1, col2 = st.columns(2)

                bm25_score = result.get("bm25_score")
                semantic_score = result.get("semantic_score")

                col1.metric(
                    "BM25 score",
                    f"{bm25_score:.4f}"
                    if bm25_score is not None else "N/A",
                )

                col2.metric(
                    "Semantic score",
                    f"{semantic_score:.4f}"
                    if semantic_score is not None else "N/A",
                )

                methods = result.get("retrieval_methods", [])
                if methods:
                    st.caption("Retrieved by: " + ", ".join(methods))

            else:
                st.metric(
                    "BM25 relevance score",
                    f"{result['score']:.4f}",
                )

                matched_terms = result.get("matched_terms", [])
                if matched_terms:
                    st.caption(
                        "Matched keywords: " + ", ".join(matched_terms)
                    )

            with st.expander("View matching text"):
                st.write(result["text"])

            st.divider()


# --------------------------------------------------
# Ask TrustRAG
# --------------------------------------------------

st.divider()
st.header("💬 Ask TrustRAG")

st.write(
    """
    Ask a question about your uploaded documents.
    TrustRAG retrieves relevant passages using hybrid retrieval,
    assembles the context, and asks the configured LLM to generate
    an answer grounded in those passages.
    """
)

ask_query = st.text_area(
    "Your question",
    placeholder="Ask a question about the information in your documents...",
    height=100,
    key="ask_query",
)

ask_col1, ask_col2 = st.columns(2)

with ask_col1:
    context_limit = st.slider(
        "Number of retrieved chunks",
        min_value=1,
        max_value=20,
        value=5,
        help="The maximum number of chunks retrieved for answer generation.",
        key="ask_context_limit",
    )

with ask_col2:
    max_context_chars = st.slider(
        "Maximum context characters",
        min_value=1000,
        max_value=20000,
        value=8000,
        step=1000,
        help="Maximum size of the assembled document context.",
        key="ask_max_context_chars",
    )

ask_button = st.button(
    "Ask TrustRAG",
    type="primary",
    use_container_width=True,
    key="ask_button",
)

if ask_button:
    if not ask_query.strip():
        st.session_state.ask_error = "Please enter a question."
        st.session_state.ask_response = None

    else:
        request_payload = {
            "query": ask_query.strip(),
            "limit": context_limit,
            "max_context_chars": max_context_chars,
        }

        try:
            with st.spinner(
                "Retrieving documents and generating a grounded answer..."
            ):
                response = requests.post(
                    f"{API_URL}/ask",
                    json=request_payload,
                    timeout=REQUEST_TIMEOUT,
                )
                response.raise_for_status()

                st.session_state.ask_response = response.json()
                st.session_state.ask_error = None

        except requests.RequestException as error:
            st.session_state.ask_error = (
                f"Question answering failed: "
                f"{get_api_error(error.response)}"
                if error.response is not None
                else f"Backend connection failed: {error}"
            )
            st.session_state.ask_response = None

        except ValueError:
            st.session_state.ask_error = (
                "The API returned an invalid response."
            )
            st.session_state.ask_response = None


# --------------------------------------------------
# Display Answer and Sources
# --------------------------------------------------

if st.session_state.ask_error:
    st.error(st.session_state.ask_error)

if st.session_state.ask_response is not None:
    answer_data = st.session_state.ask_response

    st.subheader("Generated Answer")

    st.markdown(answer_data["answer"])

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "Sources",
        answer_data.get("source_count", 0),
    )

    metric2.metric(
        "Context characters",
        answer_data.get("context_char_count", 0),
    )

    metric3.metric(
        "LLM model",
        answer_data.get("model", "Unknown"),
    )

    sources = answer_data.get("sources", [])

    st.subheader("📚 Supporting Sources")

    if not sources:
        st.warning(
            "No source passages were returned for this answer."
        )

    else:
        for source in sources:
            citation = source.get("citation", "[?]")
            filename = source.get("filename", "Unknown document")
            page_number = source.get("page_number", "Unknown")
            chunk_index = source.get("chunk_index")

            title = (
                f"{citation} {filename} — Page {page_number}"
            )

            if chunk_index is not None:
                title += f" — Chunk {chunk_index + 1}"

            with st.expander(title):
                st.write(f"**Citation:** {citation}")
                st.write(f"**Document:** {filename}")
                st.write(f"**Page:** {page_number}")

                if chunk_index is not None:
                    st.write(f"**Chunk:** {chunk_index + 1}")

                if source.get("retrieval_score") is not None:
                    st.write(
                        "**Retrieval score:** "
                        f"{source['retrieval_score']}"
                    )

                methods = source.get("retrieval_methods", [])
                if methods:
                    st.write(
                        "**Retrieval methods:** "
                        + ", ".join(methods)
                    )

                if source.get("truncated"):
                    st.caption(
                        "This source was truncated to fit the context limit."
                    )

                st.caption(
                    f"Chunk ID: {source.get('chunk_id', 'Unknown')}"
                )


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

        st.session_state.documents = response.json()["documents"]

    except requests.RequestException as error:
        display_request_error(
            error,
            getattr(error, "response", None),
        )


# --------------------------------------------------
# Display Document Library
# --------------------------------------------------

if st.session_state.documents is not None:
    documents = st.session_state.documents

    st.success(f"Found {len(documents)} documents.")

    if not documents:
        st.info("No documents have been uploaded yet.")

    else:
        st.subheader("Uploaded Documents")

        for document in documents:
            document_id = document["document_id"]

            with st.expander(document["filename"]):
                col1, col2, col3 = st.columns(3)

                col1.metric("Pages", document["page_count"])
                col2.metric("Chunks", document["chunk_count"])
                col3.metric("Document ID", document_id[:8])

                st.caption(
                    f"Uploaded at: {document['uploaded_at']}"
                )

                st.text("Full Document ID")
                st.code(document_id, language="text")

                if st.button(
                    "View Document Chunks",
                    key=f"view_{document_id}",
                ):
                    st.session_state.selected_document = document_id
                    st.rerun()

                if st.button(
                    "Delete Document",
                    key=f"delete_{document_id}",
                ):
                    try:
                        delete_response = requests.delete(
                            f"{API_URL}/documents/{document_id}",
                            timeout=15,
                        )
                        delete_response.raise_for_status()

                        st.session_state.documents = [
                            item
                            for item in st.session_state.documents
                            if item["document_id"] != document_id
                        ]

                        if st.session_state.selected_document == document_id:
                            st.session_state.selected_document = None

                        # Clear previous answers because their sources may
                        # no longer exist in the knowledge base.
                        st.session_state.ask_response = None

                        st.success("Document deleted successfully.")
                        st.rerun()

                    except requests.RequestException as error:
                        display_request_error(
                            error,
                            getattr(error, "response", None),
                        )


# --------------------------------------------------
# Selected Document Details
# --------------------------------------------------

selected_document_id = st.session_state.selected_document

if selected_document_id:
    st.divider()
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

        st.subheader(document_detail["filename"])

        col1, col2 = st.columns(2)

        col1.metric("Pages", document_detail["page_count"])
        col2.metric("Chunks", len(chunks))

        st.caption(f"Document ID: {selected_document_id}")
        st.subheader("Extracted Text Chunks")

        if not chunks:
            st.warning("No chunks found for this document.")

        else:
            for chunk in chunks:
                with st.expander(
                    f"Page {chunk['page_number']} | "
                    f"Chunk {chunk['chunk_index'] + 1}"
                ):
                    st.caption(f"Chunk ID: {chunk['chunk_id']}")
                    st.write(chunk["text"])

        if st.button("Close Document Details"):
            st.session_state.selected_document = None
            st.rerun()

    except requests.RequestException as error:
        display_request_error(
            error,
            getattr(error, "response", None),
        )

        if st.button("Close Document Details"):
            st.session_state.selected_document = None
            st.rerun()
