from pathlib import Path
import streamlit as st
from src.embeddings.embedding_model import LocalEmbeddingModel
from src.indexing.indexer import index_document
from src.rag.answer_generator import generate_answer
from src.search.semantic_search import semantic_search
from src.storage.vector_store import VectorStore

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
VECTOR_DB_DIR = PROJECT_ROOT / "data" / "vector_db"

SUPPORTED_EXTENSIONS = ["pdf", "txt", "md"]
DEFAULT_TOP_K = 5

st.set_page_config(
    page_title="Personal Knowledge Search",
    page_icon="📚",
    layout="wide",
)

@st.cache_resource
def get_embedding_model():
    """Load the local embedding model once per Streamlit process."""
    return LocalEmbeddingModel()

@st.cache_resource
def get_vector_store():
    """Reuse the persistent local ChromaDB collection."""
    return VectorStore(persist_directory=VECTOR_DB_DIR)

def save_uploaded_file(uploaded_file) -> Path:
    """Save an uploaded document to the local document directory."""
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    # Streamlit upload names are reduced to a filename, not a path.
    filename = Path(uploaded_file.name).name
    destination = DOCUMENTS_DIR / filename
    destination.write_bytes(uploaded_file.getvalue())

    return destination

def render_citations(citations):
    """Display source references returned by the RAG pipeline."""
    if not citations:
        st.caption("No source citations were returned for this answer.")
        return

    st.markdown("**Sources**")

    for citation in citations:
        source = citation.get("source", "Unknown source")
        page = citation.get("page_number")
        reference = citation.get("reference", "")

        if page is None:
            location = "Page not available"
        else:
            location = f"Page {page}"

        st.markdown(
            f"- **[{reference}]** `{source}` — {location}"
        )

def main():
    st.title("📚 Personal Knowledge Search")
    st.caption(
        "Search your documents by meaning and ask questions "
        "using locally stored knowledge."
    )

    with st.sidebar:
        st.header("Document Library")

        uploaded_files = st.file_uploader(
            "Upload documents",
            type=SUPPORTED_EXTENSIONS,
            accept_multiple_files=True,
            help="Supported formats: PDF, TXT, and Markdown.",
        )

        index_clicked = st.button(
            "Index uploaded documents",
            type="primary",
            disabled=not uploaded_files,
            use_container_width=True,
        )

        st.divider()
        st.subheader("Search settings")

        top_k = st.slider(
            "Passages to retrieve",
            min_value=1,
            max_value=10,
            value=DEFAULT_TOP_K,
        )

        st.caption("Embedding model: all-MiniLM-L6-v2")
        st.caption("Answer model: llama3.2:3b")

    # Initialize lightweight session state.
    if "last_index_results" not in st.session_state:
        st.session_state.last_index_results = []

    if index_clicked:
        try:
            with st.spinner("Loading local models and indexing documents..."):
                embedding_model = get_embedding_model()
                vector_store = get_vector_store()

                indexing_results = []

                for uploaded_file in uploaded_files:
                    try:
                        saved_path = save_uploaded_file(uploaded_file)

                        result = index_document(
                            file_path=saved_path,
                            embedding_model=embedding_model,
                            vector_store=vector_store,
                        )
                        indexing_results.append(
                            {
                                "filename": uploaded_file.name,
                                "result": result,
                                "error": None,
                            }
                        )
                    except Exception as exc:
                        indexing_results.append(
                            {
                                "filename": uploaded_file.name,
                                "result": None,
                                "error": str(exc),
                            }
                        )

                st.session_state.last_index_results = indexing_results

            st.rerun()

        except Exception as exc:
            st.error(f"Could not initialize the indexing pipeline: {exc}")

    # Show the most recent indexing summary.
    if st.session_state.last_index_results:
        st.subheader("Latest indexing results")

        for item in st.session_state.last_index_results:
            if item["error"]:
                st.error(
                    f"{item['filename']}: {item['error']}"
                )
            else:
                result = item["result"]
                st.success(
                    f"{item['filename']}: "
                    f"{result['chunks_stored']} passages indexed."
                )

    # Initialize the persistent store for counts and search.
    try:
        vector_store = get_vector_store()
        indexed_count = vector_store.count()
    except Exception as exc:
        st.error(f"Could not open the local vector database: {exc}")
        st.stop()

    metric_col, info_col = st.columns([1, 3])

    with metric_col:
        st.metric("Indexed passages", indexed_count)

    with info_col:
        st.info(
            "Your vector database is stored locally in "
            "`data/vector_db`. Uploads are saved in `data/documents`."
        )

    st.divider()

    search_tab, ask_tab = st.tabs(
        ["🔎 Semantic Search", "💬 Ask Your Documents"]
    )

    with search_tab:
        st.subheader("Find relevant passages")

        search_query = st.text_input(
            "Search query",
            placeholder="e.g. What are the main ideas in my notes?",
            key="semantic_query",
        )

        if st.button("Search documents", key="run_search"):
            if not search_query.strip():
                st.warning("Enter a search query first.")
            elif indexed_count == 0:
                st.warning("Upload and index at least one document first.")
            else:
                try:
                    with st.spinner("Searching your documents..."):
                        embedding_model = get_embedding_model()
                        results = semantic_search(
                            query=search_query,
                            embedding_model=embedding_model,
                            vector_store=vector_store,
                            top_k=top_k,
                        )

                    if not results:
                        st.info("No relevant passages were found.")
                    else:
                        st.write(f"Found {len(results)} relevant passages.")

                        for number, result in enumerate(results, start=1):
                            metadata = result.get("metadata", {})
                            source = metadata.get("source", "Unknown source")
                            page = metadata.get("page_number")
                            distance = result.get("distance")

                            title = f"{number}. {source}"
                            if page is not None:
                                title += f" — Page {page}"

                            with st.expander(title, expanded=(number == 1)):
                                st.write(result.get("text", ""))

                                if distance is not None:
                                    st.caption(
                                        f"Cosine distance: {distance:.4f}"
                                    )

                except Exception as exc:
                    st.error(f"Search failed: {exc}")

    with ask_tab:
        st.subheader("Ask a question")

        question = st.text_area(
            "Your question",
            placeholder="Ask something that your documents might answer...",
            height=100,
            key="rag_question",
        )

        if st.button("Generate answer", key="generate_answer", type="primary"):
            if not question.strip():
                st.warning("Enter a question first.")
            elif indexed_count == 0:
                st.warning("Upload and index at least one document first.")
            else:
                try:
                    with st.spinner("Retrieving evidence and generating an answer..."):
                        embedding_model = get_embedding_model()

                        results = semantic_search(
                            query=question,
                            embedding_model=embedding_model,
                            vector_store=vector_store,
                            top_k=top_k,
                        )

                        if not results:
                            answer_result = {
                                "answer": (
                                    "I couldn't find relevant information "
                                    "in your indexed documents to answer "
                                    "this question."
                                ),
                                "citations": [],
                            }
                        else:
                            answer_result = generate_answer(
                                question=question,
                                search_results=results,
                            )

                    st.markdown("### Answer")
                    st.write(answer_result["answer"])
                    render_citations(answer_result.get("citations", []))

                except Exception as exc:
                    st.error(
                        "Answer generation failed. Check that Ollama is "
                        "running and the configured model is available. "
                        f"Details: {exc}"
                    )

if __name__ == "__main__":
    main()