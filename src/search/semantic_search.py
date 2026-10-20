from src.embeddings.embedding_model import LocalEmbeddingModel
from src.storage.vector_store import VectorStore
def semantic_search(
    query: str,
    embedding_model: LocalEmbeddingModel,
    vector_store: VectorStore,
    top_k: int = 5,
) -> list[dict]:
    """
    Search indexed document chunks using a natural-language query.

    Results include passage text, source metadata, and cosine distance.
    Lower cosine distance indicates a closer match.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    if vector_store.count() == 0:
        return []

    query_embedding = embedding_model.embed_text(query.strip())

    return vector_store.search(
        query_embedding=query_embedding,
        n_results=top_k,
    )
