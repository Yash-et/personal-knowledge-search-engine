from pathlib import Path
from src.chunking.text_chunker import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    chunk_text,
)
from src.embeddings.embedding_model import LocalEmbeddingModel
from src.ingestion.document_loader import load_document
from src.storage.vector_store import VectorStore

def index_document(
    file_path: str | Path,
    embedding_model: LocalEmbeddingModel,
    vector_store: VectorStore,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> dict:
    """
    Load, chunk, embed, and store a document.

    Returns a summary of the indexing operation.
    """
    document = load_document(file_path)

    chunks = []

    for page in document["pages"]:
        page_chunks = chunk_text(
            text=page["text"],
            source=document["source"],
            page_number=page["page_number"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks.extend(page_chunks)

    if not chunks:
        return {
            "source": document["source"],
            "file_type": document["file_type"],
            "chunks_created": 0,
            "chunks_stored": 0,
        }

    embeddings = embedding_model.embed_texts(
        [chunk["text"] for chunk in chunks]
    )

    stored_count = vector_store.replace_source(
        source=document["source"],
        chunks=chunks,
        embeddings=embeddings,
    )

    return {
        "source": document["source"],
        "file_type": document["file_type"],
        "chunks_created": len(chunks),
        "chunks_stored": stored_count,
    }
