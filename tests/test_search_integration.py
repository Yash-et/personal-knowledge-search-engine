from pathlib import Path
import pytest
from src.embeddings.embedding_model import LocalEmbeddingModel
from src.storage.vector_store import VectorStore
@pytest.fixture(scope="module")
def embedding_model():
    """Load the actual local embedding model once."""
    return LocalEmbeddingModel()

@pytest.fixture
def indexed_store(tmp_path: Path, embedding_model):
    """Create a temporary database populated with real embeddings."""
    store = VectorStore(tmp_path / "integration_vectors")

    chunks = [
        {
            "chunk_id": 0,
            "text": (
                "Python is a programming language used for software "
                "development, scripting, and automation."
            ),
            "source": "python_notes.txt",
            "page_number": 1,
        },
        {
            "chunk_id": 0,
            "text": (
                "The Pacific Ocean is the largest and deepest ocean "
                "on Earth."
            ),
            "source": "ocean_facts.txt",
            "page_number": 1,
        },
        {
            "chunk_id": 0,
            "text": (
                "Machine learning uses data to train models that make "
                "predictions and identify patterns."
            ),
            "source": "machine_learning.txt",
            "page_number": 3,
        },
    ]

    embeddings = embedding_model.embed_texts(
        [chunk["text"] for chunk in chunks]
    )

    store.add_chunks(chunks, embeddings)

    return store

def test_real_embeddings_retrieve_relevant_passage(
    embedding_model,
    indexed_store,
):
    query = "What is Python used for in software development?"
    query_embedding = embedding_model.embed_text(query)

    results = indexed_store.search(
        query_embedding=query_embedding,
        n_results=3,
    )

    assert results
    assert results[0]["metadata"]["source"] == "python_notes.txt"
    assert "Python" in results[0]["text"]

def test_real_embeddings_preserve_citation_metadata(
    embedding_model,
    indexed_store,
):
    query_embedding = embedding_model.embed_text(
        "How does machine learning discover patterns in data?"
    )

    results = indexed_store.search(
        query_embedding=query_embedding,
        n_results=3,
    )

    matching_result = next(
        result
        for result in results
        if result["metadata"]["source"] == "machine_learning.txt"
    )

    assert matching_result["metadata"]["page_number"] == 3
    assert "Machine learning" in matching_result["text"]
    assert isinstance(matching_result["distance"], float)


def test_real_embeddings_have_expected_dimensions(embedding_model):
    vector = embedding_model.embed_text(
        "Testing real semantic embeddings."
    )

    assert len(vector) == 384
