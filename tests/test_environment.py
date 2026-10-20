import chromadb
import ollama
import streamlit
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

def test_required_dependencies_import():
    """Verify all required libraries are importable."""
    assert chromadb is not None
    assert ollama is not None
    assert streamlit is not None
    assert load_dotenv is not None
    assert PdfReader is not None
    assert SentenceTransformer is not None

def test_chromadb_can_store_and_retrieve(tmp_path):
    """Verify vector storage, similarity search, and persistence."""
    database_path = tmp_path / "chroma_db"

    client = chromadb.PersistentClient(path=str(database_path))
    collection = client.create_collection(name="environment_smoke_test")

    collection.add(
        ids=["sample-1", "sample-2"],
        documents=[
            "Python is used for machine learning.",
            "Gardening involves growing plants.",
        ],
        embeddings=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        metadatas=[
            {"category": "technology"},
            {"category": "gardening"},
        ],
    )

    results = collection.query(
        query_embeddings=[[0.99, 0.01, 0.0]],
        n_results=1,
    )

    assert results["ids"][0][0] == "sample-1"
    assert len(results["documents"][0]) == 1

    # Reopen the database and verify that the record persists.
    reopened_client = chromadb.PersistentClient(path=str(database_path))
    reopened_collection = reopened_client.get_collection(
        name="environment_smoke_test"
    )

    stored = reopened_collection.get(ids=["sample-1"])

    assert stored["ids"] == ["sample-1"]
    assert stored["documents"][0] == (
        "Python is used for machine learning."
    )
