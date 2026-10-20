import pytest
from src.search.semantic_search import semantic_search
from src.storage.vector_store import VectorStore
class FakeEmbeddingModel:
    def embed_text(self, text):
        normalized = text.lower()

        if "python" in normalized:
            return [1.0, 0.0, 0.0]

        if "weather" in normalized:
            return [0.0, 1.0, 0.0]

        return [0.0, 0.0, 1.0]

@pytest.fixture
def search_setup(tmp_path):
    store = VectorStore(tmp_path / "search_vectors")
    model = FakeEmbeddingModel()

    chunks = [
        {
            "chunk_id": 0,
            "text": "Python is used to build applications.",
            "source": "python.txt",
            "page_number": 1,
        },
        {
            "chunk_id": 0,
            "text": "Weather forecasts predict rain and sunshine.",
            "source": "weather.txt",
            "page_number": 2,
        },
        {
            "chunk_id": 0,
            "text": "Python supports data analysis and automation.",
            "source": "python_notes.txt",
            "page_number": 3,
        },
    ]

    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ]

    store.add_chunks(chunks, embeddings)

    return model, store

def test_search_returns_relevant_passages(search_setup):
    model, store = search_setup

    results = semantic_search(
        "Tell me about Python",
        model,
        store,
        top_k=2,
    )

    assert len(results) == 2
    assert all("Python" in result["text"] for result in results)

def test_search_preserves_source_metadata(search_setup):
    model, store = search_setup

    results = semantic_search("Python programming", model, store, top_k=2)

    assert len(results) == 2

    python_result = next(
        result
        for result in results
        if result["metadata"]["source"] == "python.txt"
    )

    assert python_result["metadata"]["page_number"] == 1
    assert python_result["text"] == "Python is used to build applications."

def test_top_k_limits_result_count(search_setup):
    model, store = search_setup

    results = semantic_search("Python", model, store, top_k=1)

    assert len(results) == 1

def test_search_on_empty_store_returns_no_results(tmp_path):
    store = VectorStore(tmp_path / "empty_vectors")
    model = FakeEmbeddingModel()

    results = semantic_search("Python", model, store)

    assert results == []

@pytest.mark.parametrize("query", ["", "   "])
def test_empty_query_is_rejected(search_setup, query):
    model, store = search_setup

    with pytest.raises(ValueError, match="query"):
        semantic_search(query, model, store)

def test_invalid_top_k_is_rejected(search_setup):
    model, store = search_setup

    with pytest.raises(ValueError, match="top_k"):
        semantic_search("Python", model, store, top_k=0)
