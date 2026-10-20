import pytest
from src.storage.vector_store import VectorStore
@pytest.fixture
def store(tmp_path):
    return VectorStore(tmp_path / "test_vectors")

@pytest.fixture
def sample_chunks():
    return [
        {
            "chunk_id": 0,
            "text": "Python is a programming language.",
            "source": "python.txt",
            "page_number": 1,
        },
        {
            "chunk_id": 1,
            "text": "Semantic search finds related content.",
            "source": "search.pdf",
            "page_number": 2,
        },
    ]

@pytest.fixture
def sample_embeddings():
    return [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

def test_empty_store_has_zero_count(store):
    assert store.count() == 0

def test_add_chunks_stores_documents_and_metadata(
    store, sample_chunks, sample_embeddings
):
    added = store.add_chunks(sample_chunks, sample_embeddings)

    assert added == 2
    assert store.count() == 2

    results = store.search([1.0, 0.0, 0.0], n_results=1)

    assert len(results) == 1
    assert results[0]["text"] == sample_chunks[0]["text"]
    assert results[0]["metadata"]["source"] == "python.txt"
    assert results[0]["metadata"]["page_number"] == 1

def test_search_returns_nearest_vector(
    store, sample_chunks, sample_embeddings
):
    store.add_chunks(sample_chunks, sample_embeddings)

    results = store.search([0.0, 1.0, 0.0], n_results=2)

    assert results[0]["text"] == sample_chunks[1]["text"]

def test_upsert_does_not_duplicate_identical_chunks(
    store, sample_chunks, sample_embeddings
):
    store.add_chunks(sample_chunks, sample_embeddings)
    store.add_chunks(sample_chunks, sample_embeddings)

    assert store.count() == 2

def test_empty_chunks_are_accepted(store):
    assert store.add_chunks([], []) == 0
    assert store.count() == 0

def test_chunk_embedding_count_must_match(store, sample_chunks):
    with pytest.raises(ValueError, match="must match"):
        store.add_chunks(sample_chunks, [[1.0, 0.0, 0.0]])

def test_invalid_chunk_is_rejected(store, sample_embeddings):
    invalid_chunks = [
        {
            "chunk_id": 0,
            "text": " ",
            "source": "notes.txt",
            "page_number": 1,
        },
        {
            "chunk_id": 1,
            "text": "Valid text",
            "source": "notes.txt",
            "page_number": 1,
        },
    ]

    with pytest.raises(ValueError, match="non-empty text"):
        store.add_chunks(invalid_chunks, sample_embeddings)

def test_inconsistent_embedding_dimensions_are_rejected(store, sample_chunks):
    with pytest.raises(ValueError, match="same dimensions"):
        store.add_chunks(
            sample_chunks,
            [[1.0, 0.0], [1.0, 0.0, 0.0]],
        )

def test_invalid_result_count_is_rejected(store):
    with pytest.raises(ValueError, match="n_results"):
        store.search([1.0, 0.0, 0.0], n_results=0)

def test_empty_query_embedding_is_rejected(store):
    with pytest.raises(ValueError, match="cannot be empty"):
        store.search([])

def test_persistence_survives_reopening(
    tmp_path, sample_chunks, sample_embeddings
):
    database_path = tmp_path / "persistent_vectors"

    first_store = VectorStore(database_path)
    first_store.add_chunks(sample_chunks, sample_embeddings)

    reopened_store = VectorStore(database_path)

    assert reopened_store.count() == 2

    results = reopened_store.search([1.0, 0.0, 0.0], n_results=1)
    assert results[0]["text"] == sample_chunks[0]["text"]

def test_invalid_chunk_identity_is_rejected():
    with pytest.raises(ValueError, match="source"):
        VectorStore.make_chunk_id("", 1, 0, "Some text")

    with pytest.raises(ValueError, match="page_number"):
        VectorStore.make_chunk_id("notes.txt", 0, 0, "Some text")

    with pytest.raises(ValueError, match="chunk_id"):
        VectorStore.make_chunk_id("notes.txt", 1, -1, "Some text")

def test_delete_source_removes_only_matching_chunks(
    store, sample_chunks, sample_embeddings
):
    store.add_chunks(sample_chunks, sample_embeddings)

    deleted_count = store.delete_source("python.txt")

    assert deleted_count == 1
    assert store.count() == 1

    remaining = store.search([0.0, 1.0, 0.0], n_results=2)
    assert len(remaining) == 1
    assert remaining[0]["metadata"]["source"] == "search.pdf"


def test_delete_missing_source_returns_zero(store):
    assert store.delete_source("missing.txt") == 0

def test_delete_source_rejects_empty_name(store):
    with pytest.raises(ValueError, match="source"):
        store.delete_source(" ")

def test_replace_source_removes_stale_chunks(
    store, sample_chunks, sample_embeddings
):
    store.add_chunks(sample_chunks, sample_embeddings)

    replacement_chunks = [
        {
            "chunk_id": 0,
            "text": "Updated Python documentation.",
            "source": "python.txt",
            "page_number": 1,
        }
    ]

    replacement_embeddings = [[0.0, 0.0, 1.0]]

    stored = store.replace_source(
        "python.txt",
        replacement_chunks,
        replacement_embeddings,
    )

    assert stored == 1
    assert store.count() == 2

    results = store.search([0.0, 0.0, 1.0], n_results=2)
    python_results = [
        result
        for result in results
        if result["metadata"]["source"] == "python.txt"
    ]

    assert len(python_results) == 1
    assert python_results[0]["text"] == "Updated Python documentation."

def test_replace_source_with_empty_chunks_deletes_source(
    store, sample_chunks, sample_embeddings
):
    store.add_chunks(sample_chunks, sample_embeddings)

    assert store.replace_source("python.txt", [], []) == 0
    assert store.count() == 1

def test_replace_source_rejects_mismatched_sources(
    store, sample_embeddings
):
    chunks = [
        {
            "chunk_id": 0,
            "text": "Other document.",
            "source": "other.txt",
            "page_number": 1,
        }
    ]

    with pytest.raises(ValueError, match="belong to the source"):
        store.replace_source("python.txt", chunks, sample_embeddings[:1])

    assert store.count() == 0

def test_replace_source_validation_preserves_existing_data(
    store, sample_chunks, sample_embeddings
):
    store.add_chunks(sample_chunks, sample_embeddings)

    invalid_chunks = [
        {
            "chunk_id": 0,
            "text": " ",
            "source": "python.txt",
            "page_number": 1,
        }
    ]

    with pytest.raises(ValueError, match="non-empty text"):
        store.replace_source(
            "python.txt",
            invalid_chunks,
            [[1.0, 0.0, 0.0]],
        )

    assert store.count() == 2
