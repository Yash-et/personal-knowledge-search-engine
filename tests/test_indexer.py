from pathlib import Path
from src.indexing.indexer import index_document
from src.storage.vector_store import VectorStore
class FakeEmbeddingModel:
    """Small deterministic substitute for pipeline unit tests."""

    def embed_texts(self, texts):
        return [
            [1.0, 0.0, 0.0]
            for _ in texts
        ]

def test_index_document_stores_chunks_and_metadata(tmp_path: Path):
    file_path = tmp_path / "notes.txt"
    file_path.write_text(
        "Python makes it possible to build useful applications.",
        encoding="utf-8",
    )

    store = VectorStore(tmp_path / "vectors")

    result = index_document(
        file_path=file_path,
        embedding_model=FakeEmbeddingModel(),
        vector_store=store,
        chunk_size=20,
        chunk_overlap=5,
    )

    assert result["source"] == "notes.txt"
    assert result["file_type"] == "txt"
    assert result["chunks_created"] > 1
    assert result["chunks_stored"] == result["chunks_created"]
    assert store.count() == result["chunks_created"]

    matches = store.search([1.0, 0.0, 0.0], n_results=1)

    assert matches[0]["metadata"]["source"] == "notes.txt"
    assert matches[0]["metadata"]["page_number"] == 1

def test_reindexing_identical_document_does_not_duplicate_chunks(
    tmp_path: Path,
):
    file_path = tmp_path / "repeat.txt"
    file_path.write_text(
        "This document should be indexed only once.",
        encoding="utf-8",
    )

    store = VectorStore(tmp_path / "vectors")
    model = FakeEmbeddingModel()

    first = index_document(file_path, model, store)
    second = index_document(file_path, model, store)

    assert first["chunks_created"] == second["chunks_created"]
    assert store.count() == first["chunks_created"]

def test_indexing_different_documents_keeps_both_sources(
    tmp_path: Path,
):
    first_file = tmp_path / "first.txt"
    second_file = tmp_path / "second.txt"

    first_file.write_text("Notes about Python.", encoding="utf-8")
    second_file.write_text("Notes about document search.", encoding="utf-8")

    store = VectorStore(tmp_path / "vectors")
    model = FakeEmbeddingModel()

    index_document(first_file, model, store)
    index_document(second_file, model, store)

    results = store.search([1.0, 0.0, 0.0], n_results=10)
    sources = {result["metadata"]["source"] for result in results}

    assert sources == {"first.txt", "second.txt"}
    assert store.count() == 2

def test_reindexing_edited_document_removes_stale_chunks(
    tmp_path: Path,
):
    file_path = tmp_path / "editable.txt"

    file_path.write_text(
        "Python is a programming language. " * 20,
        encoding="utf-8",
    )

    store = VectorStore(tmp_path / "vectors")
    model = FakeEmbeddingModel()

    first = index_document(
        file_path,
        model,
        store,
        chunk_size=30,
        chunk_overlap=5,
    )

    assert first["chunks_stored"] > 1
    old_count = store.count()

    file_path.write_text(
        "Updated document content.",
        encoding="utf-8",
    )

    second = index_document(
        file_path,
        model,
        store,
        chunk_size=30,
        chunk_overlap=5,
    )

    assert second["chunks_stored"] == 1
    assert store.count() == 1
    assert store.count() < old_count

    results = store.search([1.0, 0.0, 0.0], n_results=5)

    assert len(results) == 1
    assert results[0]["text"] == "Updated document content."
    assert results[0]["metadata"]["source"] == "editable.txt"
