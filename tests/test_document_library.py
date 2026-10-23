
from src.storage.vector_store import VectorStore


def make_chunk(source, chunk_id, text):
    return {
        "source": source,
        "page_number": 1,
        "chunk_id": chunk_id,
        "text": text,
    }


def test_list_sources_returns_empty_list_for_empty_store(tmp_path):
    store = VectorStore(persist_directory=tmp_path / "vector_db")

    assert store.list_sources() == []


def test_list_sources_groups_passages_by_filename(tmp_path):
    store = VectorStore(persist_directory=tmp_path / "vector_db")

    chunks = [
        make_chunk("notes.txt", 0, "First note passage"),
        make_chunk("report.pdf", 0, "First report passage"),
        make_chunk("notes.txt", 1, "Second note passage"),
        make_chunk("notes.txt", 2, "Third note passage"),
        make_chunk("report.pdf", 1, "Second report passage"),
    ]

    embeddings = [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.8, 0.2],
        [0.2, 0.8],
        [0.5, 0.5],
    ]

    store.add_chunks(chunks, embeddings)

    assert store.list_sources() == [
        {"source": "notes.txt", "passage_count": 3},
        {"source": "report.pdf", "passage_count": 2},
    ]


def test_list_sources_is_sorted_case_insensitively(tmp_path):
    store = VectorStore(persist_directory=tmp_path / "vector_db")

    chunks = [
        make_chunk("zebra.txt", 0, "Zebra passage"),
        make_chunk("Alpha.txt", 0, "Alpha passage"),
        make_chunk("beta.txt", 0, "Beta passage"),
    ]

    embeddings = [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.5],
    ]

    store.add_chunks(chunks, embeddings)

    sources = store.list_sources()

    assert [item["source"] for item in sources] == [
        "Alpha.txt",
        "beta.txt",
        "zebra.txt",
    ]
