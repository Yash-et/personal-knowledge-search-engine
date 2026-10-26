import json
import pytest
from src.embeddings.embedding_model import LocalEmbeddingModel
from src.evaluation.retrieval_evaluator import evaluate_retrieval
from src.storage.vector_store import VectorStore

@pytest.fixture
def evaluation_setup(tmp_path):
    dataset_path = (
        __import__("pathlib").Path(__file__).parent
        / "fixtures"
        / "retrieval_eval_dataset.json"
    )

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))

    store = VectorStore(tmp_path / "evaluation_vectors")
    model = LocalEmbeddingModel()

    documents = dataset["documents"]
    chunks = [
        {
            "chunk_id": document["chunk_id"],
            "text": document["text"],
            "source": document["source"],
            "page_number": document["page_number"],
        }
        for document in documents
    ]

    embeddings = model.embed_texts([chunk["text"] for chunk in chunks])
    store.add_chunks(chunks, embeddings)

    return dataset_path, model, store

def test_evaluator_returns_expected_report_structure(evaluation_setup):
    dataset_path, model, store = evaluation_setup

    report = evaluate_retrieval(
        dataset_path=dataset_path,
        embedding_model=model,
        vector_store=store,
        k=3,
    )

    assert report["k"] == 3
    assert report["query_count"] == 5
    assert 0.0 <= report["recall_at_k"] <= 1.0
    assert 0.0 <= report["mrr"] <= 1.0
    assert len(report["results"]) == 5

def test_evaluator_retrieves_relevant_passages(evaluation_setup):
    dataset_path, model, store = evaluation_setup

    report = evaluate_retrieval(
        dataset_path=dataset_path,
        embedding_model=model,
        vector_store=store,
        k=3,
    )

    assert report["recall_at_k"] == pytest.approx(1.0)
    assert report["mrr"] == pytest.approx(1.0)

    for result in report["results"]:
        assert result["relevant_ids"][0] in result["retrieved_ids"]

def test_evaluator_rejects_dataset_without_documents(
    tmp_path, evaluation_setup
):
    _, model, store = evaluation_setup
    empty_dataset = tmp_path / "empty.json"
    empty_dataset.write_text(
        json.dumps({"documents": [], "queries": []}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="documents"):
        evaluate_retrieval(empty_dataset, model, store)

def test_evaluator_rejects_dataset_without_queries(
    tmp_path, evaluation_setup
):
    _, model, store = evaluation_setup
    no_queries = tmp_path / "no_queries.json"
    no_queries.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "id": "a",
                        "source": "a.txt",
                        "page_number": 1,
                        "chunk_id": 0,
                        "text": "Example passage.",
                    }
                ],
                "queries": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="queries"):
        evaluate_retrieval(no_queries, model, store)

def test_evaluator_rejects_duplicate_document_ids(
    tmp_path, evaluation_setup
):
    dataset_path, model, store = evaluation_setup

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset["documents"][1]["id"] = dataset["documents"][0]["id"]

    duplicate_ids_path = tmp_path / "duplicate_ids.json"
    duplicate_ids_path.write_text(
        json.dumps(dataset),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate document ID"):
        evaluate_retrieval(
            duplicate_ids_path,
            model,
            store,
            k=3,
        )

def test_evaluator_rejects_duplicate_document_metadata(
    tmp_path, evaluation_setup
):
    dataset_path, model, store = evaluation_setup

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset["documents"][1]["source"] = dataset["documents"][0]["source"]
    dataset["documents"][1]["page_number"] = dataset["documents"][0]["page_number"]
    dataset["documents"][1]["chunk_id"] = dataset["documents"][0]["chunk_id"]

    duplicate_metadata_path = tmp_path / "duplicate_metadata.json"
    duplicate_metadata_path.write_text(
        json.dumps(dataset),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate document metadata"):
        evaluate_retrieval(
            duplicate_metadata_path,
            model,
            store,
            k=3,
        )

def test_evaluator_rejects_unknown_relevant_document_ids(
    tmp_path, evaluation_setup
):
    dataset_path, model, store = evaluation_setup

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset["queries"][0]["relevant_ids"] = ["missing-document"]

    unknown_id_path = tmp_path / "unknown_relevant_id.json"
    unknown_id_path.write_text(
        json.dumps(dataset),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown relevant document IDs"):
        evaluate_retrieval(
            unknown_id_path,
            model,
            store,
            k=3,
        )