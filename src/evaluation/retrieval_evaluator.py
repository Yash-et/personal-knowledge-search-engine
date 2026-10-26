import json
from pathlib import Path
from src.evaluation.retrieval_metrics import (
    mean_reciprocal_rank,
    recall_at_k,
)
from src.search.semantic_search import semantic_search

def evaluate_retrieval(
    dataset_path: str | Path,
    embedding_model,
    vector_store,
    k: int = 3,
) -> dict:
    """Evaluate semantic retrieval against a labeled JSON dataset."""
    dataset_path = Path(dataset_path)

    with dataset_path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    documents = dataset.get("documents", [])
    queries = dataset.get("queries", [])

    if not documents:
        raise ValueError("Evaluation dataset must contain documents.")

    if not queries:
        raise ValueError("Evaluation dataset must contain queries.")

    id_by_metadata = {}

    for document in documents:
        document_id = document["id"]
        metadata_key = (
            document["source"],
            document["page_number"],
            document["chunk_id"],
        )

        if metadata_key in id_by_metadata:
            raise ValueError(
                f"Duplicate document metadata in evaluation dataset: "
                f"{metadata_key}"
            )

        id_by_metadata[metadata_key] = document_id

    result_lists = []
    relevant_sets = []
    query_results = []

    for item in queries:
        relevant_ids = set(item["relevant_ids"])

        if not relevant_ids:
            raise ValueError("Each query must have relevant document IDs.")

        unknown_ids = relevant_ids - {
            document["id"] for document in documents
        }
        if unknown_ids:
            raise ValueError(
                f"Unknown relevant document IDs: {sorted(unknown_ids)}"
            )

        results = semantic_search(
            query=item["query"],
            embedding_model=embedding_model,
            vector_store=vector_store,
            top_k=k,
        )

        ranked_ids = []

        for result in results:
            metadata = result["metadata"]
            metadata_key = (
                metadata["source"],
                metadata["page_number"],
                metadata["chunk_id"],
            )

            document_id = id_by_metadata.get(metadata_key)

            if document_id is not None:
                ranked_ids.append(document_id)

        result_lists.append(ranked_ids)
        relevant_sets.append(relevant_ids)

        query_results.append(
            {
                "query": item["query"],
                "retrieved_ids": ranked_ids,
                "relevant_ids": sorted(relevant_ids),
                "recall_at_k": recall_at_k(ranked_ids, relevant_ids, k),
            }
        )

    return {
        "k": k,
        "query_count": len(queries),
        "recall_at_k": sum(
            result["recall_at_k"] for result in query_results
        ) / len(query_results),
        "mrr": mean_reciprocal_rank(result_lists, relevant_sets),
        "results": query_results,
    }
