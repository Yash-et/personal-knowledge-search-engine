import json
from pathlib import Path
from src.evaluation.retrieval_metrics import (
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)
from src.search.semantic_search import semantic_search

def evaluate_retrieval(
    dataset_path: str | Path,
    embedding_model,
    vector_store,
    k: int = 3,
    ks: list[int] | None = None,
) -> dict:
    """Evaluate retrieval at one or more cutoffs.

    Documents are retrieved once per query, up to the largest cutoff.
    MRR is calculated over that same maximum retrieval depth.
    """
    if ks is None:
        cutoffs = [k]
    else:
        if not ks:
            raise ValueError("ks must contain at least one cutoff.")
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
               for value in ks):
            raise ValueError("Every cutoff in ks must be a positive integer.")
        if len(set(ks)) != len(ks):
            raise ValueError("ks must not contain duplicate cutoffs.")
        cutoffs = sorted(ks)

    if ks is None and (not isinstance(k, int) or isinstance(k, bool) or k <= 0):
        raise ValueError("k must be a positive integer.")

    max_k = max(cutoffs)
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
    seen_document_ids = set()

    for document in documents:
        document_id = document["id"]

        if document_id in seen_document_ids:
            raise ValueError(
                f"Duplicate document ID in evaluation dataset: {document_id}"
            )

        seen_document_ids.add(document_id)

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

        unknown_ids = relevant_ids - seen_document_ids
        if unknown_ids:
            raise ValueError(
                f"Unknown relevant document IDs: {sorted(unknown_ids)}"
            )

        results = semantic_search(
            query=item["query"],
            embedding_model=embedding_model,
            vector_store=vector_store,
            top_k=max_k,
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

        cutoff_metrics = {
            cutoff: {
                "recall_at_k": recall_at_k(
                    ranked_ids, relevant_ids, cutoff
                ),
                "precision_at_k": precision_at_k(
                    ranked_ids, relevant_ids, cutoff
                ),
            }
            for cutoff in cutoffs
        }

        query_results.append(
            {
                "query": item["query"],
                "retrieved_ids": ranked_ids,
                "relevant_ids": sorted(relevant_ids),
                "metrics": cutoff_metrics,
                # Preserve the original single-cutoff result fields.
                "recall_at_k": cutoff_metrics[max_k]["recall_at_k"],
                "precision_at_k": cutoff_metrics[max_k]["precision_at_k"],
            }
        )

    overall_metrics = {
        cutoff: {
            "recall_at_k": sum(
                result["metrics"][cutoff]["recall_at_k"]
                for result in query_results
            ) / len(query_results),
            "precision_at_k": sum(
                result["metrics"][cutoff]["precision_at_k"]
                for result in query_results
            ) / len(query_results),
        }
        for cutoff in cutoffs
    }

    return {
        # k and the original metric fields remain compatible with
        # existing callers; they represent the maximum cutoff.
        "k": max_k,
        "ks": cutoffs,
        "query_count": len(queries),
        "metrics": overall_metrics,
        "recall_at_k": overall_metrics[max_k]["recall_at_k"],
        "precision_at_k": overall_metrics[max_k]["precision_at_k"],
        "mrr": mean_reciprocal_rank(result_lists, relevant_sets),
        "mrr_cutoff": max_k,
        "results": query_results,
    }
