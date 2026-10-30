def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Return the fraction of relevant documents retrieved in the top k."""
    if k <= 0:
        raise ValueError("k must be greater than zero.")

    if not relevant_ids:
        raise ValueError("relevant_ids must not be empty.")

    retrieved_top_k = retrieved_ids[:k]
    relevant_retrieved = set(retrieved_top_k) & relevant_ids

    return len(relevant_retrieved) / len(relevant_ids)

def reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids: set[str],
) -> float:
    """Return the reciprocal rank of the first relevant result, or zero."""
    if not relevant_ids:
        raise ValueError("relevant_ids must not be empty.")

    for rank, document_id in enumerate(retrieved_ids, start=1):
        if document_id in relevant_ids:
            return 1.0 / rank

    return 0.0

def mean_reciprocal_rank(
    result_lists: list[list[str]],
    relevant_sets: list[set[str]],
) -> float:
    """Return the mean reciprocal rank across multiple queries."""
    if not result_lists:
        raise ValueError("result_lists must not be empty.")

    if len(result_lists) != len(relevant_sets):
        raise ValueError(
            "Each result list must have a corresponding relevant set."
        )

    scores = [
        reciprocal_rank(results, relevant)
        for results, relevant in zip(result_lists, relevant_sets)
    ]

    return sum(scores) / len(scores)

def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Return the fraction of the top k results that are relevant."""
    if k <= 0:
        raise ValueError("k must be greater than zero.")

    if not relevant_ids:
        raise ValueError("relevant_ids must not be empty.")

    retrieved_top_k = retrieved_ids[:k]
    relevant_retrieved = set(retrieved_top_k) & relevant_ids

    return len(relevant_retrieved) / k
