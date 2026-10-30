import pytest
from src.evaluation.retrieval_metrics import precision_at_k
from src.evaluation.retrieval_metrics import (
    mean_reciprocal_rank,
    reciprocal_rank,
    recall_at_k,
)

def test_recall_at_k_returns_fraction_of_relevant_results():
    retrieved = ["doc-a", "doc-b", "doc-c"]
    relevant = {"doc-a", "doc-c", "doc-d"}

    assert recall_at_k(retrieved, relevant, k=2) == pytest.approx(1 / 3)
    assert recall_at_k(retrieved, relevant, k=3) == pytest.approx(2 / 3)

def test_recall_at_k_returns_one_when_all_relevant_results_are_found():
    assert recall_at_k(
        ["doc-a", "doc-b"],
        {"doc-a", "doc-b"},
        k=2,
    ) == 1.0

def test_reciprocal_rank_scores_first_relevant_result():
    assert reciprocal_rank(
        ["irrelevant", "relevant", "other"],
        {"relevant"},
    ) == pytest.approx(0.5)

def test_reciprocal_rank_returns_zero_when_no_relevant_result_is_found():
    assert reciprocal_rank(
        ["doc-a", "doc-b"],
        {"doc-c"},
    ) == 0.0

def test_mean_reciprocal_rank_averages_query_scores():
    results = [
        ["relevant-a", "other"],
        ["other", "relevant-b"],
        ["other-a", "other-b"],
    ]
    relevant = [
        {"relevant-a"},
        {"relevant-b"},
        {"relevant-c"},
    ]

    assert mean_reciprocal_rank(results, relevant) == pytest.approx(0.5)

@pytest.mark.parametrize("k", [0, -1])
def test_recall_at_k_rejects_invalid_k(k):
    with pytest.raises(ValueError, match="k"):
        recall_at_k(["doc-a"], {"doc-a"}, k)

def test_recall_at_k_rejects_empty_relevant_set():
    with pytest.raises(ValueError, match="relevant_ids"):
        recall_at_k(["doc-a"], set(), k=1)

def test_reciprocal_rank_rejects_empty_relevant_set():
    with pytest.raises(ValueError, match="relevant_ids"):
        reciprocal_rank(["doc-a"], set())

def test_mean_reciprocal_rank_rejects_empty_evaluation():
    with pytest.raises(ValueError, match="result_lists"):
        mean_reciprocal_rank([], [])

def test_mean_reciprocal_rank_rejects_mismatched_inputs():
    with pytest.raises(ValueError, match="corresponding"):
        mean_reciprocal_rank([["doc-a"]], [])

def test_recall_at_k_does_not_double_count_duplicate_results():
    retrieved = ["doc-a", "doc-a", "doc-b"]
    relevant = {"doc-a", "doc-b", "doc-c"}

    assert recall_at_k(retrieved, relevant, k=3) == pytest.approx(2 / 3)

def test_reciprocal_rank_uses_first_occurrence_of_relevant_result():
    retrieved = ["irrelevant", "doc-a", "doc-a"]
    relevant = {"doc-a"}

    assert reciprocal_rank(retrieved, relevant) == pytest.approx(0.5)

def test_precision_at_k_returns_fraction_of_relevant_results():
    retrieved = ["doc-a", "doc-b", "doc-c"]
    relevant = {"doc-a", "doc-c", "doc-d"}

    assert precision_at_k(retrieved, relevant, k=2) == pytest.approx(0.5)
    assert precision_at_k(retrieved, relevant, k=3) == pytest.approx(2 / 3)

def test_precision_at_k_returns_zero_when_no_results_are_relevant():
    assert precision_at_k(
        ["doc-a", "doc-b"],
        {"doc-c"},
        k=2,
    ) == 0.0

@pytest.mark.parametrize("k", [0, -1])
def test_precision_at_k_rejects_invalid_k(k):
    with pytest.raises(ValueError, match="k"):
        precision_at_k(["doc-a"], {"doc-a"}, k)

def test_precision_at_k_rejects_empty_relevant_set():
    with pytest.raises(ValueError, match="relevant_ids"):
        precision_at_k(["doc-a"], set(), k=1)