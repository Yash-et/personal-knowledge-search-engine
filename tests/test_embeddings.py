
import math

import pytest

from src.embeddings.embedding_model import LocalEmbeddingModel


@pytest.fixture(scope="module")
def embedding_model():
    """Load the local model once for this test module."""
    return LocalEmbeddingModel()


def test_single_text_embedding_has_expected_dimensions(embedding_model):
    vector = embedding_model.embed_text("Semantic search finds related ideas.")

    assert len(vector) == 384
    assert all(isinstance(value, float) for value in vector)


def test_embedding_values_are_finite_and_normalized(embedding_model):
    vector = embedding_model.embed_text("Local document search.")

    assert all(math.isfinite(value) for value in vector)
    assert math.isclose(
        math.sqrt(sum(value * value for value in vector)),
        1.0,
        rel_tol=1e-5,
    )


def test_batch_embedding_returns_one_vector_per_text(embedding_model):
    texts = [
        "Python programming",
        "Semantic document retrieval",
        "Local knowledge management",
    ]

    vectors = embedding_model.embed_texts(texts, batch_size=2)

    assert len(vectors) == len(texts)
    assert all(len(vector) == 384 for vector in vectors)


def test_empty_batch_returns_empty_list(embedding_model):
    assert embedding_model.embed_texts([]) == []


@pytest.mark.parametrize("text", ["", "   "])
def test_empty_text_raises_error(embedding_model, text):
    with pytest.raises(ValueError, match="non-empty"):
        embedding_model.embed_text(text)


def test_invalid_batch_size_raises_error(embedding_model):
    with pytest.raises(ValueError, match="batch_size"):
        embedding_model.embed_texts(
            ["A valid passage"],
            batch_size=0,
        )


def test_batch_with_empty_text_raises_error(embedding_model):
    with pytest.raises(ValueError, match="non-empty"):
        embedding_model.embed_texts(
            ["Valid passage", "   "]
        )


def test_similar_texts_have_higher_similarity(embedding_model):
    vectors = embedding_model.embed_texts(
        [
            "A guide to Python programming.",
            "Instructions for learning Python coding.",
            "The weather forecast predicts rain tomorrow.",
        ]
    )

    def cosine_similarity(first, second):
        return sum(a * b for a, b in zip(first, second))

    related_similarity = cosine_similarity(vectors[0], vectors[1])
    unrelated_similarity = cosine_similarity(vectors[0], vectors[2])

    assert related_similarity > unrelated_similarity
