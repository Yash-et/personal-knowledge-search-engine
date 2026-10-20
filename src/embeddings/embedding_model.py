from math import isfinite
from typing import Sequence
from sentence_transformers import SentenceTransformer
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DEVICE = "cpu"
class LocalEmbeddingModel:
    """Generate normalized text embeddings using a local sentence-transformer."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = DEFAULT_DEVICE,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name cannot be empty.")

        self.model = SentenceTransformer(
            model_name,
            device=device,
        )

    def embed_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
    ) -> list[list[float]]:
        """Convert multiple text passages into normalized embedding vectors."""
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        if not texts:
            return []

        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Every text must be a non-empty string.")

        embeddings = self.model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        result = [
            [float(value) for value in embedding]
            for embedding in embeddings
        ]

        if any(
            not isfinite(value)
            for embedding in result
            for value in embedding
        ):
            raise RuntimeError("Embedding generation produced invalid values.")

        return result

    def embed_text(self, text: str) -> list[float]:
        """Convert one text passage into a normalized embedding vector."""
        return self.embed_texts([text])[0]