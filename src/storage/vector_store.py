import hashlib
from pathlib import Path
from typing import Sequence
import chromadb
DEFAULT_COLLECTION_NAME = "personal_knowledge"
DEFAULT_DISTANCE_METRIC = "cosine"
class VectorStore:
    """Persist and search document chunk embeddings using ChromaDB."""

    def __init__(
        self,
        persist_directory: str | Path,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        if not collection_name.strip():
            raise ValueError("collection_name cannot be empty.")

        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": DEFAULT_DISTANCE_METRIC},
        )

    @staticmethod
    def make_chunk_id(
        source: str,
        page_number: int,
        chunk_id: int,
        text: str,
    ) -> str:
        """Create a deterministic ID for a source passage."""
        if not source.strip():
            raise ValueError("source cannot be empty.")
        if page_number < 1:
            raise ValueError("page_number must be at least 1.")
        if chunk_id < 0:
            raise ValueError("chunk_id cannot be negative.")

        identity = (
            f"{source}\0{page_number}\0{chunk_id}\0{text}"
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    def add_chunks(
        self,
        chunks: Sequence[dict],
        embeddings: Sequence[Sequence[float]],
    ) -> int:
        """Insert or update chunks and their precomputed embeddings."""
        if len(chunks) != len(embeddings):
            raise ValueError(
                "The number of chunks must match the number of embeddings."
            )

        if not chunks:
            return 0

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []
        vectors: list[list[float]] = []

        for chunk, embedding in zip(chunks, embeddings):
            text = chunk.get("text")
            source = chunk.get("source")
            page_number = chunk.get("page_number")
            chunk_id = chunk.get("chunk_id")

            if not isinstance(text, str) or not text.strip():
                raise ValueError("Every chunk must contain non-empty text.")
            if not isinstance(source, str) or not source.strip():
                raise ValueError("Every chunk must have a non-empty source.")
            if not isinstance(page_number, int) or page_number < 1:
                raise ValueError("Every chunk must have a valid page_number.")
            if not isinstance(chunk_id, int) or chunk_id < 0:
                raise ValueError("Every chunk must have a valid chunk_id.")
            if not embedding:
                raise ValueError("Every embedding must contain values.")

            vector = [float(value) for value in embedding]

            if any(not __import__("math").isfinite(value) for value in vector):
                raise ValueError("Embeddings must contain only finite values.")

            ids.append(
                self.make_chunk_id(
                    source, page_number, chunk_id, text
                )
            )
            documents.append(text)
            metadatas.append(
                {
                    "source": source,
                    "page_number": page_number,
                    "chunk_id": chunk_id,
                }
            )
            vectors.append(vector)

        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) != 1:
            raise ValueError("All embeddings must have the same dimensions.")

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=vectors,
        )

        return len(ids)

    def search(
        self,
        query_embedding: Sequence[float],
        n_results: int = 5,
    ) -> list[dict]:
        """Retrieve the most similar passages for a query embedding."""
        if n_results <= 0:
            raise ValueError("n_results must be greater than zero.")

        if not query_embedding:
            raise ValueError("query_embedding cannot be empty.")

        vector = [float(value) for value in query_embedding]

        if any(not __import__("math").isfinite(value) for value in vector):
            raise ValueError("Query embedding must contain finite values.")

        stored_count = self.collection.count()
        if stored_count == 0:
            return []

        response = self.collection.query(
            query_embeddings=[vector],
            n_results=min(n_results, stored_count),
            include=["documents", "metadatas", "distances"],
        )

        results: list[dict] = []

        for index, document in enumerate(response["documents"][0]):
            results.append(
                {
                    "text": document,
                    "metadata": response["metadatas"][0][index],
                    "distance": response["distances"][0][index],
                }
            )

        return results

    def count(self) -> int:
        """Return the number of stored chunks."""
        return self.collection.count()
    
    def delete_source(self, source: str) -> int:
        """Delete all indexed chunks belonging to one source filename."""
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source cannot be empty.")

        existing = self.collection.get(
            where={"source": source},
            include=["metadatas"],
        )

        ids = existing["ids"]

        if ids:
            self.collection.delete(ids=ids)

        return len(ids)

    def replace_source(
        self,
        source: str,
        chunks: Sequence[dict],
        embeddings: Sequence[Sequence[float]],
    ) -> int:
        """
        Replace all indexed chunks for a source.

        Validate the new content first so invalid input cannot delete
        previously indexed records.
        """
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source cannot be empty.")

        if len(chunks) != len(embeddings):
            raise ValueError(
                "The number of chunks must match the number of embeddings."
            )

        for chunk in chunks:
            if chunk.get("source") != source:
                raise ValueError(
                    "Every chunk must belong to the source being replaced."
                )

        # Reuse add_chunks validation without modifying the real collection.
        # Validate chunk structure, metadata, and embedding values first.
        validated_chunks = []
        validated_embeddings = []

        for chunk, embedding in zip(chunks, embeddings):
            text = chunk.get("text")
            page_number = chunk.get("page_number")
            chunk_id = chunk.get("chunk_id")

            if not isinstance(text, str) or not text.strip():
                raise ValueError("Every chunk must contain non-empty text.")
            if not isinstance(page_number, int) or page_number < 1:
                raise ValueError("Every chunk must have a valid page_number.")
            if not isinstance(chunk_id, int) or chunk_id < 0:
                raise ValueError("Every chunk must have a valid chunk_id.")
            if not embedding:
                raise ValueError("Every embedding must contain values.")

            vector = [float(value) for value in embedding]

            import math

            if any(not math.isfinite(value) for value in vector):
                raise ValueError("Embeddings must contain only finite values.")

            validated_chunks.append(chunk)
            validated_embeddings.append(vector)

        dimensions = {
            len(vector) for vector in validated_embeddings
        }

        if len(dimensions) > 1:
            raise ValueError("All embeddings must have the same dimensions.")

        if validated_embeddings:
            current_count = self.collection.count()

            if current_count:
                existing = self.collection.get(
                    limit=current_count,
                    include=["embeddings"],
                )
                existing_dimensions = {
                    len(vector)
                    for vector in existing["embeddings"]
                    if vector is not None
                }

                if (
                    existing_dimensions
                    and dimensions
                    and not dimensions.issubset(existing_dimensions)
                ):
                    raise ValueError(
                        "Embedding dimensions do not match the collection."
                    )

        self.delete_source(source)

        if not validated_chunks:
            return 0

        return self.add_chunks(
            validated_chunks,
            validated_embeddings,
        )
