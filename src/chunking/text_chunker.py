from typing import TypedDict
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100

class TextChunk(TypedDict):
    chunk_id: int
    text: str
    source: str
    page_number: int

def chunk_text(
    text: str,
    source: str,
    page_number: int,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[TextChunk]:
    """
    Split text into overlapping chunks while preserving source metadata.

    Chunk sizes and overlap are measured in characters.
    Chunk IDs start at zero and are local to this call.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    if page_number < 1:
        raise ValueError("page_number must be at least 1.")

    if not source.strip():
        raise ValueError("source cannot be empty.")

    text = text.strip()

    if not text:
        return []

    chunks: list[TextChunk] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(
                {
                    "chunk_id": len(chunks),
                    "text": chunk,
                    "source": source,
                    "page_number": page_number,
                }
            )

        if end == len(text):
            break

        start = end - chunk_overlap

    return chunks