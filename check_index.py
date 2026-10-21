
from pathlib import Path

from src.storage.vector_store import VectorStore

project_root = Path(__file__).resolve().parent
store = VectorStore(project_root / "data" / "vector_db")

print("Total indexed passages:", store.count())

results = store.collection.get(
    limit=1000,
    include=["metadatas"],
)

sources = sorted({
    metadata.get("source", "Unknown source")
    for metadata in results["metadatas"]
    if metadata
})

print("\nIndexed source filenames:")
for source in sources:
    print("-", source)

target = "FINAL MOVIE TALWAR-1.pdf.pdf"
target_count = sum(
    1
    for metadata in results["metadatas"]
    if metadata and metadata.get("source") == target
)

print(f"\nPassages belonging to {target}: {target_count}")
