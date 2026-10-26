import argparse
import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings.embedding_model import LocalEmbeddingModel
from src.evaluation.retrieval_evaluator import evaluate_retrieval
from src.storage.vector_store import VectorStore

DEFAULT_DATASET = (
    PROJECT_ROOT / "tests" / "fixtures" / "retrieval_eval_dataset.json"
)

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate semantic retrieval against a labeled dataset."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to the retrieval evaluation JSON dataset.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of top results to evaluate (default: 3).",
    )
    args = parser.parse_args()

    if args.k <= 0:
        parser.error("--k must be greater than zero.")

    if not args.dataset.is_file():
        parser.error(f"Dataset file not found: {args.dataset}")

    with args.dataset.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    documents = dataset.get("documents", [])
    if not documents:
        parser.error("The dataset must contain at least one document.")

    print("Loading local embedding model...")
    embedding_model = LocalEmbeddingModel()

    chunks = [
        {
            "chunk_id": document["chunk_id"],
            "text": document["text"],
            "source": document["source"],
            "page_number": document["page_number"],
        }
        for document in documents
    ]

    print(f"Preparing {len(chunks)} benchmark passages...")

    embeddings = embedding_model.embed_texts(
        [chunk["text"] for chunk in chunks]
    )

    # Keep evaluation vectors isolated from the application's real database.
    with tempfile.TemporaryDirectory(prefix="pks_retrieval_eval_") as temp_dir:
        vector_store = VectorStore(Path(temp_dir) / "vectors")
        vector_store.add_chunks(chunks, embeddings)

        report = evaluate_retrieval(
            dataset_path=args.dataset,
            embedding_model=embedding_model,
            vector_store=vector_store,
            k=args.k,
        )

        print("\nRetrieval Evaluation Report")
        print("=" * 60)
        print(f"Dataset:       {args.dataset}")
        print(f"Queries:       {report['query_count']}")
        print(f"Metric cutoff: k={report['k']}")
        print(f"Recall@{args.k}:      {report['recall_at_k']:.3f}")
        print(f"MRR:           {report['mrr']:.3f}")

        print("\nPer-query results")
        print("-" * 60)

        for index, result in enumerate(report["results"], start=1):
            retrieved = result["retrieved_ids"]
            relevant = set(result["relevant_ids"])
            found = set(retrieved) & relevant
            missed = relevant - set(retrieved)

            print(f"\n{index}. {result['query']}")
            print(f"   Retrieved: {retrieved}")
            print(f"   Relevant:  {result['relevant_ids']}")
            print(f"   Recall@{args.k}:  {result['recall_at_k']:.3f}")
            print(f"   Missed:    {sorted(missed) if missed else 'None'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
