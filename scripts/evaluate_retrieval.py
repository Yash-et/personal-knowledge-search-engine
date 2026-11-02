import argparse
import json
import sys
import uuid
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
EVALUATION_DB_DIR = PROJECT_ROOT / "data" / "evaluation_db"

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

    cutoff_group = parser.add_mutually_exclusive_group()
    cutoff_group.add_argument(
        "--k",
        type=int,
        help="Single retrieval cutoff (default: 3).",
    )
    cutoff_group.add_argument(
        "--ks",
        type=int,
        nargs="+",
        help="Multiple retrieval cutoffs, for example: --ks 1 3 5.",
    )

    args = parser.parse_args()

    if args.ks is not None:
        if any(k <= 0 for k in args.ks):
            parser.error("Every value supplied to --ks must be greater than zero.")
        if len(set(args.ks)) != len(args.ks):
            parser.error("--ks must not contain duplicate values.")
        cutoffs = sorted(args.ks)
    else:
        single_k = 3 if args.k is None else args.k
        if single_k <= 0:
            parser.error("--k must be greater than zero.")
        cutoffs = [single_k]

    if not args.dataset.is_file():
        parser.error(f"Dataset file not found: {args.dataset}")

    try:
        with args.dataset.open("r", encoding="utf-8") as file:
            dataset = json.load(file)
    except json.JSONDecodeError as error:
        parser.error(f"Invalid JSON dataset: {error}")

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

    # Keep evaluation isolated from the application's production database.
    collection_name = f"retrieval_eval_{uuid.uuid4().hex}"
    vector_store = VectorStore(
        EVALUATION_DB_DIR,
        collection_name=collection_name,
    )

    try:
        vector_store.add_chunks(chunks, embeddings)

        report = evaluate_retrieval(
            dataset_path=args.dataset,
            embedding_model=embedding_model,
            vector_store=vector_store,
            ks=cutoffs,
        )
    finally:
        vector_store.client.delete_collection(name=collection_name)

    print("\nRetrieval Evaluation Report")
    print("=" * 60)
    print(f"Dataset:       {args.dataset}")
    print(f"Queries:       {report['query_count']}")
    print(f"Cutoffs:       {', '.join(map(str, report['ks']))}")
    print(f"MRR@{report['mrr_cutoff']}:          {report['mrr']:.3f}")

    print("\nOverall metrics")
    print("-" * 60)
    print(f"{'Metric':<16}" + "".join(f"{'k=' + str(k):>10}" for k in report["ks"]))

    for metric_name, metric_key in (
        ("Recall@k", "recall_at_k"),
        ("Precision@k", "precision_at_k"),
    ):
        values = "".join(
            f"{report['metrics'][k][metric_key]:>10.3f}"
            for k in report["ks"]
        )
        print(f"{metric_name:<16}{values}")

    print("\nPer-query results")
    print("-" * 60)

    for index, result in enumerate(report["results"], start=1):
        retrieved = result["retrieved_ids"]
        relevant = set(result["relevant_ids"])
        missed = relevant - set(retrieved)

        print(f"\n{index}. {result['query']}")
        print(f"   Retrieved: {retrieved}")
        print(f"   Relevant:  {result['relevant_ids']}")

        for cutoff in report["ks"]:
            metrics = result["metrics"][cutoff]
            print(
                f"   k={cutoff}: "
                f"Recall={metrics['recall_at_k']:.3f}, "
                f"Precision={metrics['precision_at_k']:.3f}"
            )

        print(f"   Missed:    {sorted(missed) if missed else 'None'}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
