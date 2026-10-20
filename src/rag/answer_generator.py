import re
from typing import Any, Sequence
import ollama

DEFAULT_LLM_MODEL = "llama3.2:3b"
def generate_answer(
    question: str,
    search_results: Sequence[dict[str, Any]],
    model_name: str = DEFAULT_LLM_MODEL,
    client=None,
) -> dict[str, Any]:
    """Generate a source-grounded answer from retrieved document chunks."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must not be empty.")

    if not model_name or not model_name.strip():
        raise ValueError("Model name must not be empty.")

    if not search_results:
        return {
            "answer": (
                "I couldn't find relevant information in your indexed "
                "documents to answer this question."
            ),
            "citations": [],
        }

    passages = []
    source_map = {}

    for index, result in enumerate(search_results, start=1):
        text = result.get("text", "").strip()
        metadata = result.get("metadata", {})

        if not text:
            continue

        source = metadata.get("source", "Unknown source")
        page_number = metadata.get("page_number")

        label = f"S{index}"
        source_map[label] = {
            "source": source,
            "page_number": page_number,
        }

        passages.append(
            f"[{label}]\n"
            f"Source: {source}\n"
            f"Page: {page_number if page_number is not None else 'N/A'}\n"
            f"Content: {text}"
        )

    if not passages:
        return {
            "answer": (
                "I couldn't find usable text in the retrieved documents."
            ),
            "citations": [],
        }
    formatted_passages = "\n\n".join(passages)
    prompt = f"""
Answer the user's question using only the document passages below.

Rules:
- Use only information supported by these passages.
- Cite supporting passages with their exact labels, such as [S1].
- Do not invent facts, sources, or page numbers.
- If the passages do not contain enough evidence, say so clearly.
- Keep the answer clear and relevant.

DOCUMENT PASSAGES:
{formatted_passages}

USER QUESTION:
{question.strip()}
"""

    if client is None:
        client = ollama.Client()

    response = client.chat(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful research assistant. "
                    "Ground every factual claim in the supplied evidence."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )

    answer = response.message.content.strip()

    # Return only sources explicitly cited by the model.
    cited_labels = list(dict.fromkeys(re.findall(r"\[S(\d+)\]", answer)))
    citations = [
        {
            "source": source_map[f"S{label}"]["source"],
            "page_number": source_map[f"S{label}"]["page_number"],
            "reference": f"S{label}",
        }
        for label in cited_labels
        if f"S{label}" in source_map
    ]

    return {
        "answer": answer,
        "citations": citations,
    }
