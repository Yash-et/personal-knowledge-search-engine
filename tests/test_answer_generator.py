from types import SimpleNamespace
import pytest
from src.rag.answer_generator import generate_answer

class FakeOllamaClient:
    def __init__(self, content="Python is a programming language. [S1]"):
        self.content = content
        self.last_request = None

    def chat(self, **kwargs):
        self.last_request = kwargs
        return SimpleNamespace(
            message=SimpleNamespace(content=self.content)
        )


@pytest.fixture
def search_results():
    return [
        {
            "text": "Python is a programming language.",
            "metadata": {
                "source": "python.txt",
                "page_number": 1,
            },
            "distance": 0.1,
        },
        {
            "text": "Python supports multiple programming paradigms.",
            "metadata": {
                "source": "python_guide.pdf",
                "page_number": 3,
            },
            "distance": 0.2,
        },
    ]


def test_generate_answer_returns_answer_and_citation(search_results):
    client = FakeOllamaClient()

    result = generate_answer(
        "What is Python?",
        search_results,
        client=client,
    )

    assert result["answer"] == (
        "Python is a programming language. [S1]"
    )
    assert result["citations"] == [
        {
            "source": "python.txt",
            "page_number": 1,
            "reference": "S1",
        }
    ]


def test_prompt_contains_question_and_retrieved_sources(search_results):
    client = FakeOllamaClient()

    generate_answer(
        "What is Python?",
        search_results,
        client=client,
    )

    prompt = client.last_request["messages"][1]["content"]

    assert "What is Python?" in prompt
    assert "python.txt" in prompt
    assert "python_guide.pdf" in prompt
    assert "[S1]" in prompt
    assert "[S2]" in prompt


def test_no_results_does_not_call_model():
    client = FakeOllamaClient()

    result = generate_answer(
        "What is Python?",
        [],
        client=client,
    )

    assert result["citations"] == []
    assert "couldn't find relevant information" in result["answer"]
    assert client.last_request is None


def test_empty_question_is_rejected(search_results):
    with pytest.raises(ValueError, match="Question must not be empty"):
        generate_answer("  ", search_results)


def test_empty_model_name_is_rejected(search_results):
    with pytest.raises(ValueError, match="Model name must not be empty"):
        generate_answer(
            "What is Python?",
            search_results,
            model_name=" ",
        )


def test_multiple_citations_are_mapped(search_results):
    client = FakeOllamaClient(
        "Python is a language with multiple paradigms. [S1] [S2] [S1]"
    )

    result = generate_answer(
        "Describe Python.",
        search_results,
        client=client,
    )

    assert [c["source"] for c in result["citations"]] == [
        "python.txt",
        "python_guide.pdf",
    ]


def test_unknown_citation_label_is_ignored(search_results):
    client = FakeOllamaClient(
        "The answer is supported by the document. [S9]"
    )

    result = generate_answer(
        "What is Python?",
        search_results,
        client=client,
    )

    assert result["citations"] == []
