import pytest
from src.chunking.text_chunker import chunk_text

def test_empty_text_returns_no_chunks():
    assert chunk_text("", "notes.txt", 1) == []
    assert chunk_text("   ", "notes.txt", 1) == []

def test_short_text_creates_one_chunk():
    result = chunk_text("Hello world", "notes.txt", 1)

    assert len(result) == 1
    assert result[0] == {
        "chunk_id": 0,
        "text": "Hello world",
        "source": "notes.txt",
        "page_number": 1,
    }

def test_long_text_is_split_into_chunks():
    text = "a" * 1200

    result = chunk_text(
        text,
        "notes.txt",
        1,
        chunk_size=500,
        chunk_overlap=100,
    )

    assert [len(chunk["text"]) for chunk in result] == [500, 500, 400]
    assert "".join(
        [
            result[0]["text"],
            result[1]["text"][100:],
            result[2]["text"][100:],
        ]
    ) == text

def test_adjacent_chunks_overlap():
    text = "abcdefghijklmnopqrstuvwxyz"

    result = chunk_text(
        text,
        "notes.txt",
        1,
        chunk_size=10,
        chunk_overlap=3,
    )

    assert result[0]["text"][-3:] == result[1]["text"][:3]
    assert result[1]["text"][-3:] == result[2]["text"][:3]

def test_chunk_metadata_is_preserved():
    result = chunk_text(
        "a" * 25,
        "research.pdf",
        4,
        chunk_size=10,
        chunk_overlap=2,
    )

    assert all(chunk["source"] == "research.pdf" for chunk in result)
    assert all(chunk["page_number"] == 4 for chunk in result)
    assert [chunk["chunk_id"] for chunk in result] == list(range(len(result)))

@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [
        (0, 0),
        (-1, 0),
        (10, -1),
        (10, 10),
        (10, 11),
    ],
)
def test_invalid_chunk_settings_raise_error(chunk_size, chunk_overlap):
    with pytest.raises(ValueError):
        chunk_text(
            "Some text",
            "notes.txt",
            1,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

def test_invalid_page_number_raises_error():
    with pytest.raises(ValueError, match="page_number"):
        chunk_text("Some text", "notes.txt", 0)

def test_empty_source_raises_error():
    with pytest.raises(ValueError, match="source"):
        chunk_text("Some text", "   ", 1)
