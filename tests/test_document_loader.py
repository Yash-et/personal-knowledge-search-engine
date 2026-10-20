
from pathlib import Path

import pytest
from pypdf import PdfWriter

from src.ingestion.document_loader import (
    DocumentLoadError,
    load_document,
)


def test_load_txt_file(tmp_path: Path):
    file_path = tmp_path / "notes.txt"
    file_path.write_text(
        "Python supports semantic search.",
        encoding="utf-8",
    )

    result = load_document(file_path)

    assert result["source"] == "notes.txt"
    assert result["file_type"] == "txt"
    assert result["text"] == "Python supports semantic search."
    assert result["pages"] == [
        {
            "page_number": 1,
            "text": "Python supports semantic search.",
        }
    ]


def test_load_markdown_file(tmp_path: Path):
    file_path = tmp_path / "notes.md"
    file_path.write_text(
        "# Knowledge Base\n\nSearchable notes.",
        encoding="utf-8",
    )

    result = load_document(file_path)

    assert result["file_type"] == "md"
    assert "# Knowledge Base" in result["text"]
    assert result["pages"][0]["page_number"] == 1


def test_empty_text_file_raises_error(tmp_path: Path):
    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding="utf-8")

    with pytest.raises(DocumentLoadError, match="No extractable text"):
        load_document(file_path)


def test_missing_file_raises_error(tmp_path: Path):
    file_path = tmp_path / "missing.txt"

    with pytest.raises(DocumentLoadError, match="does not exist"):
        load_document(file_path)


def test_unsupported_file_type_raises_error(tmp_path: Path):
    file_path = tmp_path / "notes.docx"
    file_path.write_text("Unsupported for now.", encoding="utf-8")

    with pytest.raises(DocumentLoadError, match="Unsupported file type"):
        load_document(file_path)


def test_invalid_utf8_file_raises_error(tmp_path: Path):
    file_path = tmp_path / "invalid.txt"
    file_path.write_bytes(b"\xff\xfe\xfa")

    with pytest.raises(DocumentLoadError, match="valid UTF-8"):
        load_document(file_path)


def test_empty_pdf_raises_error(tmp_path: Path):
    file_path = tmp_path / "empty.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)

    with file_path.open("wb") as pdf_file:
        writer.write(pdf_file)

    with pytest.raises(DocumentLoadError, match="No extractable text"):
        load_document(file_path)


def test_invalid_pdf_raises_error(tmp_path: Path):
    file_path = tmp_path / "invalid.pdf"
    file_path.write_text("This is not a valid PDF.", encoding="utf-8")

    with pytest.raises(DocumentLoadError, match="Could not read PDF"):
        load_document(file_path)


def test_extension_is_case_insensitive(tmp_path: Path):
    file_path = tmp_path / "NOTES.TXT"
    file_path.write_text("Uppercase extension works.", encoding="utf-8")

    result = load_document(file_path)

    assert result["file_type"] == "txt"
    assert result["text"] == "Uppercase extension works."
