
from pathlib import Path
from typing import TypedDict

from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


class DocumentPage(TypedDict):
    page_number: int
    text: str


class LoadedDocument(TypedDict):
    source: str
    file_type: str
    text: str
    pages: list[DocumentPage]


class DocumentLoadError(Exception):
    """Raised when a document cannot be loaded or contains no usable text."""


def _extract_pdf(path: Path) -> list[DocumentPage]:
    """Extract text page by page from a PDF."""
    try:
        reader = PdfReader(str(path))
        pages: list[DocumentPage] = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()

            if text:
                pages.append(
                    {
                        "page_number": page_number,
                        "text": text,
                    }
                )

        return pages

    except Exception as exc:
        raise DocumentLoadError(
            f"Could not read PDF '{path.name}': {exc}"
        ) from exc


def _extract_text_file(path: Path) -> list[DocumentPage]:
    """Read a UTF-8 text or Markdown file."""
    try:
        text = path.read_text(encoding="utf-8-sig").strip()
    except UnicodeDecodeError as exc:
        raise DocumentLoadError(
            f"File '{path.name}' is not valid UTF-8 text."
        ) from exc
    except OSError as exc:
        raise DocumentLoadError(
            f"Could not read file '{path.name}': {exc}"
        ) from exc

    return [{"page_number": 1, "text": text}] if text else []


def load_document(file_path: str | Path) -> LoadedDocument:
    """
    Load a PDF, TXT, or Markdown file.

    Returns:
        source: Original filename.
        file_type: Lowercase extension without the dot.
        text: Combined extracted text.
        pages: Non-empty text sections with source page numbers.
    """
    path = Path(file_path)

    if not path.exists():
        raise DocumentLoadError(f"File does not exist: {path}")

    if not path.is_file():
        raise DocumentLoadError(f"Path is not a file: {path}")

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise DocumentLoadError(
            f"Unsupported file type '{extension}'. "
            f"Supported types: {supported}"
        )

    if extension == ".pdf":
        pages = _extract_pdf(path)
    else:
        pages = _extract_text_file(path)

    if not pages:
        raise DocumentLoadError(
            f"No extractable text found in '{path.name}'. "
            "The file may be empty, image-only, or contain no readable text."
        )

    combined_text = "\n\n".join(page["text"] for page in pages)

    return {
        "source": path.name,
        "file_type": extension[1:],
        "text": combined_text,
        "pages": pages,
    }
