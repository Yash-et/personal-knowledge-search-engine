
from pathlib import Path

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def save_uploaded_file(uploaded_file, documents_dir: str | Path) -> Path:
    """Validate and save an uploaded document using only its filename."""

    raw_name = getattr(uploaded_file, "name", None)

    if not isinstance(raw_name, str) or not raw_name.strip():
        raise ValueError("Uploaded file must have a filename.")

    # Normalize both Windows and POSIX separators before extracting the name.
    filename = Path(raw_name.replace("\\", "/")).name

    if filename in {"", ".", ".."}:
        raise ValueError("Invalid uploaded filename.")

    if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError("Unsupported file type. Use PDF, TXT, or Markdown.")

    try:
        content = uploaded_file.getvalue()
    except Exception as exc:
        raise ValueError("Could not read uploaded file contents.") from exc

    if not isinstance(content, (bytes, bytearray)):
        raise ValueError("Uploaded file contents must be bytes.")

    destination_dir = Path(documents_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    destination = destination_dir / filename
    destination.write_bytes(content)

    return destination
