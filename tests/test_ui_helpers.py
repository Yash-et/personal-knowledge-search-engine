from pathlib import Path
import pytest
from src.ui_helpers import save_uploaded_file
class FakeUploadedFile:
    def __init__(self, name, content=b"Sample document content"):
        self.name = name
        self._content = content

    def getvalue(self):
        return self._content

def test_save_txt_upload(tmp_path: Path):
    upload = FakeUploadedFile("notes.txt", b"Python notes")

    saved_path = save_uploaded_file(upload, tmp_path / "documents")

    assert saved_path.name == "notes.txt"
    assert saved_path.read_bytes() == b"Python notes"
    assert saved_path.parent.is_dir()

@pytest.mark.parametrize(
    "filename",
    ["notes.pdf", "notes.md", "NOTES.TXT"],
)
def test_supported_extensions_are_accepted(tmp_path: Path, filename: str):
    upload = FakeUploadedFile(filename)

    saved_path = save_uploaded_file(upload, tmp_path)

    assert saved_path.exists()
    assert saved_path.name == filename

@pytest.mark.parametrize(
    "filename",
    ["script.py", "image.png", "notes.docx", "no_extension"],
)
def test_unsupported_extensions_are_rejected(
    tmp_path: Path, filename: str
):
    upload = FakeUploadedFile(filename)

    with pytest.raises(ValueError, match="Unsupported file type"):
        save_uploaded_file(upload, tmp_path)

@pytest.mark.parametrize(
    "filename",
    ["", "   ", None],
)
def test_missing_filename_is_rejected(tmp_path: Path, filename):
    upload = FakeUploadedFile(filename)

    with pytest.raises(ValueError, match="filename"):
        save_uploaded_file(upload, tmp_path)

def test_path_traversal_filename_is_reduced_to_basename(tmp_path: Path):
    upload = FakeUploadedFile("../../outside.txt", b"Safe content")

    saved_path = save_uploaded_file(upload, tmp_path / "documents")

    assert saved_path.parent == tmp_path / "documents"
    assert saved_path.name == "outside.txt"
    assert saved_path.read_bytes() == b"Safe content"

def test_file_read_error_is_reported(tmp_path: Path):
    class BrokenUpload:
        name = "notes.txt"

        def getvalue(self):
            raise OSError("Simulated read failure")

    with pytest.raises(ValueError, match="Could not read"):
        save_uploaded_file(BrokenUpload(), tmp_path)

def test_non_bytes_content_is_rejected(tmp_path: Path):
    upload = FakeUploadedFile("notes.txt")
    upload.getvalue = lambda: "not bytes"

    with pytest.raises(ValueError, match="must be bytes"):
        save_uploaded_file(upload, tmp_path)