from pathlib import Path

from src.ingestion.document_loader import load_document

pdf_path = Path(
    r"C:\Users\HP\Downloads\FINAL MOVIE TALWAR-1.pdf.pdf"
)

try:
    document = load_document(pdf_path)

    print("Filename:", document["source"])
    print("File type:", document["file_type"])
    print("Pages extracted:", len(document["pages"]))
    print("Total text characters:", len(document["text"]))

    for page in document["pages"][:3]:
        print(f"\n--- Page {page['page_number']} ---")
        print(page["text"][:1000])

except Exception as exc:
    print(f"PDF extraction failed: {type(exc).__name__}: {exc}")