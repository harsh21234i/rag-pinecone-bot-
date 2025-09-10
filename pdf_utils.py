from typing import List, Tuple
from pypdf import PdfReader

def extract_pdf_text_by_page(pdf_path: str) -> List[Tuple[int, str]]:
    """Extracts text from each page of PDF."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append((i + 1, text))
    return pages
