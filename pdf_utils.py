from typing import List, Tuple
from pypdf import PdfReader

def extract_pdf_text_by_page(pdf_path: str) -> List[Tuple[int, str]]:
    """
    Returns list of (page_number_1indexed, text) for each page.
    """
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append((i + 1, text))
    return pages
