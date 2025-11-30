from functools import lru_cache
from pypdf import PdfReader

@lru_cache()
def load_pdf_text(path: str) -> str:
    text = ""
    try:
        reader = PdfReader(path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    except:
        return ""
    return text

@lru_cache()
def load_text_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except:
        return ""