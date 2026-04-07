from pypdf import PdfReader
from autostructurer.chunker import chunk_text

def process_pdf(path: str):
    reader = PdfReader(path)
    full_text = []

    for page in reader.pages:
        t = page.extract_text() or ""
        if t.strip():
            full_text.append(t)

    text = "\n".join(full_text)
    chunks = chunk_text(text)

    return [{"text": c, "t_start": 0.0, "t_end": 0.0, "ref_path": None, "source": "pdf"} for c in chunks]
