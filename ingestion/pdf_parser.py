# ingestion/pdf_parser.py
import io
from pathlib import Path
from PIL import Image
import pymupdf  # Modern replacement for fitz

# Tesseract OCR is optional — used only for scanned/image-only PDFs.
# On Streamlit Cloud and other Linux environments without Tesseract installed,
# OCR is silently skipped and PyMuPDF native text extraction is used instead.
# SEC filings are selectable text, so this covers 99% of real-world uploads.
_OCR_AVAILABLE = False
try:
    import pytesseract
    import shutil
    # Prefer system PATH first (Linux/Cloud); fall back to Windows install path
    if shutil.which("tesseract"):
        _OCR_AVAILABLE = True
    else:
        _win_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if Path(_win_path).exists():
            pytesseract.pytesseract.tesseract_cmd = _win_path
            _OCR_AVAILABLE = True
except ImportError:
    pass


def extract_raw_text(
    pdf_path: str, ocr_threshold: int = 50, dpi: int = 300
) -> list[dict]:
    """Extract text page-by-page.

    Falls back to Tesseract OCR for scanned pages only if Tesseract is
    installed and available. Otherwise skips OCR gracefully.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = []
    doc = pymupdf.open(pdf_path)
    try:
        for i, page in enumerate(doc):
            text = page.get_text().strip()

            # Scanned page fallback — only if Tesseract is available
            if len(text) < ocr_threshold and _OCR_AVAILABLE:
                pix = page.get_pixmap(dpi=dpi)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                text = pytesseract.image_to_string(
                    img, config="--psm 3"
                ).strip()

            pages.append({
                "page_number": i + 1,
                "text": text,
            })
    finally:
        doc.close()

    return pages


def extract_full_text(pdf_path: str) -> str:
    """Convenience wrapper: all pages joined into one string."""
    pages = extract_raw_text(pdf_path)
    return "\n\n".join(p["text"] for p in pages if p["text"])


def chunk_text_by_words(
    text: str, chunk_size: int = 500, overlap_pct: float = 0.10
) -> list[dict]:
    """Splits text into word chunks with overlapping words."""
    words = text.split()
    if not words:
        return []

    overlap = int(chunk_size * overlap_pct)  # 50 words
    step = max(1, chunk_size - overlap)  # 450 words

    chunks = []
    chunk_id = 0

    for i in range(0, len(words), step):
        chunk_words = words[i : i + chunk_size]
        chunks.append({
            "chunk_id": chunk_id,
            "word_count": len(chunk_words),
            "text": " ".join(chunk_words),
        })
        chunk_id += 1

        if i + chunk_size >= len(words):
            break

    return chunks


def chunk_pdf(
    pdf_path: str, chunk_size: int = 500, overlap_pct: float = 0.10
) -> list[dict]:
    """Extracts raw text and returns structured chunks with page metadata."""
    pages = extract_raw_text(pdf_path)
    all_chunks = []
    global_chunk_id = 0

    for page in pages:
        page_chunks = chunk_text_by_words(
            page["text"], chunk_size=chunk_size, overlap_pct=overlap_pct
        )
        for chunk in page_chunks:
            chunk["chunk_id"] = global_chunk_id
            chunk["page_number"] = page["page_number"]
            all_chunks.append(chunk)
            global_chunk_id += 1

    return all_chunks


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ingestion/pdf_parser.py <path_to_pdf>")
        sys.exit(1)

    pdf_file = sys.argv[1]

    # Run extraction and chunking
    chunks = chunk_pdf(pdf_file, chunk_size=500, overlap_pct=0.10)

    print(f"Successfully generated {len(chunks)} chunk(s) from {pdf_file}.\n")
    if chunks:
        print("--- Preview of First Chunk ---")
        print(f"Page: {chunks[0]['page_number']}")
        print(f"Word Count: {chunks[0]['word_count']}")
        print(f"Text Preview:\n{chunks[0]['text'][:300]}...")