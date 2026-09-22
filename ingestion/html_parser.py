# ingestion/html_parser.py
from pathlib import Path
from bs4 import BeautifulSoup
import re


def extract_html_text(html_path: str) -> str:
    """Reads an HTML file, strips scripts/styles/nav elements, and returns clean text."""
    path = Path(html_path)
    if not path.exists():
        raise FileNotFoundError(f"HTML file not found: {html_path}")

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, "html.parser")

    # Detect SEC iFrame / ixViewer interactive wrapper stubs
    if "XBRL Viewer" in html_content or "loadViewer" in html_content:
        # Check if this is just the 6KB wrapper
        visible_text = soup.get_text(separator=" ").strip()
        if len(visible_text.split()) < 100:
            raise ValueError(
                "Uploaded file is an SEC Interactive Viewer wrapper (6 KB) rather than the actual 10-K report text. "
                "On SEC EDGAR, please open the primary document (.htm) directly (or right-click 'Open Document' -> Save Link As) and upload that full file."
            )

    # Remove script, style, noscript, svg elements
    for element in soup(["script", "style", "noscript", "svg", "iframe"]):
        element.decompose()

    # Extract text with newline separator for clear block structure
    text = soup.get_text(separator="\n")

    # Clean up excessive blank lines and whitespace
    lines = [line.strip() for line in text.splitlines()]
    clean_lines = [line for line in lines if line]
    clean_text = "\n".join(clean_lines)

    if len(clean_text.split()) < 20:
        raise ValueError(
            "Uploaded HTML file contains almost no text (less than 20 words). "
            "Please ensure you uploaded the full financial report rather than a stub/link page."
        )

    return clean_text


def chunk_html(
    html_path: str, chunk_size: int = 500, overlap_pct: float = 0.10
) -> list[dict]:
    """Extracts clean text from an HTML document and splits it into word chunks."""
    full_text = extract_html_text(html_path)
    words = full_text.split()
    if not words:
        return []

    overlap = int(chunk_size * overlap_pct)
    step = max(1, chunk_size - overlap)

    chunks = []
    chunk_id = 0
    estimated_page = 1

    for i in range(0, len(words), step):
        chunk_words = words[i : i + chunk_size]
        # Estimate virtual page (approx 500 words per page)
        estimated_page = (i // 500) + 1

        chunks.append(
            {
                "chunk_id": chunk_id,
                "word_count": len(chunk_words),
                "page_number": estimated_page,
                "text": " ".join(chunk_words),
            }
        )
        chunk_id += 1

        if i + chunk_size >= len(words):
            break

    return chunks


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        chunks = chunk_html(test_file)
        print(f"Generated {len(chunks)} chunks from {test_file}")
        if chunks:
            print("Preview chunk 0:\n", chunks[0]["text"][:300])
