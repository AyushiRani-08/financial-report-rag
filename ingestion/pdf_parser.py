# ingestion/pdf_parser.py
"""
PDF ingestion: layout-aware extraction of narrative text blocks and financial tables.

Tables are detected with PyMuPDF's native `page.find_tables()` (vector-line detection,
no extra dependency) and converted into TableBlocks, so their grid survives into the
vector store instead of being flattened into a run of numbers. Text that falls inside
a detected table is dropped — it would otherwise duplicate the table as unreadable noise.

The ordered block list is handed to `table_extractor.chunk_document_blocks`, which
repeats caption, unit context and headers on every table chunk.
"""

import io
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image
import pymupdf  # Modern replacement for fitz

from ingestion.table_extractor import (
    TableBlock,
    TextBlock,
    chunk_document_blocks,
    clean_cell_text,
    extract_unit_context,
    grid_to_text,
    is_footnote_text,
    looks_like_financial_table,
    stitch_tables,
    table_block_from_grid,
)

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


BBox = Tuple[float, float, float, float]

# "Item 7. Management's Discussion" / "PART II" — the only stable PDF section markers.
_PDF_HEADING_RE = re.compile(r"^\s*(?:part\s+[ivx]+\b|item\s+\d+[a-z]?\b)", re.IGNORECASE)

_CAPTION_MAX_WORDS = 30
_CAPTION_MAX_GAP = 90.0  # points between a table and its caption


# ─────────────────────────────────────────────────────────────
# GEOMETRY HELPERS
# ─────────────────────────────────────────────────────────────

def _area(bbox: BBox) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _intersection_area(a: BBox, b: BBox) -> float:
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    return width * height


def _centre_inside(inner: BBox, outer: BBox) -> bool:
    cx = (inner[0] + inner[2]) / 2
    cy = (inner[1] + inner[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def _belongs_to_table(text_bbox: BBox, table_bbox: BBox) -> bool:
    """True when a text block is table content rather than surrounding prose."""
    if _centre_inside(text_bbox, table_bbox):
        return True
    area = _area(text_bbox)
    return area > 0 and _intersection_area(text_bbox, table_bbox) / area >= 0.5


# ─────────────────────────────────────────────────────────────
# PAGE-LEVEL EXTRACTION
# ─────────────────────────────────────────────────────────────

def _ocr_page(page, dpi: int) -> str:
    pix = page.get_pixmap(dpi=dpi)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img, config="--psm 3").strip()


def _page_text_blocks(page) -> List[Dict[str, Any]]:
    """Narrative text blocks with their bounding boxes (images skipped)."""
    blocks: List[Dict[str, Any]] = []
    try:
        raw = page.get_text("blocks")
    except Exception:
        return blocks

    for block in raw:
        if len(block) < 5:
            continue
        block_type = block[6] if len(block) > 6 else 0
        if block_type != 0:
            continue
        text = (block[4] or "").strip()
        if not text:
            continue
        blocks.append({
            "bbox": (float(block[0]), float(block[1]), float(block[2]), float(block[3])),
            "text": text,
        })
    return blocks


def _page_tables(page) -> List[Tuple[BBox, List[List[str]]]]:
    """Detected tables as (bbox, grid). Empty when detection is unavailable."""
    entries: List[Tuple[BBox, List[List[str]]]] = []
    try:
        finder = page.find_tables()
    except Exception:
        return entries

    for table in getattr(finder, "tables", None) or []:
        try:
            grid = table.extract()
        except Exception:
            continue
        if not grid or len(grid) < 2:
            continue
        cleaned = [["" if cell is None else str(cell) for cell in row] for row in grid]
        width = max((len(row) for row in cleaned), default=0)
        if width < 2:
            continue
        try:
            bbox = tuple(float(v) for v in table.bbox)
        except Exception:
            continue
        try:
            names = getattr(getattr(table, "header", None), "names", None) or []
            header_hint = [str(n or "") for n in names]
        except Exception:
            header_hint = []
        entries.append((bbox, cleaned, header_hint))
    return entries


def _surrounding_text(table_bbox: BBox, text_blocks: List[Dict[str, Any]]) -> Tuple[str, str, List[str]]:
    """Caption, unit context and footnotes around a table, taken from non-table text."""
    outside = [
        tb for tb in text_blocks
        if not _belongs_to_table(tb["bbox"], table_bbox)
    ]

    above = sorted(
        (
            (table_bbox[1] - tb["bbox"][3], tb)
            for tb in outside
            if 0 <= table_bbox[1] - tb["bbox"][3] <= _CAPTION_MAX_GAP
        ),
        key=lambda pair: pair[0],
    )

    caption = ""
    unit_context = ""
    if above:
        nearest = clean_cell_text(above[0][1]["text"])
        if nearest and len(nearest.split()) <= _CAPTION_MAX_WORDS:
            caption = nearest
        for _, tb in above[:3]:
            found = extract_unit_context(tb["text"])
            if found:
                unit_context = found
                break
    if not unit_context and caption:
        unit_context = extract_unit_context(caption) or ""

    footnotes: List[str] = []
    for tb in outside:
        gap = tb["bbox"][1] - table_bbox[3]
        if 0 <= gap <= _CAPTION_MAX_GAP and is_footnote_text(tb["text"]):
            note = clean_cell_text(tb["text"])
            if note and note not in footnotes:
                footnotes.append(note)

    return caption, unit_context, footnotes


def _normalise_ws(text: str) -> str:
    return re.sub(r"[ \t\r\f\v]+", " ", text.replace("\xa0", " ")).strip()


def _looks_like_heading(text: str) -> bool:
    return len(text.split()) <= 15 and bool(_PDF_HEADING_RE.match(text))


# ─────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────

def extract_pdf_blocks(
    pdf_path: str, ocr_threshold: int = 50, dpi: int = 300
) -> List[Any]:
    """Returns an ordered list of TextBlock / TableBlock in reading order."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    blocks: List[Any] = []
    current_section = ""

    doc = pymupdf.open(pdf_path)
    try:
        for page_index, page in enumerate(doc):
            page_no = page_index + 1
            text_blocks = _page_text_blocks(page)
            tables = _page_tables(page)

            # Scanned page fallback — only if Tesseract is available
            if not text_blocks and not tables and _OCR_AVAILABLE:
                ocr_text = _ocr_page(page, dpi)
                if len(ocr_text) >= ocr_threshold:
                    text_blocks = [{
                        "bbox": (0.0, 0.0, float(page.rect.width), float(page.rect.height)),
                        "text": ocr_text,
                    }]

            table_bboxes = [bbox for bbox, _, _ in tables]

            items: List[Dict[str, Any]] = []
            for bbox, grid, header_hint in tables:
                items.append({
                    "y": bbox[1], "x": bbox[0], "kind": "table", "bbox": bbox,
                    "grid": grid, "header_hint": header_hint,
                })
            for tb in text_blocks:
                if any(_belongs_to_table(tb["bbox"], table_bbox) for table_bbox in table_bboxes):
                    continue
                items.append({"y": tb["bbox"][1], "x": tb["bbox"][0], "kind": "text", "bbox": tb["bbox"], "text": tb["text"]})

            items.sort(key=lambda item: (round(item["y"], 1), round(item["x"], 1)))

            for item in items:
                if item["kind"] == "table":
                    if not looks_like_financial_table(item["grid"]):
                        # Layout grid, not a statement: keep its words as prose
                        # rather than losing the text hidden inside the bbox.
                        fallback = grid_to_text(item["grid"])
                        if fallback:
                            blocks.append(TextBlock(text=fallback, page_number=page_no, section=current_section))
                        continue

                    caption, unit_context, footnotes = _surrounding_text(item["bbox"], text_blocks)
                    table_block = table_block_from_grid(
                        item["grid"],
                        caption=caption,
                        unit_context=unit_context,
                        section=current_section,
                        page_start=page_no,
                        page_end=page_no,
                        header_hint=item.get("header_hint"),
                    )
                    if table_block is None:
                        continue
                    table_block.bbox = item["bbox"]
                    table_block.footnotes = footnotes
                    blocks.append(table_block)
                else:
                    text = _normalise_ws(item["text"])
                    if not text:
                        continue
                    if _looks_like_heading(text):
                        current_section = text[:120]
                    blocks.append(TextBlock(text=text, page_number=page_no, section=current_section))
    finally:
        doc.close()

    return _stitch_table_run(blocks)


def _stitch_table_run(blocks: List[Any]) -> List[Any]:
    """Stitches continuation tables that are adjacent in reading order."""
    stitched: List[Any] = []
    run: List[TableBlock] = []

    for block in blocks:
        if isinstance(block, TableBlock):
            run.append(block)
            continue
        if run:
            stitched.extend(stitch_tables(run))
            run = []
        stitched.append(block)

    if run:
        stitched.extend(stitch_tables(run))
    return stitched


def chunk_pdf(
    pdf_path: str, chunk_size: int = 300, overlap_pct: float = 0.10
) -> List[Dict[str, Any]]:
    """Extracts blocks and returns chunks with page/table/section metadata."""
    blocks = extract_pdf_blocks(pdf_path)
    return chunk_document_blocks(blocks, chunk_size=chunk_size, overlap_pct=overlap_pct)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ingestion/pdf_parser.py <path_to_pdf>")
        sys.exit(1)

    pdf_file = sys.argv[1]
    chunks = chunk_pdf(pdf_file)

    tables = sum(1 for c in chunks if c.get("is_table"))
    print(f"Generated {len(chunks)} chunk(s) from {pdf_file} ({tables} table chunks).\n")
    if chunks:
        print("--- Preview of First Chunk ---")
        print(f"Page: {chunks[0]['page_number']}")
        print(f"Word Count: {chunks[0]['word_count']}")
        print(f"Text Preview:\n{chunks[0]['text'][:300]}...")
