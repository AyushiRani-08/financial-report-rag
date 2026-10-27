"""
ingestion/table_extractor.py
Financial Table Extraction, Normalization, and Table-Aware Chunking Engine.

Handles:
1. Unit Context: Preserves and normalizes "(in millions)", "(in thousands)",
   currency symbols, and "except per share" declarations.
2. Merged & Multi-Tier Headers: Flattens hierarchical headers and resolves
   colspans so columns retain full semantic context.
3. Multi-Page Table Stitching: Joins continuation tables across consecutive
   pages with unified metadata.
4. Footnotes: Binds footnotes directly to table blocks.
5. Table-Aware Chunking: Keeps tables atomic when possible; when splitting is
   unavoidable, repeats caption, unit context, and column headers on every chunk.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# Common financial unit patterns
UNIT_PATTERNS = [
    r"\(?\s*(?:in|amounts?\s+in|dollars?\s+in|\$|usd|rs\.?|inr|eur|gbp)?\s*(?:billions?|millions?|thousands?|crores?|lakhs?)(?:\s*,?\s*except\s+(?:per\s+)?share(?:\s+amounts?|\s+data)?)?\s*\)?",
    r"\(?\s*(?:in|amounts?\s+in)?\s*(?:shares|units|percentage|basis\s+points)\s*\)?",
    r"\(?\s*except\s+(?:per\s+)?share(?:\s+amounts?|\s+data)?\s*\)?",
]
UNIT_REGEXES = [re.compile(p, re.IGNORECASE) for p in UNIT_PATTERNS]

FOOTNOTE_PREFIX_REGEX = re.compile(
    r"^(?:\*+|\(\s*[\d\w]+\s*\)|\[\s*[\d\w]+\s*\]|note\s*[\d\w]*:?|see accompanying notes)",
    re.IGNORECASE,
)


@dataclass
class TableBlock:
    """Represents an extracted financial table."""
    caption: str = ""
    unit_context: str = ""
    headers: List[str] = field(default_factory=list)
    rows: List[List[str]] = field(default_factory=list)
    footnotes: List[str] = field(default_factory=list)
    page_start: int = 1
    page_end: int = 1
    bbox: Optional[Tuple[float, float, float, float]] = None

    @property
    def col_count(self) -> int:
        if self.headers:
            return len(self.headers)
        if self.rows:
            return len(self.rows[0])
        return 0

    @property
    def row_count(self) -> int:
        return len(self.rows)


@dataclass
class TextBlock:
    """Represents narrative text outside tables."""
    text: str
    page_number: int


def clean_cell_text(text: Optional[str]) -> str:
    """Normalizes cell whitespace, removing newlines and non-breaking spaces."""
    if not text:
        return ""
    text = text.replace("\xa0", " ").replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_unit_context(text: str) -> Optional[str]:
    """
    Searches a text string (caption, header, or preceding paragraph)
    for financial unit context declarations.
    Examples:
      - "(In millions, except per share amounts)" -> "in millions, except per share amounts"
      - "Dollars in thousands" -> "dollars in thousands"
      - "Rs. in Crores" -> "Rs. in Crores"
    """
    if not text:
        return None

    for rx in UNIT_REGEXES:
        match = rx.search(text)
        if match:
            found = match.group(0).strip("() \t\r\n")
            # Filter out false positives like "in million people"
            if len(found) < 80:
                return found
    return None


def is_footnote_text(text: str) -> bool:
    """Determines if a text line is a footnote."""
    stripped = text.strip()
    return bool(FOOTNOTE_PREFIX_REGEX.match(stripped))


def flatten_multitier_headers(header_rows: List[List[str]]) -> List[str]:
    """
    Combines multi-tier header rows into single, descriptive column labels.
    Handles forward-filling empty spanning header cells.

    Example:
      Row 0: ["", "Three Months Ended", ""]
      Row 1: ["Line Item", "June 30, 2024", "June 30, 2023"]
      Output: ["Line Item", "Three Months Ended - June 30, 2024", "Three Months Ended - June 30, 2023"]
    """
    if not header_rows:
        return []
    if len(header_rows) == 1:
        return [clean_cell_text(c) or f"Col_{i+1}" for i, c in enumerate(header_rows[0])]

    # Determine max columns
    max_cols = max(len(r) for r in header_rows)

    # Normalize matrix to max_cols
    grid: List[List[str]] = []
    for r in header_rows:
        cells = [clean_cell_text(c) for c in r]
        if len(cells) < max_cols:
            cells.extend([""] * (max_cols - len(cells)))
        grid.append(cells)

    # Forward fill parent rows if spanning across sub-columns
    for row_idx in range(len(grid) - 1):
        last_val = ""
        for col_idx in range(max_cols):
            val = grid[row_idx][col_idx]
            if val:
                last_val = val
            elif last_val and (col_idx + 1 < max_cols and grid[row_idx][col_idx + 1] == ""):
                grid[row_idx][col_idx] = last_val

    # Combine downward
    final_headers: List[str] = []
    for col_idx in range(max_cols):
        parts = []
        for row_idx in range(len(grid)):
            part = grid[row_idx][col_idx]
            if part and part not in parts:
                parts.append(part)
        col_name = " - ".join(parts) if parts else f"Col_{col_idx+1}"
        final_headers.append(col_name)

    return final_headers


def table_to_markdown_chunks(
    table: TableBlock,
    max_chunk_words: int = 500,
) -> List[Dict[str, Any]]:
    """
    Converts a TableBlock into one or more Markdown chunks.
    CRITICAL: If the table exceeds max_chunk_words, repeats the Caption,
    Unit Context, and Column Headers on every single sub-chunk!
    """
    # 1. Build Header Block
    caption_part = table.caption if table.caption else "Financial Table"
    unit_part = f" | Unit Context: {table.unit_context}" if table.unit_context else ""
    page_part = (
        f" | Page: {table.page_start}"
        if table.page_start == table.page_end
        else f" | Pages: {table.page_start}-{table.page_end}"
    )

    col_count = table.col_count
    if col_count == 0:
        return []

    headers = [h or f"Col_{i+1}" for i, h in enumerate(table.headers)]
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"

    # Footnotes block
    footnote_str = ""
    if table.footnotes:
        footnote_str = "\nFootnotes:\n" + "\n".join(f"- {fn}" for fn in table.footnotes)

    # Format rows
    formatted_rows = []
    for r in table.rows:
        # Pad or trim cells to match header col_count
        cells = [clean_cell_text(c) for c in r[:col_count]]
        if len(cells) < col_count:
            cells.extend([""] * (col_count - len(cells)))
        # Escape pipe characters in cell content
        escaped_cells = [c.replace("|", "\\|") for c in cells]
        formatted_rows.append("| " + " | ".join(escaped_cells) + " |")

    # If small enough, keep as 1 atomic chunk
    full_table_text = (
        f"[{caption_part}{unit_part}{page_part}]\n"
        f"{header_line}\n"
        f"{separator_line}\n"
        + "\n".join(formatted_rows)
        + (f"\n{footnote_str}" if footnote_str else "")
    )
    full_words = len(full_table_text.split())

    if full_words <= max_chunk_words:
        return [
            {
                "text": full_table_text,
                "word_count": full_words,
                "page_number": table.page_start,
                "is_table": True,
                "table_caption": table.caption,
                "unit_context": table.unit_context,
            }
        ]

    # Split into multiple parts with repeating header block
    chunks = []
    # Calculate baseline word count of repeated header
    base_header_words = len(
        f"[{caption_part} (Part 99/99){unit_part}{page_part}]\n{header_line}\n{separator_line}\n".split()
    )
    available_words = max(50, max_chunk_words - base_header_words)

    row_batches: List[List[str]] = []
    current_batch: List[str] = []
    current_words = 0

    for row_str in formatted_rows:
        rw_words = len(row_str.split())
        if current_words + rw_words > available_words and current_batch:
            row_batches.append(current_batch)
            current_batch = [row_str]
            current_words = rw_words
        else:
            current_batch.append(row_str)
            current_words += rw_words

    if current_batch:
        row_batches.append(current_batch)

    total_parts = len(row_batches)
    for idx, batch in enumerate(row_batches, start=1):
        part_tag = f" (Part {idx}/{total_parts})"
        part_meta = f"[{caption_part}{part_tag}{unit_part}{page_part}]\n"
        part_text = (
            part_meta
            + header_line
            + "\n"
            + separator_line
            + "\n"
            + "\n".join(batch)
        )
        # Include footnotes on final sub-chunk
        if idx == total_parts and footnote_str:
            part_text += f"\n{footnote_str}"

        chunks.append(
            {
                "text": part_text,
                "word_count": len(part_text.split()),
                "page_number": table.page_start,
                "is_table": True,
                "table_caption": table.caption,
                "unit_context": table.unit_context,
            }
        )

    return chunks


def can_stitch_tables(t1: TableBlock, t2: TableBlock) -> bool:
    """
    Checks if t2 is a continuation of t1 across consecutive pages.
    Criteria:
      - Pages are consecutive (t2.page_start == t1.page_end + 1)
      - Column counts match
      - Similar column headers or continuation markers in t2 caption/headers
    """
    if t2.page_start != t1.page_end + 1:
        return False
    if t1.col_count != t2.col_count or t1.col_count == 0:
        return False

    # Check for continuation keywords
    c2_lower = t2.caption.lower()
    if any(k in c2_lower for k in ("continued", "cont'd", "statement", "balance sheet", "operations")):
        return True

    # Check if headers match closely
    if t1.headers and t2.headers:
        matches = sum(
            1 for h1, h2 in zip(t1.headers, t2.headers)
            if h1.lower() == h2.lower() or (not h1 and not h2)
        )
        if matches / len(t1.headers) >= 0.7:
            return True

    # If t2 has no distinct headers and has same col_count, it's likely continuation rows
    if not t2.headers or t2.headers == [f"Col_{i+1}" for i in range(t2.col_count)]:
        return True

    return False


def stitch_tables(tables: List[TableBlock]) -> List[TableBlock]:
    """Combines continuation tables across pages into single unified TableBlocks."""
    if len(tables) <= 1:
        return tables

    stitched: List[TableBlock] = []
    curr = tables[0]

    for next_tab in tables[1:]:
        if can_stitch_tables(curr, next_tab):
            # Merge rows
            curr.rows.extend(next_tab.rows)
            # Merge footnotes
            for fn in next_tab.footnotes:
                if fn not in curr.footnotes:
                    curr.footnotes.append(fn)
            curr.page_end = next_tab.page_end
            if not curr.unit_context and next_tab.unit_context:
                curr.unit_context = next_tab.unit_context
        else:
            stitched.append(curr)
            curr = next_tab

    stitched.append(curr)
    return stitched


def chunk_document_blocks(
    blocks: List[Any],  # List[TextBlock | TableBlock]
    chunk_size: int = 500,
    overlap_pct: float = 0.10,
) -> List[Dict[str, Any]]:
    """
    Transforms an ordered list of TextBlocks and TableBlocks into a
    structured list of chunks.

    Tables are preserved with table-aware chunking (headers/units replicated).
    Narrative text is chunked using word overlap.
    """
    final_chunks: List[Dict[str, Any]] = []
    global_id = 0

    for block in blocks:
        if isinstance(block, TableBlock):
            # Table-aware chunking
            tab_chunks = table_to_markdown_chunks(block, max_chunk_words=chunk_size)
            for tc in tab_chunks:
                tc["chunk_id"] = global_id
                final_chunks.append(tc)
                global_id += 1

        elif isinstance(block, TextBlock):
            words = block.text.split()
            if not words:
                continue

            overlap = int(chunk_size * overlap_pct)
            step = max(1, chunk_size - overlap)

            for i in range(0, len(words), step):
                chunk_words = words[i : i + chunk_size]
                final_chunks.append(
                    {
                        "chunk_id": global_id,
                        "word_count": len(chunk_words),
                        "page_number": block.page_number,
                        "text": " ".join(chunk_words),
                        "is_table": False,
                    }
                )
                global_id += 1

                if i + chunk_size >= len(words):
                    break

    return final_chunks
