# ingestion/html_parser.py
"""
HTML ingestion: entity-recursive walk that keeps `<table>` elements intact.

SEC filings are inline XBRL HTML, so the important figures live inside table cells
and the surrounding `<div>` prose. The walk emits TextBlocks for prose and TableBlocks
for real financial tables (flattening colspan, capturing caption and unit context),
then hands the ordered list to `table_extractor.chunk_document_blocks`.

Layout tables (single column, or label/value cover-page pairs with no amounts) are
rejected and walked as text so nothing is lost.

No page numbers are invented: HTML has none, so chunks carry `page_number=None` and
carry the enclosing section heading instead.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

from ingestion.table_extractor import (
    TableBlock,
    TextBlock,
    chunk_document_blocks,
    clean_cell_text,
    extract_unit_context,
    is_footnote_text,
    looks_like_financial_table,
    table_block_from_grid,
)

_SKIP_TAGS = {
    "script", "style", "noscript", "svg", "iframe", "head",
    "ix:header", "ix:hidden",          # machine-readable contexts / hidden facts
}
_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_BLOCKISH_TAGS = {"div", "p", "li", "section", "article", "header", "footer", "td", "th"}
_BLOCK_TAGS = _HEADING_TAGS | _BLOCKISH_TAGS | {
    "tr", "table", "tbody", "thead", "ul", "ol", "dl", "dt", "dd",
    "blockquote", "pre", "form", "fieldset", "nav", "aside", "main", "body", "html",
}

# "Item 7. MD&A" / "PART II" / "Note 12 — Debt" at the start of a short block.
_HTML_HEADING_RE = re.compile(
    r"^\s*(?:part\s+[ivx]+\b|item\s+\d+[a-z]?\b|note\s+\d+[a-z]?\b)",
    re.IGNORECASE,
)

_CAPTION_MAX_WORDS = 30


def _normalise_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\xa0", " ")).strip()


# ─────────────────────────────────────────────────────────────
# TABLE CONSTRUCTION
# ─────────────────────────────────────────────────────────────

def _own_rows(table_tag: Tag) -> List[Tag]:
    return [tr for tr in table_tag.find_all("tr") if tr.find_parent("table") is table_tag]


def _own_cells(row_tag: Tag) -> List[Tag]:
    cells = row_tag.find_all(["td", "th"])
    return [cell for cell in cells if cell.find_parent("tr") is row_tag]


def _row_to_cells(row_tag: Tag) -> List[str]:
    cells: List[str] = []
    for cell in _own_cells(row_tag):
        text = _normalise_ws(cell.get_text(" ", strip=True))
        try:
            span = int(cell.get("colspan", 1))
        except (TypeError, ValueError):
            span = 1
        cells.extend([text] * max(1, min(span, 20)))
    # SEC filings pad statement rows with colspan placeholders ("<td colspan=3>").
    # Collapsing the repeats recovers the logical row instead of a 40-column ghost.
    return _collapse_adjacent(cells)


def _collapse_adjacent(cells: List[str]) -> List[str]:
    collapsed: List[str] = []
    for cell in cells:
        if collapsed and collapsed[-1] == cell:
            continue
        collapsed.append(cell)
    while collapsed and not collapsed[-1]:
        collapsed.pop()
    return collapsed


def _preceding_texts(table_tag: Tag, max_items: int = 3, max_ancestors: int = 3) -> List[str]:
    """
    Short text blocks immediately before a table, nearest first.

    SEC filings nest the table inside a centering `<div>`, so the statement
    title and the "(In millions...)" line are siblings of an ancestor.
    """
    texts: List[str] = []
    node: Optional[Tag] = table_tag

    for _ in range(max_ancestors):
        parent = node.parent if node is not None else None
        if parent is None:
            break
        sibling = node.previous_sibling
        while sibling is not None and len(texts) < max_items:
            if isinstance(sibling, NavigableString):
                text = _normalise_ws(str(sibling))
                if text:
                    texts.append(text)
            elif isinstance(sibling, Tag) and sibling.name not in _SKIP_TAGS:
                if sibling.name == "table":
                    # Another table's content is not this table's caption.
                    sibling = sibling.previous_sibling
                    continue
                text = _normalise_ws(sibling.get_text(" ", strip=True))
                if text:
                    if len(text.split()) > _CAPTION_MAX_WORDS:
                        return texts
                    texts.append(text)
            sibling = sibling.previous_sibling
        if len(texts) >= max_items:
            break
        node = parent

    return texts


def _caption_and_units(table_tag: Tag) -> tuple[str, str]:
    caption_tag = table_tag.find("caption")
    if caption_tag is not None:
        caption = _normalise_ws(caption_tag.get_text(" ", strip=True))
        if caption:
            return caption, extract_unit_context(caption) or ""

    preceding = _preceding_texts(table_tag)
    unit_context = ""
    for text in preceding:
        unit_context = extract_unit_context(text) or ""
        if unit_context:
            break

    caption = ""
    for text in preceding:
        if extract_unit_context(text) is None:
            caption = text
            break
    if not caption and preceding:
        caption = preceding[0]

    return caption, unit_context


def _table_from_tag(table_tag: Tag, section: str) -> Optional[TableBlock]:
    rows = _own_rows(table_tag)
    if len(rows) < 2:
        return None

    grid = [cells for cells in (_row_to_cells(row) for row in rows) if any(cells)]
    if len(grid) < 2:
        return None

    width = max(len(row) for row in grid)
    if width < 2:
        return None

    # A statement has at least two rows of reported amounts. Cover pages,
    # address blocks and checkbox grids have words but no amount cells.
    if not looks_like_financial_table(grid):
        return None

    footnotes: List[str] = []
    while len(grid) > 2:
        non_empty = [cell for cell in grid[-1] if cell]
        if len(non_empty) == 1 and is_footnote_text(non_empty[0]):
            footnotes.insert(0, non_empty[0])
            grid.pop()
        else:
            break

    caption, unit_context = _caption_and_units(table_tag)
    caption = clean_cell_text(caption)

    table_block = table_block_from_grid(
        grid,
        caption=caption,
        unit_context=unit_context,
        section=section,
    )
    if table_block is not None:
        table_block.footnotes = footnotes
    return table_block


# ─────────────────────────────────────────────────────────────
# DOCUMENT WALK
# ─────────────────────────────────────────────────────────────

def _flush(ctx: Dict[str, Any], blocks: List[Any]) -> None:
    text = _normalise_ws(" ".join(ctx["buffer"]))
    ctx["buffer"] = []
    if text:
        blocks.append(TextBlock(text=text, page_number=None, section=ctx["section"]))


def _heading_text(node: Tag) -> str:
    text = _normalise_ws(node.get_text(" ", strip=True))
    if not text or len(text.split()) > 25:
        return ""
    return text


def _walk(node: Tag, ctx: Dict[str, Any], blocks: List[Any]) -> None:
    for child in node.children:
        if isinstance(child, Comment):
            continue
        if isinstance(child, NavigableString):
            ctx["buffer"].append(str(child))
            continue
        if not isinstance(child, Tag):
            continue

        name = (child.name or "").lower()
        if name in _SKIP_TAGS:
            continue

        if name == "br":
            ctx["buffer"].append(" ")
            continue

        if name == "table":
            _flush(ctx, blocks)
            table_block = _table_from_tag(child, ctx["section"])
            if table_block is not None:
                blocks.append(table_block)
            else:
                _walk(child, ctx, blocks)
            _flush(ctx, blocks)
            continue

        if name in _HEADING_TAGS:
            _flush(ctx, blocks)
            heading = _heading_text(child)
            if heading:
                ctx["section"] = heading[:120]
                ctx["buffer"].append(heading)
            continue

        if name in _BLOCKISH_TAGS and len(child.find_all(True, limit=20)) < 20:
            heading = _heading_text(child)
            if heading and _HTML_HEADING_RE.match(heading):
                _flush(ctx, blocks)
                ctx["section"] = heading[:120]
                ctx["buffer"].append(heading)
                continue

        if name in _BLOCK_TAGS:
            ctx["buffer"].append(" ")

        _walk(child, ctx, blocks)


def _prepare_soup(html_content: str) -> BeautifulSoup:
    soup = BeautifulSoup(html_content, "html.parser")

    # Detect SEC iFrame / ixViewer interactive wrapper stubs
    if "XBRL Viewer" in html_content or "loadViewer" in html_content:
        visible_text = soup.get_text(separator=" ").strip()
        if len(visible_text.split()) < 100:
            raise ValueError(
                "Uploaded file is an SEC Interactive Viewer wrapper (6 KB) rather than the actual 10-K report text. "
                "On SEC EDGAR, please open the primary document (.htm) directly (or right-click 'Open Document' -> Save Link As) and upload that full file."
            )

    for element in soup(["script", "style", "noscript", "svg", "iframe"]):
        element.decompose()

    return soup


def extract_html_blocks(html_path: str) -> List[Any]:
    """Returns an ordered list of TextBlock / TableBlock in reading order."""
    path = Path(html_path)
    if not path.exists():
        raise FileNotFoundError(f"HTML file not found: {html_path}")

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        html_content = f.read()

    soup = _prepare_soup(html_content)
    root = soup.body or soup

    blocks: List[Any] = []
    ctx: Dict[str, Any] = {"section": "", "buffer": []}
    _walk(root, ctx, blocks)
    _flush(ctx, blocks)

    if sum(len(b.text.split()) for b in blocks if isinstance(b, TextBlock)) < 20:
        raise ValueError(
            "Uploaded HTML file contains almost no text (less than 20 words). "
            "Please ensure you uploaded the full financial report rather than a stub/link page."
        )
    return blocks


def extract_html_text(html_path: str) -> str:
    """Flattened plain text (tables included as cell text), for previews and checks."""
    parts: List[str] = []
    for block in extract_html_blocks(html_path):
        if isinstance(block, TableBlock):
            parts.append(block.caption)
            parts.append(" ".join(block.headers))
            parts.extend(" ".join(row) for row in block.rows)
        else:
            parts.append(block.text)
    return "\n\n".join(part for part in parts if part)


def chunk_html(
    html_path: str, chunk_size: int = 300, overlap_pct: float = 0.10
) -> List[Dict[str, Any]]:
    """Chunks an HTML filing into text and table chunks with section metadata."""
    blocks = extract_html_blocks(html_path)
    return chunk_document_blocks(blocks, chunk_size=chunk_size, overlap_pct=overlap_pct)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        chunks = chunk_html(test_file)
        tables = sum(1 for c in chunks if c.get("is_table"))
        print(f"Generated {len(chunks)} chunks from {test_file} ({tables} table chunks)")
        if chunks:
            print("Preview chunk 0:\n", chunks[0]["text"][:300])
