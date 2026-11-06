"""
ui/pdf_viewer.py — Citation pills and an in-app filing viewer.

Clicking a citation renders the cited page with PyMuPDF, with the matched
passage highlighted. SEC filings are ingested as HTML, which has no page
geometry; those fall back to showing the retrieved passage plus a download.
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import Iterable, List, Optional

import streamlit as st

try:
    import pymupdf as fitz
except ImportError:  # PyMuPDF < 1.24 exposes only the fitz alias
    import fitz


RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw_pdfs"
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.,%$€£'\-]*")
_PILLS_PER_ROW = 4


# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENT RESOLUTION
# ─────────────────────────────────────────────────────────────────────────────

def resolve_source_file(paper_id: str) -> Optional[Path]:
    """Map a chunk's paper_id back to its file in data/raw_pdfs, if present."""
    paper_id = str(paper_id or "").strip()
    if not paper_id:
        return None
    # paper_id is always a bare name; reject anything that looks like a path.
    if Path(paper_id).name != paper_id or paper_id in (".", ".."):
        return None
    if not RAW_DIR.is_dir():
        return None

    files = [p for p in RAW_DIR.iterdir() if p.is_file()]
    target = paper_id.lower()

    exact = [p for p in files if p.stem.lower() == target or p.name.lower() == target]
    if exact:
        return _prefer_pdf(exact)

    # Ingested ids are not always the file stem (e.g. "aapl-20240928" was
    # indexed from aapl_20240928_10k.htm), so compare punctuation-free names.
    wanted = _normalise(paper_id)
    if len(wanted) < 5:
        return None
    fuzzy = []
    for path in files:
        candidate = _normalise(path.stem)
        if len(candidate) < 5:
            continue
        if candidate.startswith(wanted) or wanted.startswith(candidate):
            fuzzy.append(path)
    return _prefer_pdf(fuzzy) if fuzzy else None


def _normalise(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _prefer_pdf(paths: Iterable[Path]) -> Path:
    """Prefer the PDF variant, then the closest name length."""
    return sorted(
        paths,
        key=lambda p: (p.suffix.lower() != ".pdf", len(p.name)),
    )[0]


# ─────────────────────────────────────────────────────────────────────────────
# CITATION PILLS
# ─────────────────────────────────────────────────────────────────────────────

def citation_label(meta: dict) -> str:
    """Short badge label, e.g. 'AAPL 10-K · p. 42'."""
    ticker = str(meta.get("ticker") or "").upper()
    form = str(meta.get("form_type") or "").upper()
    left = " ".join(bit for bit in (ticker, form) if bit)
    if not left:
        left = Path(str(meta.get("paper_id") or "Document")).stem

    if meta.get("page_number"):
        right = f"p. {meta['page_number']}"
    elif meta.get("section"):
        right = str(meta["section"])[:28]
    elif meta.get("is_table"):
        right = "Table"
    else:
        right = "source"
    return f"{left} · {right}"


def render_citation_pills(sources: List[dict], key_prefix: str) -> None:
    """Render one clickable badge per retrieved chunk."""
    if not sources:
        return

    entries = []
    seen: dict[str, int] = {}
    for src in sources:
        meta = src.get("metadata") or {}
        label = citation_label(meta)
        seen[label] = seen.get(label, 0) + 1
        if seen[label] > 1:
            label = f"{label} ({seen[label]})"
        entries.append((label, src))

    for start in range(0, len(entries), _PILLS_PER_ROW):
        row = entries[start:start + _PILLS_PER_ROW]
        cols = st.columns(len(row), gap="small")
        for offset, (label, src) in enumerate(row):
            with cols[offset]:
                clicked = st.button(
                    label,
                    key=f"{key_prefix}_{start + offset}",
                    use_container_width=True,
                    help=(src.get("text") or "")[:300],
                )
                if clicked:
                    _open_source(src)


def _open_source(src: dict) -> None:
    meta = dict(src.get("metadata") or {})
    meta["snippet"] = src.get("text") or ""
    st.session_state["selected_source"] = meta
    st.session_state["viewer_page"] = int(meta.get("page_number") or 1)
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE RENDERING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, max_entries=24)
def render_page_png(
    pdf_path: str,
    mtime: float,          # part of the cache key; a re-indexed file re-renders
    page_number: int,
    snippet: str = "",
    zoom: float = 2.0,
) -> bytes:
    """Render one page to PNG, highlighting the part that matches the chunk."""
    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(max(0, min(page_number - 1, doc.page_count - 1)))
        for rect in _highlight_rects(page, snippet):
            try:
                annot = page.add_highlight_annot(rect)
                annot.set_colors(stroke=(1.0, 0.82, 0.2))
                annot.update()
            except Exception:
                continue
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        return pix.tobytes("png")
    finally:
        doc.close()


def _tokenise(text: str) -> List[str]:
    return [
        token.lower().strip(".,;:()[]'\"")
        for token in _TOKEN_RE.findall(text or "")
    ]


def _highlight_rects(page, snippet: str) -> List["fitz.Rect"]:
    """
    Locate the chunk's text on the page by matching word tokens, which is
    robust to the line breaks and column order that break naive search_for().
    Returns the largest matching runs, merged into line-sized rectangles.
    """
    snippet_tokens = _tokenise(snippet)
    if not snippet_tokens:
        return []

    page_tokens: List[tuple[str, "fitz.Rect"]] = []
    for word in page.get_text("words"):
        if len(word) < 5:
            continue
        rect = fitz.Rect(word[:4])
        for token in _tokenise(str(word[4])):
            if token:
                page_tokens.append((token, rect))
    if not page_tokens:
        return []

    matcher = difflib.SequenceMatcher(
        None, snippet_tokens, [token for token, _ in page_tokens], autojunk=False
    )
    blocks = [b for b in matcher.get_matching_blocks() if b.size >= 3]
    blocks.sort(key=lambda b: (-b.size, b.a))

    # A chunk can span a whole page (and every word matches). Highlight only
    # the strongest runs so the page stays readable.
    rects: List["fitz.Rect"] = []
    for block in blocks[:3]:
        merged: Optional["fitz.Rect"] = None
        for _, rect in page_tokens[block.b:block.b + block.size]:
            if (
                merged is not None
                and abs(rect.y0 - merged.y0) < 4
                and rect.x0 - merged.x1 < 60
            ):
                merged |= rect
            else:
                if merged is not None:
                    rects.append(merged)
                merged = fitz.Rect(rect)
        if merged is not None:
            rects.append(merged)

    return [r for r in rects if r.width > 2 and r.height > 2][:12]


# ─────────────────────────────────────────────────────────────────────────────
# SIDE PANEL
# ─────────────────────────────────────────────────────────────────────────────

def viewer_is_open() -> bool:
    return st.session_state.get("selected_source") is not None


_VIEWER_CSS = """
<style>
/* Pin the whole column, not just the panel: the column is a flex item in a
   row as tall as the chat, so a sticky column stays in view for its full height. */
div[data-testid="stColumn"]:has(.st-key-source_viewer),
div.stColumn:has(.st-key-source_viewer) {
    position: sticky;
    top: 3.75rem;
    align-self: flex-start;
    z-index: 1;
}
.st-key-source_viewer {
    align-self: flex-start;
}
.st-key-source_viewer_scroll {
    max-height: calc(100vh - 13rem);
    min-height: 14rem;
    overflow-y: auto;
    overscroll-behavior: contain;
    padding-right: 0.35rem;
}
.st-key-source_viewer_scroll [data-testid="stImage"] {
    margin-bottom: 0.4rem;
}
</style>
"""

_CONTINUOUS_PAGES = 5


def render_source_viewer() -> None:
    """Sticky, independently scrollable view of the cited filing."""
    source = st.session_state.get("selected_source")
    if not source:
        return

    paper_id = str(source.get("paper_id") or "Document")
    snippet = source.get("snippet") or ""
    path = resolve_source_file(paper_id)

    st.markdown(_VIEWER_CSS, unsafe_allow_html=True)

    with st.container(key="source_viewer"):
        title_col, close_col = st.columns([4, 1.5])
        with title_col:
            st.markdown(f"**{paper_id}**")
            st.caption(citation_label(source))
        with close_col:
            if st.button("Close", key="btn_close_viewer", use_container_width=True):
                st.session_state.pop("selected_source", None)
                st.rerun()

        if path is None:
            st.warning(
                "The original file for this source is not in data/raw_pdfs, "
                "so it cannot be rendered. The retrieved passage is shown below."
            )
            st.code(snippet[:4000] or "No passage text available.")
            return

        if path.suffix.lower() != ".pdf":
            st.info(
                "This filing was indexed from HTML, which has no page image. "
                "The retrieved passage and the original file are shown below."
            )
            st.code(snippet[:4000] or "No passage text available.")
            with open(path, "rb") as handle:
                st.download_button(
                    "Download source file",
                    data=handle,
                    file_name=path.name,
                    mime="text/html",
                    use_container_width=True,
                )
            return

        try:
            with fitz.open(path) as doc:
                total_pages = doc.page_count
        except Exception as exc:
            st.error(f"Could not open the source PDF: {exc}")
            return

        cited_page = int(source.get("page_number") or 1)
        page = int(st.session_state.get("viewer_page") or cited_page)
        page = max(1, min(page, total_pages))
        st.session_state["viewer_page"] = page

        if "viewer_continuous" not in st.session_state:
            st.session_state["viewer_continuous"] = True
        continuous = st.toggle("Continuous scroll", key="viewer_continuous")

        window_end = (
            min(total_pages, page + _CONTINUOUS_PAGES - 1)
            if continuous else page
        )
        label = (
            f"Pages {page}–{window_end} of {total_pages}"
            if continuous and window_end > page
            else f"Page {page} of {total_pages}"
        )

        prev_col, info_col, next_col = st.columns([1, 2.2, 1])
        with prev_col:
            if st.button("Prev", key="btn_pdf_prev", disabled=page <= 1, use_container_width=True):
                st.session_state["viewer_page"] = page - 1
                st.rerun()
        with info_col:
            st.markdown(
                f"<div style='text-align:center;padding-top:0.4rem;'>{label}</div>",
                unsafe_allow_html=True,
            )
        with next_col:
            if st.button("Next", key="btn_pdf_next", disabled=page >= total_pages, use_container_width=True):
                st.session_state["viewer_page"] = page + 1
                st.rerun()

        if continuous and window_end > page:
            back_col, fwd_col = st.columns(2)
            with back_col:
                if st.button(
                    f"Previous {_CONTINUOUS_PAGES} pages",
                    key="btn_pdf_back",
                    disabled=page <= 1,
                    use_container_width=True,
                ):
                    st.session_state["viewer_page"] = max(1, page - _CONTINUOUS_PAGES)
                    st.rerun()
            with fwd_col:
                if st.button(
                    f"Next {_CONTINUOUS_PAGES} pages",
                    key="btn_pdf_forward",
                    disabled=window_end >= total_pages,
                    use_container_width=True,
                ):
                    st.session_state["viewer_page"] = min(total_pages, page + _CONTINUOUS_PAGES)
                    st.rerun()

        zoom = 1.5 if continuous else 2.0
        with st.container(key="source_viewer_scroll"):
            for page_no in range(page, window_end + 1):
                try:
                    png = render_page_png(
                        str(path),
                        path.stat().st_mtime,
                        page_no,
                        snippet if page_no == cited_page else "",
                        zoom,
                    )
                except Exception as exc:
                    st.error(f"Could not render page {page_no}: {exc}")
                    continue
                st.image(
                    png,
                    use_container_width=True,
                    caption=f"Page {page_no} of {total_pages}",
                )

        if (
            snippet
            and page <= cited_page <= window_end
        ):
            st.caption("The cited passage is highlighted on this page.")
        st.caption(f"Source: {path.name}")
