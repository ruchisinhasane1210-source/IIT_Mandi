"""PDF Ingestor — Page-level extraction with reading order, tables, and figures.

Uses PyMuPDF (fitz) for reliable text & layout extraction, table extraction
into markdown, and embedded image extraction with OCR fallback for scanned pages.
"""
from __future__ import annotations

import io
import re
import uuid
from pathlib import Path
from typing import Any

from tracelearn.vision import caption_and_ocr, ocr_image

CHUNK_MIN_CHARS = 400
CHUNK_MAX_CHARS = 1200
CHUNK_OVERLAP = 150


def _clean_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text or "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _chunk_page_text(text: str, page_num: int, source_id: str) -> list[dict]:
    """Split page text into 200-400 token chunks with ~15% overlap."""
    clean = _clean_whitespace(text)
    if not clean:
        return []

    chunks: list[dict] = []
    start = 0
    length = len(clean)

    while start < length:
        end = min(length, start + CHUNK_MAX_CHARS)
        if end < length:
            # find good boundary (paragraph or sentence)
            window = clean[start:end]
            for boundary in ("\n\n", ".\n", ". ", "? ", "! ", "\n"):
                cut = window.rfind(boundary)
                if cut > CHUNK_MIN_CHARS:
                    end = start + cut + len(boundary)
                    break

        chunk_text = clean[start:end].strip()
        if chunk_text:
            token_count = len(chunk_text.split())
            chunks.append({
                "content": chunk_text,
                "modality": "text",
                "page_number": page_num,
                "page": page_num,
                "token_count": token_count,
                "deep_link": f"/viewer?source={source_id}&page={page_num}",
            })

        if end >= length:
            break
        next_start = end - CHUNK_OVERLAP
        start = next_start if next_start > start else end

    return chunks


def _table_to_markdown(table_data: list[list[Any]]) -> str:
    """Format a 2D table grid as standard markdown table."""
    if not table_data or not table_data[0]:
        return ""
    clean_rows: list[list[str]] = []
    for row in table_data:
        clean_rows.append([str(c or "").replace("\n", " ").strip() for c in row])

    header = clean_rows[0]
    separator = ["---"] * len(header)
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in clean_rows[1:]:
        # pad row to header length if necessary
        padded = row + [""] * (len(header) - len(row))
        lines.append("| " + " | ".join(padded[:len(header)]) + " |")
    return "\n".join(lines)


def extract_pdf(
    path: Path,
    source_id: str,
    *,
    course_id: str = "dsa_6006",
    storage_dir: Path | None = None,
) -> tuple[list[dict], list[dict]]:
    """Extract units and figures from a PDF.

    Returns:
        (units, figures)
        where units are ready for content_units insertion
        and figures are ready for figures table insertion.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        try:
            import pymupdf as fitz
        except ImportError:
            return [], []

    units: list[dict] = []
    figures: list[dict] = []

    if storage_dir is None:
        storage_dir = path.parent / "figures"
    storage_dir.mkdir(parents=True, exist_ok=True)

    with fitz.open(path) as doc:
        for page_idx, page in enumerate(doc, start=1):
            page_text = page.get_text("text") or ""

            # 1. Table extraction (PyMuPDF built-in tables)
            try:
                tables = page.find_tables()
                for t in tables:
                    table_md = _table_to_markdown(t.extract())
                    if table_md.strip():
                        units.append({
                            "content": f"Table on page {page_idx}:\n{table_md}",
                            "modality": "table",
                            "page_number": page_idx,
                            "page": page_idx,
                            "token_count": len(table_md.split()),
                            "deep_link": f"/viewer?source={source_id}&page={page_idx}",
                        })
            except Exception:
                pass

            # 2. Scanned page OCR fallback if text is sparse
            if len(_clean_whitespace(page_text)) < 40:
                try:
                    pix = page.get_pixmap(dpi=150)
                    img_bytes = pix.tobytes("png")
                    ocr_res = ocr_image(img_bytes)
                    if ocr_res.strip():
                        page_text = ocr_res
                except Exception:
                    pass

            # 3. Text chunking
            page_chunks = _chunk_page_text(page_text, page_idx, source_id)
            units.extend(page_chunks)

            # 4. Embedded image / figure extraction
            try:
                image_list = page.get_images(full=True)
                for img_info in image_list:
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    img_bytes = base_image.get("image")
                    img_ext = base_image.get("ext", "png")
                    width = base_image.get("width", 0)
                    height = base_image.get("height", 0)

                    # Filter decorative/tiny images or icons
                    if width < 80 or height < 80 or (width / (height or 1) > 10):
                        continue

                    fig_id = f"fig_{source_id}_p{page_idx}_{xref}"
                    img_filename = f"{fig_id}.{img_ext}"
                    img_path = storage_dir / img_filename
                    try:
                        img_path.write_bytes(img_bytes)
                    except Exception:
                        pass

                    # Vision caption and OCR
                    surrounding_context = page_text[:200]
                    caption, ocr_txt = caption_and_ocr(
                        img_bytes,
                        context=surrounding_context,
                        source_type="pdf",
                        page_or_slide=page_idx,
                    )

                    fig_record = {
                        "figure_id": fig_id,
                        "source_id": source_id,
                        "course_id": course_id,
                        "page_or_slide_or_ts": f"page {page_idx}",
                        "image_path": str(img_path),
                        "caption": caption,
                        "ocr_text": ocr_txt,
                        "unit_id": f"u_{fig_id}",
                    }
                    figures.append(fig_record)

                    # Also store as a retrievable figure unit
                    combined_content = f"Figure on page {page_idx}: {caption}"
                    if ocr_txt:
                        combined_content += f"\nVisible text: {ocr_txt}"

                    units.append({
                        "id": f"u_{fig_id}",
                        "content": combined_content,
                        "modality": "figure",
                        "unit_type": "figure",
                        "page_number": page_idx,
                        "page": page_idx,
                        "figure_id": fig_id,
                        "figure_reference": fig_id,
                        "token_count": len(combined_content.split()),
                        "deep_link": f"/viewer?source={source_id}&page={page_idx}",
                    })
            except Exception:
                pass

    return units, figures
