"""TraceLearn — ingestion pipeline.

Turns an uploaded file into citable ``content_units``. Extraction is
best-effort and degrades gracefully: if an optional parser (PyMuPDF,
python-pptx) is not installed, the source is flagged rather than failing
silently.

Every unit keeps its origin — ``page_number`` for PDFs, ``slide_number`` for
decks, ``timestamp_start/end`` for transcripts — which is what makes the
citation viewer (page / slide / timestamp) possible.
"""
from __future__ import annotations

import re
from pathlib import Path

from .db import execute, insert, jdumps, query_one
from .embeddings import embed_batch, save_unit_embedding

CHUNK_CHARS = 900
CHUNK_OVERLAP = 120

TEXT_EXT = {".txt", ".md", ".markdown", ".csv"}
PDF_EXT = {".pdf"}
PPTX_EXT = {".pptx"}
VIDEO_EXT = {".mp4", ".mov", ".webm", ".mkv"}
AUDIO_EXT = {".mp3", ".wav", ".m4a"}

SOURCE_TYPE_BY_EXT = {
    **{e: "pdf" for e in TEXT_EXT},
    **{e: "pdf" for e in PDF_EXT},
    **{e: "slides" for e in PPTX_EXT},
    **{e: "video" for e in VIDEO_EXT},
    **{e: "video" for e in AUDIO_EXT},
    ".png": "image", ".jpg": "image", ".jpeg": "image",
}

SUPPORTED_EXT = set(SOURCE_TYPE_BY_EXT)


def detect_type(filename: str) -> str | None:
    return SOURCE_TYPE_BY_EXT.get(Path(filename).suffix.lower())


# ------------------------------------------------------------- extraction

def _chunk_text(text: str) -> list[str]:
    text = re.sub(r"\n{3,}", "\n\n", (text or "").strip())
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + CHUNK_CHARS)
        # prefer to break on a sentence or paragraph boundary
        if end < len(text):
            window = text[start:end]
            for boundary in ("\n\n", ". ", "\n"):
                cut = window.rfind(boundary)
                if cut > CHUNK_CHARS * 0.5:
                    end = start + cut + len(boundary)
                    break
        chunks.append(text[start:end].strip())
        next_start = end - CHUNK_OVERLAP
        start = next_start if next_start > start else end
    return [c for c in chunks if c]


def extract_pdf(path: Path) -> list[dict]:
    try:
        import fitz  # PyMuPDF
    except Exception:
        return []
    pages: list[dict] = []
    try:
        with fitz.open(path) as doc:
            for page_index, page in enumerate(doc, start=1):
                text = page.get_text("text") or ""
                for chunk in _chunk_text(text):
                    pages.append({"content": chunk, "page_number": page_index})
    except Exception:
        return []
    return pages


def extract_pptx(path: Path) -> list[dict]:
    try:
        from pptx import Presentation
    except Exception:
        return []
    units: list[dict] = []
    try:
        deck = Presentation(str(path))
        for index, slide in enumerate(deck.slides, start=1):
            parts = []
            for shape in slide.shapes:
                if shape.has_text_frame and shape.text_frame.text:
                    parts.append(shape.text_frame.text)
            if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
                notes = slide.notes_slide.notes_text_frame.text
                if notes:
                    parts.append(f"(speaker notes) {notes}")
            content = "\n".join(parts).strip()
            if content:
                units.append({"content": content, "slide_number": index})
    except Exception:
        return []
    return units


def extract_text_file(path: Path) -> list[dict]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    return [{"content": c} for c in _chunk_text(text)]


def extract(path: Path, source_type: str) -> list[dict]:
    ext = path.suffix.lower()
    if ext in TEXT_EXT:
        return extract_text_file(path)
    if ext in PDF_EXT or (source_type == "pdf" and ext == ".pdf"):
        return extract_pdf(path)
    if ext in PPTX_EXT:
        return extract_pptx(path)
    return []


# ------------------------------------------------------------- pipeline

def process_source(source_id: str) -> dict:
    """Extract -> chunk -> embed -> persist. Safe to call in a background thread."""
    source = query_one("SELECT * FROM sources WHERE id = ?", (source_id,))
    if not source:
        return {"ok": False, "error": "source not found"}

    execute("UPDATE sources SET processing_status = 'processing', progress = 10 WHERE id = ?",
            (source_id,))

    path = Path(source.get("storage_path") or "")
    if not path.exists():
        execute("UPDATE sources SET processing_status = 'failed', error = ? WHERE id = ?",
                ("stored file missing", source_id))
        return {"ok": False, "error": "missing file"}

    pieces = extract(path, source.get("source_type") or "pdf")
    if not pieces:
        execute(
            "UPDATE sources SET processing_status = 'pending', progress = 0, error = ? WHERE id = ?",
            ("no text extracted — install PyMuPDF/python-pptx or add a transcript", source_id),
        )
        return {"ok": False, "error": "no extractable text"}

    execute("UPDATE sources SET progress = 45 WHERE id = ?", (source_id,))

    texts = [p["content"] for p in pieces]
    vectors = embed_batch(texts)

    for piece, vector in zip(pieces, vectors):
        unit_id = insert(
            "content_units",
            {
                "source_id": source_id,
                "course_id": source["course_id"],
                "content": piece["content"],
                "unit_type": "text" if "page_number" in piece else (
                    "slide" if "slide_number" in piece else "text"
                ),
                "page_number": piece.get("page_number"),
                "slide_number": piece.get("slide_number"),
                "timestamp_start": piece.get("timestamp_start"),
                "timestamp_end": piece.get("timestamp_end"),
                "topic_ids": jdumps([]),
                "concept_ids": jdumps([]),
                "metadata": jdumps({"extractor": path.suffix.lower().lstrip(".")}),
            },
        )
        save_unit_embedding(unit_id, vector)

    execute("UPDATE sources SET processing_status = 'done', progress = 100, error = NULL WHERE id = ?",
            (source_id,))
    return {"ok": True, "units": len(pieces)}
