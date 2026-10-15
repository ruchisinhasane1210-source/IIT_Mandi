"""TraceLearn — vision captioning for figures and diagrams.

Uses Gemini's multimodal API to caption extracted images from PDFs and slides.
Falls back to a placeholder caption in demo mode or when no key is present.
"""
from __future__ import annotations

import base64
from typing import Any

import requests

from .config import settings


# ------------------------------------------------------------------ public

def caption_image(
    image_bytes: bytes,
    *,
    context: str = "",
    source_type: str = "figure",
    page_or_slide: int | None = None,
) -> str:
    """Return a natural-language caption for the given image bytes.

    In demo mode or when GEMINI_API_KEY is absent, returns a placeholder.
    """
    if not image_bytes:
        return _placeholder_caption(context, source_type, page_or_slide)
    if not settings.gemini_key or settings.force_demo:
        return _placeholder_caption(context, source_type, page_or_slide)
    try:
        return _gemini_vision_caption(image_bytes, context=context)
    except Exception:
        return _placeholder_caption(context, source_type, page_or_slide)


def ocr_image(image_bytes: bytes) -> str:
    """Extract text visible in the image using Gemini vision.

    Returns empty string in demo mode or on failure.
    """
    if not image_bytes:
        return ""
    if not settings.gemini_key or settings.force_demo:
        return ""
    try:
        return _gemini_ocr(image_bytes)
    except Exception:
        return ""


def caption_and_ocr(
    image_bytes: bytes,
    *,
    context: str = "",
    source_type: str = "figure",
    page_or_slide: int | None = None,
) -> tuple[str, str]:
    """Return (caption, ocr_text) for an image in a single API call."""
    if not image_bytes:
        return _placeholder_caption(context, source_type, page_or_slide), ""
    if not settings.gemini_key or settings.force_demo:
        return _placeholder_caption(context, source_type, page_or_slide), ""
    try:
        caption = _gemini_vision_caption(image_bytes, context=context)
        ocr = _gemini_ocr(image_bytes)
        return caption, ocr
    except Exception:
        return _placeholder_caption(context, source_type, page_or_slide), ""


# ----------------------------------------------------------------- internals

def _gemini_vision_caption(image_bytes: bytes, *, context: str = "") -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    # Detect image mime type from magic bytes
    mime = _detect_mime(image_bytes)
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.0-flash:generateContent?key={settings.gemini_key}"
    )
    parts: list[dict[str, Any]] = [
        {"inline_data": {"mime_type": mime, "data": b64}},
        {
            "text": (
                "Describe this figure or diagram in 2-4 sentences suitable for a student studying "
                "the topic. Include: what it shows, any labels or axes, key relationships shown, "
                "and how the visual illustrates the underlying concept. Be specific and accurate."
                + (f" Context from surrounding text: {context[:200]}" if context else "")
            )
        },
    ]
    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 300},
    }
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


def _gemini_ocr(image_bytes: bytes) -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    mime = _detect_mime(image_bytes)
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.0-flash:generateContent?key={settings.gemini_key}"
    )
    parts: list[dict[str, Any]] = [
        {"inline_data": {"mime_type": mime, "data": b64}},
        {
            "text": (
                "Extract all visible text from this image exactly as it appears. "
                "Include labels, axis titles, legend text, annotations, and any other text. "
                "Return only the extracted text with no commentary."
            )
        },
    ]
    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 512},
    }
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


def _detect_mime(image_bytes: bytes) -> str:
    """Detect image MIME type from magic bytes."""
    if image_bytes[:4] == b"\x89PNG":
        return "image/png"
    if image_bytes[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if image_bytes[:4] == b"GIF8":
        return "image/gif"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"  # safe default


def _placeholder_caption(
    context: str, source_type: str, page_or_slide: int | None
) -> str:
    loc = f" (page/slide {page_or_slide})" if page_or_slide else ""
    base = f"Figure from {source_type}{loc}."
    if context:
        base += f" Related to: {context[:100].strip()}."
    base += " (Add GEMINI_API_KEY to enable automatic captioning.)"
    return base
