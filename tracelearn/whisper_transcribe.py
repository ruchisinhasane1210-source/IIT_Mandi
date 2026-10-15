"""TraceLearn — audio/video transcription.

Priority order:
  1. openai-whisper  (local, free) if installed and WHISPER_MODEL is set.
  2. Gemini audio API if GEMINI_API_KEY is set (files < 20 MB).
  3. Stub — returns a helpful placeholder so the rest of ingestion continues.

Every returned segment has: {text, start, end}  (start/end in seconds).
"""
from __future__ import annotations

import base64
import mimetypes
import os
import re
from pathlib import Path

import requests

from .config import settings

# Extensions we treat as audio/video
VIDEO_AUDIO_EXT = {".mp4", ".mov", ".webm", ".mkv", ".mp3", ".wav", ".m4a", ".ogg"}
_MAX_INLINE_BYTES = 20 * 1024 * 1024  # 20 MB


# ------------------------------------------------------------------ public

def transcribe(audio_path: Path) -> list[dict]:
    """Transcribe an audio or video file.

    Returns a list of segment dicts: [{text, start, end}, ...]
    Guaranteed to return at least one segment (placeholder) so callers
    always have something to chunk.
    """
    if not audio_path.exists():
        return _stub(audio_path.name, reason="file not found")

    if settings.force_demo or (
        not settings.gemini_key and not _whisper_available()
    ):
        return _stub(audio_path.name, reason="no transcription backend configured")

    if _whisper_available():
        try:
            return _whisper_transcribe(audio_path)
        except Exception:
            pass  # fall through

    if settings.gemini_key:
        try:
            return _gemini_transcribe(audio_path)
        except Exception:
            pass

    return _stub(audio_path.name, reason="transcription failed")


def download_youtube(url: str, out_dir: Path) -> Path | None:
    """Download audio from a YouTube URL using yt-dlp.

    Returns the path to the downloaded audio file, or None on failure.
    yt-dlp is already installed in this environment.
    """
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    out_template = str(out_dir / "%(id)s.%(ext)s")

    opts = {
        "format": "bestaudio/best",
        "outtmpl": out_template,
        "quiet": True,
        "no_warnings": True,
        # Try to extract audio to mp3 if ffmpeg is available; ignore errors
        "postprocessors": [],
    }
    # Add ffmpeg post-processor only if ffmpeg is available
    if _ffmpeg_available():
        opts["postprocessors"] = [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "64",
            }
        ]

    try:
        with __import__("yt_dlp").YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_id = info.get("id", "unknown")
            for ext in ("mp3", "m4a", "webm", "opus", "ogg", "wav"):
                p = out_dir / f"{video_id}.{ext}"
                if p.exists():
                    return p
    except Exception:
        pass
    return None


# ---------------------------------------------------------------- providers

def _whisper_available() -> bool:
    if not os.getenv("WHISPER_MODEL", "").strip():
        return False
    try:
        import whisper  # noqa: F401
        return True
    except ImportError:
        return False


def _whisper_transcribe(audio_path: Path) -> list[dict]:
    import whisper

    model_name = os.getenv("WHISPER_MODEL", "base")
    model = whisper.load_model(model_name)
    result = model.transcribe(str(audio_path), word_timestamps=False)
    segments = [
        {"text": seg["text"].strip(), "start": float(seg["start"]), "end": float(seg["end"])}
        for seg in result.get("segments", [])
        if seg.get("text", "").strip()
    ]
    return segments or _stub(audio_path.name, reason="whisper returned no segments")


def _gemini_transcribe(audio_path: Path) -> list[dict]:
    file_bytes = audio_path.read_bytes()
    if len(file_bytes) > _MAX_INLINE_BYTES:
        return _stub(
            audio_path.name,
            reason=f"file too large for Gemini inline audio ({len(file_bytes) // 1024 // 1024} MB > 20 MB)",
        )

    mime, _ = mimetypes.guess_type(str(audio_path))
    mime = mime or "audio/mpeg"
    b64 = base64.b64encode(file_bytes).decode("utf-8")

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.0-flash:generateContent?key={settings.gemini_key}"
    )
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"inline_data": {"mime_type": mime, "data": b64}},
                    {
                        "text": (
                            "Transcribe this audio precisely. "
                            "Format each segment as: [START - END] Text\n"
                            "where START and END are decimal seconds. "
                            "Example: [0.0 - 12.5] The heap property requires each parent...\n"
                            "Include all spoken content including equations and code."
                        )
                    },
                ],
            }
        ],
        "generationConfig": {"temperature": 0.0},
    }
    resp = requests.post(url, json=payload, timeout=180)
    resp.raise_for_status()
    data = resp.json()
    raw = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    segments = _parse_transcript(raw)
    return segments or _stub(audio_path.name, reason="Gemini returned no parseable segments")


# --------------------------------------------------------------- parsing

_SEG_PATTERN = re.compile(
    r"\[(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\]\s*(.+)"
)


def _parse_transcript(raw: str) -> list[dict]:
    segments: list[dict] = []
    for line in raw.splitlines():
        m = _SEG_PATTERN.match(line.strip())
        if m:
            start, end, text = float(m.group(1)), float(m.group(2)), m.group(3).strip()
            if text:
                segments.append({"text": text, "start": start, "end": end})
    if not segments:
        # Fallback: treat whole text as one block
        clean = re.sub(r"\[.*?\]", "", raw).strip()
        if clean:
            segments = [{"text": clean, "start": 0.0, "end": 0.0}]
    return segments


def _stub(filename: str, reason: str = "") -> list[dict]:
    msg = (
        f"[Transcription unavailable for '{filename}'"
        + (f": {reason}" if reason else "")
        + ". Add GEMINI_API_KEY or set WHISPER_MODEL=base to enable.]"
    )
    return [{"text": msg, "start": 0.0, "end": 0.0}]


def _ffmpeg_available() -> bool:
    import shutil
    return shutil.which("ffmpeg") is not None


# --------------------------------------------------------------- chunker

def chunk_transcript(segments: list[dict], chunk_seconds: float = 60.0) -> list[dict]:
    """Group transcript segments into fixed-duration chunks.

    Returns list of {content, timestamp_start, timestamp_end}.
    """
    if not segments:
        return []
    chunks: list[dict] = []
    current_texts: list[str] = []
    chunk_start = segments[0]["start"]
    chunk_end = segments[0]["end"]

    for seg in segments:
        current_texts.append(seg["text"])
        chunk_end = seg["end"]
        # Start a new chunk every chunk_seconds
        if chunk_end - chunk_start >= chunk_seconds:
            text = " ".join(current_texts).strip()
            if text:
                chunks.append(
                    {
                        "content": text,
                        "timestamp_start": round(chunk_start, 2),
                        "timestamp_end": round(chunk_end, 2),
                    }
                )
            current_texts = []
            chunk_start = chunk_end

    # Flush last chunk
    if current_texts:
        text = " ".join(current_texts).strip()
        if text:
            chunks.append(
                {
                    "content": text,
                    "timestamp_start": round(chunk_start, 2),
                    "timestamp_end": round(chunk_end, 2),
                }
            )
    return chunks
