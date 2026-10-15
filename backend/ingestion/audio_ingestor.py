"""Audio Ingestor — Audio transcription, 30-60s chunking, and deep links."""
from __future__ import annotations

from pathlib import Path

from tracelearn.whisper_transcribe import chunk_transcript, transcribe


def extract_audio(
    path: Path,
    source_id: str,
    *,
    course_id: str = "dsa_6006",
) -> tuple[list[dict], list[dict]]:
    """Transcribe audio into 30-60s chunked units with precise timestamps."""
    units: list[dict] = []
    figures: list[dict] = []

    segments = transcribe(path)
    chunks = chunk_transcript(segments, chunk_seconds=45.0)

    for chunk in chunks:
        t_start = chunk["timestamp_start"]
        t_end = chunk["timestamp_end"]
        content = chunk["content"].strip()
        if content:
            deep_link = f"/viewer?source={source_id}&t={round(float(t_start), 2)}"
            units.append({
                "content": content,
                "modality": "transcript",
                "unit_type": "transcript",
                "timestamp_start": t_start,
                "timestamp_end": t_end,
                "start_ts": t_start,
                "end_ts": t_end,
                "token_count": len(content.split()),
                "deep_link": deep_link,
            })

    return units, figures
