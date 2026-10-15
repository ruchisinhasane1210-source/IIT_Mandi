"""Video Ingestor — Audio transcription, 30-60s chunking, keyframe diagrams, and deep links.

Supports local video files and YouTube URLs.
Produces transcript units and captioned keyframe figure units.
"""
from __future__ import annotations

import os
from pathlib import Path

from tracelearn.vision import caption_and_ocr
from tracelearn.whisper_transcribe import chunk_transcript, download_youtube, transcribe


def extract_video(
    path_or_url: str | Path,
    source_id: str,
    *,
    course_id: str = "dsa_6006",
    storage_dir: Path | None = None,
) -> tuple[list[dict], list[dict]]:
    """Transcribe video and extract keyframes.

    Returns:
        (units, figures)
    """
    path_str = str(path_or_url)
    units: list[dict] = []
    figures: list[dict] = []

    if storage_dir is None:
        storage_dir = Path("storage") / course_id / "media"
    storage_dir.mkdir(parents=True, exist_ok=True)

    # 1. Handle YouTube or remote URLs
    local_path: Path | None = None
    if path_str.startswith("http://") or path_str.startswith("https://"):
        local_path = download_youtube(path_str, storage_dir)
        if not local_path:
            # Fallback stub for URL without network or failed download
            units.append({
                "content": f"Video source from URL: {path_str}. Audio transcription queued or unavailable.",
                "modality": "transcript",
                "unit_type": "transcript",
                "timestamp_start": 0.0,
                "timestamp_end": 60.0,
                "start_ts": 0.0,
                "end_ts": 60.0,
                "token_count": 15,
                "deep_link": f"/viewer?source={source_id}&t=0.0",
            })
            return units, figures
    else:
        local_path = Path(path_str)

    # 2. Transcribe audio
    segments = transcribe(local_path)
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

    # 3. Keyframe diagram extraction (if video has visual frames)
    # Check if OpenCV is available for frame extraction
    try:
        import cv2

        cap = cv2.VideoCapture(str(local_path))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0

            # Sample keyframe every 60 seconds (max 5 keyframes to avoid excessive overhead)
            sample_interval = max(30.0, duration / 6.0) if duration > 30 else 15.0
            cur_time = sample_interval / 2.0

            frame_idx = 1
            while cur_time < duration and frame_idx <= 5:
                cap.set(cv2.CAP_PROP_POS_MSEC, cur_time * 1000.0)
                ret, frame = cap.read()
                if ret and frame is not None:
                    # Save frame
                    fig_id = f"fig_{source_id}_t{int(cur_time)}"
                    frame_path = storage_dir / f"{fig_id}.jpg"
                    cv2.imwrite(str(frame_path), frame)

                    # Encode to bytes for captioning
                    _, buf = cv2.imencode(".jpg", frame)
                    img_bytes = buf.tobytes()

                    caption, ocr_txt = caption_and_ocr(
                        img_bytes,
                        context=f"Lecture video frame at {round(cur_time, 1)}s",
                        source_type="video",
                        page_or_slide=None,
                    )

                    fig_record = {
                        "figure_id": fig_id,
                        "source_id": source_id,
                        "course_id": course_id,
                        "page_or_slide_or_ts": f"timestamp {round(cur_time, 1)}s",
                        "image_path": str(frame_path),
                        "caption": caption,
                        "ocr_text": ocr_txt,
                        "unit_id": f"u_{fig_id}",
                    }
                    figures.append(fig_record)

                    combined_caption = f"Keyframe at {round(cur_time, 1)}s: {caption}"
                    if ocr_txt:
                        combined_caption += f"\nVisible text: {ocr_txt}"

                    units.append({
                        "id": f"u_{fig_id}",
                        "content": combined_caption,
                        "modality": "figure",
                        "unit_type": "figure",
                        "timestamp_start": cur_time,
                        "timestamp_end": cur_time + 5.0,
                        "start_ts": cur_time,
                        "end_ts": cur_time + 5.0,
                        "figure_id": fig_id,
                        "figure_reference": fig_id,
                        "token_count": len(combined_caption.split()),
                        "deep_link": f"/viewer?source={source_id}&t={round(cur_time, 2)}",
                    })
                    frame_idx += 1
                cur_time += sample_interval

            cap.release()
    except Exception:
        pass

    return units, figures
