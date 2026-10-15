"""Backend Ingestion Package for Track D Multimodal Ingestion."""
from __future__ import annotations

from .pdf_ingestor import extract_pdf
from .pptx_ingestor import extract_pptx
from .video_ingestor import extract_video
from .audio_ingestor import extract_audio
from .orchestrator import IngestionOrchestrator, orchestrator

__all__ = [
    "extract_pdf",
    "extract_pptx",
    "extract_video",
    "extract_audio",
    "IngestionOrchestrator",
    "orchestrator",
]
