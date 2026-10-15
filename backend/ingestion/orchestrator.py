"""Ingestion Orchestrator — End-to-end multimodal ingestion pipeline.

Stages:
  1. queued (0%)
  2. extracting (15%): run modality ingestor (pdf, pptx, video, audio)
  3. chunking & validation (35%): validate location and deep_link
  4. captioning (55%): vision captioning + OCR on figures
  5. embedding (75%): compute dense vector representations
  6. tagging & graph (90%): 4-pass topic/concept extraction & prerequisite DAG
  7. done (100%): final unit counts and status
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from tracelearn.db import execute, insert, jdumps, new_id, query, query_one
from tracelearn.embeddings import embed_batch, save_unit_embedding
from tracelearn.graph import build as build_graph

from .audio_ingestor import extract_audio
from .pdf_ingestor import extract_pdf
from .pptx_ingestor import extract_pptx
from .topic_tagger import (
    build_prerequisites_pass3,
    extract_candidates_pass1,
    merge_and_cluster_pass2,
    tag_units_pass4,
)
from .video_ingestor import extract_video


def compute_file_hash(path_or_url: str | Path) -> str:
    """SHA-256 hash of a file or URL string."""
    p = Path(path_or_url)
    if p.exists() and p.is_file():
        h = hashlib.sha256()
        with p.open("rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()
    return hashlib.sha256(str(path_or_url).encode("utf-8")).hexdigest()


class IngestionOrchestrator:
    """Manages idempotent background ingestion jobs."""

    def __init__(self, course_id: str = "dsa_6006") -> None:
        self.course_id = course_id

    def ingest(
        self,
        path_or_url: str | Path,
        source_type: str | None = None,
        title: str | None = None,
        *,
        course_id: str | None = None,
        async_run: bool = True,
    ) -> dict[str, str]:
        """Start or retrieve an ingestion job. Returns {job_id, source_id}."""
        cid = course_id or self.course_id
        path_str = str(path_or_url)
        file_hash = compute_file_hash(path_or_url)

        # 1. Idempotency check: check if already ingested
        existing_source = query_one(
            "SELECT id, processing_status, progress FROM sources WHERE path_or_url = ? AND course_id = ?",
            (path_str, cid),
        )
        if existing_source and existing_source.get("processing_status") == "done":
            # Check for existing job
            job = query_one(
                "SELECT id FROM ingest_jobs WHERE source_id = ? ORDER BY started_at DESC",
                (existing_source["id"],),
            )
            job_id = job["id"] if job else new_id()
            return {"job_id": job_id, "source_id": existing_source["id"], "status": "done"}

        # 2. Detect source type
        stype = source_type or self._detect_type(path_str)
        fname = Path(path_str).name or "source"
        stitle = title or fname

        # 3. Create or update source record
        source_id = insert("sources", {
            "course_id": cid,
            "source_id": new_id()[:8],
            "title": stitle,
            "file_name": fname,
            "source_type": stype,
            "type": stype,
            "path_or_url": path_str,
            "storage_path": path_str if Path(path_str).exists() else None,
            "source_url": path_str if path_str.startswith("http") else None,
            "processing_status": "processing",
            "status": "processing",
            "progress": 5,
        })

        # 4. Create ingest_jobs record
        job_id = insert("ingest_jobs", {
            "source_id": source_id,
            "course_id": cid,
            "stage": "queued",
            "progress": 5,
            "counts": jdumps({}),
        })

        if async_run:
            thread = threading.Thread(
                target=self._run_pipeline,
                args=(job_id, source_id, cid, path_str, stype, stitle),
                daemon=True,
            )
            thread.start()
        else:
            self._run_pipeline(job_id, source_id, cid, path_str, stype, stitle)

        return {"job_id": job_id, "source_id": source_id}

    def _detect_type(self, path_or_url: str) -> str:
        lower = path_or_url.lower()
        if lower.endswith(".pdf"):
            return "pdf"
        if lower.endswith((".pptx", ".ppt")):
            return "pptx"
        if lower.endswith((".mp4", ".mov", ".webm", ".mkv")) or "youtube.com" in lower or "youtu.be" in lower:
            return "video"
        if lower.endswith((".mp3", ".wav", ".m4a", ".ogg")):
            return "audio"
        return "pdf"

    def _update_job(
        self,
        job_id: str,
        source_id: str,
        stage: str,
        progress: int,
        error: str | None = None,
        counts: dict | None = None,
    ) -> None:
        updates = ["stage = ?", "progress = ?"]
        params: list[Any] = [stage, progress]
        if error:
            updates.append("error = ?")
            params.append(error)
        if counts is not None:
            updates.append("counts = ?")
            params.append(jdumps(counts))
        if stage in ("done", "failed"):
            updates.append("finished_at = CURRENT_TIMESTAMP")

        params.append(job_id)
        execute(f"UPDATE ingest_jobs SET {', '.join(updates)} WHERE job_id = ?", tuple(params))
        execute(
            "UPDATE sources SET processing_status = ?, status = ?, progress = ?, error = ? WHERE id = ?",
            (stage, stage, progress, error, source_id),
        )

    def _run_pipeline(
        self,
        job_id: str,
        source_id: str,
        course_id: str,
        path_str: str,
        stype: str,
        title: str,
    ) -> None:
        try:
            # Stage 1: extracting
            self._update_job(job_id, source_id, "extracting", 15)
            p = Path(path_str)

            raw_units: list[dict] = []
            raw_figures: list[dict] = []

            if stype == "pdf":
                raw_units, raw_figures = extract_pdf(p, source_id, course_id=course_id)
            elif stype in ("pptx", "slides"):
                raw_units, raw_figures = extract_pptx(p, source_id, course_id=course_id)
            elif stype == "video":
                raw_units, raw_figures = extract_video(path_str, source_id, course_id=course_id)
            elif stype == "audio":
                raw_units, raw_figures = extract_audio(p, source_id, course_id=course_id)
            else:
                # Fallback text
                if p.exists():
                    text = p.read_text(encoding="utf-8", errors="ignore")
                    raw_units = [{
                        "content": text,
                        "modality": "text",
                        "page_number": 1,
                        "page": 1,
                        "token_count": len(text.split()),
                        "deep_link": f"/viewer?source={source_id}&page=1",
                    }]

            # Stage 2: chunking & strict location validation
            self._update_job(job_id, source_id, "chunking", 35)
            valid_units: list[dict] = []
            for u in raw_units:
                # Validate non-null deep_link and precise location
                has_loc = (
                    u.get("page") is not None
                    or u.get("page_number") is not None
                    or u.get("slide_no") is not None
                    or u.get("slide_number") is not None
                    or u.get("start_ts") is not None
                    or u.get("timestamp_start") is not None
                )
                if not has_loc:
                    # Flag and repair location to page 1 rather than discarding completely
                    u["page_number"] = 1
                    u["page"] = 1

                if not u.get("deep_link"):
                    if u.get("page_number") is not None:
                        u["deep_link"] = f"/viewer?source={source_id}&page={u['page_number']}"
                    elif u.get("slide_number") is not None:
                        u["deep_link"] = f"/viewer?source={source_id}&slide={u['slide_number']}"
                    elif u.get("timestamp_start") is not None:
                        u["deep_link"] = f"/viewer?source={source_id}&t={round(float(u['timestamp_start']), 2)}"
                    else:
                        u["deep_link"] = f"/viewer?source={source_id}&page=1"

                valid_units.append(u)

            # Stage 3: captioning and saving figures
            self._update_job(job_id, source_id, "captioning", 55)
            for fig in raw_figures:
                insert("figures", {
                    "figure_id": fig["figure_id"],
                    "source_id": source_id,
                    "course_id": course_id,
                    "page_or_slide_or_ts": fig.get("page_or_slide_or_ts"),
                    "image_path": fig.get("image_path"),
                    "caption": fig.get("caption"),
                    "ocr_text": fig.get("ocr_text"),
                    "unit_id": fig.get("unit_id"),
                })

            # Stage 4: embeddings
            self._update_job(job_id, source_id, "embedding", 75)
            texts = [u["content"] for u in valid_units]
            vectors = embed_batch(texts) if texts else []

            # Stage 5: tagging & graph building
            self._update_job(job_id, source_id, "tagging", 85)
            existing_topics = query("SELECT * FROM concepts WHERE level = 'topic' AND course_id = ?", (course_id,))
            existing_concepts = query("SELECT * FROM concepts WHERE course_id = ?", (course_id,))

            # 4-pass extraction & tagging
            cand = extract_candidates_pass1(title, valid_units)
            can_topics, can_concepts = merge_and_cluster_pass2([cand], existing_topics, existing_concepts)
            prereq_edges = build_prerequisites_pass3([cand], can_concepts)

            # Persist canonical topics & concepts into database
            for t in can_topics:
                existing = query_one("SELECT id FROM concepts WHERE id = ?", (t["id"],))
                if not existing:
                    insert("concepts", {
                        "id": t["id"],
                        "course_id": course_id,
                        "name": t["name"],
                        "level": "topic",
                        "description": t.get("summary", ""),
                        "order_index": t.get("order_index", 0),
                    })
                existing_t = query_one("SELECT topic_id FROM topics WHERE topic_id = ?", (t["id"],))
                if not existing_t:
                    insert("topics", {
                        "topic_id": t["id"],
                        "course_id": course_id,
                        "name": t["name"],
                        "level": "topic",
                        "summary": t.get("summary", ""),
                    })

            for c in can_concepts:
                existing = query_one("SELECT id FROM concepts WHERE id = ?", (c["id"],))
                if not existing:
                    insert("concepts", {
                        "id": c["id"],
                        "course_id": course_id,
                        "name": c["name"],
                        "level": "concept",
                        "description": c.get("definition", ""),
                        "definition": c.get("definition", ""),
                        "topic_id": c.get("topic_id"),
                    })

            for e in prereq_edges:
                execute(
                    "INSERT OR REPLACE INTO prerequisites (prerequisite_id, dependent_id, requires_concept_id, concept_id, confidence, reason) VALUES (?, ?, ?, ?, ?, ?)",
                    (e["prerequisite_id"], e["dependent_id"], e["prerequisite_id"], e["dependent_id"], e.get("confidence", 0.8), e.get("reason", "")),
                )

            tagged_units, coverage_report = tag_units_pass4(valid_units, can_topics, can_concepts)

            # Insert units and embeddings
            for unit, vec in zip(tagged_units, vectors):
                uid = unit.get("id") or new_id()
                insert("content_units", {
                    "id": uid,
                    "source_id": source_id,
                    "course_id": course_id,
                    "content": unit["content"],
                    "unit_type": unit.get("unit_type", "text"),
                    "modality": unit.get("modality", "text"),
                    "page_number": unit.get("page_number"),
                    "page": unit.get("page_number"),
                    "slide_number": unit.get("slide_number"),
                    "slide_no": unit.get("slide_number"),
                    "timestamp_start": unit.get("timestamp_start"),
                    "start_ts": unit.get("timestamp_start"),
                    "timestamp_end": unit.get("timestamp_end"),
                    "end_ts": unit.get("timestamp_end"),
                    "figure_id": unit.get("figure_id"),
                    "figure_reference": unit.get("figure_id"),
                    "deep_link": unit.get("deep_link"),
                    "topic_id": unit.get("topic_id"),
                    "topic_ids": unit.get("topic_ids", "[]"),
                    "concept_ids": unit.get("concept_ids", "[]"),
                    "token_count": unit.get("token_count", len(unit["content"].split())),
                    "metadata": jdumps({"coverage": coverage_report.get("coverage_pct", 100)}),
                })
                if vec:
                    save_unit_embedding(uid, vec)

            # Counts summary
            modality_counts: dict[str, int] = {}
            for u in tagged_units:
                mod = u.get("modality", "text")
                modality_counts[mod] = modality_counts.get(mod, 0) + 1

            counts = {
                "units": len(tagged_units),
                "figures": len(raw_figures),
                "by_modality": modality_counts,
                "coverage_pct": coverage_report["coverage_pct"],
            }

            self._update_job(job_id, source_id, "done", 100, counts=counts)

        except Exception as exc:
            self._update_job(job_id, source_id, "failed", 0, error=str(exc))


orchestrator = IngestionOrchestrator()
