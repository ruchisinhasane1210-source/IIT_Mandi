"""End-to-end Demo Course Seed Script.

Ingests a small sample course consisting of:
  1. Lecture video/audio (sample_30s.wav)
  2. Lecture slides (sample_3slide.pptx)
  3. Textbook chapter (sample_3page.pdf)

Extracts units, captions figures, computes embeddings, generates 4-pass topic/concept/prereq
graph, tags every unit, and prints detailed ingestion stats.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.ingestion import orchestrator
from tracelearn import create_app
from tracelearn.db import insert, query, query_one
from tests.generate_fixtures import generate_all_fixtures

COURSE_ID = "demo_multimodal_course"


def seed_demo_course() -> dict:
    start_time = time.time()
    app = create_app()

    with app.app_context():
        print(f"Creating / ensuring course '{COURSE_ID}'...")
        existing_course = query_one("SELECT id FROM courses WHERE id = ?", (COURSE_ID,))
        if not existing_course:
            insert("courses", {
                "id": COURSE_ID,
                "title": "Multimodal Algorithms & Data Structures",
                "description": "Demo course ingesting lecture video, slide deck, and textbook chapter.",
                "exam_date": "2026-11-01",
                "daily_study_minutes": 30,
            })

        print("Generating / verifying fixture files...")
        fixtures = generate_all_fixtures()

        print("\nIngesting materials end-to-end:")
        # 1. Textbook chapter PDF
        print("  [1/3] Ingesting textbook chapter (PDF)...")
        r_pdf = orchestrator.ingest(
            fixtures["pdf"],
            source_type="pdf",
            title="Chapter 8: Binary Heaps Textbook",
            course_id=COURSE_ID,
            async_run=False,
        )
        print(f"        -> Source ID: {r_pdf['source_id']}")

        # 2. Lecture slides PPTX
        print("  [2/3] Ingesting lecture slides (PPTX)...")
        r_pptx = orchestrator.ingest(
            fixtures["pptx"],
            source_type="pptx",
            title="Lecture 8: Binary Heaps Deck",
            course_id=COURSE_ID,
            async_run=False,
        )
        print(f"        -> Source ID: {r_pptx['source_id']}")

        # 3. Lecture audio/video
        print("  [3/3] Ingesting lecture audio/video...")
        r_media = orchestrator.ingest(
            fixtures["audio"],
            source_type="audio",
            title="Lecture 8 Audio Recording",
            course_id=COURSE_ID,
            async_run=False,
        )
        print(f"        -> Source ID: {r_media['source_id']}")

        elapsed = round(time.time() - start_time, 2)

        # Compute ingestion stats
        units = query("SELECT id, modality, topic_id, concept_ids, deep_link, page, slide_no, start_ts FROM content_units WHERE course_id = ?", (COURSE_ID,))
        topics = query("SELECT id, name FROM concepts WHERE course_id = ? AND level = 'topic'", (COURSE_ID,))
        concepts = query("SELECT id, name FROM concepts WHERE course_id = ? AND level != 'topic'", (COURSE_ID,))
        prereqs = query("SELECT * FROM prerequisites")
        figures = query("SELECT figure_id, caption, ocr_text FROM figures WHERE course_id = ?", (COURSE_ID,))

        by_modality: dict[str, int] = {}
        tagged_count = 0
        for u in units:
            m = u.get("modality", "text")
            by_modality[m] = by_modality.get(m, 0) + 1
            if u.get("topic_id"):
                tagged_count += 1

        coverage_pct = (tagged_count / len(units) * 100.0) if units else 100.0

        stats = {
            "course_id": COURSE_ID,
            "total_units": len(units),
            "units_by_modality": by_modality,
            "figures_count": len(figures),
            "topics_count": len(topics),
            "concepts_count": len(concepts),
            "prerequisite_edges": len(prereqs),
            "tagging_coverage_pct": round(coverage_pct, 1),
            "time_taken_seconds": elapsed,
        }

        print("\n" + "=" * 60)
        print("DEMO COURSE INGESTION STATS")
        print("=" * 60)
        print(f"Course ID:             {stats['course_id']}")
        print(f"Total Content Units:   {stats['total_units']}")
        print(f"Units by Modality:     {stats['units_by_modality']}")
        print(f"Captioned Figures:     {stats['figures_count']}")
        print(f"Topics Extracted:      {stats['topics_count']}")
        print(f"Concepts Extracted:    {stats['concepts_count']}")
        print(f"Prerequisite Edges:    {stats['prerequisite_edges']}")
        print(f"Tagging Coverage:      {stats['tagging_coverage_pct']}%")
        print(f"Total Time Taken:      {stats['time_taken_seconds']}s")
        print("=" * 60 + "\n")

        return stats


if __name__ == "__main__":
    seed_demo_course()
