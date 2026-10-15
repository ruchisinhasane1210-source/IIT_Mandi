"""TraceLearn — JSON API."""
from __future__ import annotations

import threading
import time
from pathlib import Path

from flask import Blueprint, jsonify, request, send_file
import requests as _requests

from .. import assessment as assessment_mod
from .. import evaluation as evaluation_mod
from .. import rag
from ..config import settings
from ..db import execute, insert, jdumps, jloads, new_id, query, query_one
from ..graph import build as build_graph
from ..ingest import SUPPORTED_EXT, detect_type, process_source
from ..supabase_storage import build_storage, generate_source_name, source_storage_path
from ..learner import (
    log_misconception,
    mastery_map,
    record_attempt,
    recommend,
    study_schedule,
    weak_evidence,
)
from ..seed import COURSE_ID, USER_ID, ensure_seeded, reset


def requests_get(url, **kwargs):
    return _requests.get(url, **kwargs)

bp = Blueprint("api", __name__)


def _ok(payload: dict, status: int = 200):
    return jsonify(payload), status


def _topic_of(unit_id: str) -> str | None:
    row = query_one("SELECT topic_ids FROM content_units WHERE id = ?", (unit_id,))
    if not row:
        return None
    topics = jloads(row.get("topic_ids"), []) or []
    return topics[0] if topics else None


# ------------------------------------------------------------------ health

@bp.get("/health")
def health():
    from ..db import health as db_health

    return _ok({
        "ok": True,
        "app": settings.summary(),
        "db": db_health(),
        "questions": query_one("SELECT COUNT(*) AS n FROM questions")["n"],
    })


# -------------------------------------------------------------------- chat

@bp.post("/courses/<course_id>/chat")
def chat(course_id: str):
    payload = request.get_json(silent=True) or {}
    question = (payload.get("message") or "").strip()
    if not question:
        return _ok({"error": "message is required"}, 400)

    state = mastery_map(course_id, USER_ID)
    level = "beginner" if not state else (
        "beginner" if sum(r["p_mastery"] for r in state.values()) / len(state) < 0.4
        else "intermediate"
    )

    result = rag.answer(question, course_id, student_level=level)

    insert("chat_messages", {
        "course_id": course_id, "user_id": USER_ID, "role": "user", "content": question,
    })
    insert("chat_messages", {
        "course_id": course_id, "user_id": USER_ID, "role": "assistant",
        "content": result["answer"],
        "grounding_status": result["grounding_status"],
        "grounding_score": result["grounding_score"],
        "citations": jdumps(result["citations"]),
        "outside_knowledge": result.get("outside_knowledge"),
    })

    # learner hook: a question that the material cannot answer signals confusion
    topic_touched = None
    if result["citations"]:
        topic_touched = _topic_of(result["citations"][0]["id"])
    if topic_touched and result["grounding_status"] == rag.NOT_COVERED:
        weak_evidence(course_id, topic_touched, USER_ID)

    result["topic_touched"] = topic_touched
    return _ok(result)


# ------------------------------------------------------------ knowledge map

@bp.get("/courses/<course_id>/knowledge-graph")
def knowledge_graph(course_id: str):
    return _ok(build_graph(course_id, USER_ID))


# -------------------------------------------------------------- learner API

@bp.get("/students/<user_id>/mastery")
def student_mastery(user_id: str):
    state = mastery_map(COURSE_ID, user_id)
    return _ok({
        "topics": [
            {"topic_id": t, "name": r["name"], "p_mastery": round(r["p_mastery"], 3),
             "attempts": r["attempts"], "correct": r["correct"]}
            for t, r in state.items()
        ]
    })


@bp.get("/students/<user_id>/misconceptions")
def student_misconceptions(user_id: str):
    rows = query(
        "SELECT m.*, c.name AS topic_name FROM misconceptions m "
        "LEFT JOIN concepts c ON c.id = m.topic_id WHERE m.user_id = ? ORDER BY m.count DESC",
        (user_id,),
    )
    return _ok({"misconceptions": rows})


@bp.get("/students/<user_id>/recommendations")
def student_recommendations(user_id: str):
    return _ok(recommend(COURSE_ID, user_id))


@bp.get("/students/<user_id>/schedule")
def student_schedule(user_id: str):
    return _ok({"schedule": study_schedule(COURSE_ID, user_id)})


# ------------------------------------------------------------- assessments

@bp.post("/courses/<course_id>/assessments/generate")
def generate_assessment(course_id: str):
    payload = request.get_json(silent=True) or {}
    result = assessment_mod.generate(
        course_id,
        user_id=payload.get("user_id", USER_ID),
        num=int(payload.get("number_of_questions") or 5),
        topics=payload.get("topics"),
        kinds=payload.get("question_types"),
        kind=payload.get("kind") or "quiz",
    )
    return _ok(result)


@bp.get("/assessments/<assessment_id>")
def get_assessment(assessment_id: str):
    found = assessment_mod.get(assessment_id)
    if not found:
        return _ok({"error": "not found"}, 404)
    return _ok(found)


@bp.post("/assessments/<assessment_id>/submit")
def submit_assessment(assessment_id: str):
    payload = request.get_json(silent=True) or {}
    report = assessment_mod.submit(assessment_id, payload.get("answers") or {})
    if not report:
        return _ok({"error": "not found"}, 404)
    return _ok(report)


@bp.post("/courses/<course_id>/diagnostic")
def diagnostic(course_id: str):
    """Cold start: a 10-question adaptive diagnostic across the course."""
    result = assessment_mod.generate(course_id, num=10, kind="diagnostic")
    return _ok(result)


# --------------------------------------------------------------- ingestion

@bp.post("/courses/<course_id>/sources")
def upload_source(course_id: str):
    file = request.files.get("file")
    if not file or not file.filename:
        return _ok({"error": "no file provided"}, 400)

    source_type = detect_type(file.filename)
    if not source_type:
        return _ok({"error": f"unsupported file type. Supported: {sorted(SUPPORTED_EXT)}"}, 415)

    safe_name = Path(file.filename).name
    storage_backend = build_storage()

    if storage_backend and settings.storage_ready:
        # Read into memory once; Supabase Storage wants bytes.
        file_bytes = file.read()
        content_type = file.mimetype or "application/octet-stream"
        up = storage_backend.signed_upload(file_bytes, safe_name, content_type)
        source_storage_key = up["storage_path"]
        storage_url = up.get("storage_url")
        local_dest = None
    else:
        local_dest = settings.storage_dir / course_id / f"{new_id()[:8]}_{safe_name}"
        local_dest.parent.mkdir(parents=True, exist_ok=True)
        file.save(local_dest)
        source_storage_key = str(local_dest)
        storage_url = None

    source_id = insert("sources", {
        "course_id": course_id,
        "source_id": new_id()[:8],
        "file_name": safe_name,
        "source_type": source_type,
        "storage_path": source_storage_key,
        "storage_url": storage_url,
        "mime_type": file.mimetype,
        "processing_status": "processing",
        "progress": 5,
    })

    job_id = insert("jobs", {
        "course_id": course_id, "source_id": source_id, "kind": "ingest",
        "status": "processing", "step": "queued", "progress": 5,
    })

    def _run() -> None:
        try:
            # If we used Supabase Storage, the file is already remote; remind
            # process_source to be tolerant of a missing local path.
            source = query_one("SELECT storage_path, source_type FROM sources WHERE id = ?", (source_id,))
            if source and source.get("storage_path"):
                remote = source["storage_path"]
                if not remote.startswith("tracelearn/"):
                    # Non-local storage path; process_source only needs the key
                    # and will skip local extraction (falls back to pending).
                    pass
            process_source(source_id)
        finally:
            source = query_one("SELECT processing_status, progress FROM sources WHERE id = ?",
                               (source_id,))
            execute(
                "UPDATE jobs SET status = ?, progress = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                ("done" if source and source["processing_status"] == "done" else "failed",
                 (source or {}).get("progress", 0), job_id),
            )

    threading.Thread(target=_run, daemon=True).start()

    return _ok({"source_id": source_id, "job_id": job_id}, 202)


@bp.get("/jobs/<job_id>")
def get_job(job_id: str):
    job = query_one("SELECT * FROM jobs WHERE id = ?", (job_id,))
    if not job:
        return _ok({"error": "not found"}, 404)
    source = query_one("SELECT * FROM sources WHERE id = ?", (job["source_id"],))
    return _ok({"job": job, "source": source})


@bp.get("/units/<unit_id>")
def get_unit(unit_id: str):
    """Everything the citation viewer needs to open one citable location."""
    unit = query_one("SELECT * FROM content_units WHERE id = ?", (unit_id,))
    if not unit:
        return _ok({"error": "not found"}, 404)

    source = query_one("SELECT * FROM sources WHERE id = ?", (unit["source_id"],))
    if unit.get("timestamp_start") is not None:
        label = f"Play lecture from {_clock(unit['timestamp_start'])}"
        source_type = "video"
    elif unit.get("slide_number") is not None:
        label = f"Open slide {unit['slide_number']}"
        source_type = "slide"
    elif unit.get("page_number") is not None:
        label = f"Open page {unit['page_number']}"
        source_type = "pdf"
    else:
        label = "Open source"
        source_type = unit.get("unit_type") or "pdf"

    return _ok({
        "id": unit["id"],
        "content": unit["content"],
        "unit_type": unit.get("unit_type"),
        "source_type": source_type,
        "source_id": unit.get("source_id"),
        "file_name": (source or {}).get("file_name"),
        "page_number": unit.get("page_number"),
        "slide_number": unit.get("slide_number"),
        "timestamp_start": unit.get("timestamp_start"),
        "timestamp_end": unit.get("timestamp_end"),
        "topics": jloads(unit.get("topic_ids"), []) or [],
        "jump_url": f"/api/sources/{unit['source_id']}/file",
        "location": {"label": label},
    })


def _clock(seconds) -> str:
    total = int(max(0, float(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


@bp.get("/sources/<source_id>/file")
def source_file(source_id: str):
    source = query_one("SELECT * FROM sources WHERE id = ?", (source_id,))
    if not source or not source.get("storage_path"):
        return _ok({"error": "not found"}, 404)

    storage_backend = build_storage()
    if storage_backend and settings.storage_ready and source.get("storage_url"):
        # Stream from Supabase Storage instead of local disk.
        try:
            resp = requests_get(source["storage_url"], timeout=60, stream=True)
            if not resp.ok:
                return _ok({"error": "file missing in storage"}, 404)
            from flask import Response
            def _stream():
                for chunk in resp.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        yield chunk
            return Response(
                _stream(),
                mimetype=source.get("mime_type") or "application/octet-stream",
                headers={"Content-Disposition": f'attachment; filename="{source["file_name"]}"'},
            )
        except Exception:
            pass

    path = Path(source["storage_path"])
    if not path.exists():
        return _ok({"error": "file missing on disk"}, 404)
    return send_file(path, download_name=source["file_name"])


# -------------------------------------------------------------- evaluation

@bp.post("/evaluation/run")
def run_evaluation():
    return _ok(evaluation_mod.run_all(COURSE_ID))


@bp.get("/evaluation/<run_id>")
def get_evaluation(run_id: str):
    row = query_one("SELECT * FROM eval_runs WHERE id = ?", (run_id,))
    if not row:
        return _ok({"error": "not found"}, 404)
    row["metrics"] = jloads(row.get("metrics"), {})
    return _ok(row)


# --------------------------------------------------------------------- dev

@bp.post("/dev/reset")
def dev_reset():
    reset()
    ensure_seeded()
    return _ok({"ok": True, "reseeded": True})
