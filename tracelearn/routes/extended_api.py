"""TraceLearn Extended API — Track D Multimodal Ingestion, Knowledge Graph, and Retrieval."""
from __future__ import annotations

from pathlib import Path
from flask import Blueprint, jsonify, request

from backend.ingestion.orchestrator import orchestrator
from tracelearn import rag
from tracelearn.config import settings
from tracelearn.db import execute, jdumps, jloads, new_id, query, query_one

bp = Blueprint("extended_api", __name__)


def _ok(payload: dict, status: int = 200):
    return jsonify(payload), status


# ------------------------------------------------------------------ Ingestion

@bp.post("/ingest/upload")
def ingest_upload():
    course_id = request.form.get("course_id") or "dsa_6006"
    file = request.files.get("file")
    url = request.form.get("url") or (request.get_json(silent=True) or {}).get("url")
    title = request.form.get("title") or (request.get_json(silent=True) or {}).get("title")

    if file and file.filename:
        safe_name = Path(file.filename).name
        dest = settings.storage_dir / course_id / f"{new_id()[:8]}_{safe_name}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        file.save(dest)
        res = orchestrator.ingest(dest, title=title or safe_name, course_id=course_id)
        return _ok(res, 202)

    if url:
        res = orchestrator.ingest(url, title=title or url, course_id=course_id)
        return _ok(res, 202)

    return _ok({"error": "No file or URL provided"}, 400)


@bp.get("/ingest/status/<job_id>")
def ingest_status(job_id: str):
    job = query_one("SELECT * FROM ingest_jobs WHERE job_id = ?", (job_id,))
    if not job:
        # Fallback to jobs table if old-style job
        job = query_one("SELECT id as job_id, * FROM jobs WHERE id = ?", (job_id,))
        if not job:
            return _ok({"error": "Job not found"}, 404)

    counts = jloads(job.get("counts"), {})
    return _ok({
        "job_id": job.get("job_id") or job.get("id"),
        "source_id": job.get("source_id"),
        "stage": job.get("stage") or job.get("status"),
        "progress": job.get("progress", 0),
        "error": job.get("error"),
        "counts": counts,
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
    })


@bp.get("/sources")
def list_sources():
    sources = query("SELECT * FROM sources ORDER BY created_at DESC")
    # Attach unit counts to each source
    result = []
    for s in sources:
        sid = s["id"]
        s_alt = s.get("source_id")
        n = query_one(
            "SELECT COUNT(*) as cnt FROM content_units WHERE source_id = ? OR source_id = ?",
            (sid, s_alt),
        )["cnt"]
        s_dict = dict(s)
        s_dict["unit_count"] = n
        s_dict["title"] = s.get("title") or s.get("file_name")
        s_dict["type"] = s.get("source_type") or s.get("type")
        s_dict["status"] = s.get("processing_status") or s.get("status")
        result.append(s_dict)

    return _ok({"sources": result})


@bp.delete("/sources/<source_id>")
def delete_source(source_id: str):
    # Find all units belonging to this source
    units = query(
        "SELECT id FROM content_units WHERE source_id = ? OR source_id = ?",
        (source_id, source_id),
    )
    unit_ids = [u["id"] for u in units]

    # Delete embeddings
    if unit_ids:
        placeholders = ", ".join("?" for _ in unit_ids)
        try:
            execute(f"DELETE FROM unit_embeddings WHERE unit_id IN ({placeholders})", unit_ids)
        except Exception:
            pass

    execute("DELETE FROM figures WHERE source_id = ?", (source_id,))
    execute("DELETE FROM content_units WHERE source_id = ? OR source_id = ?", (source_id, source_id))
    execute("DELETE FROM ingest_jobs WHERE source_id = ?", (source_id,))
    execute("DELETE FROM sources WHERE id = ? OR source_id = ?", (source_id, source_id))

    return _ok({"ok": True, "deleted_source_id": source_id, "deleted_units": len(unit_ids)})


# ------------------------------------------------------------- Deep Links / Source Excerpt

@bp.get("/source/<unit_id>")
def get_source_unit(unit_id: str):
    unit = query_one("SELECT * FROM content_units WHERE id = ?", (unit_id,))
    if not unit:
        return _ok({"error": "Unit not found"}, 404)

    source = query_one("SELECT * FROM sources WHERE id = ? OR source_id = ?", (unit["source_id"], unit["source_id"]))
    source_title = (source or {}).get("title") or (source or {}).get("file_name", "Unknown Source")

    # Neighbors: previous and next units in the same source
    sid = unit["source_id"]
    all_source_units = query(
        "SELECT id FROM content_units WHERE source_id = ? ORDER BY id ASC",
        (sid,),
    )
    all_ids = [u["id"] for u in all_source_units]
    prev_id = None
    next_id = None
    if unit_id in all_ids:
        idx = all_ids.index(unit_id)
        if idx > 0:
            prev_id = all_ids[idx - 1]
        if idx + 1 < len(all_ids):
            next_id = all_ids[idx + 1]

    location = {
        "page": unit.get("page_number") or unit.get("page"),
        "slide": unit.get("slide_number") or unit.get("slide_no"),
        "timestamp_start": unit.get("timestamp_start") or unit.get("start_ts"),
        "timestamp_end": unit.get("timestamp_end") or unit.get("end_ts"),
    }

    deep_link = unit.get("deep_link")
    if not deep_link:
        if location["page"] is not None:
            deep_link = f"/viewer?source={sid}&page={location['page']}"
        elif location["slide"] is not None:
            deep_link = f"/viewer?source={sid}&slide={location['slide']}"
        elif location["timestamp_start"] is not None:
            deep_link = f"/viewer?source={sid}&t={round(float(location['timestamp_start']), 2)}"
        else:
            deep_link = f"/viewer?source={sid}"

    return _ok({
        "unit_id": unit["id"],
        "id": unit["id"],
        "excerpt": (unit.get("content") or "")[:260],
        "content": unit.get("content") or "",
        "modality": unit.get("modality") or unit.get("unit_type", "text"),
        "location": location,
        "deep_link": deep_link,
        "source_id": sid,
        "source_title": source_title,
        "neighbors": {
            "prev": prev_id,
            "next": next_id,
        },
    })


# ------------------------------------------------------------- Knowledge Graph

@bp.get("/knowledge/graph")
def knowledge_graph_endpoint():
    course_id = request.args.get("course_id") or "dsa_6006"
    topics = query("SELECT * FROM concepts WHERE level = 'topic' AND course_id = ?", (course_id,))
    concepts = query("SELECT * FROM concepts WHERE level != 'topic' AND course_id = ?", (course_id,))
    edges = query("SELECT * FROM prerequisites")

    # Calculate unit counts per node
    units = query("SELECT id, topic_ids, concept_ids, topic_id FROM content_units WHERE course_id = ?", (course_id,))
    unit_counts: dict[str, int] = {}
    for u in units:
        # direct topic_id
        if u.get("topic_id"):
            unit_counts[u["topic_id"]] = unit_counts.get(u["topic_id"], 0) + 1
        for tid in jloads(u.get("topic_ids"), []) or []:
            unit_counts[tid] = unit_counts.get(tid, 0) + 1
        for cid in jloads(u.get("concept_ids"), []) or []:
            unit_counts[cid] = unit_counts.get(cid, 0) + 1

    clean_edges = []
    for e in edges:
        src = e.get("prerequisite_id") or e.get("requires_concept_id")
        dst = e.get("dependent_id") or e.get("concept_id")
        if src and dst:
            clean_edges.append({
                "source": src,
                "target": dst,
                "confidence": e.get("confidence", 0.8),
                "reason": e.get("reason", ""),
            })

    return _ok({
        "topics": topics,
        "concepts": concepts,
        "prerequisite_edges": clean_edges,
        "unit_counts_per_node": unit_counts,
    })


@bp.get("/knowledge/topic/<topic_id>")
def knowledge_topic(topic_id: str):
    topic = query_one("SELECT * FROM concepts WHERE id = ? AND level = 'topic'", (topic_id,))
    if not topic:
        topic = query_one("SELECT topic_id as id, name, summary as description FROM topics WHERE topic_id = ?", (topic_id,))
        if not topic:
            return _ok({"error": "Topic not found"}, 404)

    # Subtopics and concepts under this topic
    subtopics = query("SELECT * FROM concepts WHERE parent_id = ?", (topic_id,))
    concepts = query("SELECT * FROM concepts WHERE topic_id = ? OR parent_id = ?", (topic_id, topic_id))

    # Units tagged with this topic
    all_units = query("SELECT * FROM content_units")
    matched_units = []
    for u in all_units:
        tids = jloads(u.get("topic_ids"), []) or []
        if u.get("topic_id") == topic_id or topic_id in tids:
            source = query_one("SELECT title, file_name FROM sources WHERE id = ?", (u["source_id"],))
            u_dict = dict(u)
            u_dict["source_title"] = (source or {}).get("title") or (source or {}).get("file_name", "")
            matched_units.append(u_dict)

    return _ok({
        "topic": topic,
        "subtopics": subtopics,
        "concepts": concepts,
        "units": matched_units,
    })


@bp.get("/knowledge/concept/<concept_id>")
def knowledge_concept(concept_id: str):
    concept = query_one("SELECT * FROM concepts WHERE id = ?", (concept_id,))
    if not concept:
        return _ok({"error": "Concept not found"}, 404)

    # Prerequisites (what this concept requires)
    prereq_rows = query(
        "SELECT * FROM prerequisites WHERE dependent_id = ? OR concept_id = ?",
        (concept_id, concept_id),
    )
    prereq_ids = [r.get("prerequisite_id") or r.get("requires_concept_id") for r in prereq_rows]
    prerequisites = query(
        f"SELECT * FROM concepts WHERE id IN ({', '.join('?' for _ in prereq_ids)})",
        prereq_ids,
    ) if prereq_ids else []

    # Dependents (what requires this concept)
    dep_rows = query(
        "SELECT * FROM prerequisites WHERE prerequisite_id = ? OR requires_concept_id = ?",
        (concept_id, concept_id),
    )
    dep_ids = [r.get("dependent_id") or r.get("concept_id") for r in dep_rows]
    dependents = query(
        f"SELECT * FROM concepts WHERE id IN ({', '.join('?' for _ in dep_ids)})",
        dep_ids,
    ) if dep_ids else []

    # Units tagged with this concept
    all_units = query("SELECT * FROM content_units")
    matched_units = []
    for u in all_units:
        cids = jloads(u.get("concept_ids"), []) or []
        if concept_id in cids:
            source = query_one("SELECT title, file_name FROM sources WHERE id = ?", (u["source_id"],))
            u_dict = dict(u)
            u_dict["source_title"] = (source or {}).get("title") or (source or {}).get("file_name", "")
            matched_units.append(u_dict)

    return _ok({
        "concept": concept,
        "definition": concept.get("definition") or concept.get("description", ""),
        "prerequisites": prerequisites,
        "dependents": dependents,
        "units": matched_units,
    })


# ------------------------------------------------------------- Retrieval Debug

@bp.post("/retrieve/debug")
def retrieve_debug_endpoint():
    payload = request.get_json(silent=True) or {}
    query_text = (payload.get("query") or payload.get("question") or "").strip()
    if not query_text:
        return _ok({"error": "query is required"}, 400)

    filters = payload.get("filters") or {}
    k = int(payload.get("k") or payload.get("top_k") or 5)
    course_id = payload.get("course_id") or "dsa_6006"

    debug_res = rag.retrieve_debug(query_text, filters=filters, k=k, course_id=course_id)
    return _ok(debug_res)


# ---------------------------------------------------------------- Tutor Chat (Stage 3)

@bp.post("/chat")
def tutor_chat_endpoint():
    """Stage 3 Source-Grounded Tutor Chat API.
    
    Compatible with both new Stage 3 schema and previous /courses/<id>/chat requests.
    """
    payload = request.get_json(silent=True) or {}
    message = (payload.get("message") or "").strip()
    if not message:
        return _ok({"error": "message is required"}, 400)

    session_id = payload.get("session_id")
    student_id = payload.get("student_id") or "demo_learner"
    course_id = payload.get("course_id") or "dsa_6006"
    scope = payload.get("scope") or {}
    allow_outside = bool(payload.get("allow_outside_knowledge", False))
    language = payload.get("language") or "auto"

    from backend.tutor import pipeline
    res = pipeline.process(
        message,
        session_id=session_id,
        student_id=student_id,
        course_id=course_id,
        scope=scope,
        allow_outside_knowledge=allow_outside,
        language=language,
    )
    return _ok(res)


@bp.get("/chat/session/<session_id>")
def tutor_session_history(session_id: str):
    """Retrieve full conversation history with structured citations and support signals."""
    rows = query(
        "SELECT * FROM tutor_messages WHERE session_id = ? ORDER BY created_at ASC",
        (session_id,),
    )
    messages = []
    for r in rows:
        messages.append({
            "id": r["id"],
            "role": r["role"],
            "content": r["content"],
            "grounded_answer": r.get("grounded_answer"),
            "outside_knowledge": r.get("outside_knowledge"),
            "support_level": r.get("support_level"),
            "citations": jloads(r.get("citations"), []),
            "flags": jloads(r.get("flags"), []),
            "metrics": jloads(r.get("metrics"), {}),
            "created_at": r.get("created_at"),
        })
    return _ok({"session_id": session_id, "messages": messages})


@bp.get("/citation/<unit_id>")
def citation_unit_endpoint(unit_id: str):
    """Alias for /source/<unit_id> providing precise location and excerpt for citations."""
    return get_source_unit(unit_id)

