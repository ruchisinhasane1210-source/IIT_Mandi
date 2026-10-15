"""TraceLearn — server-rendered pages."""
from __future__ import annotations

import time

from flask import Blueprint, render_template

from .. import assessment as assessment_mod
from .. import evaluation as evaluation_mod
from ..db import jloads, query
from ..graph import build as build_graph
from ..learner import (
    mastery_map,
    overall_mastery,
    recommend,
    risk_propagation,
    study_schedule,
)
from ..seed import COURSE_ID, USER_ID, counts

bp = Blueprint("pages", __name__)

_eval_cache: dict[str, tuple[float, dict]] = {}
_EVAL_TTL = 300  # seconds


def _mastery_rows(course_id: str) -> list[dict]:
    state = mastery_map(course_id, USER_ID)
    risks = risk_propagation(course_id, USER_ID)
    rows = []
    for topic_id, record in state.items():
        rows.append(
            {
                "id": topic_id,
                "name": record["name"],
                "p_mastery": round(float(record["p_mastery"]), 3),
                "risk_mastery": round(risks.get(topic_id, record["p_mastery"]), 3),
                "attempts": record["attempts"],
                "correct": record["correct"],
                "misconceptions": record["misconception_count"],
                "band": _band(record["p_mastery"]),
            }
        )
    rows.sort(key=lambda r: r["p_mastery"])
    return rows


def _band(p: float) -> str:
    if p >= 0.7:
        return "high"
    if p >= 0.4:
        return "mid"
    return "low"


def _sources(course_id: str) -> list[dict]:
    return query(
        "SELECT * FROM sources WHERE course_id = ? ORDER BY created_at", (course_id,)
    )


@bp.get("/")
def dashboard():
    course = query("SELECT * FROM courses WHERE id = ?", (COURSE_ID,))
    return render_template(
        "dashboard.html",
        active="dashboard",
        course=course[0] if course else None,
        counts=counts(),
        overall=overall_mastery(COURSE_ID, USER_ID),
        mastery=_mastery_rows(COURSE_ID),
        recommendation=recommend(COURSE_ID, USER_ID),
        sources=_sources(COURSE_ID),
    )


@bp.get("/upload")
def upload():
    return render_template("upload.html", active="upload", sources=_sources(COURSE_ID))


@bp.get("/chat")
def chat():
    history = query(
        "SELECT * FROM chat_messages WHERE course_id = ? ORDER BY created_at LIMIT 50",
        (COURSE_ID,),
    )
    for message in history:
        message["citations"] = jloads(message.get("citations"), []) or []
    return render_template("chat.html", active="chat", history=history)


@bp.get("/graph")
def graph():
    return render_template(
        "graph.html",
        active="graph",
        graph=build_graph(COURSE_ID, USER_ID),
        mastery=_mastery_rows(COURSE_ID),
    )


@bp.get("/assessment")
def assessment_page():
    return render_template("assessment.html", active="assessment", course_id=COURSE_ID)


@bp.get("/results")
def results():
    return render_template(
        "results.html",
        active="assessment",
        report=assessment_mod.latest_report(COURSE_ID, USER_ID),
    )


@bp.get("/mastery")
def mastery_page():
    return render_template(
        "mastery.html",
        active="mastery",
        overall=overall_mastery(COURSE_ID, USER_ID),
        mastery=_mastery_rows(COURSE_ID),
        schedule=study_schedule(COURSE_ID, USER_ID),
        misconceptions=query(
            "SELECT * FROM misconceptions WHERE course_id = ? ORDER BY count DESC", (COURSE_ID,)
        ),
        recommendation=recommend(COURSE_ID, USER_ID),
    )


@bp.get("/eval")
def eval_page():
    cached = _eval_cache.get(COURSE_ID)
    if cached and time.time() - cached[0] < _EVAL_TTL:
        results = cached[1]
    else:
        results = evaluation_mod.run_all(COURSE_ID)
        _eval_cache[COURSE_ID] = (time.time(), results)
    return render_template("eval.html", active="eval", results=results)
