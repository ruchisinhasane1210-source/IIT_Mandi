"""TraceLearn Stage 3 Learner-Model Hook Interface.

Stage 3 provides the interface and a no-op/logging implementation.
Stage 4 will replace this with real BKT/IRT parameter updates and posterior mastery estimates.
"""
from __future__ import annotations

import logging
from typing import Any

from tracelearn.db import insert, query, query_one

logger = logging.getLogger("tracelearn.learner")


def get_student_state(student_id: str, concept_ids: list[str]) -> dict[str, float]:
    """Retrieve mastery estimates for the given concept IDs for a student.

    Returns a dict mapping concept_id -> mastery probability in [0.0, 1.0].
    Defaults to 0.5 when unknown or unobserved.
    """
    if not concept_ids:
        return {}

    state: dict[str, float] = {}
    for cid in concept_ids:
        # Check if there is an existing mastery record in mastery_records table
        row = query_one(
            "SELECT p_mastery FROM mastery_records WHERE user_id = ? AND topic_id = ?",
            (student_id, cid),
        )
        if row and row.get("p_mastery") is not None:
            state[cid] = float(row["p_mastery"])
        else:
            state[cid] = 0.5

    return state


def record_interaction(student_id: str, event: dict[str, Any]) -> None:
    """Record a learning interaction event.

    Event schema:
    {
        "type": "chat",
        "concept_ids": [...],
        "support_level": "supported|partial|unsupported",
        "signals": ["asked_clarification" | "repeated_question" | "expressed_confusion" | "correct_restatement", ...],
        "session_id": "...",
        "turn_id": "...",
        "details": {...}
    }
    """
    logger.info(
        "Learner interaction recorded for student %s: type=%s, concepts=%s, signals=%s",
        student_id,
        event.get("type"),
        event.get("concept_ids"),
        event.get("signals"),
    )
