"""TraceLearn Tutor Retrieval Stage.

Calls retrieve(query, filters, k) from Stage 2.
Augments results:
- Includes figure units when the query relates to a diagram, graph, figure, or illustration.
- Appends neighboring units (prev / next) when the top hit is a fragment or short chunk.
"""
from __future__ import annotations

import re
from typing import Any

from tracelearn.db import jloads, query, query_one
from tracelearn.rag import retrieve


def retrieve_with_context(
    query_text: str,
    course_id: str = "dsa_6006",
    *,
    scope: dict[str, Any] | None = None,
    k: int = 5,
) -> list[dict]:
    """Execute grounded retrieval with figure inclusion and neighboring context."""
    scope = scope or {}
    filters: dict[str, Any] = {}

    # Map scope fields
    if scope.get("source_ids"):
        # If single source_id, filter by it
        sids = scope["source_ids"]
        if len(sids) == 1:
            filters["source_id"] = sids[0]
    elif scope.get("source_id"):
        filters["source_id"] = scope["source_id"]

    if scope.get("topic_ids"):
        tids = scope["topic_ids"]
        if len(tids) == 1:
            filters["topic_id"] = tids[0]
    elif scope.get("topic_id"):
        filters["topic_id"] = scope["topic_id"]

    if scope.get("page_range"):
        filters["page_range"] = scope["page_range"]

    # 1. Base retrieval via Stage 2 retrieve()
    hits = retrieve(query_text, course_id=course_id, filters=filters, top_k=k)
    if not hits:
        return []

    # 2. Check if query is asking about diagrams/figures
    q_lower = query_text.lower()
    figure_terms = ["diagram", "figure", "illustration", "drawing", "visual", "chart", "tree", "picture", "show"]
    is_figure_query = any(term in q_lower for term in figure_terms)

    # If figure-related and top hits lack a figure, look for relevant figures in the same course/topic
    has_figure = any(h.get("modality") == "figure" or h.get("source_type") == "figure" for h in hits)
    if is_figure_query and not has_figure:
        # Search specifically for figure units
        fig_filters = dict(filters)
        fig_filters["modality"] = "figure"
        fig_hits = retrieve(query_text, course_id=course_id, filters=fig_filters, top_k=2)
        if fig_hits:
            hits.insert(1, fig_hits[0])  # Insert at rank 2
            hits = hits[:k]

    # 3. Augment top hit with neighbor if it's a short fragment (< 120 chars)
    if hits:
        top_unit = hits[0]
        content_len = len(top_unit.get("content", ""))
        if content_len < 120 and top_unit.get("source_id"):
            neighbors = _fetch_neighbors(top_unit["id"], top_unit["source_id"])
            if neighbors.get("next"):
                next_u = neighbors["next"]
                if not any(h["id"] == next_u["id"] for h in hits):
                    hits.append(next_u)

    return hits


def _fetch_neighbors(unit_id: str, source_id: str) -> dict[str, dict | None]:
    """Fetch adjacent units from the same source."""
    rows = query(
        "SELECT * FROM content_units WHERE source_id = ? ORDER BY page_number, slide_number, timestamp_start, id",
        (source_id,),
    )
    if not rows:
        return {"prev": None, "next": None}

    idx = None
    for i, r in enumerate(rows):
        if r["id"] == unit_id:
            idx = i
            break

    prev_doc = rows[idx - 1] if idx is not None and idx > 0 else None
    next_doc = rows[idx + 1] if idx is not None and idx < len(rows) - 1 else None

    from tracelearn.rag import _as_evidence
    return {
        "prev": _as_evidence(prev_doc, 0.4) if prev_doc else None,
        "next": _as_evidence(next_doc, 0.4) if next_doc else None,
    }
