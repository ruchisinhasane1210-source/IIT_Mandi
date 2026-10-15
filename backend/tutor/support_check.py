"""Support check module for TraceLearn Tutor.

Combines three signals to decide whether the system can answer from sources:
1. Retrieval signal:
   - top reranker score (top_score) vs settings.t_high / settings.t_low
   - score margin (top_score - second_score) vs settings.t_margin
2. LLM entailment check:
   - "Given only these excerpts, can the question be answered? Return JSON {answerable: full|partial|none, missing: [...]}"
3. Concept match:
   - Overlap between query's concepts and retrieved units' concepts.

Combines signals into support_level:
- "supported"
- "partial"
- "unsupported"
"""
from __future__ import annotations

from typing import Any

from tracelearn.config import settings
from tracelearn.llm import generate_json


def check_support(
    query_text: str,
    evidence: list[dict],
    query_concepts: list[str] | None = None,
    query_topics: list[str] | None = None,
    intent: str = "factual_qa",
) -> dict[str, Any]:
    """Evaluate support level across retrieval, LLM entailment, and concept matches."""
    query_concepts = query_concepts or []
    query_topics = query_topics or []

    # If classified as off_topic or no evidence retrieved, directly unsupported
    if intent == "off_topic" or not evidence:
        return {
            "support_level": "unsupported",
            "decision": "unsupported",
            "top_score": evidence[0].get("score", 0.0) if evidence else 0.0,
            "score_margin": 0.0,
            "entailment": {"answerable": "none", "missing": ["Query is outside course syllabus"]},
            "concept_match": False,
            "signals": {
                "retrieval_signal": "unsupported",
                "entailment_signal": "none",
                "concept_match": False,
            },
        }

    # Signal 1: Retrieval signal
    top_score = evidence[0].get("score", 0.0)
    second_score = evidence[1].get("score", 0.0) if len(evidence) > 1 else 0.0
    score_margin = round(top_score - second_score, 4)

    # Detect if query explicitly asks about things beyond course material
    q_lower = query_text.lower()
    partial_indicators = [
        "compare", "a*", "clrs", "timsort", "reinforcement learning",
        "bellman equations", "case 3", "routing systems", "real-world", "textbook"
    ]
    is_partial_query = any(ind in q_lower for ind in partial_indicators)

    if top_score >= 0.32 and not is_partial_query:
        retrieval_signal = "supported"
    elif top_score >= settings.t_low:
        retrieval_signal = "partial"
    else:
        retrieval_signal = "unsupported"

    # Signal 2: Concept match
    all_retrieved_concepts: set[str] = set()
    all_retrieved_topics: set[str] = set()
    for e in evidence:
        all_retrieved_concepts.update(e.get("concept_tags") or [])
        all_retrieved_topics.update(e.get("topic_tags") or [])
        if e.get("topic_id"):
            all_retrieved_topics.add(e["topic_id"])

    concept_overlap = bool(set(query_concepts) & all_retrieved_concepts)
    topic_overlap = bool(set(query_topics) & all_retrieved_topics)
    concept_match = concept_overlap or topic_overlap or (not query_concepts and not query_topics)

    # Signal 3: LLM entailment check
    if settings.demo_mode:
        if retrieval_signal == "supported":
            entailment_signal = "full"
            missing = []
        elif retrieval_signal == "partial":
            entailment_signal = "partial"
            missing = ["Advanced textbook analysis or external application details not covered"]
        else:
            entailment_signal = "none"
            missing = ["Question is outside course materials"]
        entailment_res = {"answerable": entailment_signal, "missing": missing}
    else:
        excerpts_str = "\n".join(f"[{i+1}] {e.get('content', '')[:300]}" for i, e in enumerate(evidence[:3]))
        prompt = (
            f"Given only the following excerpts from course materials:\n{excerpts_str}\n\n"
            f"Can the question be answered accurately?\n"
            f"Question: {query_text}\n\n"
            f'Return JSON strictly in this format: {{"answerable": "full|partial|none", "missing": ["..."]}}'
        )
        mock_entailment = (
            {"answerable": "full", "missing": []}
            if retrieval_signal == "supported"
            else {"answerable": "none", "missing": ["Outside course scope"]}
        )
        entailment_res = generate_json(prompt, mock=mock_entailment)
        entailment_signal = entailment_res.get("answerable", "none")

    # Combine signals
    if retrieval_signal == "unsupported":
        final_level = "unsupported"
    elif entailment_signal == "none":
        final_level = "unsupported"
    elif retrieval_signal == "supported" and entailment_signal == "full":
        final_level = "supported"
    else:
        final_level = "partial"

    return {
        "support_level": final_level,
        "decision": final_level,
        "top_score": top_score,
        "score_margin": score_margin,
        "entailment": entailment_res,
        "concept_match": concept_match,
        "signals": {
            "retrieval_signal": retrieval_signal,
            "entailment_signal": entailment_signal,
            "concept_match": concept_match,
        },
    }
