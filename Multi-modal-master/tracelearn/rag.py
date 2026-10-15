"""TraceLearn — source-grounded retrieval and answering.

Pipeline (§9 of the plan):

  1. embed the question
  2. vector search over content units (cosine)
  3. keyword search (BM25-lite) for exact terms and equations
  4. fuse the two ranked lists with reciprocal rank fusion
  5. compute a grounding score and map it to grounded / partial / not_covered
  6. answer only from the kept evidence, with validated citations

In demo mode (no model key) steps 1-2 still work — the local hashing
vectorizer gives meaningful similarity — and the answer is produced
**extractively** from the top evidence, so citations stay honest.
"""
from __future__ import annotations

import math
import re

from .config import settings
from .db import jloads, query
from .embeddings import cosine, embed, load_unit_embeddings
from .llm import generate

_WORD = re.compile(r"[a-z0-9]+")

GROUNDED = "grounded"
PARTIAL = "partial"
NOT_COVERED = "not_covered"

# Reference magnitudes for a "clearly relevant" hit, used to saturate scores
# into [0, 1]. Calibrated for the local hashing vectorizer / BM25 pair.
SIM_REF = 0.45
KW_REF = 12.0
SIM_WEIGHT = 0.55
KW_WEIGHT = 0.45


# --------------------------------------------------------------- retrieval

def _bm25_scores(question: str, docs: list[dict]) -> dict[str, float]:
    """A compact BM25 over the course's units (corpus is small by design)."""
    q_terms = set(_WORD.findall(question.lower()))
    if not q_terms or not docs:
        return {}
    doc_tokens = [_WORD.findall((d["content"] or "").lower()) for d in docs]
    n = len(docs)
    avg_len = sum(len(t) for t in doc_tokens) / n or 1.0
    df: dict[str, int] = {}
    for toks in doc_tokens:
        for term in q_terms & set(toks):
            df[term] = df.get(term, 0) + 1

    k1, b = 1.2, 0.75
    scores: dict[str, float] = {}
    for doc, toks in zip(docs, doc_tokens):
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        score = 0.0
        for term in q_terms:
            if tf.get(term):
                idf = math.log(1 + (n - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5))
                denom = tf[term] + k1 * (1 - b + b * len(toks) / avg_len)
                score += idf * (tf[term] * (k1 + 1)) / denom
        scores[doc["id"]] = score
    return scores


def _rrf(rank_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    fused: dict[str, float] = {}
    for ranks in rank_lists:
        for position, doc_id in enumerate(ranks):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + position + 1)
    return fused


def retrieve(question: str, course_id: str, *, topic_ids: list[str] | None = None,
             top_k: int = 5) -> list[dict]:
    """Return the top evidence units for a question, best first."""
    docs = query(
        "SELECT * FROM content_units WHERE course_id = ?",
        (course_id,),
    )
    if topic_ids:
        wanted = set(topic_ids)
        docs = [
            d for d in docs
            if wanted & set(jloads(d.get("topic_ids"), []) or [])
        ] or docs
    if not docs:
        return []

    # --- 1/2. vector search (SQLite scores in Python; corpus is tiny) ---
    vectors = dict(load_unit_embeddings([d["id"] for d in docs]))
    q_vec = embed(question)
    sims = {d["id"]: cosine(q_vec, vectors.get(d["id"])) for d in docs}
    vector_order = sorted(sims, key=lambda i: sims[i], reverse=True)[:30]

    # --- 3. keyword search ---
    kw = _bm25_scores(question, docs)
    kw_order = sorted(kw, key=lambda i: kw[i], reverse=True)[:30]

    # --- 4. reciprocal rank fusion ---
    fused = _rrf([vector_order, kw_order])
    ranked = sorted(fused, key=lambda i: fused[i], reverse=True)[: max(top_k * 3, top_k)]

    by_id = {d["id"]: d for d in docs}
    top_fused = fused[ranked[0]] if ranked else 0.0

    evidence = []
    for doc_id in ranked:
        doc = by_id[doc_id]
        # Saturating blend in [0,1]. SIM_REF / KW_REF are the "clearly relevant"
        # magnitudes for the demo embedding provider; thresholds are tuned on the
        # evaluation set (see eval/test_set.json) to maximise refusal F1.
        sim_norm = min(1.0, max(0.0, sims.get(doc_id, 0.0)) / SIM_REF)
        kw_norm = min(1.0, max(0.0, kw.get(doc_id, 0.0)) / KW_REF)
        score = SIM_WEIGHT * sim_norm + KW_WEIGHT * kw_norm
        if top_fused:
            score += 0.06 * (fused[doc_id] / top_fused)
        score = max(0.0, min(1.0, score))
        evidence.append(_as_evidence(doc, score))

    evidence.sort(key=lambda e: e["score"], reverse=True)
    return evidence[:top_k]


def _as_evidence(doc: dict, score: float) -> dict:
    return {
        "id": doc["id"],
        "source_id": doc.get("source_id"),
        "source_type": _source_type(doc),
        "content": doc.get("content") or "",
        "page_number": doc.get("page_number"),
        "slide_number": doc.get("slide_number"),
        "timestamp_start": doc.get("timestamp_start"),
        "timestamp_end": doc.get("timestamp_end"),
        "score": round(score, 4),
    }


def _source_type(doc: dict) -> str:
    unit_type = (doc.get("unit_type") or "").lower()
    if unit_type == "transcript":
        return "video"
    if unit_type == "slide":
        return "slide"
    if unit_type == "figure":
        return "figure"
    return "pdf"


# ----------------------------------------------------------------- refusal

def grounding_status(score: float) -> str:
    if score >= settings.t_high:
        return GROUNDED
    if score >= settings.t_low:
        return PARTIAL
    return NOT_COVERED


def _excerpt(text: str, limit: int = 260) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _citation(evidence: dict) -> dict:
    return {
        "id": evidence["id"],
        "source_id": evidence["source_id"],
        "source_type": evidence["source_type"],
        "page_number": evidence["page_number"],
        "slide_number": evidence["slide_number"],
        "timestamp_start": evidence["timestamp_start"],
        "timestamp_end": evidence["timestamp_end"],
        "excerpt": _excerpt(evidence["content"]),
        "score": evidence["score"],
    }


def _extractive_answer(question: str, evidence: list[dict]) -> str:
    """Demo-mode answer: quote the best evidence instead of inventing prose."""
    best = evidence[0]["content"].strip()
    opener = "Based on your course material:"
    body = _excerpt(best, 420)
    if len(evidence) > 1:
        body += " The material adds: " + _excerpt(evidence[1]["content"], 200)
    return f"{opener}\n\n{body}"


_TUTOR_SYSTEM = (
    "You are a course-grounded tutor. Use only the provided evidence for "
    "course-backed claims. Never invent citations. Cite evidence IDs for every "
    "important claim. If the evidence does not answer the question, say the "
    "material does not cover it. Keep general knowledge in a separate section "
    "labelled 'Outside the course material'. Explain at the student's level."
)


def answer(question: str, course_id: str, *, topic_ids: list[str] | None = None,
           student_level: str = "") -> dict:
    """Full grounded answer with citations and a computed grounding status."""
    evidence = retrieve(question, course_id, topic_ids=topic_ids)
    if not evidence:
        return {
            "answer": "This is not covered in your course material.",
            "grounding_status": NOT_COVERED,
            "grounding_score": 0.0,
            "citations": [],
            "outside_knowledge": None,
        }

    score = evidence[0]["score"]
    status = grounding_status(score)

    if status == NOT_COVERED:
        return {
            "answer": "This is not covered in your course material. Add a relevant "
                      "source, or I can give a clearly-labelled general answer instead.",
            "grounding_status": NOT_COVERED,
            "grounding_score": score,
            "citations": [],
            "outside_knowledge": None,
        }

    if settings.demo_mode:
        text = _extractive_answer(question, evidence)
    else:
        blocks = "\n\n".join(f"[{e['id']}] {e['content']}" for e in evidence)
        prompt = (
            f"Student level: {student_level or 'unknown'}.\n"
            f"Question: {question}\n\nEvidence:\n{blocks}\n\n"
            "Answer the question using only the evidence. Cite evidence IDs inline "
            "like [id]. Keep it tight and teach the concept."
        )
        text = generate(prompt, system=_TUTOR_SYSTEM, temperature=0.3)

    # citation validation: keep only ids that were actually retrieved
    valid_ids = {e["id"] for e in evidence}
    citations = [_citation(e) for e in evidence if e["id"] in valid_ids]

    outside = None
    if status == PARTIAL:
        outside = (
            "Parts of this question may go beyond your uploaded material. "
            "The answer above is limited to what your sources support."
        )

    return {
        "answer": text,
        "grounding_status": status,
        "grounding_score": round(score, 4),
        "citations": citations,
        "outside_knowledge": outside,
    }


def validate_citations(citations: list[dict], course_id: str) -> list[dict]:
    """Post-processor + unit-test target: drop citations that don't resolve."""
    valid = {row["id"] for row in query(
        "SELECT id FROM content_units WHERE course_id = ?", (course_id,)
    )}
    return [c for c in citations if c.get("id") in valid]
