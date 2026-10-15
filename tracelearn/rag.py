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


try:
    from rank_bm25 import BM25Okapi
    HAVE_RANK_BM25 = True
except ImportError:
    HAVE_RANK_BM25 = False


# --------------------------------------------------------------- retrieval

def _bm25_scores(question: str, docs: list[dict]) -> dict[str, float]:
    """Rank BM25 scores over units using rank_bm25 or built-in formula."""
    q_terms = [t.lower() for t in _WORD.findall(question)]
    if not q_terms or not docs:
        return {}
    
    if HAVE_RANK_BM25:
        tokenized_corpus = [[t.lower() for t in _WORD.findall(d.get("content", ""))] for d in docs]
        bm25 = BM25Okapi(tokenized_corpus)
        doc_scores = bm25.get_scores(q_terms)
        return {doc["id"]: float(sc) for doc, sc in zip(docs, doc_scores)}

    # Fallback BM25
    doc_tokens = [_WORD.findall((d["content"] or "").lower()) for d in docs]
    n = len(docs)
    avg_len = sum(len(t) for t in doc_tokens) / n or 1.0
    q_set = set(q_terms)
    df: dict[str, int] = {}
    for toks in doc_tokens:
        for term in q_set & set(toks):
            df[term] = df.get(term, 0) + 1

    k1, b = 1.2, 0.75
    scores: dict[str, float] = {}
    for doc, toks in zip(docs, doc_tokens):
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        score = 0.0
        for term in q_set:
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


def retrieve(
    question: str,
    course_id: str = "dsa_6006",
    *,
    topic_ids: list[str] | None = None,
    top_k: int = 5,
    filters: dict | None = None,
    k: int | None = None,
) -> list[dict]:
    """Return the top evidence units for a question with optional filters and reranking."""
    if k is not None:
        top_k = k

    docs = query(
        "SELECT * FROM content_units WHERE course_id = ?",
        (course_id,),
    )
    if not docs:
        return []

    # Apply dynamic filters
    fil = filters or {}
    if topic_ids:
        wanted_topics = set(topic_ids)
        docs = [d for d in docs if wanted_topics & set(jloads(d.get("topic_ids"), []) or [])] or docs
    if fil.get("source_id"):
        docs = [d for d in docs if d.get("source_id") == fil["source_id"]]
    if fil.get("topic_id"):
        t_target = fil["topic_id"]
        docs = [d for d in docs if d.get("topic_id") == t_target or t_target in (jloads(d.get("topic_ids"), []) or [])]
    if fil.get("concept_id"):
        c_target = fil["concept_id"]
        docs = [d for d in docs if c_target in (jloads(d.get("concept_ids"), []) or [])]
    if fil.get("modality"):
        m_target = fil["modality"].lower()
        docs = [d for d in docs if (d.get("modality") or _source_type(d)).lower() == m_target]
    if fil.get("page_range"):
        p_min, p_max = fil["page_range"]
        docs = [d for d in docs if d.get("page_number") is not None and p_min <= d["page_number"] <= p_max]
    if fil.get("time_range"):
        t_min, t_max = fil["time_range"]
        docs = [d for d in docs if d.get("timestamp_start") is not None and t_min <= d["timestamp_start"] <= t_max]

    if not docs:
        return []

    # --- 1/2. vector search (SQLite scores in Python; corpus is small) ---
    vectors = dict(load_unit_embeddings([d["id"] for d in docs]))
    q_vec = embed(question)
    sims = {d["id"]: cosine(q_vec, vectors.get(d["id"])) for d in docs}
    vector_order = sorted(sims, key=lambda i: sims[i], reverse=True)[:30]

    # --- 3. keyword search (BM25) ---
    kw = _bm25_scores(question, docs)
    kw_order = sorted(kw, key=lambda i: kw[i], reverse=True)[:30]

    # --- 4. reciprocal rank fusion (RRF) ---
    fused = _rrf([vector_order, kw_order])
    ranked = sorted(fused, key=lambda i: fused[i], reverse=True)[: max(top_k * 3, top_k)]

    by_id = {d["id"]: d for d in docs}
    top_fused = fused[ranked[0]] if ranked else 0.0

    evidence = []
    for doc_id in ranked:
        doc = by_id[doc_id]
        sim_norm = min(1.0, max(0.0, sims.get(doc_id, 0.0)) / SIM_REF)
        kw_norm = min(1.0, max(0.0, kw.get(doc_id, 0.0)) / KW_REF)
        score = SIM_WEIGHT * sim_norm + KW_WEIGHT * kw_norm
        if top_fused:
            score += 0.06 * (fused[doc_id] / top_fused)
        score = max(0.0, min(1.0, score))
        evidence.append(_as_evidence(doc, score))

    evidence.sort(key=lambda e: e["score"], reverse=True)
    return evidence[:top_k]


def compute_support_score(hits: list[dict]) -> dict:
    """Compute support score (top score + score margin) for refusal decisions."""
    if not hits:
        return {"support_score": 0.0, "top_score": 0.0, "score_margin": 0.0, "grounding_status": NOT_COVERED}
    top_score = hits[0].get("score", 0.0)
    second_score = hits[1].get("score", 0.0) if len(hits) > 1 else 0.0
    margin = round(top_score - second_score, 4)
    support_score = round(top_score + 0.5 * margin, 4)
    status = grounding_status(top_score)
    return {
        "support_score": support_score,
        "top_score": top_score,
        "score_margin": margin,
        "grounding_status": status,
    }


def retrieve_debug(
    query: str,
    filters: dict | None = None,
    k: int = 5,
    course_id: str = "dsa_6006",
) -> dict:
    """Debug endpoint representation with hit details and support score."""
    hits = retrieve(query, course_id, filters=filters, top_k=k)
    support = compute_support_score(hits)
    return {
        "query": query,
        "filters": filters or {},
        "k": k,
        "hits": hits,
        "support": support,
        "support_score": support["support_score"],
        "grounding_status": support["grounding_status"],
    }


def _as_evidence(doc: dict, score: float) -> dict:
    source_type = _source_type(doc)
    modality = doc.get("modality") or source_type
    sid = doc.get("source_id", "")
    pnum = doc.get("page_number")
    snum = doc.get("slide_number")
    tstart = doc.get("timestamp_start")

    # Construct deep_link if missing
    deep_link = doc.get("deep_link")
    if not deep_link:
        if pnum is not None:
            deep_link = f"/viewer?source={sid}&page={pnum}"
        elif snum is not None:
            deep_link = f"/viewer?source={sid}&slide={snum}"
        elif tstart is not None:
            deep_link = f"/viewer?source={sid}&t={round(float(tstart), 2)}"
        else:
            deep_link = f"/viewer?source={sid}"

    location = {
        "page": pnum,
        "slide": snum,
        "timestamp_start": tstart,
        "timestamp_end": doc.get("timestamp_end"),
    }

    return {
        "id": doc["id"],
        "unit_id": doc["id"],
        "source_id": sid,
        "source_type": source_type,
        "modality": modality,
        "content": doc.get("content") or "",
        "text": doc.get("content") or "",
        "page_number": pnum,
        "slide_number": snum,
        "timestamp_start": tstart,
        "timestamp_end": doc.get("timestamp_end"),
        "location": location,
        "deep_link": deep_link,
        "topic_id": doc.get("topic_id"),
        "topic_tags": jloads(doc.get("topic_ids"), []) or [],
        "concept_tags": jloads(doc.get("concept_ids"), []) or [],
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
