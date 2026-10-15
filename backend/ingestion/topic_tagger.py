"""Topic, Concept, and Prerequisite Extractor and Tagging Pipeline (Section 2.5).

Implements:
  - Pass 1 (per source): Pydantic-validated extraction of topics, subtopics, concepts.
  - Pass 2 (global merge): Deduplication & clustering into canonical hierarchy.
  - Pass 3 (prerequisites): DAG proposal with cycle detection & weakest-edge removal.
  - Pass 4 (tagging): Tag every unit with 1 topic_id and 1-5 concept_ids with 100% coverage.
"""
from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field

from tracelearn.config import settings
from tracelearn.embeddings import cosine, embed, embed_batch
from tracelearn.llm import generate


# ------------------------------------------------------------- Pydantic schemas

class ConceptDef(BaseModel):
    name: str = Field(description="Name of the concept")
    definition: str = Field(description="One-line definition of the concept")
    topic_name: str = Field(description="Parent topic or subtopic name")


class TopicDef(BaseModel):
    name: str = Field(description="Topic name")
    summary: str = Field(default="", description="Summary of the topic")
    subtopics: list[str] = Field(default_factory=list, description="List of subtopic names")


class PrereqDef(BaseModel):
    concept: str = Field(description="Dependent concept that requires prior knowledge")
    requires: str = Field(description="Required prerequisite concept")
    confidence: float = Field(default=0.8, description="Confidence score between 0.0 and 1.0")
    reason: str = Field(default="", description="One-line justification")


class SourceExtraction(BaseModel):
    topics: list[TopicDef] = Field(default_factory=list)
    concepts: list[ConceptDef] = Field(default_factory=list)
    prerequisites: list[PrereqDef] = Field(default_factory=list)


# ----------------------------------------------------------------------- Pass 1

def extract_candidates_pass1(
    source_title: str,
    units: list[dict],
) -> SourceExtraction:
    """Pass 1: Extract candidate topics, subtopics, and concepts with definitions."""
    if not units:
        return SourceExtraction()

    # Sample representative excerpts across the document
    sample_texts = [u.get("content", "") for u in units if u.get("content")]
    if len(sample_texts) > 10:
        step = max(1, len(sample_texts) // 10)
        sample_texts = sample_texts[::step][:10]

    joined_text = "\n---\n".join(sample_texts[:8])[:3000]

    prompt = f"""You are a curriculum analyst. Extract the topics, subtopics, key concepts, and prerequisite relationships from this course material:

Title: {source_title}
Material excerpts:
{joined_text}

Return a valid JSON object matching this schema:
{{
  "topics": [
    {{"name": "Topic Name", "summary": "Short summary", "subtopics": ["Subtopic 1", "Subtopic 2"]}}
  ],
  "concepts": [
    {{"name": "Concept Name", "definition": "One-line definition", "topic_name": "Topic Name"}}
  ],
  "prerequisites": [
    {{"concept": "Concept A", "requires": "Concept B", "confidence": 0.85, "reason": "Reason why A requires B"}}
  ]
}}
Only return the JSON object, nothing else.
"""

    if settings.llm_ready:
        for _ in range(2):  # retry on invalid json
            try:
                raw = generate(prompt, temperature=0.1)
                clean_json = _extract_json_block(raw)
                data = json.loads(clean_json)
                return SourceExtraction.model_validate(data)
            except Exception:
                pass

    # Heuristic fallback in demo mode or when LLM is unavailable
    return _heuristic_candidates(source_title, units)


def _heuristic_candidates(source_title: str, units: list[dict]) -> SourceExtraction:
    title_clean = re.sub(r"[_\-.]+", " ", source_title).title()
    topics = [
        TopicDef(name=title_clean, summary=f"Key concepts from {source_title}", subtopics=["Fundamentals", "Applications"])
    ]
    # Extract noun phrases or capitalized terms from the first few units
    concepts: list[ConceptDef] = []
    seen_concepts = set()

    terms = [
        ("Binary Heap", "A complete binary tree with heap-order property stored in an array."),
        ("Heapify", "Process of maintaining heap property by sifting elements down or up."),
        ("Merge Sort", "Divide-and-conquer sorting algorithm with O(n log n) runtime."),
        ("Hashing", "Mapping keys to slots in a table using a hash function."),
        ("Open Addressing", "Collision resolution where all elements are stored directly in hash table."),
        ("Breadth-First Search", "Graph traversal layer by layer to find shortest unweighted paths."),
        ("Dijkstra's Algorithm", "Single-source shortest path algorithm on non-negative weighted graphs."),
        ("Dynamic Programming", "Solving problems with optimal substructure and overlapping subproblems."),
    ]
    for name, definition in terms:
        for u in units:
            if name.lower() in (u.get("content") or "").lower():
                if name not in seen_concepts:
                    seen_concepts.add(name)
                    concepts.append(ConceptDef(name=name, definition=definition, topic_name=title_clean))
                break

    if not concepts:
        concepts.append(ConceptDef(name=f"{title_clean} Core", definition=f"Core concept of {title_clean}", topic_name=title_clean))

    prereqs: list[PrereqDef] = []
    if len(concepts) >= 2:
        prereqs.append(PrereqDef(
            concept=concepts[1].name,
            requires=concepts[0].name,
            confidence=0.8,
            reason="Foundational concept needed first"
        ))

    return SourceExtraction(topics=topics, concepts=concepts, prerequisites=prereqs)


# ----------------------------------------------------------------------- Pass 2

def merge_and_cluster_pass2(
    extractions: list[SourceExtraction],
    existing_topics: list[dict],
    existing_concepts: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Pass 2: Global merge and deduplication of topics and concepts."""
    canonical_topics: dict[str, dict] = {t["name"].lower(): t for t in existing_topics}
    canonical_concepts: dict[str, dict] = {c["name"].lower(): c for c in existing_concepts}

    for ext in extractions:
        for t in ext.topics:
            t_key = t.name.lower().strip()
            if t_key not in canonical_topics:
                t_id = f"t_{re.sub(r'[^a-z0-9]+', '_', t_key).strip('_')}"
                canonical_topics[t_key] = {
                    "id": t_id,
                    "topic_id": t_id,
                    "name": t.name,
                    "level": "topic",
                    "summary": t.summary,
                    "order_index": len(canonical_topics),
                }

        for c in ext.concepts:
            c_key = c.name.lower().strip()
            if c_key not in canonical_concepts:
                # Find matching parent topic
                parent_topic_id = None
                p_key = c.topic_name.lower().strip()
                if p_key in canonical_topics:
                    parent_topic_id = canonical_topics[p_key]["id"]
                elif canonical_topics:
                    parent_topic_id = next(iter(canonical_topics.values()))["id"]

                c_id = f"c_{re.sub(r'[^a-z0-9]+', '_', c_key).strip('_')}"
                canonical_concepts[c_key] = {
                    "id": c_id,
                    "concept_id": c_id,
                    "name": c.name,
                    "definition": c.definition,
                    "description": c.definition,
                    "topic_id": parent_topic_id,
                    "level": "concept",
                    "order_index": len(canonical_concepts),
                }

    return list(canonical_topics.values()), list(canonical_concepts.values())


# ----------------------------------------------------------------------- Pass 3

def build_prerequisites_pass3(
    extractions: list[SourceExtraction],
    canonical_concepts: list[dict],
    confidence_threshold: float = 0.5,
) -> list[dict]:
    """Pass 3: Build prerequisite DAG, filter by confidence, and break cycles."""
    concept_map = {c["name"].lower(): c["id"] for c in canonical_concepts}
    edges: list[dict] = []

    for ext in extractions:
        for p in ext.prerequisites:
            if p.confidence < confidence_threshold:
                continue
            src = concept_map.get(p.requires.lower())
            dst = concept_map.get(p.concept.lower())
            if src and dst and src != dst:
                edges.append({
                    "prerequisite_id": src,
                    "dependent_id": dst,
                    "requires_concept_id": src,
                    "concept_id": dst,
                    "confidence": p.confidence,
                    "reason": p.reason,
                })

    # Break any cycles by dropping weakest edge
    return _break_cycles_dag(edges)


def _break_cycles_dag(edges: list[dict]) -> list[dict]:
    """Greedy cycle repair on prerequisite edges."""
    kept = list(edges)
    all_nodes = {e["prerequisite_id"] for e in kept} | {e["dependent_id"] for e in kept}

    for _ in range(len(kept)):
        # Check if DAG via Kahn's algorithm
        in_degree = {n: 0 for n in all_nodes}
        adj = {n: [] for n in all_nodes}
        for e in kept:
            adj[e["prerequisite_id"]].append(e["dependent_id"])
            in_degree[e["dependent_id"]] += 1

        queue = [n for n, d in in_degree.items() if d == 0]
        visited = 0
        while queue:
            cur = queue.pop(0)
            visited += 1
            for nxt in adj[cur]:
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)

        if visited == len(all_nodes):
            return kept  # valid DAG

        # Find a cycle node and remove weakest edge in the cycle
        cycle_nodes = {n for n, d in in_degree.items() if d > 0}
        candidates = [e for e in kept if e["prerequisite_id"] in cycle_nodes and e["dependent_id"] in cycle_nodes]
        if not candidates:
            return kept
        weakest = min(candidates, key=lambda e: e.get("confidence", 0.5))
        kept.remove(weakest)

    return kept


# ----------------------------------------------------------------------- Pass 4

def tag_units_pass4(
    units: list[dict],
    canonical_topics: list[dict],
    canonical_concepts: list[dict],
) -> tuple[list[dict], dict]:
    """Pass 4: Tag every unit with exactly 1 topic_id and 1-5 concept_ids.

    Returns:
        (tagged_units, coverage_report)
    """
    if not canonical_topics:
        canonical_topics = [{"id": "t_general", "name": "General"}]
    if not canonical_concepts:
        canonical_concepts = [{"id": "c_general", "name": "General Concept", "topic_id": canonical_topics[0]["id"]}]

    concept_texts = [f"{c['name']}: {c.get('definition', '')}" for c in canonical_concepts]
    topic_texts = [t["name"] for t in canonical_topics]

    # Pre-embed concepts and topics
    concept_vecs = embed_batch(concept_texts)
    topic_vecs = embed_batch(topic_texts)

    tagged_units: list[dict] = []
    low_confidence_count = 0
    flagged_units: list[str] = []

    for idx, u in enumerate(units):
        content = u.get("content", "")
        u_vec = embed(content[:400])

        # 1. Match primary topic
        topic_sims = [cosine(u_vec, tv) for tv in topic_vecs]
        best_t_idx = int(max(range(len(topic_sims)), key=lambda i: topic_sims[i]))
        best_topic = canonical_topics[best_t_idx]
        topic_id = best_topic["id"]

        # 2. Match 1-5 concepts
        concept_sims = [(cosine(u_vec, cv), c) for cv, c in zip(concept_vecs, canonical_concepts)]
        concept_sims.sort(key=lambda x: x[0], reverse=True)

        # Select top concepts with positive similarity
        top_concepts = [c["id"] for score, c in concept_sims[:3] if score > 0.15]
        if not top_concepts:
            top_concepts = [concept_sims[0][1]["id"]]

        top_score = concept_sims[0][0]
        if top_score < 0.25:
            low_confidence_count += 1
            flagged_units.append(u.get("id", f"unit_{idx}"))

        u["topic_id"] = topic_id
        u["topic_ids"] = json.dumps([topic_id])
        u["concept_ids"] = json.dumps(top_concepts)
        tagged_units.append(u)

    total_units = len(units)
    coverage_pct = 100.0 if total_units > 0 else 0.0

    coverage_report = {
        "total_units": total_units,
        "tagged_units": total_units,
        "coverage_pct": coverage_pct,
        "low_confidence_count": low_confidence_count,
        "flagged_units": flagged_units,
    }

    return tagged_units, coverage_report


# ------------------------------------------------------------- internal helpers

def _extract_json_block(text: str) -> str:
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        return m.group(1).strip()
    return text.strip()
