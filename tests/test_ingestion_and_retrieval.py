"""Unit and Integration Tests for Track D Stage 2:
Multimodal Ingestion, Knowledge Base, and Retrieval Layer.
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from backend.ingestion import (
    extract_audio,
    extract_pdf,
    extract_pptx,
    orchestrator,
)
from backend.ingestion.topic_tagger import (
    build_prerequisites_pass3,
    extract_candidates_pass1,
    merge_and_cluster_pass2,
    tag_units_pass4,
)
from tracelearn import create_app, rag
from tracelearn.db import query, query_one
from tracelearn.graph import _topological

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="module")
def app():
    app = create_app()
    app.config["TESTING"] = True
    yield app


@pytest.fixture(scope="module")
def client(app):
    return app.test_client()


# ------------------------------------------------------------- 1. Ingestor tests

def test_pdf_ingestor():
    pdf_path = FIXTURES_DIR / "sample_3page.pdf"
    assert pdf_path.exists(), "PDF fixture missing"

    units, figures = extract_pdf(pdf_path, source_id="test_pdf_01")
    assert len(units) >= 3, f"Expected at least 3 units, got {len(units)}"

    # Assert every unit has a non-null deep link and valid page number
    pages_seen = set()
    for u in units:
        assert u.get("deep_link") is not None, f"Unit {u} missing deep_link"
        assert u.get("deep_link").startswith("/viewer?source=test_pdf_01&page="), "Invalid deep link schema"
        page = u.get("page_number") or u.get("page")
        assert page is not None and 1 <= page <= 3, f"Invalid page number {page}"
        pages_seen.add(page)

    assert {1, 2, 3}.issubset(pages_seen), f"Expected pages 1, 2, 3 to be extracted, got {pages_seen}"


def test_pptx_ingestor():
    pptx_path = FIXTURES_DIR / "sample_3slide.pptx"
    assert pptx_path.exists(), "PPTX fixture missing"

    units, figures = extract_pptx(pptx_path, source_id="test_pptx_01")
    assert len(units) >= 3, f"Expected at least 3 units, got {len(units)}"

    # Assert every unit has non-null deep link and valid slide number
    slides_seen = set()
    for u in units:
        assert u.get("deep_link") is not None, f"Unit {u} missing deep_link"
        assert u.get("deep_link").startswith("/viewer?source=test_pptx_01&slide="), "Invalid slide deep link"
        slide = u.get("slide_number") or u.get("slide_no")
        assert slide is not None and 1 <= slide <= 3, f"Invalid slide number {slide}"
        slides_seen.add(slide)

    assert {1, 2, 3} == slides_seen, f"Expected slides 1, 2, 3, got {slides_seen}"


def test_audio_ingestor():
    audio_path = FIXTURES_DIR / "sample_30s.wav"
    assert audio_path.exists(), "Audio fixture missing"

    units, figures = extract_audio(audio_path, source_id="test_audio_01")
    assert len(units) >= 1, "Expected at least 1 audio transcript unit"

    for u in units:
        assert u.get("deep_link") is not None, f"Unit {u} missing deep_link"
        assert u.get("deep_link").startswith("/viewer?source=test_audio_01&t="), "Invalid audio deep link"
        t_start = u.get("timestamp_start") or u.get("start_ts")
        assert t_start is not None and t_start >= 0.0, "Missing timestamp on audio unit"


# ------------------------------------------------------------- 2. Pipeline & Graph Assertions

def test_100_percent_tagging_coverage(app):
    with app.app_context():
        units = query("SELECT id, topic_id, topic_ids, concept_ids FROM content_units")
        assert len(units) > 0, "No units found in database"

        untagged = []
        for u in units:
            has_topic = bool(u.get("topic_id") or json.loads(u.get("topic_ids") or "[]"))
            has_concept = bool(json.loads(u.get("concept_ids") or "[]"))
            if not (has_topic and has_concept):
                untagged.append(u["id"])

        coverage = ((len(units) - len(untagged)) / len(units)) * 100.0
        assert coverage == 100.0, f"Expected 100% tagging coverage, found untagged units: {untagged}"


def test_prerequisite_graph_no_cycles(app):
    with app.app_context():
        concepts = query("SELECT id FROM concepts")
        known_nodes = {c["id"] for c in concepts}
        edges = query("SELECT prerequisite_id, dependent_id, confidence FROM prerequisites")

        clean_edges = [
            {"source": e["prerequisite_id"], "target": e["dependent_id"], "confidence": e["confidence"]}
            for e in edges
            if e["prerequisite_id"] in known_nodes and e["dependent_id"] in known_nodes
        ]

        topological_order = _topological(known_nodes, clean_edges)
        assert topological_order is not None, "Prerequisite graph contains cycles!"


def test_orchestrator_idempotency(app):
    with app.app_context():
        pdf_path = FIXTURES_DIR / "sample_3page.pdf"
        res1 = orchestrator.ingest(pdf_path, title="Idempotent Test Course", async_run=False)
        assert "source_id" in res1, "First ingestion failed"

        initial_count = query_one(
            "SELECT COUNT(*) as n FROM content_units WHERE source_id = ?",
            (res1["source_id"],),
        )["n"]

        # Ingest the same file path again
        res2 = orchestrator.ingest(pdf_path, title="Idempotent Test Course", async_run=False)
        assert res2["source_id"] == res1["source_id"], "Expected same source_id for identical file"

        final_count = query_one(
            "SELECT COUNT(*) as n FROM content_units WHERE source_id = ?",
            (res1["source_id"],),
        )["n"]
        assert initial_count == final_count, "Re-ingestion duplicated units!"


# ------------------------------------------------------------- 3. Retrieval Sanity Set (Recall@5 & MRR)

def test_retrieval_sanity_recall_and_mrr(app):
    """Run retrieval against the 10 grounded test queries from eval/test_set.json.

    Reports Recall@5 and Mean Reciprocal Rank (MRR).
    """
    test_set_path = Path("eval/test_set.json")
    assert test_set_path.exists(), "eval/test_set.json not found"

    with test_set_path.open("r", encoding="utf-8") as f:
        all_cases = json.load(f)

    grounded_queries = [c for c in all_cases if c.get("category") == "grounded"][:10]
    assert len(grounded_queries) >= 10, f"Expected at least 10 grounded test queries, got {len(grounded_queries)}"

    reciprocal_ranks = []
    hits_at_5 = 0

    with app.app_context():
        for case in grounded_queries:
            q = case["question"]
            expected = set(case.get("expected_units") or [])

            results = rag.retrieve(q, course_id="dsa_6006", top_k=5)
            hit_ids = [r["id"] for r in results]

            # Check if any expected unit is in top 5
            found_rank = None
            for rank_idx, doc_id in enumerate(hit_ids, start=1):
                if doc_id in expected:
                    found_rank = rank_idx
                    break

            if found_rank is not None:
                hits_at_5 += 1
                reciprocal_ranks.append(1.0 / found_rank)
            else:
                reciprocal_ranks.append(0.0)

    recall_at_5 = (hits_at_5 / len(grounded_queries)) * 100.0
    mrr = sum(reciprocal_ranks) / len(grounded_queries)

    print(f"\n================ RETRIEVAL SANITY RESULTS ================")
    print(f"Total test queries: {len(grounded_queries)}")
    print(f"Recall@5:           {recall_at_5:.1f}% ({hits_at_5}/{len(grounded_queries)})")
    print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
    print(f"==========================================================\n")

    assert recall_at_5 >= 70.0, f"Recall@5 was too low: {recall_at_5}%"
    assert mrr >= 0.50, f"MRR was too low: {mrr}"


# ------------------------------------------------------------- 4. API Endpoints

def test_api_sources_and_unit(client):
    r = client.get("/sources")
    assert r.status_code == 200
    sources = r.json.get("sources", [])
    assert len(sources) > 0

    r = client.get("/source/u_heap_1")
    assert r.status_code == 200
    data = r.json
    assert data["deep_link"] == "/viewer?source=ocw_6006_lec08&slide=12"
    assert "neighbors" in data


def test_api_knowledge_graph_and_debug(client):
    r = client.get("/knowledge/graph")
    assert r.status_code == 200
    assert "topics" in r.json
    assert "concepts" in r.json
    assert "prerequisite_edges" in r.json
    assert "unit_counts_per_node" in r.json

    r = client.post("/retrieve/debug", json={"query": "binary heap array children index", "k": 3})
    assert r.status_code == 200
    debug_data = r.json
    assert len(debug_data.get("hits", [])) == 3
    assert "support_score" in debug_data
    assert "grounding_status" in debug_data
