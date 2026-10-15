"""TraceLearn Tutor Pipeline Orchestrator (Track D Section 2).

Connects:
1. Query understanding (language, rewriting, intent, concept mapping)
2. Retrieval (filters, figure inclusion, neighbors)
3. Support check (retrieval score margin + LLM entailment + concept match)
4. Grounded generation (strict excerpt use, outside knowledge separation)
5. Faithfulness guard (verification, tag checking, hallucination removal)
6. Response assembly & learner hook emission
"""
from __future__ import annotations

import logging
from typing import Any

from backend.learner.interface import record_interaction
from backend.tutor.generator import generate_grounded_response
from backend.tutor.query_understanding import rewrite_and_understand_query
from backend.tutor.retriever import retrieve_with_context
from backend.tutor.support_check import check_support
from backend.tutor.verifier import verify_and_guard
from tracelearn.db import execute, insert, jdumps, jloads, new_id, query

logger = logging.getLogger("tracelearn.tutor")


class TutorPipeline:
    """End-to-end tutor chat pipeline."""

    def process(
        self,
        message: str,
        *,
        session_id: str | None = None,
        student_id: str = "demo_learner",
        course_id: str = "dsa_6006",
        scope: dict[str, Any] | None = None,
        allow_outside_knowledge: bool = False,
        language: str = "auto",
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        session_id = session_id or f"sess_{new_id()[:8]}"
        history = history or self._load_session_history(session_id)

        # Step 1: Query understanding
        qu_result = rewrite_and_understand_query(message, history=history, client_language=language)
        retrieval_query = qu_result["retrieval_query"]
        reply_language = qu_result["reply_language"]
        intent = qu_result["intent"]
        query_concepts = qu_result["concept_ids"]
        query_topics = qu_result["topic_ids"]

        # Fast-path for quiz requests (Assessment module stub)
        if intent == "quiz_request":
            return self._handle_quiz_request(session_id, student_id, query_concepts)

        # Fast-path for small talk
        if intent == "small_talk":
            return self._handle_small_talk(session_id, student_id)

        # Step 2: Retrieval
        evidence = retrieve_with_context(
            retrieval_query,
            course_id=course_id,
            scope=scope,
            k=5,
        )

        # Step 3: Support Check
        support_res = check_support(
            retrieval_query,
            evidence,
            query_concepts=query_concepts,
            query_topics=query_topics,
            intent=intent,
        )
        support_level = support_res["support_level"]

        # Log refusal or partial answer to refusal_logs table
        if support_level in ("unsupported", "partial"):
            self._log_refusal(
                retrieval_query,
                course_id=course_id,
                student_id=student_id,
                session_id=session_id,
                support_signals=support_res["signals"],
                decision=support_level,
                reason="Below support threshold or missing evidence in corpus",
            )

        # Step 4: Generation
        gen_result = generate_grounded_response(
            retrieval_query,
            evidence,
            support_level,
            student_id=student_id,
            course_id=course_id,
            allow_outside_knowledge=allow_outside_knowledge,
            reply_language=reply_language,
            query_concepts=query_concepts,
            missing_points=support_res.get("entailment", {}).get("missing", []),
        )

        # Step 5: Post-generation Verification (Faithfulness guard)
        if support_level != "unsupported" and gen_result["grounded_answer"]:
            verified = verify_and_guard(
                gen_result["grounded_answer"],
                gen_result.get("source_map", {}),
                gen_result.get("citations", []),
                existing_flags=gen_result.get("flags", []),
            )
            final_grounded_answer = verified["verified_answer"]
            final_citations = verified["filtered_citations"]
            final_flags = verified["flags"]
            faithfulness_score = verified["faithfulness_score"]

            # If partial support, ensure the explicit "not covered" note is attached
            if support_level == "partial" and "not cover" not in final_grounded_answer.lower():
                not_covered_stmt = (
                    " Note: The course material does not cover textbook-specific proofs, "
                    "external benchmarks, or real-world implementation details for this topic."
                )
                final_grounded_answer = f"{final_grounded_answer}{not_covered_stmt}"

            # Re-assemble full answer if outside knowledge exists
            if gen_result.get("outside_knowledge"):
                final_answer = (
                    f"{final_grounded_answer}\n\n### Not from your course material:\n"
                    f"{gen_result['outside_knowledge']}"
                )
            else:
                final_answer = final_grounded_answer
        else:
            final_answer = gen_result["answer"]
            final_grounded_answer = gen_result["grounded_answer"]
            final_citations = []
            final_flags = gen_result.get("flags", [])
            faithfulness_score = 1.0

        top_score = support_res.get("top_score", 0.0)

        # Step 6: Learner Model Hook Emission
        learner_signals = self._detect_learner_signals(message, qu_result, support_level)
        record_interaction(
            student_id,
            {
                "type": "chat",
                "session_id": session_id,
                "concept_ids": query_concepts,
                "support_level": support_level,
                "signals": learner_signals,
            },
        )

        # Step 7: Persistence in tutor_messages
        self._persist_turn(
            session_id=session_id,
            student_id=student_id,
            user_message=message,
            assistant_answer=final_answer,
            grounded_answer=final_grounded_answer,
            outside_knowledge=gen_result.get("outside_knowledge"),
            support_level=support_level,
            citations=final_citations,
            flags=final_flags,
            metrics={"faithfulness": faithfulness_score, "top_retrieval_score": top_score},
        )

        # Backwards compatible response: contains both new contract & old fields
        response_data = {
            "session_id": session_id,
            "answer": final_answer,
            "reply": final_answer,  # old alias
            "grounded_answer": final_grounded_answer,
            "outside_knowledge": gen_result.get("outside_knowledge"),
            "support_level": support_level,
            "grounding_status": (
                "grounded" if support_level == "supported"
                else ("partial" if support_level == "partial" else "not_covered")
            ),  # old alias
            "grounding_score": top_score,  # old alias
            "citations": final_citations,
            "flags": final_flags,
            "related_concepts": gen_result.get("related_concepts", []),
            "suggested_followups": gen_result.get("suggested_followups", []),
            "metrics": {
                "faithfulness": faithfulness_score,
                "top_retrieval_score": top_score,
            },
        }

        return response_data

    def _handle_quiz_request(self, session_id: str, student_id: str, concepts: list[str]) -> dict[str, Any]:
        msg = (
            "Practice assessment module is coming soon in Stage 5! "
            "In the meantime, feel free to ask me to explain any concept or trace an algorithm."
        )
        return {
            "session_id": session_id,
            "answer": msg,
            "reply": msg,
            "grounded_answer": "",
            "outside_knowledge": None,
            "support_level": "supported",
            "grounding_status": "grounded",
            "grounding_score": 1.0,
            "citations": [],
            "flags": [],
            "related_concepts": [],
            "suggested_followups": ["Explain binary heap insert", "How does Dijkstra work?"],
            "metrics": {"faithfulness": 1.0, "top_retrieval_score": 1.0},
        }

    def _handle_small_talk(self, session_id: str, student_id: str) -> dict[str, Any]:
        msg = "Hello! I am your TraceLearn tutor for this course. What concept or problem can I help you study today?"
        return {
            "session_id": session_id,
            "answer": msg,
            "reply": msg,
            "grounded_answer": "",
            "outside_knowledge": None,
            "support_level": "supported",
            "grounding_status": "grounded",
            "grounding_score": 1.0,
            "citations": [],
            "flags": [],
            "related_concepts": [],
            "suggested_followups": ["What is a binary heap?", "Explain hash table open addressing"],
            "metrics": {"faithfulness": 1.0, "top_retrieval_score": 1.0},
        }

    def _detect_learner_signals(self, message: str, qu: dict, support_level: str) -> list[str]:
        signals = []
        lower = message.lower()
        if any(c in lower for c in ["don't understand", "dont understand", "confused", "unclear", "samajh nahi"]):
            signals.append("expressed_confusion")
        if qu.get("is_followup"):
            signals.append("asked_clarification")
        if any(r in lower for r in ["so basically", "in other words", "does that mean", "yani"]):
            signals.append("correct_restatement")
        return signals

    def _load_session_history(self, session_id: str) -> list[dict[str, str]]:
        rows = query(
            "SELECT role, content FROM tutor_messages WHERE session_id = ? ORDER BY created_at ASC LIMIT 10",
            (session_id,),
        )
        return [{"role": r["role"], "content": r["content"]} for r in rows]

    def _persist_turn(
        self,
        session_id: str,
        student_id: str,
        user_message: str,
        assistant_answer: str,
        grounded_answer: str,
        outside_knowledge: str | None,
        support_level: str,
        citations: list[dict],
        flags: list[dict],
        metrics: dict,
    ) -> None:
        try:
            # User message
            insert("tutor_messages", {
                "session_id": session_id,
                "student_id": student_id,
                "role": "user",
                "content": user_message,
                "support_level": None,
                "citations": None,
                "flags": None,
                "metrics": None,
            })
            # Assistant message
            insert("tutor_messages", {
                "session_id": session_id,
                "student_id": student_id,
                "role": "assistant",
                "content": assistant_answer,
                "grounded_answer": grounded_answer,
                "outside_knowledge": outside_knowledge,
                "support_level": support_level,
                "citations": jdumps(citations),
                "flags": jdumps(flags),
                "metrics": jdumps(metrics),
            })
        except Exception as e:
            logger.warning("Failed to persist tutor turn: %s", e)

    def _log_refusal(
        self,
        query_text: str,
        course_id: str,
        student_id: str,
        session_id: str,
        support_signals: dict,
        decision: str,
        reason: str,
    ) -> None:
        try:
            insert("refusal_logs", {
                "query": query_text,
                "course_id": course_id,
                "student_id": student_id,
                "session_id": session_id,
                "support_signals": jdumps(support_signals),
                "decision": decision,
                "reason": reason,
            })
        except Exception as e:
            logger.warning("Failed to log refusal: %s", e)


pipeline = TutorPipeline()
