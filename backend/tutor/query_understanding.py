"""Query understanding step for TraceLearn Tutor.

Responsibilities:
1. Detect language (English / Hindi / Hinglish). If non-English, translate to English for retrieval
   while preserving the target reply language.
2. Rewrite follow-up questions into standalone queries using the last few conversation turns.
3. Classify intent: concept_explanation | factual_qa | problem_help | quiz_request | small_talk | off_topic.
4. Map query to topic_ids / concept_ids via knowledge graph and keyword index.
"""
from __future__ import annotations

import re
from typing import Any

from tracelearn.config import settings
from tracelearn.db import jloads, query
from tracelearn.llm import generate_json


# Common Hindi / Hinglish markers
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_HINGLISH_WORDS = {
    "kya", "hai", "kaise", "samjhao", "batao", "hota", "hoti", "hote", "kyu", "kyun",
    "karo", "karna", "iska", "iski", "iske", "nahi", "nahin", "aur", "ye", "yeh", "woh",
    "bhi", "tha", "thi", "the", "kuch", "apne", "hum", "tum", "mera", "meri"
}

_GREETINGS = {
    "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
    "how are you", "what's up", "who are you", "what can you do", "namaste"
}


def detect_language(text: str) -> str:
    """Detect whether text is 'en', 'hi', or 'hinglish'."""
    if not text:
        return "en"
    if _DEVANAGARI_RE.search(text):
        return "hi"
    words = re.findall(r"[a-zA-Z]+", text.lower())
    if not words:
        return "en"
    hinglish_count = sum(1 for w in words if w in _HINGLISH_WORDS)
    if hinglish_count >= 2 or (len(words) <= 4 and hinglish_count >= 1):
        return "hinglish"
    return "en"


def rewrite_and_understand_query(
    message: str,
    history: list[dict[str, str]] | None = None,
    client_language: str = "auto",
) -> dict[str, Any]:
    """Parse, detect language, rewrite follow-ups, classify intent, and map concepts."""
    raw_message = (message or "").strip()
    history = history or []

    # 1. Language detection
    detected_lang = detect_language(raw_message)
    reply_language = detected_lang if client_language in ("auto", None, "") else client_language

    # Translate if non-English
    retrieval_query = raw_message
    if reply_language in ("hi", "hinglish") and settings.enable_multilingual:
        # If in demo mode without live LLM, translate common patterns or use english words
        if settings.demo_mode:
            # Strip hinglish stop words to leave core english concepts
            words = [w for w in re.findall(r"[a-zA-Z0-9]+", raw_message) if w.lower() not in _HINGLISH_WORDS]
            if words:
                retrieval_query = " ".join(words)
        else:
            prompt = (
                f"Translate the following student question into a clear, concise English search query for computer science retrieval:\n"
                f"Question: {raw_message}\n"
                f'Return JSON: {{"english_query": "..."}}'
            )
            res = generate_json(prompt, mock={"english_query": raw_message})
            retrieval_query = res.get("english_query") or raw_message

    # 2. Standalone rewrite for follow-up questions
    is_followup = False
    lower_q = raw_message.lower().strip()
    followup_patterns = [
        "again", "simply", "more detail", "what about", "how about", "why is that",
        "can you explain", "elaborate", "give an example", "i don't get it",
        "explain it again", "samjhao fir se", "aur batao"
    ]
    if history and (len(lower_q.split()) < 6 or any(p in lower_q for p in followup_patterns)):
        is_followup = True
        # Extract last relevant user question and assistant context
        last_turns = history[-4:]
        last_context = "\n".join(f"{t.get('role', 'user')}: {t.get('content', '')}" for t in last_turns)
        if settings.demo_mode:
            # Heuristic rewrite: combine recent user message with current query
            prev_user_queries = [t.get("content", "") for t in last_turns if t.get("role") == "user"]
            if prev_user_queries:
                retrieval_query = f"{prev_user_queries[-1]} {retrieval_query}"
        else:
            prompt = (
                f"Given the conversation turns:\n{last_context}\n\n"
                f"Rewrite the student's follow-up question: '{retrieval_query}' into a complete, standalone question.\n"
                f'Return JSON: {{"standalone_query": "..."}}'
            )
            res = generate_json(prompt, mock={"standalone_query": retrieval_query})
            retrieval_query = res.get("standalone_query") or retrieval_query

    # 3. Classify intent
    intent = _classify_intent(raw_message, lower_q)

    # 4. Map to topic_ids and concept_ids from knowledge graph
    matched_topics, matched_concepts = _match_concepts(retrieval_query)

    return {
        "raw_message": raw_message,
        "retrieval_query": retrieval_query,
        "reply_language": reply_language,
        "intent": intent,
        "is_followup": is_followup,
        "topic_ids": matched_topics,
        "concept_ids": matched_concepts,
    }


def _classify_intent(raw: str, lower: str) -> str:
    # Quiz request
    if any(k in lower for k in ["quiz", "test me", "practice question", "give me a problem", "mcq"]):
        return "quiz_request"

    # Small talk
    cleaned = re.sub(r"[^\w\s]", "", lower).strip()
    if cleaned in _GREETINGS or lower.startswith(("hi ", "hello ", "hey ")):
        return "small_talk"

    # Obvious off-topic queries
    off_topic_keywords = [
        "recipe", "bake", "sourdough", "capital city", "fifa", "world cup",
        "president", "ibuprofen", "medicine", "combustion cycle", "photosynthesis"
    ]
    if any(k in lower for k in off_topic_keywords):
        return "off_topic"

    # Problem help
    if any(k in lower for k in ["how to solve", "step by step", "algorithm for", "trace", "pseudo"]):
        return "problem_help"

    # Concept explanation vs factual QA
    if any(k in lower for k in ["why", "explain", "how does", "what is the intuition", "concept"]):
        return "concept_explanation"

    return "factual_qa"


def _match_concepts(query_text: str) -> tuple[list[str], list[str]]:
    """Match tokens against topics and concepts table."""
    q_lower = query_text.lower()
    matched_topics: set[str] = set()
    matched_concepts: set[str] = set()

    # Query topics
    try:
        topics = query("SELECT id, name FROM topics")
        for t in topics:
            name = (t.get("name") or "").lower()
            if name and (name in q_lower or any(word in q_lower for word in name.split() if len(word) > 4)):
                matched_topics.add(t["id"])
    except Exception:
        pass

    # Query concepts
    try:
        concepts = query("SELECT id, name, topic_id FROM concepts")
        for c in concepts:
            name = (c.get("name") or "").lower()
            if name and (name in q_lower or any(word in q_lower for word in name.split() if len(word) > 4)):
                matched_concepts.add(c["id"])
                if c.get("topic_id"):
                    matched_topics.add(c["topic_id"])
    except Exception:
        pass

    return list(matched_topics), list(matched_concepts)
