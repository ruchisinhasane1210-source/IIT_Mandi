"""TraceLearn — assessment engine.

Generates an adaptive quiz (§12.5), serves questions without leaking answer
keys, grades submissions, applies BKT updates and returns a cited report.
"""
from __future__ import annotations

from .db import execute, insert, jdumps, jloads, new_id, query, query_one
from .learner import GUESS_MCQ, bkt_update, guess_for, mastery_map, recommend

LEVEL_OF_DIFFICULTY = 3


def _ordered_topics(course_id: str, user_id: str) -> list[str]:
    """Weakest topics first, weighted by prerequisite readiness."""
    state = mastery_map(course_id, user_id)
    if not state:
        return []
    return sorted(state, key=lambda t: state[t]["p_mastery"])


def _predicted_correct(p: float, difficulty: int, question_type: str) -> float:
    """Rough P(correct) from mastery and difficulty — drives adaptive pick."""
    guess = guess_for(question_type)
    slip = 0.10
    skill = p ** (1.0 + 0.25 * (difficulty - 3))
    return skill * (1 - slip) + (1 - skill) * guess


def generate(course_id: str, *, user_id: str = "demo_learner", num: int = 5,
             topics: list[str] | None = None, kinds: list[str] | None = None,
             kind: str = "quiz") -> dict:
    pool = query("SELECT * FROM questions WHERE course_id = ?", (course_id,))
    if topics:
        pool = [q for q in pool if q["topic_id"] in topics]
    if kinds:
        pool = [q for q in pool if q["question_type"] in kinds]
    if not pool:
        return {"assessment_id": None, "questions": []}

    state = mastery_map(course_id, user_id)
    target_topics = topics or _ordered_topics(course_id, user_id)

    chosen: list[dict] = []
    used: set[str] = set()
    last_topic: str | None = None
    streak = 0

    for _ in range(min(num, len(pool))):
        best, best_score = None, None
        for topic in target_topics:
            if topic == last_topic and streak >= 2:
                continue  # never ask the same topic 3x in a row
            candidates = [q for q in pool if q["topic_id"] == topic and q["id"] not in used]
            if not candidates:
                continue
            p = float(state.get(topic, {}).get("p_mastery", 0.3))
            for cand in candidates:
                predicted = _predicted_correct(p, cand.get("difficulty") or 3, cand["question_type"])
                # aim for ~0.7 predicted correctness, and prioritise weak topics
                score = abs(predicted - 0.7) - 0.35 * (1 - p)
                if best_score is None or score < best_score:
                    best, best_score = (topic, cand), score
        if best is None:
            break
        topic, question = best
        chosen.append(question)
        used.add(question["id"])
        streak = streak + 1 if topic == last_topic else 1
        last_topic = topic

    assessment_id = insert(
        "assessments",
        {
            "user_id": user_id,
            "course_id": course_id,
            "kind": kind,
            "topic_ids": jdumps(sorted({q["topic_id"] for q in chosen})),
            "status": "active",
        },
    )
    for question in chosen:
        insert("attempts", {
            "assessment_id": assessment_id,
            "user_id": user_id,
            "question_id": question["id"],
            "is_correct": None,
        })

    return {"assessment_id": assessment_id, "questions": [public(q) for q in chosen]}


def public(question: dict) -> dict:
    """Strip the answer key and misconceptions before sending to the browser."""
    options = jloads(question.get("options"), []) or []
    return {
        "id": question["id"],
        "question_text": question["question_text"],
        "question_type": question["question_type"],
        "options": [{"text": o["text"]} for o in options],
        "difficulty": question.get("difficulty") or LEVEL_OF_DIFFICULTY,
        "topic_id": question["topic_id"],
    }


def get(assessment_id: str) -> dict | None:
    assessment = query_one("SELECT * FROM assessments WHERE id = ?", (assessment_id,))
    if not assessment:
        return None
    question_ids = [
        r["question_id"]
        for r in query("SELECT question_id FROM attempts WHERE assessment_id = ?", (assessment_id,))
    ]
    questions = []
    for qid in question_ids:
        q = query_one("SELECT * FROM questions WHERE id = ?", (qid,))
        if q:
            questions.append(public(q))
    assessment["questions"] = questions
    return assessment


def _normalise(value: str) -> str:
    return " ".join(str(value or "").lower().replace(",", "").split())


def _grade_one(question: dict, answer: str) -> tuple[bool, str | None, str]:
    """Return (is_correct, misconception_name, feedback)."""
    correct_answer = _normalise(question.get("correct_answer"))
    given = _normalise(answer)
    options = jloads(question.get("options"), []) or []

    if question["question_type"] == "mcq":
        for opt in options:
            if _normalise(opt["text"]) == given:
                if opt.get("is_correct"):
                    return True, None, "Correct."
                return False, opt.get("misconception"), "Not quite — see the explanation."
        return False, None, "Your choice was not one of the listed options."

    # free-text / numerical: tolerant comparison
    if given == correct_answer or correct_answer in given or given in correct_answer:
        return True, None, "Correct."
    return False, None, "Not quite — compare with the model answer."


def submit(assessment_id: str, answers: dict[str, str]) -> dict:
    assessment = query_one("SELECT * FROM assessments WHERE id = ?", (assessment_id,))
    if not assessment:
        return {}
    course_id = assessment["course_id"]
    user_id = assessment["user_id"]

    results = []
    correct_count = 0
    topic_stats: dict[str, dict] = {}
    misconceptions: list[dict] = []

    attempt_rows = query(
        "SELECT * FROM attempts WHERE assessment_id = ? AND is_correct IS NULL",
        (assessment_id,),
    )
    for attempt in attempt_rows:
        question = query_one("SELECT * FROM questions WHERE id = ?", (attempt["question_id"],))
        if not question:
            continue
        answer = answers.get(question["id"], "")
        is_correct, misconception, feedback = _grade_one(question, answer)
        correct_count += 1 if is_correct else 0

        execute(
            """UPDATE attempts SET submitted_answer = ?, is_correct = ?, misconception = ?,
               feedback = ? WHERE id = ?""",
            (answer, 1 if is_correct else 0, misconception, feedback, attempt["id"]),
        )

        posterior = _apply(course_id, question["topic_id"], is_correct,
                           question["question_type"], misconception, user_id)

        name = query_one("SELECT name FROM concepts WHERE id = ?", (question["topic_id"],))
        topic_name = name["name"] if name else question["topic_id"]
        stat = topic_stats.setdefault(topic_name, {"topic_id": question["topic_id"],
                                                   "correct": 0, "total": 0})
        stat["total"] += 1
        stat["correct"] += 1 if is_correct else 0

        if misconception:
            misconceptions.append({"topic": topic_name, "name": misconception})

        results.append({
            "question_id": question["id"],
            "question_text": question["question_text"],
            "submitted_answer": answer,
            "correct_answer": question["correct_answer"],
            "is_correct": is_correct,
            "feedback": feedback,
            "misconception": misconception,
            "explanation": question["explanation"],
            "topic": topic_name,
            "citation": jloads(question.get("source_citation"), {}),
            "p_mastery": round(posterior, 3),
        })

    total = len(results)
    score = round(100 * correct_count / total, 1) if total else 0.0
    execute(
        "UPDATE assessments SET status = 'completed', score = ?, completed_at = CURRENT_TIMESTAMP WHERE id = ?",
        (score, assessment_id),
    )

    weak = [t["topic"] for t in topic_stats.values() if t["correct"] / t["total"] < 0.6]
    return {
        "assessment_id": assessment_id,
        "score": score,
        "correct": correct_count,
        "total": total,
        "topic_stats": list(topic_stats.values()),
        "weak_topics": weak,
        "misconceptions": misconceptions,
        "results": results,
        "recommendation": recommend(course_id, user_id),
        "next_actions": _next_actions(weak, misconceptions),
    }


def _apply(course_id: str, topic_id: str, correct: bool, question_type: str,
           misconception: str | None, user_id: str) -> float:
    from .learner import record_attempt

    return record_attempt(
        course_id, topic_id, correct,
        question_type=question_type, misconception=misconception, user_id=user_id,
    )


def _next_actions(weak: list[str], misconceptions: list[dict]) -> list[str]:
    actions: list[str] = []
    if weak:
        actions.append(f"Re-read your sources on {', '.join(weak[:2])} before the next quiz.")
    if misconceptions:
        actions.append(f"Targeted remediation for: {misconceptions[0]['name']}.")
    actions.append("Run a 5-question quiz on your weakest topic to confirm the gain.")
    return actions


def latest_report(course_id: str, user_id: str = "demo_learner") -> dict | None:
    row = query_one(
        "SELECT * FROM assessments WHERE course_id = ? AND user_id = ? AND status = 'completed' "
        "ORDER BY created_at DESC LIMIT 1",
        (course_id, user_id),
    )
    if not row:
        return None
    results = []
    topic_stats: dict[str, dict] = {}
    for attempt in query("SELECT * FROM attempts WHERE assessment_id = ?", (row["id"],)):
        question = query_one("SELECT * FROM questions WHERE id = ?", (attempt["question_id"],))
        if not question:
            continue
        name = query_one("SELECT name FROM concepts WHERE id = ?", (question["topic_id"],))
        topic_name = name["name"] if name else (question["topic_id"] or "General")
        unit_ids = jloads(question.get("source_unit_ids"), []) or []
        is_correct = bool(attempt.get("is_correct"))

        stat = topic_stats.setdefault(topic_name, {"topic": topic_name, "correct": 0, "total": 0})
        stat["total"] += 1
        stat["correct"] += 1 if is_correct else 0

        results.append({
            "question_text": question["question_text"],
            "submitted_answer": attempt.get("submitted_answer"),
            "correct_answer": question["correct_answer"],
            "is_correct": is_correct,
            "misconception": attempt.get("misconception"),
            "explanation": question["explanation"],
            "citation": jloads(question.get("source_citation"), {}),
            "topic": topic_name,
            "unit_id": unit_ids[0] if unit_ids else None,
        })

    weak = [s["topic"] for s in topic_stats.values() if s["correct"] / s["total"] < 0.6]
    mistakes = [{"name": r["misconception"]} for r in results if r["misconception"]]
    return {
        "score": row.get("score"),
        "results": results,
        "created_at": row.get("created_at"),
        "topic_stats": list(topic_stats.values()),
        "weak_topics": weak,
        "next_actions": _next_actions(weak, mistakes),
    }
