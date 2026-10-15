"""TraceLearn — the learner model.

Bayesian Knowledge Tracing per topic (§12), plus prerequisite-aware
recommendation, misconception tracking and a simple forgetting-curve schedule.

BKT has exactly the properties the spec demands: a wrong answer *lowers*
mastery, a correct one raises it, and the posterior always stays in [0, 1].
"""
from __future__ import annotations

import math

from .db import execute, jdumps, query, query_one, scalar

# defaults (§12.1)
GUESS_MCQ = 0.20
GUESS_FREE = 0.05
SLIP = 0.10
LEARN = 0.15
PRIOR = 0.30


def bkt_update(p: float, correct: bool, guess: float = GUESS_MCQ,
               slip: float = SLIP, learn: float = LEARN) -> float:
    """One BKT step. Returns the posterior after applying learning."""
    if correct:
        post = (p * (1 - slip)) / (p * (1 - slip) + (1 - p) * guess)
    else:
        post = (p * slip) / (p * slip + (1 - p) * (1 - guess))
    return post + (1 - post) * learn


def guess_for(question_type: str | None) -> float:
    return GUESS_MCQ if (question_type or "") == "mcq" else GUESS_FREE


def band(p: float) -> str:
    if p >= 0.7:
        return "high"
    if p >= 0.4:
        return "mid"
    return "low"


# ------------------------------------------------------------ state access

def mastery_map(course_id: str, user_id: str = "demo_learner") -> dict[str, dict]:
    rows = query(
        """
        SELECT m.topic_id, m.p_mastery, m.attempts, m.correct,
               m.misconception_count, c.name
        FROM mastery_records m
        LEFT JOIN concepts c ON c.id = m.topic_id
        WHERE m.course_id = ? AND m.user_id = ?
        """,
        (course_id, user_id),
    )
    return {r["topic_id"]: r for r in rows}


def overall_mastery(course_id: str, user_id: str = "demo_learner") -> float:
    value = scalar(
        "SELECT AVG(p_mastery) AS p FROM mastery_records WHERE course_id = ? AND user_id = ?",
        (course_id, user_id),
        default=0.0,
    )
    return round(float(value or 0.0), 3)


def prerequisites_of(topic_id: str) -> list[str]:
    rows = query("SELECT prerequisite_id FROM prerequisites WHERE dependent_id = ?", (topic_id,))
    return [r["prerequisite_id"] for r in rows]


def dependents_of(topic_id: str) -> list[str]:
    rows = query("SELECT dependent_id FROM prerequisites WHERE prerequisite_id = ?", (topic_id,))
    return [r["dependent_id"] for r in rows]


# --------------------------------------------------------------- updates

def record_attempt(course_id: str, topic_id: str, correct: bool, *,
                   question_type: str = "mcq", confidence: int | None = None,
                   misconception: str | None = None,
                   user_id: str = "demo_learner") -> float:
    """Apply one observation to a topic and persist the new posterior."""
    row = query_one(
        "SELECT * FROM mastery_records WHERE course_id = ? AND user_id = ? AND topic_id = ?",
        (course_id, user_id, topic_id),
    )
    prior = float(row["p_mastery"]) if row else PRIOR
    posterior = bkt_update(prior, correct, guess=guess_for(question_type))
    posterior = max(0.0, min(1.0, posterior))

    if row:
        execute(
            """
            UPDATE mastery_records
            SET p_mastery = ?, attempts = attempts + 1,
                correct = correct + ?, misconception_count = misconception_count + ?,
                last_practiced_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (posterior, 1 if correct else 0, 1 if misconception else 0, row["id"]),
        )
    else:
        from .db import insert

        insert(
            "mastery_records",
            {
                "user_id": user_id,
                "course_id": course_id,
                "topic_id": topic_id,
                "p_mastery": posterior,
                "attempts": 1,
                "correct": 1 if correct else 0,
                "misconception_count": 1 if misconception else 0,
            },
        )

    if misconception:
        log_misconception(course_id, topic_id, misconception, user_id=user_id)

    return posterior


def log_misconception(course_id: str, topic_id: str, name: str,
                      user_id: str = "demo_learner") -> None:
    from .db import insert

    existing = query_one(
        "SELECT id, count FROM misconceptions WHERE user_id = ? AND course_id = ? AND name = ?",
        (user_id, course_id, name),
    )
    if existing:
        execute("UPDATE misconceptions SET count = count + 1 WHERE id = ?", (existing["id"],))
    else:
        insert(
            "misconceptions",
            {
                "user_id": user_id,
                "course_id": course_id,
                "topic_id": topic_id,
                "name": name,
                "count": 1,
            },
        )


def weak_evidence(course_id: str, topic_id: str, user_id: str = "demo_learner") -> None:
    """A chat turn that signals confusion: a half-weight incorrect observation."""
    row = query_one(
        "SELECT id, p_mastery FROM mastery_records WHERE course_id = ? AND user_id = ? AND topic_id = ?",
        (course_id, user_id, topic_id),
    )
    if not row:
        return
    posterior = bkt_update(float(row["p_mastery"]), False)
    # halve the step so chat never moves the model as hard as a graded answer
    blended = (float(row["p_mastery"]) + posterior) / 2
    execute("UPDATE mastery_records SET p_mastery = ? WHERE id = ?", (blended, row["id"]))


# ------------------------------------------------------ recommendations

def recommend(course_id: str, user_id: str = "demo_learner") -> dict:
    """Prerequisite-aware 'what should I study next' (§12.3).

    When a topic is weak, its weakest prerequisite wins — because practising
    the dependent topic is wasted effort until the foundation is there.
    """
    state = mastery_map(course_id, user_id)
    if not state:
        return {}

    weakest = min(state, key=lambda t: state[t]["p_mastery"])
    weak_p = state[weakest]["p_mastery"]

    prereqs = prerequisites_of(weakest)
    blocking = [
        p for p in prereqs
        if p in state and state[p]["p_mastery"] < 0.6
    ]
    if blocking:
        target = min(blocking, key=lambda t: state[t]["p_mastery"])
        return {
            "kind": "prerequisite",
            "target_topic": target,
            "target_name": state[target]["name"],
            "p_mastery": state[target]["p_mastery"],
            "blocked_topic": weakest,
            "blocked_name": state[weakest]["name"],
            "reason": (
                f"You are missing {state[target]['name']}, which "
                f"{state[weakest]['name']} depends on. Closing that gap first "
                f"will unblock the harder topic."
            ),
        }

    return {
        "kind": "practice",
        "target_topic": weakest,
        "target_name": state[weakest]["name"],
        "p_mastery": weak_p,
        "blocked_topic": None,
        "blocked_name": None,
        "reason": (
            f"{state[weakest]['name']} is your weakest topic at "
            f"{round(weak_p * 100)}% mastery. Targeted practice will move it most."
        ),
    }


def risk_propagation(course_id: str, user_id: str = "demo_learner") -> dict[str, float]:
    """Nudge dependents of weak topics so the flow map shows risk spreading."""
    state = mastery_map(course_id, user_id)
    adjusted = {t: r["p_mastery"] for t, r in state.items()}
    for topic, record in state.items():
        if record["p_mastery"] >= 0.6:
            continue
        for dep in dependents_of(topic):
            if dep in adjusted:
                gap = 0.6 - record["p_mastery"]
                adjusted[dep] = max(0.0, adjusted[dep] - 0.15 * gap)
    return adjusted


# ---------------------------------------------------------------- schedule

def study_schedule(course_id: str, user_id: str = "demo_learner",
                   days: int = 3) -> list[dict]:
    """A tiny forgetting-curve plan prioritising weak and decaying topics."""
    state = mastery_map(course_id, user_id)
    if not state:
        return []

    # priority = weakness x staleness (all topics share a flat decay here)
    scored = sorted(
        state.values(),
        key=lambda r: (1 - r["p_mastery"]) + 0.25 * math.log1p(r["attempts"]),
        reverse=True,
    )
    plan: list[dict] = []
    for i, topic in enumerate(scored):
        plan.append(
            {
                "day": f"Day {i % days + 1}",
                "topic_id": topic["topic_id"],
                "topic_name": topic["name"],
                "minutes": 20 + int((1 - topic["p_mastery"]) * 20),
                "kind": "review" if topic["p_mastery"] < 0.4 else "practice",
                "mastery": round(topic["p_mastery"], 2),
            }
        )
    return plan
