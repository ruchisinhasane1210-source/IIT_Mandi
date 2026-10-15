"""TraceLearn — demo corpus.

Seeds a complete MIT 6.006 (DSA) course so every page renders with real-looking
data before any file is uploaded or any key is configured. ``ensure_seeded()``
is idempotent: it exits immediately once the course row exists.
"""
from __future__ import annotations

from .db import execute, insert, jdumps, new_id, query_one, scalar

COURSE_ID = "dsa_6006"
USER_ID = "demo_learner"

# ------------------------------------------------------------------ topics
# (key, name, order, description)
TOPICS = [
    ("t_analysis", "Algorithm Analysis", 0, "Asymptotics, RAM model, recurrence solving."),
    ("t_arrays", "Arrays & Dynamic Arrays", 1, "Static arrays, amortised doubling, Python lists."),
    ("t_hashing", "Hashing", 2, "Hash functions, chaining, open addressing, load factor."),
    ("t_sorting", "Sorting", 3, "Merge sort, radix sort, comparison lower bounds."),
    ("t_heaps", "Heaps & Priority Queues", 4, "Binary heaps, heapify, build-heap in O(n)."),
    ("t_graphs", "Graphs & Traversal", 5, "Representations, BFS, DFS, topological order."),
    ("t_shortest", "Shortest Paths", 6, "Dijkstra, Bellman-Ford, weighted graphs."),
    ("t_dp", "Dynamic Programming", 7, "Overlapping subproblems, memoisation, tabulation."),
]

# prerequisite -> dependent
PREREQS = [
    ("t_analysis", "t_arrays"),
    ("t_arrays", "t_hashing"),
    ("t_arrays", "t_sorting"),
    ("t_arrays", "t_heaps"),
    ("t_arrays", "t_graphs"),
    ("t_heaps", "t_shortest"),
    ("t_graphs", "t_shortest"),
    ("t_analysis", "t_dp"),
    ("t_graphs", "t_dp"),
]

# topic -> (p_mastery, attempts, correct, misconceptions, last_practiced_days_ago)
MASTERY = {
    "t_analysis": (0.82, 6, 5, 0, 2),
    "t_arrays": (0.74, 5, 4, 0, 3),
    "t_hashing": (0.41, 7, 3, 2, 1),
    "t_sorting": (0.63, 5, 3, 1, 4),
    "t_heaps": (0.35, 6, 2, 3, 1),
    "t_graphs": (0.52, 4, 2, 1, 6),
    "t_shortest": (0.22, 3, 1, 2, 8),
    "t_dp": (0.18, 2, 1, 1, 9),
}

# ---------------------------------------------------------- content units
# (key, source_key, unit_type, content, page, slide, ts_start, ts_end)
UNITS = [
    ("u_heap_1", "ocw_6006_lec08", "slide",
     "A binary heap is a complete binary tree stored implicitly in an array. "
     "Index i has children at 2i+1 and 2i+2 and parent at floor((i-1)/2). "
     "The heap property requires each parent to be smaller than its children "
     "(min-heap) or larger (max-heap).",
     None, 12, None, None),
    ("u_heap_2", "opendatastructures", "text",
     "Building a heap from an arbitrary array takes O(n), not O(n log n). "
     "The tighter bound follows from the observation that most nodes sit near "
     "the leaves and sift down only a short distance; summing the costs over all "
     "nodes telescopes to a linear total.",
     None, None, None, None),
    ("u_heap_3", "ocw_6006_lec08", "slide",
     "Figure description: the diagram shows the same binary heap drawn twice — "
     "once as a tree with parent-to-child arrows and once as a flat array. "
     "The node labelled 4 sits at index 3 in the array with children at indices "
     "8 and 9, illustrating the 2i+1 / 2i+2 indexing rule.",
     None, 18, None, None),
    ("u_hash_1", "ocw_6006_lec04", "slide",
     "Open addressing stores every element in the table itself. On collision we "
     "probe according to a fixed sequence. Linear probing has good cache "
     "behaviour but suffers primary clustering; double hashing removes that "
     "clustering but costs an extra hash computation per probe.",
     None, 7, None, None),
    ("u_hash_2", "opendatastructures", "text",
     "The load factor alpha = n/m bounds performance. With chaining, the expected "
     "probe length is 1 + alpha. With open addressing the expected number of probes "
     "for an unsuccessful search is at most 1/(1 - alpha), so the table must never "
     "become full.",
     None, None, None, None),
    ("u_sort_1", "ocw_6006_lec03", "transcript",
     "Merge sort splits the array in half, recursively sorts each half, then merges "
     "the two sorted runs in linear time. The recurrence is T of n equals 2 T of n "
     "over 2 plus theta of n, which solves to theta of n log n by the master theorem.",
     None, None, 762.0, 798.0),
    ("u_sort_2", "ocw_6006_lec03", "transcript",
     "Because radix sort does not compare keys, it beats the n log n comparison "
     "lower bound. Sorting n integers of b bits in r-bit digits takes "
     "theta of n times b over r passes, each pass linear.",
     None, None, 1810.0, 1849.0),
    ("u_graph_1", "opendatastructures", "text",
     "Breadth-first search explores a graph in layers by distance from the source. "
     "On an unweighted graph the first time BFS reaches a vertex it has found a "
     "shortest path, because every edge contributes exactly one unit of distance.",
     None, None, None, None),
    ("u_graph_2", "opendatastructures", "text",
     "Dijkstra's algorithm generalises BFS to weighted graphs by always extracting "
     "the unsettled vertex of minimum tentative distance from a priority queue. "
     "A binary heap gives O((V + E) log V) overall.",
     None, None, None, None),
    ("u_short_1", "ocw_6006_lec03", "slide",
     "Dijkstra requires non-negative edge weights. A single negative edge can make "
     "the greedy extract-min step finalise a distance that a later path would have "
     "improved. Bellman-Ford relaxes every edge V-1 times and tolerates negatives, "
     "at O(V times E).",
     None, None, None, None),
    ("u_dp_1", "opendatastructures", "text",
     "Dynamic programming applies when a problem has optimal substructure and "
     "overlapping subproblems. Memoisation caches subproblem results top-down; "
     "tabulation fills a table bottom-up. Both turn exponential recursion into "
     "polynomial time by solving each subproblem once.",
     None, None, None, None),
    ("u_analysis_1", "opendatastructures", "text",
     "Big-O upper-bounds growth; big-Theta bounds it tightly from both sides. "
     "Dropping constant factors is valid because asymptotics describe behaviour "
     "as n grows without bound, not runtime on one particular machine.",
     None, None, None, None),
]

# ---------------------------------------------------------------- questions
# topic -> list of question dicts
QUESTIONS = [
    {
        "topic": "t_heaps",
        "type": "mcq",
        "difficulty": 3,
        "text": "Why can a binary heap be built from an arbitrary array in O(n) rather than O(n log n)?",
        "options": [
            ("Because most nodes are near the leaves and sift down only a short distance.", True, None),
            ("Because heapify skips the leaves entirely and leaves are half the array.", False,
             "Confuses counting nodes with counting work — skipping leaves does not by itself give linear time."),
            ("Because each sift-down is O(1) in the worst case.", False,
             "Believes sift-down is constant time — it is O(log n) in the worst case."),
            ("Because the array is already partially sorted after insertion.", False,
             "Assumes build-heap relies on pre-existing order — it works on any array."),
        ],
        "answer": "Because most nodes are near the leaves and sift down only a short distance.",
        "explanation": "The sum of sift-down heights over all nodes telescopes to a linear total.",
        "units": ["u_heap_2", "u_heap_1"],
        "citation": {"slide_number": 12, "source_id": "ocw_6006_lec08"},
    },
    {
        "topic": "t_heaps",
        "type": "mcq",
        "difficulty": 2,
        "text": "In an array-backed binary heap, where are the children of the node at index i?",
        "options": [
            ("At indices 2i+1 and 2i+2.", True, None),
            ("At indices 2i and 2i+1.", False,
             "Off-by-one indexing — the 1-based formulas applied to a 0-based array."),
            ("At indices i/2 and i/2+1.", False,
             "Confuses the child relation with the parent relation."),
            ("At indices i+1 and i+2.", False,
             "Believes children are adjacent in the array."),
        ],
        "answer": "At indices 2i+1 and 2i+2.",
        "explanation": "0-based implicit layout: left child 2i+1, right child 2i+2.",
        "units": ["u_heap_1", "u_heap_3"],
        "citation": {"slide_number": 18, "source_id": "ocw_6006_lec08"},
    },
    {
        "topic": "t_heaps",
        "type": "short_answer",
        "difficulty": 2,
        "text": "State the heap property for a min-heap.",
        "options": [],
        "answer": "Every parent is less than or equal to both of its children.",
        "explanation": "Min-heap: parent <= children, so the minimum sits at the root.",
        "units": ["u_heap_1"],
        "citation": {"slide_number": 12, "source_id": "ocw_6006_lec08"},
    },
    {
        "topic": "t_hashing",
        "type": "mcq",
        "difficulty": 3,
        "text": "Which collision strategy avoids primary clustering?",
        "options": [
            ("Double hashing.", True, None),
            ("Linear probing.", False, "Believes linear probing avoids clustering — it is the classic cause of it."),
            ("Separate chaining with a good hash.", False,
             "Confuses chaining (which sidesteps probing) with fixing clustering in open addressing."),
            ("Increasing the table size.", False,
             "Believes clustering is a capacity problem rather than a probe-sequence problem."),
        ],
        "answer": "Double hashing.",
        "explanation": "Double hashing spreads probes across the table, breaking up runs.",
        "units": ["u_hash_1"],
        "citation": {"slide_number": 7, "source_id": "ocw_6006_lec04"},
    },
    {
        "topic": "t_hashing",
        "type": "numerical",
        "difficulty": 3,
        "text": "With open addressing and load factor alpha = 0.5, what is the expected number of probes for an unsuccessful search? Give a number.",
        "options": [],
        "answer": "2",
        "explanation": "1/(1 - alpha) = 1/0.5 = 2.",
        "units": ["u_hash_2"],
        "citation": {"page_number": 84, "source_id": "opendatastructures"},
    },
    {
        "topic": "t_sorting",
        "type": "mcq",
        "difficulty": 2,
        "text": "What does the merge-sort recurrence T(n) = 2T(n/2) + Θ(n) solve to?",
        "options": [
            ("Θ(n log n).", True, None),
            ("Θ(n).", False, "Drops the log factor — the merge cost at each level adds up."),
            ("Θ(n²).", False, "Confuses merge sort with quadratic insertion/bubble sort."),
            ("Θ(log n).", False, "Confuses the depth of recursion with the total work."),
        ],
        "answer": "Θ(n log n).",
        "explanation": "log n levels, each costing Θ(n) to merge.",
        "units": ["u_sort_1"],
        "citation": {"timestamp_start": 762.0, "source_id": "ocw_6006_lec03"},
    },
    {
        "topic": "t_sorting",
        "type": "short_answer",
        "difficulty": 4,
        "text": "Why can radix sort beat the Ω(n log n) comparison lower bound?",
        "options": [],
        "answer": "Because it does not compare keys; it distributes them by digit.",
        "explanation": "The lower bound applies only to comparison-based sorts.",
        "units": ["u_sort_2"],
        "citation": {"timestamp_start": 1810.0, "source_id": "ocw_6006_lec03"},
    },
    {
        "topic": "t_graphs",
        "type": "mcq",
        "difficulty": 2,
        "text": "Why does BFS find shortest paths on an unweighted graph?",
        "options": [
            ("It explores vertices in layers by distance, so the first time it reaches a vertex the path is shortest.", True, None),
            ("Because it uses a stack to backtrack optimally.", False,
             "Confuses BFS's queue with DFS's stack."),
            ("Because unweighted graphs have no cycles.", False,
             "Believes unweighted implies acyclic — the two are unrelated."),
            ("Because it sorts the edges by weight first.", False,
             "Imports Dijkstra's idea into an algorithm that ignores weights."),
        ],
        "answer": "It explores vertices in layers by distance, so the first time it reaches a vertex the path is shortest.",
        "explanation": "Each edge contributes exactly one unit of distance in BFS's layer order.",
        "units": ["u_graph_1"],
        "citation": {"source_id": "opendatastructures", "page_number": 92},
    },
    {
        "topic": "t_shortest",
        "type": "mcq",
        "difficulty": 3,
        "text": "What is the key limitation of Dijkstra's algorithm?",
        "options": [
            ("It requires non-negative edge weights.", True, None),
            ("It cannot handle disconnected graphs.", False,
             "Believes unreachable vertices break the algorithm — it simply leaves them unsettled."),
            ("It only works on trees.", False, "Confuses shortest paths with tree traversal."),
            ("It requires all edge weights to be equal.", False,
             "Believes it cannot handle weights at all — that is what makes it useful."),
        ],
        "answer": "It requires non-negative edge weights.",
        "explanation": "A negative edge can invalidate the greedy extract-min finalisation.",
        "units": ["u_short_1", "u_graph_2"],
        "citation": {"slide_number": 22, "source_id": "ocw_6006_lec03"},
    },
    {
        "topic": "t_shortest",
        "type": "numerical",
        "difficulty": 4,
        "text": "Dijkstra with a binary heap runs in O((V + E) log V). For V = 1000 and E = 5000, which term dominates?",
        "options": [],
        "answer": "E log V",
        "explanation": "E log V ≈ 5000 × 10 = 50000 versus V log V ≈ 10000, so the edge term dominates.",
        "units": ["u_graph_2"],
        "citation": {"source_id": "opendatastructures"},
    },
    {
        "topic": "t_dp",
        "type": "mcq",
        "difficulty": 2,
        "text": "Which two properties must a problem have for dynamic programming to apply?",
        "options": [
            ("Optimal substructure and overlapping subproblems.", True, None),
            ("Sorted input and distinct keys.", False, "Imports requirements from searching and sorting."),
            ("Non-negative weights and a DAG.", False, "Imports requirements from shortest paths."),
            ("Amortised operations and a suitable potential function.", False,
             "Confuses DP with amortised analysis."),
        ],
        "answer": "Optimal substructure and overlapping subproblems.",
        "explanation": "Overlap makes caching worthwhile; optimal substructure makes the recursion correct.",
        "units": ["u_dp_1"],
        "citation": {"source_id": "opendatastructures"},
    },
    {
        "topic": "t_analysis",
        "type": "mcq",
        "difficulty": 1,
        "text": "Why is dropping constant factors valid in asymptotic analysis?",
        "options": [
            ("Because asymptotics describe growth as n becomes large, not runtime on one machine.", True, None),
            ("Because constants are always negligible in practice.", False,
             "Believes constants never matter — they can dominate at realistic input sizes."),
            ("Because big-O is a lower bound on runtime.", False,
             "Confuses big-O (upper bound) with big-Omega."),
            ("Because CPUs normalise instruction costs.", False, "Invents a hardware claim."),
        ],
        "answer": "Because asymptotics describe growth as n becomes large, not runtime on one machine.",
        "explanation": "Constants and lower-order terms are dominated as n -> infinity.",
        "units": ["u_analysis_1"],
        "citation": {"source_id": "opendatastructures"},
    },
]


def is_seeded() -> bool:
    return query_one("SELECT id FROM courses WHERE id = ?", (COURSE_ID,)) is not None


def ensure_seeded() -> bool:
    """Insert the demo course once. Returns True if it inserted anything."""
    if is_seeded():
        return False

    insert("users", {"id": USER_ID, "name": "Demo Learner", "preferred_language": "English"})
    insert(
        "courses",
        {
            "id": COURSE_ID,
            "owner_id": USER_ID,
            "title": "MIT 6.006 — Introduction to Algorithms",
            "description": "Six lectures on data structures and algorithms, ingested from "
                           "MIT OpenCourseWare plus the Open Data Structures textbook.",
            "exam_date": "2026-10-25",
            "daily_study_minutes": 45,
        },
    )

    # topics + prerequisite graph
    for key, name, order, description in TOPICS:
        insert(
            "concepts",
            {
                "id": key,
                "course_id": COURSE_ID,
                "name": name,
                "level": "topic",
                "description": description,
                "order_index": order,
            },
        )
    for prereq, dependent in PREREQS:
        execute(
            "INSERT INTO prerequisites (prerequisite_id, dependent_id, confidence) VALUES (?, ?, ?)",
            (prereq, dependent, 0.8),
        )

    # mastery + misconceptions
    for topic, (p, attempts, correct, misc, days) in MASTERY.items():
        insert(
            "mastery_records",
            {
                "user_id": USER_ID,
                "course_id": COURSE_ID,
                "topic_id": topic,
                "p_mastery": p,
                "attempts": attempts,
                "correct": correct,
                "misconception_count": misc,
                "last_practiced_at": None,
            },
        )

    for topic, label in (
        ("t_heaps", "Confuses build-heap with repeated insertion"),
        ("t_heaps", "Off-by-one in implicit array indexing (1-based vs 0-based)"),
        ("t_hashing", "Believes linear probing avoids clustering"),
        ("t_shortest", "Applies Dijkstra to graphs with negative edges"),
        ("t_dp", "Confuses dynamic programming with amortised analysis"),
    ):
        insert(
            "misconceptions",
            {
                "user_id": USER_ID,
                "course_id": COURSE_ID,
                "topic_id": topic,
                "name": label,
                "count": 2,
            },
        )

    # sources
    source_ids = {
        "ocw_6006_lec03": ("Lecture 3: Sorting", "video", "ocw_6006_lec03.mp4", "video/mp4"),
        "ocw_6006_lec04": ("Lecture 4: Hashing", "slide", "lec04_hashing.pdf", "application/pdf"),
        "ocw_6006_lec08": ("Lecture 8: Binary Heaps", "slide", "lec08_heaps.pdf", "application/pdf"),
        "opendatastructures": ("Open Data Structures", "pdf", "opendatastructures.pdf", "application/pdf"),
    }
    for sid, (title, stype, fname, mime) in source_ids.items():
        insert(
            "sources",
            {
                "id": sid,
                "course_id": COURSE_ID,
                "source_id": sid,
                "file_name": fname,
                "source_type": stype,
                "mime_type": mime,
                "license": "CC BY-NC-SA 4.0",
                "attribution": f"MIT OpenCourseWare, 6.006 — {title}",
                "processing_status": "done",
                "progress": 100,
                "duration_seconds": 5000.0 if stype == "video" else None,
                "num_pages": 320 if stype == "pdf" else None,
            },
        )

    # content units (the citable layer)
    unit_topics = {
        "u_heap_1": ["t_heaps"], "u_heap_2": ["t_heaps"], "u_heap_3": ["t_heaps"],
        "u_hash_1": ["t_hashing"], "u_hash_2": ["t_hashing"],
        "u_sort_1": ["t_sorting"], "u_sort_2": ["t_sorting"],
        "u_graph_1": ["t_graphs"], "u_graph_2": ["t_graphs", "t_shortest"],
        "u_short_1": ["t_shortest"], "u_dp_1": ["t_dp"], "u_analysis_1": ["t_analysis"],
    }
    for key, src, utype, content, page, slide, ts, te in UNITS:
        insert(
            "content_units",
            {
                "id": key,
                "source_id": src,
                "course_id": COURSE_ID,
                "content": content,
                "unit_type": utype,
                "page_number": page,
                "slide_number": slide,
                "timestamp_start": ts,
                "timestamp_end": te,
                "topic_ids": jdumps(unit_topics.get(key, [])),
                "concept_ids": jdumps(unit_topics.get(key, [])),
                "metadata": jdumps({}),
            },
        )

    # Embed every seeded unit so retrieval works with no API key at all.
    from .embeddings import embed_batch, save_unit_embedding

    keys = [u[0] for u in UNITS]
    texts = [u[3] for u in UNITS]
    for unit_key, vector in zip(keys, embed_batch(texts)):
        save_unit_embedding(unit_key, vector)

    # questions
    for q in QUESTIONS:
        options = [
            {"text": text, "is_correct": correct, "misconception": misc}
            for text, correct, misc in q["options"]
        ]
        insert(
            "questions",
            {
                "course_id": COURSE_ID,
                "question_text": q["text"],
                "question_type": q["type"],
                "options": jdumps(options),
                "correct_answer": q["answer"],
                "explanation": q["explanation"],
                "difficulty": q["difficulty"],
                "topic_id": q["topic"],
                "source_unit_ids": jdumps(q["units"]),
                "source_citation": jdumps(q["citation"]),
                "verification_status": "verified",
                "verification_log": jdumps(
                    {"evidence": "pass", "blind_solve": "pass", "ambiguity": "pass", "dedupe": "pass"}
                ),
                "question_hash": new_id()[:12],
            },
        )

    # a short study plan so the mastery page has something to show
    plan = [
        ("day1", "t_heaps", 25, "review", "Weakest core topic and a prerequisite for shortest paths."),
        ("day1", "t_shortest", 20, "practice", "Depends on heaps; risk of blocked progress."),
        ("day2", "t_hashing", 20, "flashcards", "Misconception recorded — targeted remediation."),
        ("day3", "t_dp", 25, "review", "Lowest mastery overall; exam weight is high."),
    ]
    for date, topic, minutes, kind, reason in plan:
        insert(
            "study_plan_items",
            {
                "user_id": USER_ID,
                "course_id": COURSE_ID,
                "plan_date": date,
                "topic_id": topic,
                "minutes": minutes,
                "kind": kind,
                "reason": reason,
            },
        )

    return True


def reset() -> None:
    """Drop the demo rows (used by the /api/dev/reset endpoint)."""
    for table in (
        "study_plan_items", "misconceptions", "mastery_records", "attempts",
        "assessments", "questions", "content_units", "prerequisites",
        "concepts", "sources", "courses", "users",
    ):
        execute(f"DELETE FROM {table}")


def counts() -> dict:
    def c(table: str) -> int:
        return scalar(f"SELECT COUNT(*) AS n FROM {table}", default=0) or 0

    return {
        "units": c("content_units"),
        "topics": c("concepts"),
        "questions": c("questions"),
        "sources": c("sources"),
    }
