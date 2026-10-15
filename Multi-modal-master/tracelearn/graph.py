"""TraceLearn — concept graph construction and layout.

Nodes are topics/concepts, edges are prerequisite -> dependent. The graph is
validated as a DAG (cycles are broken by dropping the weakest edge) and laid
out in layers by longest path from the roots, so the UI can render a clean
left-to-right flow map with plain SVG — no graph library needed.
"""
from __future__ import annotations

from .db import query
from .learner import band, mastery_map, risk_propagation

NODE_W, NODE_H = 178, 64


def build(course_id: str, user_id: str = "demo_learner") -> dict:
    concepts = query(
        "SELECT * FROM concepts WHERE course_id = ? ORDER BY order_index", (course_id,)
    )
    edge_rows = query("SELECT * FROM prerequisites")
    state = mastery_map(course_id, user_id)
    risks = risk_propagation(course_id, user_id)

    known = {c["id"] for c in concepts}
    edges = [
        {
            "source": e["prerequisite_id"],
            "target": e["dependent_id"],
            "confidence": e.get("confidence") or 0.5,
        }
        for e in edge_rows
        if e["prerequisite_id"] in known and e["dependent_id"] in known
    ]

    edges = _break_cycles(edges, known)
    levels = _assign_levels(concepts, edges)
    _place(concepts, levels)

    nodes = []
    for c in concepts:
        record = state.get(c["id"])
        p = float(record["p_mastery"]) if record else 0.3
        nodes.append(
            {
                "id": c["id"],
                "name": c["name"],
                "description": c.get("description"),
                "x": c["_x"],
                "y": c["_y"],
                "level": c["_level"],
                "p_mastery": round(p, 3),
                "risk_mastery": round(risks.get(c["id"], p), 3),
                "band": band(p),
                "attempts": (record or {}).get("attempts", 0),
            }
        )

    # attach endpoint coordinates so the template can draw edges directly
    coords = {n["id"]: (n["x"], n["y"]) for n in nodes}
    for edge in edges:
        src = coords.get(edge["source"])
        dst = coords.get(edge["target"])
        if not src or not dst:
            continue
        edge["x1"], edge["y1"] = src[0] + NODE_W, src[1] + NODE_H / 2
        edge["x2"], edge["y2"] = dst[0], dst[1] + NODE_H / 2
        edge["cx"] = (edge["x1"] + edge["x2"]) / 2

    return {
        "nodes": nodes,
        "edges": edges,
        "node_w": NODE_W,
        "node_h": NODE_H,
        "width": max((n["x"] for n in nodes), default=0) + NODE_W + 60,
        "height": max((n["y"] for n in nodes), default=0) + NODE_H + 60,
    }


def _break_cycles(edges: list[dict], known: set[str]) -> list[dict]:
    """Greedy DAG repair: while a cycle exists, drop its weakest edge."""
    kept = list(edges)
    for _ in range(len(kept)):
        order = _topological(known, kept)
        if order is not None:
            return kept
        cycle = _find_cycle(known, kept)
        if not cycle:
            return kept
        cyc = set(cycle)
        candidates = [e for e in kept if e["source"] in cyc and e["target"] in cyc]
        if not candidates:
            return kept
        weakest = min(candidates, key=lambda e: e["confidence"])
        kept.remove(weakest)
    return kept


def _topological(nodes: set[str], edges: list[dict]) -> list[str] | None:
    indeg = {n: 0 for n in nodes}
    adj: dict[str, list[str]] = {n: [] for n in nodes}
    for e in edges:
        adj[e["source"]].append(e["target"])
        indeg[e["target"]] += 1
    queue = [n for n in nodes if indeg[n] == 0]
    out: list[str] = []
    while queue:
        node = queue.pop(0)
        out.append(node)
        for nxt in adj[node]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)
    return out if len(out) == len(nodes) else None


def _find_cycle(nodes: set[str], edges: list[dict]) -> list[str] | None:
    adj: dict[str, list[str]] = {n: [] for n in nodes}
    for e in edges:
        adj[e["source"]].append(e["target"])
    state: dict[str, int] = {}

    def visit(node: str, stack: list[str]) -> list[str] | None:
        state[node] = 1
        stack.append(node)
        for nxt in adj[node]:
            if state.get(nxt, 0) == 0:
                found = visit(nxt, stack)
                if found:
                    return found
            elif state.get(nxt) == 1:
                return stack[stack.index(nxt):]
        stack.pop()
        state[node] = 2
        return None

    for node in nodes:
        if state.get(node, 0) == 0:
            found = visit(node, [])
            if found:
                return found
    return None


def _assign_levels(concepts: list[dict], edges: list[dict]) -> dict[str, int]:
    indeg = {c["id"]: 0 for c in concepts}
    adj: dict[str, list[str]] = {c["id"]: [] for c in concepts}
    for e in edges:
        adj[e["source"]].append(e["target"])
        indeg[e["target"]] += 1
    level = {c["id"]: 0 for c in concepts}
    queue = [c["id"] for c in concepts if indeg[c["id"]] == 0]
    while queue:
        node = queue.pop(0)
        for nxt in adj[node]:
            level[nxt] = max(level[nxt], level[node] + 1)
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)
    return level


def _place(concepts: list[dict], levels: dict[str, int]) -> None:
    """Column per level, evenly spaced rows, vertically centred per column."""
    columns: dict[int, list[dict]] = {}
    for c in concepts:
        lvl = levels.get(c["id"], 0)
        c["_level"] = lvl
        columns.setdefault(lvl, []).append(c)

    dx, dy = 250, 108
    tallest = max((len(v) for v in columns.values()), default=1)
    for lvl, items in columns.items():
        items.sort(key=lambda c: c.get("order_index", 0))
        offset = (tallest - len(items)) * dy / 2
        for row, c in enumerate(items):
            c["_x"] = 40 + lvl * dx
            c["_y"] = 40 + offset + row * dy
