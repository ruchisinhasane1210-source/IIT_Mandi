"""TraceLearn — embeddings.

Two providers behind one function:

  * Gemini ``text-embedding-004`` when GEMINI_API_KEY is set.
  * A local **hashing vectorizer** otherwise. This is not a toy: it is the
    classic "hashing trick" (token -> bucket via a stable hash, tf weighting,
    L2 normalisation) so cosine similarity is genuinely meaningful in demo
    mode and retrieval returns relevant chunks without any API key.
"""
from __future__ import annotations

import hashlib
import math
import re

import requests

from .config import settings
from .db import get_db, query, to_embedding

_WORD = re.compile(r"[a-z0-9]+")

_STOP = {
    "the", "a", "an", "of", "to", "and", "in", "is", "are", "it", "that",
    "for", "on", "as", "with", "by", "be", "this", "at", "or", "from", "we",
    "you", "can", "if", "not", "but", "so", "then", "than", "which", "when",
}


def _tokens(text: str) -> list[str]:
    words = [w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1]
    # include bigrams so multi-word concepts ("binary heap") cohere
    bigrams = [f"{a}_{b}" for a, b in zip(words, words[1:])]
    return words + bigrams


def _hash_index(token: str, dim: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dim


def _local_embed(text: str, dim: int) -> list[float]:
    vec = [0.0] * dim
    tokens = _tokens(text or "")
    if not tokens:
        return vec
    counts: dict[str, int] = {}
    for tok in tokens:
        counts[tok] = counts.get(tok, 0) + 1
    for tok, count in counts.items():
        idx = _hash_index(tok, dim)
        # sublinear tf keeps very common words from dominating
        vec[idx] += 1.0 + math.log(count)
    norm = math.sqrt(sum(v * v for v in vec))
    if norm:
        vec = [v / norm for v in vec]
    return vec


def _gemini_embed(texts: list[str]) -> list[list[float]]:
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{settings.gemini_embed_model}:batchEmbedContents?key={settings.gemini_key}"
    )
    payload = {
        "requests": [
            {
                "model": f"models/{settings.gemini_embed_model}",
                "content": {"parts": [{"text": t or " "}]},
            }
            for t in texts
        ]
    }
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return [item["values"] for item in data["embeddings"]]


def embed(text: str) -> list[float]:
    return embed_batch([text])[0]


def embed_batch(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    if settings.gemini_key and not settings.force_demo:
        try:
            return _gemini_embed(texts)
        except Exception:
            pass  # fall through to the local provider
    return [_local_embed(t, settings.embed_dim) for t in texts]


def cosine(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(v * v for v in a[:n]))
    nb = math.sqrt(sum(v * v for v in b[:n]))
    if not na or not nb:
        return 0.0
    return dot / (na * nb)


# --------------------------------------------------------- vector storage

def save_unit_embedding(unit_id: str, vector: list[float]) -> None:
    conn, pg = get_db()
    cur = conn.cursor()
    try:
        if pg:
            cur.execute(
                "UPDATE content_units SET embedding = %s::vector WHERE id = %s",
                (to_embedding(vector, True), unit_id),
            )
        else:
            cur.execute(
                "UPDATE content_units SET embedding = ? WHERE id = ?",
                (to_embedding(vector, False), unit_id),
            )
        conn.commit()
    finally:
        cur.close()


def _parse_vector(raw) -> list[float] | None:
    """Embeddings come back as a JSON list (SQLite) or a pgvector string."""
    if raw is None:
        return None
    if isinstance(raw, list):
        return [float(x) for x in raw]
    text = str(raw).strip()
    if text.startswith("["):
        try:
            return [float(x) for x in text.strip("[]").split(",") if x]
        except ValueError:
            return None
    return None


def load_unit_embeddings(unit_ids: list[str] | None = None) -> list[tuple[str, list[float]]]:
    """Used by the SQLite fallback path: pull embeddings and score in Python."""
    statement = "SELECT id, embedding FROM content_units WHERE embedding IS NOT NULL"
    params: list = []
    if unit_ids:
        marks = ", ".join("?" for _ in unit_ids)
        statement += f" AND id IN ({marks})"
        params = list(unit_ids)
    out: list[tuple[str, list[float]]] = []
    for row in query(statement, params):
        vec = _parse_vector(row.get("embedding"))
        if vec:
            out.append((row["id"], vec))
    return out
