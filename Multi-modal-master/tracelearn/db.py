"""TraceLearn — database layer.

One API, two backends:

  * SQLite   (default, zero config, file at ./tracelearn.db)
  * Postgres (Supabase) as soon as DATABASE_URL is set

Queries are written once with ``?`` placeholders and ``%s`` is swapped in
for Postgres, so route/service code never branches on the engine. The
canonical ``schema.sql`` is transparently down-converted for SQLite at boot.
"""
from __future__ import annotations

import json
import re
import sqlite3
import uuid
from contextlib import contextmanager
from typing import Any, Iterable

from flask import g

from .config import settings

try:
    import psycopg2
    import psycopg2.extras

    HAVE_PG = True
except Exception:  # pragma: no cover - optional dependency
    HAVE_PG = False


# ---------------------------------------------------------------- helpers

def new_id() -> str:
    return str(uuid.uuid4())


def jloads(raw: Any, default: Any = None) -> Any:
    """Tolerant JSON decode: handles None, already-decoded values and bad JSON."""
    if raw is None or raw == "":
        return default
    if isinstance(raw, (list, dict)):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return default


def jdumps(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def to_embedding(value: list[float] | None, pg: bool) -> Any:
    """Embeddings travel as a vector literal for pgvector, JSON text for SQLite."""
    if value is None:
        return None
    if pg:
        return "[" + ",".join(f"{x:.6f}" for x in value) + "]"
    return json.dumps(value)


# ------------------------------------------------------- schema conversion

_PG_ONLY = (
    r"CREATE EXTENSION[^;]*;",
    r"CREATE INDEX[^;]*USING\s+ivfflat[^;]*;",
    r"CREATE INDEX[^;]*USING\s+GIN[^;]*;",
)


def sqlite_schema(sql: str) -> str:
    """Down-convert the canonical Postgres DDL into SQLite-compatible DDL."""
    for pattern in _PG_ONLY:
        sql = re.sub(pattern, "", sql, flags=re.I | re.S)
    replacements = (
        (r"VECTOR\(\d+\)", "TEXT"),
        (r"\bJSONB\b", "TEXT"),
        (r"\bTSVECTOR\b", "TEXT"),
        (r"\bUUID\b", "TEXT"),
        (r"\bFLOAT\b", "REAL"),
        (r"\bBOOLEAN\b", "INTEGER"),
        (r"\bNOW\(\)", "CURRENT_TIMESTAMP"),
        (r"::\w+", ""),
    )
    for pattern, repl in replacements:
        sql = re.sub(pattern, repl, sql, flags=re.I)
    return sql


# ------------------------------------------------------------- connections

def open_connection() -> tuple[Any, bool]:
    """Raw connection + whether it is Postgres. Used outside request context."""
    if settings.db_is_postgres and HAVE_PG:
        conn = psycopg2.connect(settings.database_url)
        conn.autocommit = True
        return conn, True
    conn = sqlite3.connect(settings.sqlite_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn, False


def get_db() -> tuple[Any, bool]:
    """Request-scoped connection (cached on flask.g)."""
    if "db_conn" not in g:
        g.db_conn, g.db_pg = open_connection()
    return g.db_conn, g.db_pg


def close_db(_exc: BaseException | None = None) -> None:
    conn = g.pop("db_conn", None)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass


# ----------------------------------------------------------- query helpers

def _sql(statement: str, pg: bool) -> str:
    return statement.replace("?", "%s") if pg else statement


def _cursor(conn: Any, pg: bool):
    if pg:
        return conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    return conn.cursor()


def query(statement: str, params: Iterable[Any] = ()) -> list[dict]:
    """Run a SELECT and return a list of plain dicts."""
    conn, pg = get_db()
    cur = _cursor(conn, pg)
    try:
        cur.execute(_sql(statement, pg), tuple(params))
        rows = cur.fetchall()
    finally:
        cur.close()
    if pg:
        return [dict(r) for r in rows]
    return [dict(r) for r in rows]


def query_one(statement: str, params: Iterable[Any] = ()) -> dict | None:
    rows = query(statement, params)
    return rows[0] if rows else None


def scalar(statement: str, params: Iterable[Any] = (), default: Any = None) -> Any:
    row = query_one(statement, params)
    if not row:
        return default
    return next(iter(row.values()))


def execute(statement: str, params: Iterable[Any] = ()) -> int:
    """Run an INSERT/UPDATE/DELETE and commit. Returns affected row count."""
    conn, pg = get_db()
    cur = _cursor(conn, pg)
    try:
        cur.execute(_sql(statement, pg), tuple(params))
        affected = cur.rowcount
        conn.commit()
    finally:
        cur.close()
    return affected


def insert(table: str, data: dict, *, pg: bool | None = None) -> str:
    """INSERT a dict, returning its id (generating one when absent)."""
    data = dict(data)
    data.setdefault("id", new_id())
    conn, pg_flag = get_db() if pg is None else (None, pg)
    columns = ", ".join(data.keys())
    placeholders = ", ".join("?" for _ in data)
    statement = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"

    if pg_flag:
        conn, _ = get_db()
        cur = _cursor(conn, True)
        try:
            cur.execute(_sql(statement, True), tuple(data.values()))
            conn.commit()
        finally:
            cur.close()
        return data["id"]

    execute(statement, tuple(data.values()))
    return data["id"]


# ------------------------------------------------------------------ boot

def init_db() -> dict:
    """Create every table. Idempotent: safe to call on each boot."""
    conn, pg = open_connection()
    ddl = settings.schema_path.read_text(encoding="utf-8")
    try:
        cur = conn.cursor()
        if pg:
            for chunk in ddl.split(";"):
                if chunk.strip():
                    cur.execute(chunk)
        else:
            conn.executescript(sqlite_schema(ddl))
        if not pg:
            conn.commit()
    finally:
        conn.close()
    return {"engine": "postgres" if pg else "sqlite"}


@contextmanager
def transaction():
    """Group several writes atomically."""
    conn, pg = get_db()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def health() -> dict:
    try:
        one = scalar("SELECT 1 AS ok", default=None)
        return {"ok": one == 1, "engine": settings.db_label}
    except Exception as exc:  # pragma: no cover
        return {"ok": False, "engine": settings.db_label, "error": str(exc)}
