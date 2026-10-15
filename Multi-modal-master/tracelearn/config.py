"""TraceLearn — configuration loaded from the environment.

Everything here has a sane default so the app boots with no .env at all
(demo mode). See .env.example for the full list of knobs.
"""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env")
except ImportError:  # python-dotenv is optional in a bare environment
    pass


def _flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _num(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip())
    except (TypeError, ValueError):
        return default


class Settings:
    """Single source of truth for runtime configuration."""

    def __init__(self) -> None:
        self.base_dir = BASE_DIR

        # --- paths ---
        self.storage_dir = BASE_DIR / "storage"
        self.data_dir = BASE_DIR / "data"
        self.sqlite_path = BASE_DIR / "tracelearn.db"
        self.schema_path = BASE_DIR / "schema.sql"

        # --- Flask ---
        self.secret_key = os.getenv("SESSION_SECRET_KEY") or "tracelearn-dev-key"

        # --- Supabase ---
        self.database_url = os.getenv("DATABASE_URL", "").strip()
        self.supabase_url = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
        self.supabase_key = os.getenv("SUPABASE_SECRET_KEY", "").strip()
        self.supabase_anon_key = os.getenv("SUPABASE_ANON_KEY", "").strip()
        self.supabase_bucket = os.getenv("SUPABASE_STORAGE_BUCKET", "tracelearn")

        # --- Models ---
        self.gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.gemini_chat_model = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-flash")
        self.gemini_embed_model = os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004")
        self.groq_key = os.getenv("GROQ_API_KEY", "").strip()
        self.groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.embed_dim = int(_num("EMBED_DIM", 768))

        # --- Grounding thresholds, tuned on eval/test_set.json to maximise
        # refusal F1 with the local provider (unsupported <= 0.28, grounded >= 0.37).
        # Re-tune after switching to real embeddings: see eval notes in the README.
        self.t_high = _num("T_HIGH", 0.36)
        self.t_low = _num("T_LOW", 0.28)

        # --- Demo / mock ---
        self.force_demo = _flag("DEMO_MODE", False)

        self.storage_dir.mkdir(parents=True, exist_ok=True)

    # ---- derived flags -------------------------------------------------

    @property
    def llm_ready(self) -> bool:
        return bool(self.gemini_key or self.groq_key) and not self.force_demo

    @property
    def demo_mode(self) -> bool:
        """True when we have no real model to call, so we serve seeded data."""
        return not self.llm_ready

    @property
    def db_is_postgres(self) -> bool:
        return bool(self.database_url)

    @property
    def db_label(self) -> str:
        return "Supabase Postgres" if self.db_is_postgres else "SQLite (local)"

    @property
    def storage_ready(self) -> bool:
        return bool(self.supabase_url and (self.supabase_key or self.supabase_anon_key)) and not self.force_demo

    @property
    def supabase_rest_ready(self) -> bool:
        """True when the project URL + anon (or service) key is present."""
        return bool(self.supabase_url and (self.supabase_key or self.supabase_anon_key)) and not self.force_demo

    def summary(self) -> dict:
        """Small dict the UI shows in the status pill / settings page."""
        return {
            "demo_mode": self.demo_mode,
            "db": self.db_label,
            "storage": "Supabase Storage" if self.storage_ready else "Local disk",
            "llm": (
                f"Gemini · {self.gemini_chat_model}"
                if self.gemini_key
                else (f"Groq · {self.groq_model}" if self.groq_key else "Demo (no key)")
            ),
            "embeddings": (
                f"Gemini · {self.gemini_embed_model} ({self.embed_dim}d)"
                if self.gemini_key
                else f"Hash stub ({self.embed_dim}d)"
            ),
            "supabase": (
                "connected"
                if self.supabase_url and (self.supabase_key or self.supabase_anon_key)
                else "not configured"
            ),
        }


settings = Settings()
