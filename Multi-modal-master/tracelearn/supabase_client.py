"""
TraceLearn — Supabase helper.

Used only by the ingestion path so uploaded media can live in Supabase
Storage instead of the local ``storage/`` directory when the project is
configured. All other code paths still read/write local paths, so this
module is optional: if it is not wired in, nothing breaks.

Environment (from config.settings):
    SUPABASE_URL
    SUPABASE_SECRET_KEY   (service role, for Storage admin writes)
    SUPABASE_ANON_KEY     (public anon, safe to ship to clients)
    SUPABASE_STORAGE_BUCKET
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from .config import settings


class SupabaseStorage:
    def __init__(self) -> None:
        self._bucket = settings.supabase_bucket
        self._base = settings.supabase_url.removesuffix("/")

    # ---------------------------------------------------------------- public
    def signed_upload(self, file_bytes: bytes, filename: str, content_type: str | None = None) -> dict[str, Any]:
        """
        Return a dict describing the stored file for the caller to persist
        into the ``sources`` row:
            {
                "storage_path": "tracelearn/<id>/<name>",
                "storage_url":  "<public or signed url>",
            }
        """
        raise NotImplementedError

    def object_url(self, path: str) -> str:
        """
        Public or signed URL for an already-stored object. ``path`` is the
        storage path as persisted in the DB (e.g. ``tracelearn/<id>/<name>``).
        """
        raise NotImplementedError


def build_storage() -> SupabaseStorage | None:
    """
    Return a storage backend when Supabase is configured. Right now this
    returns ``None`` unless you wire a concrete implementation; the caller
    then falls back to the local ``storage/`` path.
    """
    if settings.supabase_rest_ready and settings.supabase_url and (settings.supabase_key or settings.supabase_anon_key):
        # Wire the real client here once you pick the Supabase SDK / REST path.
        return None
    return None


def source_storage_path(course_id: str, name: str) -> str:
    """
    Stable logical path used both locally and as the Supabase object key.
    """
    return f"{settings.supabase_bucket}/{course_id}/{name}"


def generate_source_name(original_name: str) -> str:
    safe = Path(original_name).name or "upload"
    return f"{uuid.uuid4().hex[:8]}_{safe}"
