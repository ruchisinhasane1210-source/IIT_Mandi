"""
TraceLearn — Supabase Storage backend (REST, no SDK).

Wires up the ingestion path so uploaded media lands in Supabase Storage
instead of the local ``storage/`` directory whenever the project is
configured. Uses only stdlib + requests, which is already a dependency.

Environment (from config.settings):
    SUPABASE_URL
    SUPABASE_SECRET_KEY   (service role, for Storage admin writes)
    SUPABASE_ANON_KEY     (public anon, safe to ship to clients)
    SUPABASE_STORAGE_BUCKET
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import requests

from .config import settings


class SupabaseStorage:
    def __init__(self) -> None:
        self._bucket = settings.supabase_bucket
        self._base = settings.supabase_url.removesuffix("/")

    # ---------------------------------------------------------------- public
    def signed_upload(self, file_bytes: bytes, filename: str, content_type: str | None = None) -> dict[str, Any]:
        """
        Upload ``file_bytes`` into Supabase Storage ``bucket`` and return a
        small dict the caller can persist into the ``sources`` row:

            {
                "storage_path": "tracelearn/<course>/<file-key>",
                "storage_url":  "<signed or public url>",
            }

        The stored object key is deterministic-ish so the same source name in
        the same course always maps to the same logical path; we still prepend
        a short random prefix so distinct uploads never collide.
        """
        key = f"{self._bucket}/{generate_source_name(filename)}"
        headers = self._auth_headers()
        url = f"{self._base}/storage/v1/object/{self._bucket}/{key}"

        resp = requests.put(
            url,
            headers={**headers, "x-supabase-name": filename, "x-supabase-content-type": content_type or "application/octet-stream"},
            data=file_bytes,
            timeout=60,
        )
        if not resp.ok:
            raise RuntimeError(f"Supabase Storage upload failed: {resp.status_code} {resp.text}")

        public_url = f"{self._base}/storage/v1/object/public/{self._bucket}/{key}"
        # Try to fetch a signed URL; fall back to the public one if the bucket
        # is public. Both are acceptable for the traced media use-case.
        signed_url = self._signed_url(key)
        return {
            "storage_path": key,
            "storage_url": signed_url or public_url,
            "object_key": key,
            "public_url": public_url,
        }

    def object_url(self, path: str) -> str:
        """
        Public or signed URL for an already-stored object. ``path`` is the
        storage path as persisted in the DB (e.g. ``tracelearn/<course>/<key>``).
        """
        if path.startswith(self._bucket + "/"):
            key = path[len(self._bucket) + 1:]
        else:
            key = path
        signed = self._signed_url(key)
        if signed:
            return signed
        return f"{self._base}/storage/v1/object/public/{self._bucket}/{key}"

    # ---------------------------------------------------- internals
    def _auth_headers(self) -> dict[str, str]:
        key = settings.supabase_key or settings.supabase_anon_key or ""
        if not key:
            raise RuntimeError("Supabase key not configured")
        return {"Authorization": f"Bearer {key}", "apiKey": key}

    def _signed_url(self, key: str) -> str | None:
        """
        Ask Supabase for a short-lived signed URL. If the bucket is public or
        the anon token lacks the permission, fall back to public URL (handled
        by the caller).
        """
        try:
            url = f"{self._base}/storage/v1/object/sign/{self._bucket}/{key}?download=false"
            resp = requests.get(url, headers=self._auth_headers(), timeout=20)
            if resp.ok:
                payload = resp.json()
                return payload.get("signedUrl") or payload.get("signedurl")
        except Exception:
            pass
        return None


def build_storage() -> SupabaseStorage | None:
    """
    Return a storage backend when Supabase is configured. ``None`` otherwise,
    which tells the caller to keep using the local ``storage/`` directory.
    """
    if settings.supabase_rest_ready and settings.supabase_url and (settings.supabase_key or settings.supabase_anon_key):
        return SupabaseStorage()
    return None


def source_storage_path(course_id: str, name: str) -> str:
    return f"{settings.supabase_bucket}/{course_id}/{name}"


def generate_source_name(original_name: str) -> str:
    safe = Path(original_name).name or "upload"
    return f"{uuid.uuid4().hex[:8]}_{safe}"
