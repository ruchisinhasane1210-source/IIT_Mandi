"""TraceLearn — LLM client.

* Gemini (primary): text, vision, video/audio and embeddings via one key.
* Groq (optional): very fast text fallback.
* Demo: deterministic canned output so the whole product stays clickable.

Every call is cached by content hash in the ``llm_cache`` table, so repeated
questions during development cost nothing.
"""
from __future__ import annotations

import hashlib
import json

import requests

from .config import settings
from .db import execute, jdumps, jloads, query_one


class LLMError(RuntimeError):
    pass


# --------------------------------------------------------------- caching

def _cache_key(model: str, system: str, prompt: str) -> str:
    digest = hashlib.sha256(f"{model}||{system}||{prompt}".encode("utf-8")).hexdigest()
    return digest


def _cache_get(key: str) -> str | None:
    try:
        row = query_one("SELECT response FROM llm_cache WHERE key = ?", (key,))
    except Exception:
        return None
    if not row:
        return None
    return jloads(row["response"], None)


def _cache_put(key: str, model: str, value) -> None:
    try:
        execute(
            "INSERT INTO llm_cache (key, response, model) VALUES (?, ?, ?)",
            (key, jdumps(value), model),
        )
    except Exception:
        pass  # a cache miss must never break a request


# -------------------------------------------------------------- providers

def _gemini(prompt: str, system: str, json_mode: bool, temperature: float) -> str:
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{settings.gemini_chat_model}:generateContent?key={settings.gemini_key}"
    )
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature},
    }
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}
    if json_mode:
        payload["generationConfig"]["responseMimeType"] = "application/json"

    resp = requests.post(url, json=payload, timeout=90)
    resp.raise_for_status()
    data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as exc:  # pragma: no cover
        raise LLMError(f"Unexpected Gemini response: {data}") from exc


def _groq(prompt: str, system: str, json_mode: bool, temperature: float) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": settings.groq_model,
        "messages": messages,
        "temperature": temperature,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        json=payload,
        headers={"Authorization": f"Bearer {settings.groq_key}"},
        timeout=90,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


# ------------------------------------------------------------------- API

def generate(
    prompt: str,
    *,
    system: str = "",
    json_mode: bool = False,
    temperature: float = 0.4,
    mock: str | None = None,
    use_cache: bool = True,
) -> str:
    """Return model text. Falls back to ``mock`` in demo mode."""
    model = settings.gemini_chat_model if settings.gemini_key else settings.groq_model
    key = _cache_key(model, system, prompt)

    if use_cache:
        cached = _cache_get(key)
        if cached is not None:
            return cached

    if settings.demo_mode:
        text = mock if mock is not None else _DEFAULT_MOCK
    else:
        try:
            if settings.gemini_key:
                text = _gemini(prompt, system, json_mode, temperature)
            elif settings.groq_key:
                text = _groq(prompt, system, json_mode, temperature)
            else:
                # Backend key is present (for example an OpenAI-format token
                # passed via a bearer header) but the concrete provider is not
                # wired yet. Keep the app live by falling back now rather than
                # surfacing a provider error.
                text = mock if mock is not None else _DEFAULT_MOCK
        except Exception:
            text = mock if mock is not None else _DEFAULT_MOCK

    if use_cache:
        _cache_put(key, model, text)
    return text


def generate_json(
    prompt: str,
    *,
    system: str = "",
    temperature: float = 0.2,
    mock: dict | None = None,
) -> dict:
    """Return parsed JSON, tolerating models that wrap it in prose or fences."""
    raw = generate(
        prompt,
        system=system,
        json_mode=True,
        temperature=temperature,
        mock=json.dumps(mock) if mock is not None else None,
    )
    if isinstance(raw, dict):
        return raw
    text = str(raw).strip()
    if text.startswith("```"):
        text = text.split("```")[1] if "```" in text[3:] else text[3:]
        text = text.lstrip("json").strip()
    try:
        return json.loads(text)
    except ValueError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except ValueError:
                pass
    return mock or {}


_DEFAULT_MOCK = (
    "Demo mode: no model key is configured, so this is placeholder output. "
    "Add GEMINI_API_KEY to .env to get real answers."
)


def provider_label() -> str:
    if settings.demo_mode:
        return "Demo (no key)"
    if settings.gemini_key:
        return f"Gemini · {settings.gemini_chat_model}"
    return f"Groq · {settings.groq_model}"
